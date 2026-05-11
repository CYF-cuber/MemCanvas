#!/usr/bin/env python3
"""
SVG vector graphics evaluation for Memory Canvas on ScienceQA.

Tests whether storing canvas content as SVG (vector graphics) instead of
raster formats (PNG/WebP/AVIF) can significantly reduce storage while
maintaining VLM inference accuracy.

Key insight:
- Text-only canvases: SVG stores text as XML elements, much smaller than rendered pixels
- Image-containing canvases: SVG must embed raster data as base64, potentially larger

Pipeline:
1. Load existing memory index (with CLIP embeddings)
2. Load ScienceQA training data to re-render as SVG
3. Re-render each canvas as SVG from original data
4. Measure SVG sizes vs original PNG
5. Rasterize SVGs using cairosvg for VLM evaluation
6. Compare accuracy

Outputs results and report under /home/cyf/codex.
"""

import argparse
import base64
import io
import json
import os
import pickle
import sys
import time
import xml.sax.saxutils as saxutils
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import cairosvg

# Import experiment components (pickle requires MemoryEntry class).
sys.path.insert(0, "/home/cyf/memory/memory_canvas/experiments")
from scienceqa_qwen25vl_full_experiment import (  # noqa: E402
    MemoryIndex,
    MemoryEntry,  # noqa: F401
    CLIPLargeMemoryBuilder,
    Qwen25VLEvaluator,
    ExperimentConfig,
    ScienceQADataLoader,
)

DEFAULT_MEMORY_INDEX = "/home/cyf/memory/experiments/scienceqa_qwen25vl_full/memory_index_qwen25vl_full.pkl"
DEFAULT_OUTPUT_ROOT = "/home/cyf/codex"

# ScienceQA paths
SCIENCEQA_DIR = "/home/cyf/memory/ScienceQA"
IMAGES_DIR = f"{SCIENCEQA_DIR}/data/scienceqa/images"
CAPTIONS_PATH = f"{SCIENCEQA_DIR}/data/captions.json"


def xml_escape(text: str) -> str:
    """Escape text for SVG XML."""
    return saxutils.escape(text)


class SVGCanvasRenderer:
    """Renders ScienceQA problem content as SVG, mirroring DynamicCanvas layout.

    Produces an SVG document with the same visual structure as the PIL-based
    DynamicCanvas: title, separators, context/hint, embedded image, question,
    choices, lecture, and solution.
    """

    def __init__(
        self,
        patch_size: int = 640,
        font_size: int = 18,
        padding: int = 25,
        content_gap: int = 12,
    ):
        self.patch_size = patch_size
        self.font_size = font_size
        self.padding = padding
        self.content_gap = content_gap
        self.content_width = patch_size - 2 * padding

    def _estimate_text_lines(self, text: str, font_size: int) -> List[str]:
        """Wrap text by estimating character widths.

        CJK characters ≈ font_size px wide, ASCII ≈ 0.6 × font_size.
        """
        lines: List[str] = []
        for para in text.split("\n"):
            if not para.strip():
                lines.append("")
                continue
            current = ""
            width = 0.0
            for ch in para:
                cw = font_size if ord(ch) > 127 else font_size * 0.6
                if width + cw <= self.content_width:
                    current += ch
                    width += cw
                else:
                    if current:
                        lines.append(current)
                    current = ch
                    width = cw
            if current:
                lines.append(current)
        return lines

    def render_problem_svg(
        self,
        pid: str,
        problem: Dict,
        caption: str,
        image_path: Optional[str],
    ) -> bytes:
        """Render a ScienceQA problem as SVG bytes."""
        elements: List[str] = []
        cursor_y = self.padding

        def add_text(text: str, fs: int = None, color: str = "black"):
            nonlocal cursor_y
            fs = fs or self.font_size
            line_height = int(fs * 1.3)
            for line in self._estimate_text_lines(text, fs):
                if not line:
                    cursor_y += line_height
                    continue
                baseline_y = cursor_y + fs
                escaped = xml_escape(line)
                elements.append(
                    f'<text x="{self.padding}" y="{baseline_y}" '
                    f'font-size="{fs}px" '
                    f'font-family="WenQuanYi Zen Hei, DejaVu Sans, sans-serif" '
                    f'fill="{color}">{escaped}</text>'
                )
                cursor_y += line_height
            cursor_y += self.content_gap

        def add_separator():
            nonlocal cursor_y
            y = cursor_y + 10
            elements.append(
                f'<line x1="{self.padding}" y1="{y}" '
                f'x2="{self.patch_size - self.padding}" y2="{y}" '
                f'stroke="rgb(200,200,200)" stroke-width="1"/>'
            )
            cursor_y += 20

        def add_image(img_path: str, max_h: int = 280):
            nonlocal cursor_y
            try:
                img = Image.open(img_path)
            except Exception:
                return

            max_w = self.content_width
            img_w, img_h = img.size
            scale = min(max_w / img_w, max_h / img_h, 1.0)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            if scale < 1.0:
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Ensure RGB
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            x = self.padding + (self.content_width - new_w) // 2
            elements.append(
                f'<image x="{x}" y="{cursor_y}" width="{new_w}" height="{new_h}" '
                f'xlink:href="data:image/png;base64,{b64}"/>'
            )
            cursor_y += new_h + self.content_gap

        # === Render canvas content (mirrors render_to_canvas in experiment) ===
        subject = problem.get("subject", "Unknown")
        topic = problem.get("topic", "Unknown")
        add_text(f"Q#{pid}: {subject} - {topic}", fs=22)
        add_separator()

        hint = problem.get("hint", "")
        if hint:
            add_text(f"[Context] {hint}", fs=16)

        if image_path:
            add_image(image_path, max_h=280)

        if caption:
            add_text(f"[Image] {caption}", fs=16)

        add_separator()

        question = problem.get("question", "")
        add_text(f"Question: {question}", fs=18)

        choices = problem.get("choices", [])
        answer_idx = problem.get("answer", -1)
        choice_labels = ["A", "B", "C", "D", "E", "F"]
        for i, choice in enumerate(choices):
            label = choice_labels[i] if i < len(choice_labels) else str(i)
            marker = "\u2713" if i == answer_idx else "\u25CB"
            add_text(f"  {marker} {label}. {choice}", fs=16)

        lecture = problem.get("lecture", "")
        if lecture:
            add_separator()
            add_text("Background:", fs=18)
            add_text(lecture[:500], fs=14)

        solution = problem.get("solution", "")
        if solution:
            add_separator()
            add_text("Solution:", fs=18)
            add_text(solution[:300], fs=14)

        # Build final SVG
        total_height = max(cursor_y + self.padding, self.patch_size)
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{self.patch_size}" height="{total_height}">',
            f'<rect width="{self.patch_size}" height="{total_height}" fill="white"/>',
        ]
        svg_parts.extend(elements)
        svg_parts.append("</svg>")
        return "\n".join(svg_parts).encode("utf-8")


def build_output_dir(root: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(root) / f"svg_eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_memory_index(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    memories = data["memories"]
    embeddings = data["embeddings"]
    embedding_dim = data.get("embedding_dim", 768)
    return memories, embeddings, embedding_dim


def precompute_queries_and_retrieval(
    test_pids: List[str],
    test_data: Dict,
    text_encoder: CLIPLargeMemoryBuilder,
    mem_index: MemoryIndex,
    top_k: int,
    threshold: float,
) -> Dict[str, List[Tuple[int, float]]]:
    """Return mapping pid -> list of (memory_index, similarity)."""
    retrieved_map: Dict[str, List[Tuple[int, float]]] = {}
    mem_id_to_index = {id(m): i for i, m in enumerate(mem_index.memories)}

    for pid in tqdm(test_pids, desc="Precompute retrieval"):
        problem = test_data[pid]
        question_text = (
            problem.get("question", "") + " " + problem.get("hint", "")
        ).strip()
        query_embedding = text_encoder.encode_text_query(question_text)
        retrieved = mem_index.search(
            query_embedding, top_k=top_k, threshold=threshold
        )

        filtered = []
        for m, s in retrieved:
            if m.pid == pid:
                continue
            filtered.append((mem_id_to_index[id(m)], s))
        retrieved_map[pid] = filtered
    return retrieved_map


def eval_baseline(
    vlm: Qwen25VLEvaluator,
    test_pids: List[str],
    test_data: Dict,
) -> Dict:
    choice_labels = ["A", "B", "C", "D", "E", "F"]
    correct = 0
    total = 0
    for pid in tqdm(test_pids, desc="Eval baseline"):
        problem = test_data[pid]
        answer_idx = problem.get("answer", 0)
        correct_answer = (
            choice_labels[answer_idx] if answer_idx < len(choice_labels) else "A"
        )
        pred = vlm.predict(problem, retrieved_memories=None)
        if pred == correct_answer:
            correct += 1
        total += 1
    acc = (correct / total) * 100 if total else 0.0
    return {"correct": correct, "total": total, "accuracy": acc}


def eval_memory_format(
    label: str,
    memories: List,
    vlm: Qwen25VLEvaluator,
    test_pids: List[str],
    test_data: Dict,
    retrieved_map: Dict[str, List[Tuple[int, float]]],
) -> Dict:
    choice_labels = ["A", "B", "C", "D", "E", "F"]
    correct = 0
    total = 0
    for pid in tqdm(test_pids, desc=f"Eval {label}"):
        problem = test_data[pid]
        answer_idx = problem.get("answer", 0)
        correct_answer = (
            choice_labels[answer_idx] if answer_idx < len(choice_labels) else "A"
        )

        retrieved = []
        for mem_idx, sim in retrieved_map[pid]:
            retrieved.append((memories[mem_idx], sim))

        pred = vlm.predict(problem, retrieved_memories=retrieved)
        if pred == correct_answer:
            correct += 1
        total += 1
    acc = (correct / total) * 100 if total else 0.0
    return {"correct": correct, "total": total, "accuracy": acc}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SVG vector graphics evaluation for Memory Canvas on ScienceQA"
    )
    parser.add_argument("--memory-index", default=DEFAULT_MEMORY_INDEX)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--max-test-samples", type=int, default=0, help="0 means full test set"
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline eval (use previous result: 78.73%%)",
    )
    args = parser.parse_args()

    output_dir = build_output_dir(args.output_root)
    results_path = output_dir / "svg_eval_results.json"
    report_path = output_dir / "svg_eval_report.md"
    config_path = output_dir / "config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Output dir: {output_dir}")

    # ── Load memory index ──
    print(f"Loading memory index: {args.memory_index}")
    memories, embeddings, embedding_dim = load_memory_index(args.memory_index)

    mem_index = MemoryIndex(embedding_dim=embedding_dim)
    mem_index.memories = memories
    mem_index.embeddings = embeddings
    print(f"Loaded {len(memories)} memories")

    # ── Load ScienceQA data ──
    loader = ScienceQADataLoader()
    train_data = loader.get_split("train")
    test_data = loader.get_split("test")

    captions: Dict[str, str] = {}
    if os.path.exists(CAPTIONS_PATH):
        with open(CAPTIONS_PATH, "r", encoding="utf-8") as f:
            captions = json.load(f).get("captions", {})

    # Test pids
    try:
        test_pids = sorted(test_data.keys(), key=lambda x: int(x))
    except Exception:
        test_pids = sorted(test_data.keys())

    if args.max_test_samples and args.max_test_samples > 0:
        test_pids = test_pids[: args.max_test_samples]

    print(f"Test samples: {len(test_pids)}")

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Re-render all memories as SVG
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Phase 1: Re-render memories as SVG")
    print("=" * 60)

    svg_renderer = SVGCanvasRenderer(
        patch_size=640, font_size=18, padding=25, content_gap=12
    )

    svg_bytes_list: List[bytes] = []
    png_total = 0
    svg_total = 0
    svg_total_text_only = 0
    svg_total_with_image = 0
    count_text_only = 0
    count_with_image = 0
    failed = 0

    for mem in tqdm(memories, desc="Render SVG"):
        pid = mem.pid
        problem = train_data.get(pid)

        if problem is None:
            # Fallback: keep original PNG bytes
            svg_bytes_list.append(mem.canvas_image_bytes)
            png_total += len(mem.canvas_image_bytes)
            svg_total += len(mem.canvas_image_bytes)
            failed += 1
            continue

        caption = captions.get(pid, "")
        image_path = loader.get_image_path(pid, problem)

        try:
            svg_b = svg_renderer.render_problem_svg(pid, problem, caption, image_path)
        except Exception as e:
            print(f"\nWarn: SVG render failed for {pid}: {e}")
            svg_bytes_list.append(mem.canvas_image_bytes)
            png_total += len(mem.canvas_image_bytes)
            svg_total += len(mem.canvas_image_bytes)
            failed += 1
            continue

        svg_bytes_list.append(svg_b)
        png_total += len(mem.canvas_image_bytes)
        svg_total += len(svg_b)

        if mem.has_image:
            svg_total_with_image += len(svg_b)
            count_with_image += 1
        else:
            svg_total_text_only += len(svg_b)
            count_text_only += 1

    print(f"\nSVG rendering complete:")
    print(f"  Total memories: {len(memories)}")
    print(f"  Failed: {failed}")
    print(
        f"  PNG total: {png_total / 1024 / 1024:.1f} MB, "
        f"avg: {png_total / len(memories) / 1024:.1f} KB"
    )
    print(
        f"  SVG total: {svg_total / 1024 / 1024:.1f} MB, "
        f"avg: {svg_total / len(memories) / 1024:.1f} KB"
    )
    print(f"  Ratio vs PNG: {svg_total / png_total * 100:.2f}%")
    if count_text_only > 0:
        print(
            f"  Text-only ({count_text_only}): "
            f"avg {svg_total_text_only / count_text_only / 1024:.1f} KB"
        )
    if count_with_image > 0:
        print(
            f"  With image ({count_with_image}): "
            f"avg {svg_total_with_image / count_with_image / 1024:.1f} KB"
        )

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Rasterize SVGs to PNG bytes for VLM evaluation
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Phase 2: Rasterize SVGs")
    print("=" * 60)

    rasterized_png_list: List[bytes] = []
    raster_failed = 0

    for svg_b, mem in tqdm(
        zip(svg_bytes_list, memories), total=len(memories), desc="Rasterize SVG"
    ):
        # If we didn't render SVG (failed), use original PNG
        if svg_b is mem.canvas_image_bytes:
            rasterized_png_list.append(mem.canvas_image_bytes)
            continue
        try:
            png_b = cairosvg.svg2png(bytestring=svg_b)
            rasterized_png_list.append(png_b)
        except Exception as e:
            print(f"\nWarn: Rasterization failed for {mem.pid}: {e}")
            rasterized_png_list.append(mem.canvas_image_bytes)
            raster_failed += 1

    print(f"Rasterization complete. Failed: {raster_failed}")

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Evaluation
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Phase 3: Evaluation")
    print("=" * 60)

    config = ExperimentConfig(
        max_memories_to_retrieve=args.top_k,
        similarity_threshold=args.threshold,
    )
    print("Loading CLIP text encoder...")
    text_encoder = CLIPLargeMemoryBuilder(config)
    print("Loading VLM evaluator...")
    vlm = Qwen25VLEvaluator(config)

    # Precompute retrieval
    retrieved_map = precompute_queries_and_retrieval(
        test_pids, test_data, text_encoder, mem_index, args.top_k, args.threshold
    )

    # Baseline (no memory)
    if args.skip_baseline:
        print("Skipping baseline eval (using previous result: 78.73%)")
        baseline = {"correct": 3339, "total": 4241, "accuracy": 78.73143126621079}
    else:
        baseline = eval_baseline(vlm, test_pids, test_data)

    # Original PNG accuracy
    original_bytes_backup = [m.canvas_image_bytes for m in memories]

    print("\n--- Evaluating PNG (original) ---")
    png_result = eval_memory_format(
        "PNG (original)", memories, vlm, test_pids, test_data, retrieved_map
    )

    # Swap to rasterized SVG bytes for evaluation
    print("\n--- Evaluating SVG (rasterized) ---")
    for m, rb in zip(memories, rasterized_png_list):
        m.canvas_image_bytes = rb

    svg_result = eval_memory_format(
        "SVG (rasterized)", memories, vlm, test_pids, test_data, retrieved_map
    )

    # Restore original bytes
    for m, ob in zip(memories, original_bytes_backup):
        m.canvas_image_bytes = ob

    # ══════════════════════════════════════════════════════════════
    # Results & Report
    # ══════════════════════════════════════════════════════════════
    results = {
        "baseline": baseline,
        "png": {
            "accuracy": png_result,
            "size_stats": {
                "total_bytes": png_total,
                "avg_kb": png_total / len(memories) / 1024,
            },
        },
        "svg": {
            "accuracy": svg_result,
            "size_stats": {
                "total_bytes": svg_total,
                "avg_kb": svg_total / len(memories) / 1024,
                "ratio_vs_png": svg_total / png_total if png_total else 0,
                "text_only_count": count_text_only,
                "text_only_avg_kb": (
                    svg_total_text_only / count_text_only / 1024
                    if count_text_only
                    else 0
                ),
                "with_image_count": count_with_image,
                "with_image_avg_kb": (
                    svg_total_with_image / count_with_image / 1024
                    if count_with_image
                    else 0
                ),
            },
        },
        "render_stats": {
            "total_memories": len(memories),
            "svg_render_failures": failed,
            "rasterization_failures": raster_failed,
        },
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Build Markdown report
    svg_ratio_pct = svg_total / png_total * 100 if png_total else 0
    lines = [
        "# SVG Vector Graphics Evaluation Report",
        "",
        f"Memory index: `{args.memory_index}`",
        f"Total memories: {len(memories)}",
        f"Test samples: {len(test_pids)}",
        f"Top-K: {args.top_k}, Threshold: {args.threshold}",
        f"SVG render failures: {failed}, Rasterization failures: {raster_failed}",
        "",
        "## Storage Size Comparison",
        "",
        "| Format | Total (MB) | Avg (KB) | Ratio vs PNG |",
        "|--------|----------:|---------:|-------------:|",
        f"| PNG (original) | {png_total / 1024 / 1024:.1f} | "
        f"{png_total / len(memories) / 1024:.1f} | 100.00% |",
        f"| SVG | {svg_total / 1024 / 1024:.1f} | "
        f"{svg_total / len(memories) / 1024:.1f} | {svg_ratio_pct:.2f}% |",
        "",
        "### SVG Size by Content Type",
        "",
    ]

    if count_text_only > 0:
        lines.append(
            f"- **Text-only** canvases ({count_text_only}): "
            f"avg {svg_total_text_only / count_text_only / 1024:.1f} KB"
        )
    if count_with_image > 0:
        lines.append(
            f"- **With image** canvases ({count_with_image}): "
            f"avg {svg_total_with_image / count_with_image / 1024:.1f} KB"
        )

    lines.extend(
        [
            "",
            "## Accuracy Comparison",
            "",
            "| Method | Accuracy | Correct / Total |",
            "|--------|--------:|:---------------:|",
            f"| Baseline (no memory) | {baseline['accuracy']:.2f}% | "
            f"{baseline['correct']}/{baseline['total']} |",
            f"| PNG (original) | {png_result['accuracy']:.2f}% | "
            f"{png_result['correct']}/{png_result['total']} |",
            f"| SVG (rasterized) | {svg_result['accuracy']:.2f}% | "
            f"{svg_result['correct']}/{svg_result['total']} |",
            "",
            "## Cross-format Comparison (from previous experiments)",
            "",
            "| Format | Avg KB | Accuracy |",
            "|--------|------:|--------:|",
            "| PNG | 81.1 | 80.71% |",
            "| WebP-85 | 47.0 | 80.64% |",
            "| AVIF-50 | 30.1 | 80.48% |",
            f"| **SVG** | **{svg_total / len(memories) / 1024:.1f}** | "
            f"**{svg_result['accuracy']:.2f}%** |",
        ]
    )

    report_text = "\n".join(lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nResults saved: {results_path}")
    print(f"Report saved: {report_path}")

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"Baseline:      {baseline['accuracy']:.2f}%")
    print(
        f"PNG (original): {png_result['accuracy']:.2f}% "
        f"(avg {png_total / len(memories) / 1024:.1f} KB)"
    )
    print(
        f"SVG:            {svg_result['accuracy']:.2f}% "
        f"(avg {svg_total / len(memories) / 1024:.1f} KB, "
        f"{svg_ratio_pct:.2f}% of PNG)"
    )
    if count_text_only > 0:
        print(
            f"  Text-only SVG: avg {svg_total_text_only / count_text_only / 1024:.1f} KB"
        )
    if count_with_image > 0:
        print(
            f"  With-image SVG: avg {svg_total_with_image / count_with_image / 1024:.1f} KB"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
