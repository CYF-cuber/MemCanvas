#!/usr/bin/env python3
"""
Full-dataset accuracy evaluation for Memory Canvas with compressed memory images.

Outputs results and report under /home/cyf/codex.
"""

import argparse
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

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


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def compress_bytes(orig_bytes: bytes, save_kwargs: Dict) -> bytes:
    img = Image.open(io.BytesIO(orig_bytes))
    img = ensure_rgb(img)
    out = io.BytesIO()
    img.save(out, **save_kwargs)
    return out.getvalue()


def format_supported(fmt: str) -> bool:
    """Check if PIL can encode the format by doing a small in-memory save."""
    img = Image.new("RGB", (2, 2), (255, 255, 255))
    out = io.BytesIO()
    try:
        img.save(out, format=fmt)
    except Exception:
        return False
    return True


def build_output_dir(root: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(root) / f"compression_full_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_memory_index(path: str) -> Tuple[List, np.ndarray, int]:
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
        question_text = (problem.get("question", "") + " " + problem.get("hint", "")).strip()
        query_embedding = text_encoder.encode_text_query(question_text)
        retrieved = mem_index.search(query_embedding, top_k=top_k, threshold=threshold)

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
        correct_answer = choice_labels[answer_idx] if answer_idx < len(choice_labels) else "A"

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
        correct_answer = choice_labels[answer_idx] if answer_idx < len(choice_labels) else "A"

        retrieved = []
        for mem_idx, sim in retrieved_map[pid]:
            retrieved.append((memories[mem_idx], sim))

        pred = vlm.predict(problem, retrieved_memories=retrieved)
        if pred == correct_answer:
            correct += 1
        total += 1

    acc = (correct / total) * 100 if total else 0.0
    return {"correct": correct, "total": total, "accuracy": acc}


def compress_all(memories: List, label: str, save_kwargs: Dict) -> Tuple[List[bytes], Dict]:
    original_total = 0
    compressed_total = 0
    out_bytes: List[bytes] = []

    for mem in tqdm(memories, desc=f"Compress {label}"):
        b = mem.canvas_image_bytes
        original_total += len(b)
        if label == "PNG":
            out_bytes.append(b)
            compressed_total += len(b)
            continue
        cb = compress_bytes(b, save_kwargs)
        out_bytes.append(cb)
        compressed_total += len(cb)

    avg_kb = (compressed_total / len(memories)) / 1024 if memories else 0
    ratio = (compressed_total / original_total) if original_total else 0
    stats = {
        "format": label,
        "original_total_bytes": original_total,
        "compressed_total_bytes": compressed_total,
        "avg_kb": avg_kb,
        "ratio": ratio,
    }
    return out_bytes, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Full evaluation for compressed memory images")
    parser.add_argument("--memory-index", default=DEFAULT_MEMORY_INDEX)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-test-samples", type=int, default=0, help="0 means full test set")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--formats", nargs="+", default=["png", "webp-85", "avif-50"],
                        help="Formats: png, webp-85, webp-75, webp-lossless, avif-50, avif-35")

    args = parser.parse_args()

    output_dir = build_output_dir(args.output_root)
    report_path = output_dir / "compression_full_report.md"
    results_path = output_dir / "compression_full_results.json"
    config_path = output_dir / "config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Output dir: {output_dir}")

    # Load memory index
    print(f"Loading memory index: {args.memory_index}")
    memories, embeddings, embedding_dim = load_memory_index(args.memory_index)

    mem_index = MemoryIndex(embedding_dim=embedding_dim)
    mem_index.memories = memories
    mem_index.embeddings = embeddings

    # Load test split
    loader = ScienceQADataLoader()
    test_data = loader.get_split("test")
    try:
        test_pids = sorted(test_data.keys(), key=lambda x: int(x))
    except Exception:
        test_pids = sorted(test_data.keys())

    if args.max_test_samples and args.max_test_samples > 0:
        test_pids = test_pids[: args.max_test_samples]

    print(f"Test samples: {len(test_pids)}")

    # Initialize models
    config = ExperimentConfig(
        max_memories_to_retrieve=args.top_k,
        similarity_threshold=args.threshold,
    )
    print("Loading CLIP text encoder...")
    text_encoder = CLIPLargeMemoryBuilder(config)
    print("Loading VLM evaluator...")
    vlm = Qwen25VLEvaluator(config)

    # Precompute retrieval list for each test pid
    retrieved_map = precompute_queries_and_retrieval(
        test_pids, test_data, text_encoder, mem_index, args.top_k, args.threshold
    )

    # Baseline eval (no memory)
    baseline = eval_baseline(vlm, test_pids, test_data)

    format_configs = {
        "png": ("PNG", {"format": "PNG", "optimize": True}),
        "webp-85": ("WEBP-85", {"format": "WEBP", "quality": 85, "method": 6}),
        "webp-75": ("WEBP-75", {"format": "WEBP", "quality": 75, "method": 6}),
        "webp-lossless": ("WEBP-lossless", {"format": "WEBP", "lossless": True, "method": 6}),
        "avif-50": ("AVIF-50", {"format": "AVIF", "quality": 50}),
        "avif-35": ("AVIF-35", {"format": "AVIF", "quality": 35}),
    }

    results = {
        "baseline": baseline,
        "formats": {},
    }

    original_bytes = [m.canvas_image_bytes for m in memories]

    for fmt in args.formats:
        if fmt not in format_configs:
            print(f"Unknown format: {fmt}, skipping")
            continue
        label, save_kwargs = format_configs[fmt]

        if label != "PNG" and not format_supported(save_kwargs["format"]):
            print(f"Format not supported by PIL: {label}, skipping")
            continue

        print(f"\n=== Format: {label} ===")
        compressed_bytes, stats = compress_all(memories, label, save_kwargs)

        # Swap bytes for evaluation
        for m, b in zip(memories, compressed_bytes):
            m.canvas_image_bytes = b

        mem_result = eval_memory_format(label, memories, vlm, test_pids, test_data, retrieved_map)

        results["formats"][label] = {
            "size_stats": stats,
            "accuracy": mem_result,
        }

    # Restore original bytes
    for m, b in zip(memories, original_bytes):
        m.canvas_image_bytes = b

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Report
    lines = []
    lines.append("# Compression Full Evaluation Report")
    lines.append("")
    lines.append(f"Memory index: {args.memory_index}")
    lines.append(f"Test samples: {len(test_pids)}")
    lines.append(f"Top-K: {args.top_k}")
    lines.append(f"Threshold: {args.threshold}")
    lines.append("")
    lines.append("## Baseline (no memory)")
    lines.append(f"Accuracy: {baseline['accuracy']:.2f}% ({baseline['correct']}/{baseline['total']})")
    lines.append("")
    lines.append("## Memory accuracy by format")
    lines.append("| Format | Avg KB | Ratio vs PNG | Accuracy | Correct/Total |")
    lines.append("|---|---:|---:|---:|---:|")

    for label, data in results["formats"].items():
        stats = data["size_stats"]
        acc = data["accuracy"]
        ratio_pct = stats["ratio"] * 100 if stats.get("ratio") else 0
        lines.append(
            f"| {label} | {stats['avg_kb']:.1f} | {ratio_pct:.2f}% | {acc['accuracy']:.2f}% | {acc['correct']}/{acc['total']} |"
        )

    report_text = "\n".join(lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nResults saved: {results_path}")
    print(f"Report saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
