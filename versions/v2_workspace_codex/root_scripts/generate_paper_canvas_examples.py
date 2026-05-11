#!/usr/bin/env python3
"""
Generate 3 canvas examples per benchmark using the enhanced rendering pipeline.

Uses add_html() for formatted text and add_tall_image() for tall infographics.
Outputs to /home/cyf/codex/paper_canvas_examples/<benchmark>/
"""

import gzip
import io
import json
import os
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/cyf/memory")
from memory_canvas.dynamic_canvas import DynamicCanvas, DynamicCanvasConfig

OUTPUT_DIR = Path("/home/cyf/codex/paper_canvas_examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def combine_patches(canvas: DynamicCanvas) -> Image.Image:
    """Combine canvas patches into a compact image (no whitespace waste)."""
    return canvas.get_compact_image()


def save_canvas(img: Image.Image, path: Path):
    """Save canvas image and print info."""
    img.save(str(path))
    w, h = img.size
    kb = path.stat().st_size / 1024
    print(f"  Saved: {path.name} ({w}x{h}, {kb:.1f} KB)")


def new_canvas(patch_size=640, font_size=16, padding=20, content_gap=10) -> DynamicCanvas:
    return DynamicCanvas(DynamicCanvasConfig(
        patch_size=patch_size, font_size=font_size,
        padding=padding, content_gap=content_gap,
        show_patch_boundary=False,
    ))


# ===========================================================================
# 1. ScienceQA
# ===========================================================================
def generate_scienceqa():
    print("\n=== ScienceQA (3 examples) ===")
    out = OUTPUT_DIR / "scienceqa"
    out.mkdir(exist_ok=True)

    # Load memory index
    sys.path.insert(0, "/home/cyf/memory/memory_canvas/experiments")
    from scienceqa_qwen25vl_full_experiment import MemoryEntry, ScienceQADataLoader
    # Make pickle happy: MemoryEntry was pickled from __main__
    import __main__
    __main__.MemoryEntry = MemoryEntry

    pkl_path = "/home/cyf/memory/experiments/scienceqa_qwen25vl_full/memory_index_qwen25vl_full.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    memories = data["memories"]

    loader = ScienceQADataLoader()
    train_data = loader.get_split("train")

    # Select 3 diverse examples (different subjects, with images)
    priority = ["natural science", "social science", "language science"]
    subjects_seen = set()
    selected = []
    for mem in memories:
        if mem.pid not in train_data:
            continue
        problem = train_data[mem.pid]
        subject = problem.get("subject", "unknown")
        has_image = problem.get("image") is not None and problem.get("image") != ""
        if subject not in subjects_seen and mem.canvas_image_bytes and has_image:
            subjects_seen.add(subject)
            selected.append((mem, problem))
            if len(selected) >= 3:
                break

    for i, (mem, problem) in enumerate(selected):
        canvas = new_canvas(font_size=16, padding=22, content_gap=10)

        subject = problem.get("subject", "")
        topic = problem.get("topic", "")
        pid = mem.pid

        # Header
        canvas.add_html(f"**Q#{pid}:** {subject} — {topic}", content_type="markdown")
        canvas.add_separator()

        # Hint / context
        hint = problem.get("hint", "")
        if hint:
            canvas.add_html(f"*Context:* {hint}", content_type="markdown")

        # Image
        img_path = f"/home/cyf/memory/ScienceQA/data/scienceqa/images/train/{pid}/image.png"
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            canvas.add_image(img, max_height=280)

        canvas.add_separator()

        # Question + choices
        question = problem.get("question", "")
        canvas.add_html(f"**Question:** {question}", content_type="markdown")

        choices = problem.get("choices", [])
        answer_idx = problem.get("answer", -1)
        labels = ["A", "B", "C", "D", "E", "F"]
        choice_lines = []
        for ci, ch in enumerate(choices):
            label = labels[ci] if ci < len(labels) else str(ci)
            marker = "\u2713" if ci == answer_idx else "\u25cb"
            choice_lines.append(f"  {marker} **{label}.** {ch}")
        canvas.add_html("\n\n".join(choice_lines), content_type="markdown")

        # Lecture / solution
        lecture = problem.get("lecture", "")
        if lecture:
            canvas.add_separator()
            canvas.add_html(f"**Background:**\n\n{lecture[:400]}", content_type="markdown")

        solution = problem.get("solution", "")
        if solution:
            canvas.add_separator()
            canvas.add_html(f"**Solution:**\n\n{solution[:250]}", content_type="markdown")

        img_out = combine_patches(canvas)
        fname = f"scienceqa_{i+1}_{subject.replace(' ', '_')}_{topic.replace(' ', '_')}.png"
        save_canvas(img_out, out / fname)


# ===========================================================================
# 2. OK-VQA
# ===========================================================================
def generate_okvqa():
    print("\n=== OK-VQA (3 examples) ===")
    out = OUTPUT_DIR / "okvqa"
    out.mkdir(exist_ok=True)

    with open("/home/cyf/codex/okvqa_data/okvqa_cached.pkl", "rb") as f:
        data = pickle.load(f)
    train_samples = data["train"]

    # Pick 3 samples that have images
    selected = []
    for s in train_samples[:200]:
        if os.path.exists(s.get("image_path", "")):
            selected.append(s)
            if len(selected) >= 3:
                break

    for i, sample in enumerate(selected):
        canvas = new_canvas(font_size=16, padding=22, content_gap=10)

        # Header
        qid = sample.get("question_id", "")
        canvas.add_html(f"**[OK-VQA]** QID: {qid}", content_type="markdown")
        canvas.add_separator()

        # Image
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
            canvas.add_image(img, max_height=350)
        except Exception:
            pass

        # Caption
        caption = sample.get("caption", "")
        if caption:
            canvas.add_html(f"*Caption:* {caption}", content_type="markdown")

        canvas.add_separator()

        # Question
        canvas.add_html(f"**Q:** {sample['question']}", content_type="markdown")

        # Answers
        answers = sample.get("answers", [])
        if answers:
            counts = Counter(a.strip().lower() for a in answers)
            top = counts.most_common(1)[0][0]
            others = [a for a, c in counts.most_common(3)[1:] if c >= 2]
            ans_md = f"\u2713 **A:** {top}"
            if others:
                ans_md += f"\n\nAlso accepted: {', '.join(others)}"
            canvas.add_html(ans_md, content_type="markdown")

        img_out = combine_patches(canvas)
        save_canvas(img_out, out / f"okvqa_{i+1}_qid{qid}.png")


# ===========================================================================
# 3. HotpotQA
# ===========================================================================
def generate_hotpotqa():
    print("\n=== HotpotQA (3 examples) ===")
    out = OUTPUT_DIR / "hotpotqa"
    out.mkdir(exist_ok=True)

    with open("/home/cyf/codex/hotpotqa_data/hotpotqa_meta.pkl", "rb") as f:
        data = pickle.load(f)
    train_data = data["train"]

    # Pick 3: one comparison, one bridge-easy, one bridge-hard
    targets = [("comparison", None), ("bridge", "easy"), ("bridge", "hard")]
    selected = []
    used_types = set()
    for sample in train_data[:500]:
        key = (sample.get("type", ""), sample.get("level", ""))
        for t_type, t_level in targets:
            if key[0] == t_type and (t_level is None or key[1] == t_level):
                tag = f"{t_type}_{t_level or 'any'}"
                if tag not in used_types:
                    used_types.add(tag)
                    selected.append(sample)
                    break
        if len(selected) >= 3:
            break

    for i, sample in enumerate(selected):
        canvas = new_canvas(font_size=14, padding=20, content_gap=8)

        qtype = sample.get("type", "")
        level = sample.get("level", "")

        # Header
        canvas.add_html(
            f"**[HotpotQA]** Type: {qtype} | Level: {level}",
            content_type="markdown",
        )
        canvas.add_separator()

        # Supporting context
        sf_titles = set(t for t, _ in sample.get("supporting_facts", []))
        context_parts = []
        for para in sample.get("paragraphs", []):
            title = para["title"]
            if title in sf_titles:
                text = para["text"][:300]
                if len(para["text"]) > 300:
                    text += "..."
                context_parts.append(f"**[{title}]**\n\n{text}")

        if context_parts:
            canvas.add_html("\n\n".join(context_parts), content_type="markdown")

        canvas.add_separator()

        # Q & A
        canvas.add_html(f"**Q:** {sample['question']}", content_type="markdown")
        canvas.add_html(f"\u2713 **A:** {sample['answer']}", content_type="markdown")

        img_out = combine_patches(canvas)
        save_canvas(img_out, out / f"hotpotqa_{i+1}_{qtype}_{level}.png")


# ===========================================================================
# 4. InfographicVQA
# ===========================================================================
def generate_infographicvqa():
    print("\n=== InfographicVQA (3 examples) ===")
    out = OUTPUT_DIR / "infographicvqa"
    out.mkdir(exist_ok=True)

    with open("/home/cyf/codex/infographicvqa_data/infographicvqa_meta.pkl", "rb") as f:
        data = pickle.load(f)
    train_data = data["train"]

    # Pick 3 with images of varying sizes
    img_dir = Path("/home/cyf/codex/infographicvqa_data/images_train")
    selected = []
    for sample in train_data[:100]:
        ip = sample.get("image_path", "")
        if ip and os.path.exists(ip):
            selected.append(sample)
            if len(selected) >= 3:
                break

    for i, sample in enumerate(selected):
        canvas = new_canvas(font_size=16, padding=20, content_gap=10)

        # Header
        canvas.add_html("**[InfographicVQA]** Training Example", content_type="markdown")
        canvas.add_separator()

        # Infographic: smart segmented cropping
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
            w, h = img.size
            print(f"  Sample {i+1}: image {w}x{h}, aspect={h/w:.2f}")
            canvas.add_tall_image(img, max_sections=3, overlap=50)
        except Exception as e:
            print(f"  Warning: image load failed: {e}")

        canvas.add_separator()

        # Q & A
        canvas.add_html(f"**Q:** {sample['question']}", content_type="markdown")
        answers = sample.get("answers", [])
        if answers:
            ans = answers[0] if isinstance(answers, list) else str(answers)
            canvas.add_html(f"\u2713 **A:** {ans}", content_type="markdown")

        img_out = combine_patches(canvas)
        save_canvas(img_out, out / f"infographicvqa_{i+1}.png")


# ===========================================================================
# 5. MMQA
# ===========================================================================
def generate_mmqa():
    print("\n=== MMQA (3 examples) ===")
    out = OUTPUT_DIR / "mmqa"
    out.mkdir(exist_ok=True)

    with open("/home/cyf/codex/mmqa_data/mmqa_parsed.pkl", "rb") as f:
        parsed = pickle.load(f)

    train_data = parsed["train"]
    tables = parsed["tables"]
    texts = parsed["texts"]
    images_meta = parsed["images"]
    images_dir = Path("/home/cyf/codex/mmqa_data/final_dataset_images")

    def load_mmqa_image(doc_id: str) -> Optional[Image.Image]:
        if doc_id not in images_meta:
            return None
        path = images_dir / images_meta[doc_id]["path"]
        if not path.exists():
            return None
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    def table_to_list(table_data, max_rows=6):
        headers = [h["column_name"] for h in table_data["header"]]
        rows = []
        for row in table_data["table_rows"][:max_rows]:
            cells = [cell["text"][:40] for cell in row]
            rows.append(cells)
        return headers, rows

    def table_to_markdown(table_data, max_rows=5):
        header = [h["column_name"] for h in table_data["header"]]
        rows = [[cell["text"] for cell in row] for row in table_data["table_rows"]]
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows[:max_rows]:
            cells = [c[:50] for c in row]
            lines.append("| " + " | ".join(cells) + " |")
        if len(rows) > max_rows:
            lines.append(f"... ({len(rows)} rows total)")
        return "\n".join(lines)

    # Pick 3 diverse types: TextQ, TableQ, ImageQ (or Compose)
    target_types = ["TextQ", "TableQ", "ImageQ", "Compose(TextQ,TableQ)",
                    "Compose(ImageQ,TableQ)", "ImageListQ"]
    type_examples = {}
    for sample in train_data[:2000]:
        qtype = sample["metadata"]["type"]
        if qtype not in type_examples:
            type_examples[qtype] = sample
        if len(type_examples) >= 6:
            break

    # Select 3 diverse
    picked = []
    for t in target_types:
        if t in type_examples and len(picked) < 3:
            picked.append(type_examples[t])
    # Fill if needed
    if len(picked) < 3:
        for t, s in type_examples.items():
            if s not in picked:
                picked.append(s)
            if len(picked) >= 3:
                break

    for i, sample in enumerate(picked):
        canvas = new_canvas(font_size=14, padding=20, content_gap=8)

        modalities = sample["metadata"]["modalities"]
        qtype = sample["metadata"]["type"]

        # Header
        canvas.add_html(
            f"**[{qtype}]** Modalities: {', '.join(modalities)}",
            content_type="markdown",
        )
        canvas.add_separator()

        # Context sections
        for ctx in sample.get("supporting_context", []):
            doc_id = ctx["doc_id"]
            doc_part = ctx["doc_part"]

            if doc_part == "text" and doc_id in texts:
                text_doc = texts[doc_id]
                title = text_doc.get("title", "")
                passage = text_doc.get("text", "")[:300]
                canvas.add_html(f"**[Text] {title}**\n\n{passage}", content_type="markdown")

            elif doc_part == "table" and doc_id in tables:
                table_doc = tables[doc_id]
                title = table_doc.get("title", "")
                canvas.add_html(f"**[Table] {title}**", content_type="markdown")
                try:
                    headers, rows = table_to_list(table_doc["table"], max_rows=6)
                    if headers and rows:
                        canvas.add_table(rows, headers=headers)
                except Exception:
                    md = table_to_markdown(table_doc["table"], max_rows=5)
                    canvas.add_html(md, content_type="markdown")

            elif doc_part == "image":
                img = load_mmqa_image(doc_id)
                title = images_meta.get(doc_id, {}).get("title", "")
                if img is not None:
                    canvas.add_html(f"**[Image] {title}**", content_type="markdown")
                    canvas.add_image(img, max_height=300)
                else:
                    canvas.add_html(
                        f"**[Image] {title}** *(not available)*",
                        content_type="markdown",
                    )

        canvas.add_separator()

        # Q & A
        canvas.add_html(f"**Q:** {sample['question']}", content_type="markdown")
        answers = sample.get("answers", [])
        if answers:
            ans_str = str(answers[0]["answer"])
            canvas.add_html(f"\u2713 **A:** {ans_str}", content_type="markdown")

        img_out = combine_patches(canvas)
        safe_type = qtype.replace("(", "_").replace(")", "_").replace(",", "_")
        save_canvas(img_out, out / f"mmqa_{i+1}_{safe_type}.png")


# ===========================================================================
# 6. LoCoMo
# ===========================================================================
def generate_locomo():
    print("\n=== LoCoMo (3 examples) ===")
    out = OUTPUT_DIR / "locomo"
    out.mkdir(exist_ok=True)

    with open("/home/cyf/codex/datasets/locomo/locomo10.json") as f:
        locomo_data = json.load(f)

    count = 0
    for item in locomo_data[:3]:
        sample_id = item["sample_id"]
        conversation = item.get("conversation", {})

        # Find first session with dialog turns
        for key in sorted(conversation.keys()):
            if key.endswith("_date_time"):
                continue
            turns = conversation[key]
            if not isinstance(turns, list) or not turns:
                continue

            # Build a canvas from the first few turns of this session
            canvas = new_canvas(font_size=16, padding=22, content_gap=10)

            canvas.add_html(
                f"**[LoCoMo]** {sample_id} | {key}",
                content_type="markdown",
            )
            canvas.add_separator()

            # Render dialog turns
            dialog_md = []
            for turn in turns[:6]:
                speaker = turn.get("speaker", "?")
                dia_id = turn.get("dia_id", "")
                text = turn.get("text", "")
                dialog_md.append(f"**{speaker}** ({dia_id}): {text}")

            canvas.add_html("\n\n".join(dialog_md), content_type="markdown")

            if len(turns) > 6:
                canvas.add_html(
                    f"*... ({len(turns)} turns total)*", content_type="markdown"
                )

            img_out = combine_patches(canvas)
            count += 1
            save_canvas(
                img_out,
                out / f"locomo_{count}_{sample_id}_{key}.png",
            )
            break  # one session per conversation

        if count >= 3:
            break


# ===========================================================================
# 7. MSR-VTT
# ===========================================================================
def generate_msrvtt():
    print("\n=== MSR-VTT (3 examples) ===")
    out = OUTPUT_DIR / "msrvtt"
    out.mkdir(exist_ok=True)

    # Load annotation for captions
    ann_path = "/home/cyf/codex/datasets/msrvtt/MSRVTT/annotation/MSR_VTT.json"
    with open(ann_path) as f:
        ann = json.load(f)

    # Build video_id -> caption map
    vid_captions = {}
    for sent in ann.get("sentences", []):
        vid = sent["video_id"]
        if vid not in vid_captions:
            vid_captions[vid] = sent["caption"]

    videos_dir = Path("/home/cyf/codex/datasets/msrvtt/MSRVTT/videos/all")

    # Pick 3 test videos
    test_vids = ["video7010", "video7050", "video7100"]
    for i, vid_id in enumerate(test_vids):
        vid_path = videos_dir / f"{vid_id}.mp4"
        if not vid_path.exists():
            print(f"  Skipping {vid_id}: video not found")
            continue

        # Extract 4 keyframes using ffmpeg
        frames = extract_frames(str(vid_path), n_frames=4)
        if not frames:
            print(f"  Skipping {vid_id}: no frames extracted")
            continue

        caption = vid_captions.get(vid_id, "")

        canvas = new_canvas(font_size=16, padding=20, content_gap=10)

        # Header
        canvas.add_html(
            f"**[MSR-VTT]** {vid_id}", content_type="markdown"
        )
        canvas.add_separator()

        # Frame grid: 2x2
        grid = make_frame_grid(frames, frame_size=(290, 163))
        canvas.add_image(grid, max_height=400)

        canvas.add_separator()

        # Caption
        if caption:
            canvas.add_html(f"**Caption:** {caption}", content_type="markdown")

        img_out = combine_patches(canvas)
        save_canvas(img_out, out / f"msrvtt_{i+1}_{vid_id}.png")


def extract_frames(video_path: str, n_frames: int = 4) -> List[Image.Image]:
    """Extract evenly-spaced frames from a video using ffmpeg."""
    import tempfile
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        # Get duration
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            duration = float(result.stdout.strip())
        except Exception:
            duration = 10.0

        # Extract frames at evenly spaced timestamps
        for fi in range(n_frames):
            t = duration * (fi + 0.5) / n_frames
            out_path = os.path.join(tmpdir, f"frame_{fi}.jpg")
            cmd = [
                "ffmpeg", "-ss", str(t), "-i", video_path,
                "-vframes", "1", "-q:v", "2", out_path,
                "-y", "-loglevel", "quiet"
            ]
            try:
                subprocess.run(cmd, timeout=10)
                if os.path.exists(out_path):
                    frames.append(Image.open(out_path).convert("RGB"))
            except Exception:
                pass
    return frames


def make_frame_grid(
    frames: List[Image.Image],
    frame_size: Tuple[int, int] = (290, 163),
    grid: Tuple[int, int] = (2, 2),
) -> Image.Image:
    """Arrange frames in a 2x2 grid."""
    cols, rows = grid
    fw, fh = frame_size
    gap = 4
    grid_w = cols * fw + (cols - 1) * gap
    grid_h = rows * fh + (rows - 1) * gap
    canvas_img = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))

    for idx in range(rows * cols):
        col = idx % cols
        row = idx // cols
        x = col * (fw + gap)
        y = row * (fh + gap)
        if idx < len(frames):
            resized = frames[idx].resize((fw, fh), Image.Resampling.LANCZOS)
            canvas_img.paste(resized, (x, y))

    return canvas_img


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks", nargs="*",
        default=["scienceqa", "okvqa", "hotpotqa", "infographicvqa", "mmqa", "locomo", "msrvtt"],
        help="Which benchmarks to generate",
    )
    args = parser.parse_args()

    dispatch = {
        "scienceqa": generate_scienceqa,
        "okvqa": generate_okvqa,
        "hotpotqa": generate_hotpotqa,
        "infographicvqa": generate_infographicvqa,
        "mmqa": generate_mmqa,
        "locomo": generate_locomo,
        "msrvtt": generate_msrvtt,
    }

    for bench in args.benchmarks:
        if bench in dispatch:
            try:
                dispatch[bench]()
            except Exception as e:
                print(f"  ERROR in {bench}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  Unknown benchmark: {bench}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Paper Canvas Examples (New Rendering)")
    print(f"{'='*60}")
    for subdir in sorted(OUTPUT_DIR.iterdir()):
        if subdir.is_dir() and subdir.name != "memocr_comparison" and subdir.name != "final":
            files = sorted(subdir.glob("*.png"))
            print(f"  {subdir.name}: {len(files)} files")
            for f in files:
                img = Image.open(f)
                print(f"    - {f.name} ({img.width}x{img.height})")
    print(f"\nAll saved to: {OUTPUT_DIR}")
