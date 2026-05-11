#!/usr/bin/env python3
"""
Canvas0415: Single-column, large-font, large-image canvases.
Key changes vs rebuild_all_canvases.py:
  - Single-column layout only (no two-column)
  - Larger fonts (14-18pt vs 12-15pt)
  - Larger images (max_dim=800 vs 400)
  - Wider ref_width (800 vs 600)

Usage:
    python rebuild_canvas0415.py --benchmark scienceqa
    python rebuild_canvas0415.py --benchmark all
"""
import argparse, io, json, os, pickle, sys, time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/codex")
from smart_canvas_layout import (
    measure_text, measure_image, measure_table, choose_best_layout, render_layout,
    layout_single_column, BlockType, ContentBlock,
)

# ============================================================
# Canvas0415 parameters — larger, clearer, single-column
# ============================================================
FONT_HEADER = 14       # was 12
FONT_HINT = 16         # was 14
FONT_BODY = 16         # was 14
FONT_QA = 18           # was 15
FONT_SOLUTION = 15     # was 13
FONT_TABLE = 14        # was 12
REF_WIDTH = 800        # was 600
IMG_MAX_DIM = 800      # was 400 — images fill more canvas space
CANVAS_WIDTH = 830     # REF_WIDTH + 2 * PADDING(15)

BASE_DIR = Path("/home/cyf/codex/canvas0415")


# ============================================================
# ScienceQA
# ============================================================
def rebuild_scienceqa():
    print("\n=== ScienceQA Canvas0415 ===")
    CANVAS_DIR = BASE_DIR / "scienceqa"
    CANVAS_DIR.mkdir(parents=True, exist_ok=True)

    cache = Path("/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl")
    with open(cache, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        train = data.get("train", data)
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        train, _ = data
    else:
        train = data

    from datasets import load_dataset
    hf_ds = load_dataset("derek-thomas/ScienceQA", split="train")

    done_marker = CANVAS_DIR / "done.txt"
    if done_marker.exists():
        print(f"  Already done: {done_marker.read_text().strip()} canvases")
        return int(done_marker.read_text().strip())

    n = len(train)
    print(f"  Rendering {n} canvases (single-column, large font)...")
    for i in tqdm(range(n), desc="ScienceQA canvas0415"):
        out = CANVAS_DIR / f"{i:05d}.png"
        if out.exists():
            continue
        p = train[i]
        blocks = []

        # Header
        subj = p.get("subject", "")
        topic = p.get("topic", "")
        if subj or topic:
            blocks.append(measure_text(f"[{subj}] {topic}", font_size=FONT_HEADER, ref_width=REF_WIDTH))

        # Hint/context
        hint = p.get("hint", "")
        if hint:
            blocks.append(measure_text(hint, font_size=FONT_HINT, ref_width=REF_WIDTH))

        # Image — large, no size restriction
        if i < len(hf_ds) and hf_ds[i].get("image") is not None:
            img = hf_ds[i]["image"].convert("RGB")
            blocks.append(measure_image(img, max_dim=IMG_MAX_DIM))

        # Knowledge/lecture
        lecture = p.get("lecture", "")
        if lecture:
            blocks.append(measure_text(lecture[:500], font_size=FONT_BODY, ref_width=REF_WIDTH))

        # Question + choices + answer
        q = p.get("question", "")
        choices = p.get("choices", [])
        answer_idx = p.get("answer", 0)
        choice_text = "\n".join(
            f"{'>>> ' if j == answer_idx else '    '}{chr(65+j)}. {c}"
            for j, c in enumerate(choices)
        )
        blocks.append(measure_text(f"Q: {q}\n{choice_text}", font_size=FONT_QA, ref_width=REF_WIDTH))

        # Solution
        solution = p.get("solution", "")
        if solution:
            blocks.append(measure_text(f"Solution: {solution[:300]}", font_size=FONT_SOLUTION, ref_width=REF_WIDTH))

        if not blocks:
            blocks.append(measure_text("(empty)", font_size=FONT_BODY, ref_width=REF_WIDTH))

        # Force single-column layout
        layout = layout_single_column(blocks, target_width=CANVAS_WIDTH)
        img_out = render_layout(layout)
        buf = io.BytesIO()
        img_out.save(buf, format="PNG", optimize=True)
        out.write_bytes(buf.getvalue())

    done_marker.write_text(str(n))
    print(f"  Done: {n} canvases → {CANVAS_DIR}")
    return n


# ============================================================
# OK-VQA
# ============================================================
def rebuild_okvqa():
    print("\n=== OK-VQA Canvas0415 ===")
    CANVAS_DIR = BASE_DIR / "okvqa"
    CANVAS_DIR.mkdir(parents=True, exist_ok=True)

    cache_file = Path("/home/cyf/codex/okvqa_data/okvqa_cached.pkl")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    train = data["train"]

    done_marker = CANVAS_DIR / "done.txt"
    if done_marker.exists():
        print(f"  Already done: {done_marker.read_text().strip()}")
        return int(done_marker.read_text().strip())

    n = len(train)
    print(f"  Rendering {n} canvases...")
    for i in tqdm(range(n), desc="OK-VQA canvas0415"):
        out = CANVAS_DIR / f"{i:05d}.png"
        if out.exists():
            continue
        s = train[i]
        blocks = []

        # Image — large
        img_path = s.get("image_path", "")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            blocks.append(measure_image(img, max_dim=IMG_MAX_DIM))

        # Question + answers
        q = s.get("question", "")
        answers = s.get("answers", [])
        ans_text = ", ".join(answers[:5]) if answers else ""
        blocks.append(measure_text(f"Q: {q}\n>>> A: {ans_text}", font_size=FONT_QA, ref_width=REF_WIDTH))

        # Caption
        cap = s.get("caption", "")
        if cap:
            blocks.append(measure_text(f"Caption: {cap[:200]}", font_size=FONT_SOLUTION, ref_width=REF_WIDTH))

        if not blocks:
            blocks.append(measure_text("(empty)", font_size=FONT_BODY, ref_width=REF_WIDTH))

        layout = layout_single_column(blocks, target_width=CANVAS_WIDTH)
        img_out = render_layout(layout)
        buf = io.BytesIO()
        img_out.save(buf, format="PNG", optimize=True)
        out.write_bytes(buf.getvalue())

    done_marker.write_text(str(n))
    print(f"  Done: {n}")
    return n


# ============================================================
# MMQA
# ============================================================
def _table_to_text(table_data: dict, max_rows: int = 6) -> str:
    headers = [h["column_name"] for h in table_data["header"]]
    rows = [[cell["text"][:40] for cell in row] for row in table_data["table_rows"][:max_rows]]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    if len(table_data["table_rows"]) > max_rows:
        lines.append(f"... ({len(table_data['table_rows'])} rows total)")
    return "\n".join(lines)


def rebuild_mmqa():
    print("\n=== MMQA Canvas0415 ===")
    CANVAS_DIR = BASE_DIR / "mmqa"
    CANVAS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR = Path("/home/cyf/codex/mmqa_data/final_dataset_images")

    cache_file = Path("/home/cyf/codex/mmqa_data/mmqa_parsed.pkl")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)

    train = data["train"]
    tables = data["tables"]
    texts = data["texts"]
    images_meta = data["images"]

    done_marker = CANVAS_DIR / "done.txt"
    if done_marker.exists():
        print(f"  Already done: {done_marker.read_text().strip()}")
        return int(done_marker.read_text().strip())

    n = len(train)
    print(f"  Rendering {n} canvases...")
    for i in tqdm(range(n), desc="MMQA canvas0415"):
        out = CANVAS_DIR / f"{i:05d}.png"
        if out.exists():
            continue
        s = train[i]
        blocks = []

        # Header
        modalities = s.get("metadata", {}).get("modalities", [])
        qtype = s.get("metadata", {}).get("type", "")
        if qtype or modalities:
            header = f"[{qtype}] Modalities: {', '.join(modalities)}"
            blocks.append(measure_text(header, font_size=FONT_HEADER, ref_width=REF_WIDTH))

        # Supporting context
        for ctx in s.get("supporting_context", [])[:3]:
            doc_id = ctx["doc_id"]
            doc_part = ctx["doc_part"]

            if doc_part == "text" and doc_id in texts:
                text_doc = texts[doc_id]
                title = text_doc.get("title", "")
                passage = text_doc.get("text", "")[:300]
                blocks.append(measure_text(f"[Text] {title}\n{passage}", font_size=FONT_BODY, ref_width=REF_WIDTH))

            elif doc_part == "table" and doc_id in tables:
                table_doc = tables[doc_id]
                title = table_doc.get("title", "")
                if title:
                    blocks.append(measure_text(f"[Table] {title}", font_size=FONT_HEADER, ref_width=REF_WIDTH))
                # Render as visual table grid instead of raw markdown text
                headers = [h["column_name"] for h in table_doc["table"]["header"]]
                rows = [[cell["text"][:40] for cell in row] for row in table_doc["table"]["table_rows"][:8]]
                table_data = [headers] + rows
                if len(table_doc["table"]["table_rows"]) > 8:
                    table_data.append([f"...({len(table_doc['table']['table_rows'])} rows)"] + [""] * (len(headers) - 1))
                blocks.append(measure_table(table_data, font_size=FONT_TABLE))

            elif doc_part == "image" and doc_id in images_meta:
                img_info = images_meta[doc_id]
                img_path = IMAGES_DIR / img_info["path"]
                if img_path.exists():
                    try:
                        img = Image.open(img_path).convert("RGB")
                        title = img_info.get("title", "")
                        blocks.append(measure_text(f"[Image] {title}", font_size=FONT_HEADER, ref_width=REF_WIDTH))
                        blocks.append(measure_image(img, max_dim=IMG_MAX_DIM))
                    except Exception:
                        pass

        # Question + answer
        q = s.get("question", "")
        answers = s.get("answers", [])
        ans = str(answers[0]["answer"]) if answers else ""
        blocks.append(measure_text(f"Q: {q}\n>>> A: {ans}", font_size=FONT_QA, ref_width=REF_WIDTH))

        if not blocks:
            blocks.append(measure_text("(empty)", font_size=FONT_BODY, ref_width=REF_WIDTH))

        layout = layout_single_column(blocks, target_width=CANVAS_WIDTH)
        img_out = render_layout(layout)
        buf = io.BytesIO()
        img_out.save(buf, format="PNG", optimize=True)
        out.write_bytes(buf.getvalue())

    done_marker.write_text(str(n))
    print(f"  Done: {n}")
    return n


# ============================================================
# HotpotQA
# ============================================================
def rebuild_hotpotqa():
    print("\n=== HotpotQA Canvas0415 ===")
    CANVAS_DIR = BASE_DIR / "hotpotqa"
    CANVAS_DIR.mkdir(parents=True, exist_ok=True)

    cache_file = Path("/home/cyf/codex/hotpotqa_data/hotpotqa_meta.pkl")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)

    train = data if isinstance(data, list) else data.get("train", data)

    done_marker = CANVAS_DIR / "done.txt"
    if done_marker.exists():
        print(f"  Already done: {done_marker.read_text().strip()}")
        return int(done_marker.read_text().strip())

    n = len(train)
    print(f"  Rendering {n} canvases...")
    for i in tqdm(range(n), desc="HotpotQA canvas0415"):
        out = CANVAS_DIR / f"{i:05d}.png"
        if out.exists():
            continue
        s = train[i]
        blocks = []

        # Type + level header
        qtype = s.get("type", "")
        level = s.get("level", "")
        if qtype or level:
            blocks.append(measure_text(f"[{qtype}] Level: {level}", font_size=FONT_HEADER, ref_width=REF_WIDTH))

        # Supporting context paragraphs
        for para in s.get("paragraphs", [])[:4]:
            title = para["title"]
            text = para["text"][:400]
            blocks.append(measure_text(f"[{title}]\n{text}", font_size=FONT_BODY, ref_width=REF_WIDTH))

        # Question + answer
        q = s.get("question", "")
        ans = s.get("answer", "")
        blocks.append(measure_text(f"Q: {q}\n>>> A: {ans}", font_size=FONT_QA, ref_width=REF_WIDTH))

        if not blocks:
            blocks.append(measure_text("(empty)", font_size=FONT_BODY, ref_width=REF_WIDTH))

        layout = layout_single_column(blocks, target_width=CANVAS_WIDTH)
        img_out = render_layout(layout)
        buf = io.BytesIO()
        img_out.save(buf, format="PNG", optimize=True)
        out.write_bytes(buf.getvalue())

    done_marker.write_text(str(n))
    print(f"  Done: {n}")
    return n


# ============================================================
# CLIP Embeddings (shared)
# ============================================================
def compute_clip_embeddings(canvas_dir: Path, n: int):
    """Compute CLIP image + text embeddings for all canvases."""
    import torch
    from transformers import CLIPProcessor, CLIPModel

    img_emb_path = canvas_dir / "clip_img_emb.npy"
    if img_emb_path.exists():
        print(f"  CLIP embeddings already computed: {img_emb_path}")
        return

    print(f"  Computing CLIP embeddings for {n} canvases...")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    all_emb = []
    bs = 32
    for i in tqdm(range(0, n, bs), desc="CLIP img"):
        imgs = []
        for j in range(i, min(i + bs, n)):
            imgs.append(Image.open(canvas_dir / f"{j:05d}.png").convert("RGB"))
        inp = proc(images=imgs, return_tensors="pt", padding=True)
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            feat = clip.get_image_features(**inp)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        all_emb.append(feat.cpu().numpy())
    emb = np.concatenate(all_emb)
    np.save(img_emb_path, emb)
    print(f"  Saved: {emb.shape} → {img_emb_path}")

    del clip, proc
    torch.cuda.empty_cache()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["scienceqa", "okvqa", "mmqa", "hotpotqa", "all"], default="scienceqa")
    parser.add_argument("--embed", action="store_true", help="Also compute CLIP embeddings")
    args = parser.parse_args()

    benchmarks = ["scienceqa", "okvqa", "mmqa", "hotpotqa"] if args.benchmark == "all" else [args.benchmark]

    for bm in benchmarks:
        if bm == "scienceqa":
            n = rebuild_scienceqa()
        elif bm == "okvqa":
            n = rebuild_okvqa()
        elif bm == "mmqa":
            n = rebuild_mmqa()
        elif bm == "hotpotqa":
            n = rebuild_hotpotqa()

        if args.embed:
            compute_clip_embeddings(BASE_DIR / bm, n)


if __name__ == "__main__":
    main()
