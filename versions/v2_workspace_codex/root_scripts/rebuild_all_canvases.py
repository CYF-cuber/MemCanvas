#!/usr/bin/env python3
"""
Rebuild canvases for ScienceQA/OK-VQA/MMQA/InfographicVQA using SmartCanvasLayout.
Phase 1: CPU rendering (all 4 benchmarks in parallel-ish)
Phase 2: GPU CLIP embedding
"""
import io, json, os, pickle, sys, time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/codex")
from smart_canvas_layout import measure_text, measure_image, measure_table, choose_best_layout, render_layout, layout_single_column, BlockType, ContentBlock

# ============================================================
# ScienceQA
# ============================================================
def rebuild_scienceqa():
    print("\n=== ScienceQA SmartCanvas Rebuild ===")
    DATA_DIR = Path("/home/cyf/codex/okvqa_data")  # ScienceQA uses its own cache
    CANVAS_DIR = Path("/home/cyf/codex/scienceqa_smart_canvases")
    CANVAS_DIR.mkdir(exist_ok=True)

    # Load data
    cache = Path("/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl")
    with open(cache, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        train = data.get("train", data)
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        train, _ = data
    else:
        train = data

    # Load images from HF dataset
    from datasets import load_dataset
    hf_ds = load_dataset("derek-thomas/ScienceQA", split="train")

    done_marker = CANVAS_DIR / "done.txt"
    if done_marker.exists():
        print(f"  Already done: {done_marker.read_text().strip()} canvases")
        return int(done_marker.read_text().strip())

    n = len(train)
    print(f"  Rendering {n} canvases...")
    for i in tqdm(range(n), desc="ScienceQA"):
        out = CANVAS_DIR / f"{i:05d}.png"
        if out.exists():
            continue
        p = train[i]
        blocks = []

        # Header
        subj = p.get("subject", "")
        topic = p.get("topic", "")
        if subj or topic:
            blocks.append(measure_text(f"[{subj}] {topic}", font_size=12, ref_width=600))

        # Hint/context
        hint = p.get("hint", "")
        if hint:
            blocks.append(measure_text(hint, font_size=14, ref_width=600))

        # Image
        if i < len(hf_ds) and hf_ds[i].get("image") is not None:
            img = hf_ds[i]["image"].convert("RGB")
            blocks.append(measure_image(img, max_dim=400))

        # Knowledge/lecture
        lecture = p.get("lecture", "")
        if lecture:
            blocks.append(measure_text(lecture[:500], font_size=14, ref_width=600))

        # Question + choices + answer
        q = p.get("question", "")
        choices = p.get("choices", [])
        answer_idx = p.get("answer", 0)
        choice_text = "\n".join(f"{'✓ ' if j == answer_idx else '  '}{chr(65+j)}. {c}" for j, c in enumerate(choices))
        blocks.append(measure_text(f"Q: {q}\n{choice_text}", font_size=15, ref_width=600))

        # Solution
        solution = p.get("solution", "")
        if solution:
            blocks.append(measure_text(f"Solution: {solution[:300]}", font_size=13, ref_width=600))

        if not blocks:
            blocks.append(measure_text("(empty)", font_size=14, ref_width=600))

        layout = choose_best_layout(blocks)
        img_out = render_layout(layout)
        buf = io.BytesIO()
        img_out.save(buf, format="PNG", optimize=True)
        out.write_bytes(buf.getvalue())

    done_marker.write_text(str(n))
    print(f"  Done: {n} canvases")
    return n


# ============================================================
# OK-VQA
# ============================================================
def rebuild_okvqa():
    print("\n=== OK-VQA SmartCanvas Rebuild ===")
    DATA_DIR = Path("/home/cyf/codex/okvqa_data")
    CANVAS_DIR = DATA_DIR / "canvases_smart"
    CANVAS_DIR.mkdir(exist_ok=True)

    cache_file = DATA_DIR / "okvqa_cached.pkl"
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    train = data["train"]

    done_marker = CANVAS_DIR / "done.txt"
    if done_marker.exists():
        print(f"  Already done: {done_marker.read_text().strip()}")
        return int(done_marker.read_text().strip())

    n = len(train)
    print(f"  Rendering {n} canvases...")
    for i in tqdm(range(n), desc="OK-VQA"):
        out = CANVAS_DIR / f"{i:05d}.png"
        if out.exists():
            continue
        s = train[i]
        blocks = []

        # Image
        img_path = s.get("image_path", "")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            blocks.append(measure_image(img, max_dim=400))

        # Question + answers
        q = s.get("question", "")
        answers = s.get("answers", [])
        ans_text = ", ".join(answers[:5]) if answers else ""
        blocks.append(measure_text(f"Q: {q}\n✓ A: {ans_text}", font_size=15, ref_width=600))

        # Caption
        cap = s.get("caption", "")
        if cap:
            blocks.append(measure_text(f"Caption: {cap[:200]}", font_size=13, ref_width=600))

        if not blocks:
            blocks.append(measure_text("(empty)", font_size=14, ref_width=600))

        layout = choose_best_layout(blocks)
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
    """Convert MMQA table dict to readable text for canvas rendering."""
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
    print("\n=== MMQA SmartCanvas Rebuild ===")
    DATA_DIR = Path("/home/cyf/codex/mmqa_data")
    CANVAS_DIR = DATA_DIR / "canvases_smart"
    CANVAS_DIR.mkdir(exist_ok=True)
    IMAGES_DIR = DATA_DIR / "final_dataset_images"

    cache_file = DATA_DIR / "mmqa_parsed.pkl"
    with open(cache_file, "rb") as f:
        data = pickle.load(f)

    # MMQA data is a dict with 'train', 'tables', 'texts', 'images' keys
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
    for i in tqdm(range(n), desc="MMQA"):
        out = CANVAS_DIR / f"{i:05d}.png"
        if out.exists():
            continue
        s = train[i]
        blocks = []

        # Header with question type and modalities
        modalities = s.get("metadata", {}).get("modalities", [])
        qtype = s.get("metadata", {}).get("type", "")
        if qtype or modalities:
            header = f"[{qtype}] Modalities: {', '.join(modalities)}"
            blocks.append(measure_text(header, font_size=12, ref_width=600))

        # Supporting context: resolve via tables/texts/images dicts
        for ctx in s.get("supporting_context", [])[:3]:
            doc_id = ctx["doc_id"]
            doc_part = ctx["doc_part"]

            if doc_part == "text" and doc_id in texts:
                text_doc = texts[doc_id]
                title = text_doc.get("title", "")
                passage = text_doc.get("text", "")[:300]
                blocks.append(measure_text(f"[Text] {title}\n{passage}", font_size=13, ref_width=600))

            elif doc_part == "table" and doc_id in tables:
                table_doc = tables[doc_id]
                title = table_doc.get("title", "")
                if title:
                    blocks.append(measure_text(f"[Table] {title}", font_size=12, ref_width=600))
                headers = [h["column_name"] for h in table_doc["table"]["header"]]
                rows = [[cell["text"][:40] for cell in row] for row in table_doc["table"]["table_rows"][:8]]
                table_data = [headers] + rows
                if len(table_doc["table"]["table_rows"]) > 8:
                    table_data.append([f"...({len(table_doc['table']['table_rows'])} rows)"] + [""] * (len(headers) - 1))
                blocks.append(measure_table(table_data, font_size=12))

            elif doc_part == "image" and doc_id in images_meta:
                img_info = images_meta[doc_id]
                img_path = IMAGES_DIR / img_info["path"]
                if img_path.exists():
                    try:
                        img = Image.open(img_path).convert("RGB")
                        title = img_info.get("title", "")
                        blocks.append(measure_text(f"[Image] {title}", font_size=12, ref_width=600))
                        blocks.append(measure_image(img, max_dim=350))
                    except Exception:
                        pass

        # Question + answer
        q = s.get("question", "")
        answers = s.get("answers", [])
        ans = str(answers[0]["answer"]) if answers else ""
        blocks.append(measure_text(f"Q: {q}\n✓ A: {ans}", font_size=15, ref_width=600))

        if not blocks:
            blocks.append(measure_text("(empty)", font_size=14, ref_width=600))

        layout = choose_best_layout(blocks)
        img_out = render_layout(layout)
        buf = io.BytesIO()
        img_out.save(buf, format="PNG", optimize=True)
        out.write_bytes(buf.getvalue())

    done_marker.write_text(str(n))
    print(f"  Done: {n}")
    return n


# ============================================================
# InfographicVQA
# ============================================================
def rebuild_infographicvqa():
    print("\n=== InfographicVQA SmartCanvas Rebuild ===")
    DATA_DIR = Path("/home/cyf/codex/infographicvqa_data")
    CANVAS_DIR = DATA_DIR / "canvases_smart"
    CANVAS_DIR.mkdir(exist_ok=True)

    cache_file = DATA_DIR / "infographicvqa_meta.pkl"
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    train = data if isinstance(data, list) else data.get("train", data)

    done_marker = CANVAS_DIR / "done.txt"
    if done_marker.exists():
        print(f"  Already done: {done_marker.read_text().strip()}")
        return int(done_marker.read_text().strip())

    n = len(train)
    print(f"  Rendering {n} canvases...")
    for i in tqdm(range(n), desc="InfographicVQA"):
        out = CANVAS_DIR / f"{i:05d}.png"
        if out.exists():
            continue
        s = train[i]
        blocks = []

        # Infographic image
        img_path = s.get("image_path", "")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            # For tall infographics, use segmented cropping
            w, h = img.size
            if h > 2 * w:
                # Split into sections
                n_sections = min(4, max(2, h // w))
                section_h = h // n_sections
                for si in range(n_sections):
                    crop = img.crop((0, si * section_h, w, min((si+1) * section_h, h)))
                    blocks.append(measure_image(crop, max_dim=500))
            else:
                blocks.append(measure_image(img, max_dim=500))

        # Question + answer
        q = s.get("question", "")
        ans = s.get("answer", "")
        blocks.append(measure_text(f"Q: {q}\n✓ A: {ans}", font_size=15, ref_width=600))

        if not blocks:
            blocks.append(measure_text("(empty)", font_size=14, ref_width=600))

        layout = choose_best_layout(blocks)
        img_out = render_layout(layout)
        buf = io.BytesIO()
        img_out.save(buf, format="PNG", optimize=True)
        out.write_bytes(buf.getvalue())

    done_marker.write_text(str(n))
    print(f"  Done: {n}")
    return n


# ============================================================
# CLIP Embedding (GPU)
# ============================================================
def compute_clip_embeddings(canvas_dir, output_prefix, n_canvases):
    """Compute CLIP image embeddings for SmartCanvas PNG files."""
    import torch
    from transformers import CLIPProcessor, CLIPModel

    emb_file = Path(f"{output_prefix}_img_emb.npy")
    if emb_file.exists():
        print(f"  Embeddings exist: {emb_file}")
        return np.load(emb_file)

    print(f"  Computing CLIP embeddings for {n_canvases} canvases...")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    all_emb = []
    bs = 32
    for i in tqdm(range(0, n_canvases, bs), desc="CLIP emb"):
        imgs = []
        for j in range(i, min(i+bs, n_canvases)):
            p = Path(canvas_dir) / f"{j:05d}.png"
            imgs.append(Image.open(p).convert("RGB"))
        inp = proc(images=imgs, return_tensors="pt", padding=True)
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            feat = clip.get_image_features(**inp)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        all_emb.append(feat.cpu().numpy())

    emb = np.concatenate(all_emb)
    np.save(emb_file, emb)
    print(f"  Saved: {emb.shape}")

    del clip, proc
    torch.cuda.empty_cache()
    return emb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["render", "embed", "all"], default="all")
    parser.add_argument("--benchmark", choices=["scienceqa", "okvqa", "mmqa", "infographicvqa", "all"], default="all")
    args = parser.parse_args()

    counts = {}
    if args.phase in ("render", "all"):
        if args.benchmark in ("scienceqa", "all"):
            counts["scienceqa"] = rebuild_scienceqa()
        if args.benchmark in ("okvqa", "all"):
            counts["okvqa"] = rebuild_okvqa()
        if args.benchmark in ("mmqa", "all"):
            counts["mmqa"] = rebuild_mmqa()
        if args.benchmark in ("infographicvqa", "all"):
            counts["infographicvqa"] = rebuild_infographicvqa()

    if args.phase in ("embed", "all"):
        if args.benchmark in ("scienceqa", "all") or "scienceqa" in counts:
            n = counts.get("scienceqa") or int(Path("/home/cyf/codex/scienceqa_smart_canvases/done.txt").read_text())
            compute_clip_embeddings(
                "/home/cyf/codex/scienceqa_smart_canvases",
                "/home/cyf/codex/scienceqa_smart_canvases/clip", n)
        if args.benchmark in ("okvqa", "all") or "okvqa" in counts:
            n = counts.get("okvqa") or int(Path("/home/cyf/codex/okvqa_data/canvases_smart/done.txt").read_text())
            compute_clip_embeddings(
                "/home/cyf/codex/okvqa_data/canvases_smart",
                "/home/cyf/codex/okvqa_data/canvases_smart/clip", n)
        if args.benchmark in ("mmqa", "all") or "mmqa" in counts:
            n = counts.get("mmqa") or int(Path("/home/cyf/codex/mmqa_data/canvases_smart/done.txt").read_text())
            compute_clip_embeddings(
                "/home/cyf/codex/mmqa_data/canvases_smart",
                "/home/cyf/codex/mmqa_data/canvases_smart/clip", n)
        if args.benchmark in ("infographicvqa", "all") or "infographicvqa" in counts:
            n = counts.get("infographicvqa") or int(Path("/home/cyf/codex/infographicvqa_data/canvases_smart/done.txt").read_text())
            compute_clip_embeddings(
                "/home/cyf/codex/infographicvqa_data/canvases_smart",
                "/home/cyf/codex/infographicvqa_data/canvases_smart/clip", n)

    print("\n=== All done ===")
