#!/usr/bin/env python3
"""
Build SmartCanvas images from compressed HotpotQA texts.
Uses the same smart_canvas_layout as the original experiment.
"""
import io, json, os, pickle, sys, time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/codex")
from smart_canvas_layout import measure_text, choose_best_layout, render_layout

OUTPUT_DIR = Path("/home/cyf/memcanvas0402")
MAX_SAMPLES = 3000  # match compression subset


def build_canvases(level):
    print(f"\n=== Building {level} compressed canvases ===")
    src_dir = OUTPUT_DIR / f"hotpotqa_{level}"
    cache_file = src_dir / "compressed_texts.pkl"
    canvas_dir = src_dir / "canvases"
    canvas_dir.mkdir(parents=True, exist_ok=True)

    done_marker = canvas_dir / "done.txt"
    if done_marker.exists():
        n = int(done_marker.read_text().strip())
        print(f"  Already done: {n} canvases")
        return n

    # Load compressed data
    compressed = pickle.load(open(cache_file, "rb"))
    n = len(compressed)
    print(f"  {n} compressed samples loaded")

    sizes = []
    for i in tqdm(range(n), desc=f"Canvas ({level})"):
        out = canvas_dir / f"{i:05d}.png"
        if out.exists():
            continue
        if i not in compressed:
            continue
        s = compressed[i]

        blocks = []

        # Supporting paragraphs (compressed)
        sf_titles = set(t for t, _ in s.get("supporting_facts", []))
        for p in s["paragraphs"]:
            title = p["title"]
            text = p["text"]
            if not text or len(text) < 5:
                continue
            # Mark supporting facts
            prefix = f"[{title}]"
            if title in sf_titles:
                prefix = f"★ [{title}]"
            blocks.append(measure_text(f"{prefix}\n{text}", font_size=14, ref_width=600))

        # Question + Answer
        qa_text = f"Q: {s['question']}\nA: {s['answer']}"
        blocks.append(measure_text(qa_text, font_size=16, ref_width=600))

        if not blocks:
            # Fallback: minimal canvas
            blocks.append(measure_text(f"Q: {s['question']}\nA: {s['answer']}", font_size=16, ref_width=600))

        layout = choose_best_layout(blocks)
        img_out = render_layout(layout)

        buf = io.BytesIO()
        img_out.save(buf, format="PNG", optimize=True)
        out.write_bytes(buf.getvalue())
        sizes.append(len(buf.getvalue()))

    done_marker.write_text(str(n))
    if sizes:
        print(f"  Avg canvas size: {np.mean(sizes)/1024:.1f} KB")
    print(f"  Done: {n} canvases in {canvas_dir}")
    return n


def compute_clip_embeddings(canvas_dir, n_canvases):
    """Compute CLIP image embeddings for canvas PNGs."""
    import torch
    from transformers import CLIPProcessor, CLIPModel

    emb_file = Path(canvas_dir).parent / "clip_img_emb.npy"
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
        for j in range(i, min(i + bs, n_canvases)):
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
    print(f"  Saved: {emb.shape} -> {emb_file}")

    del clip, proc
    torch.cuda.empty_cache()
    return emb


def compute_text_embeddings(compressed_data, n_samples):
    """Compute CLIP text embeddings for Q+A strings (for hybrid retrieval)."""
    import torch
    from transformers import CLIPProcessor, CLIPModel

    parent_dir = None  # Will be set per-level
    # Actually we need the output path, let's just return embs
    print(f"  Computing CLIP text embeddings for {n_samples} samples...")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    all_emb = []
    bs = 64
    texts = []
    for i in range(n_samples):
        if i in compressed_data:
            s = compressed_data[i]
            t = f"{s['question']} {s['answer']}"
        else:
            t = ""
        texts.append(t[:77])  # CLIP max tokens

    for i in tqdm(range(0, len(texts), bs), desc="CLIP text"):
        batch = texts[i:i + bs]
        inp = proc(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            feat = clip.get_text_features(**inp)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        all_emb.append(feat.cpu().numpy())

    emb = np.concatenate(all_emb)
    del clip, proc
    torch.cuda.empty_cache()
    return emb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True, choices=["light", "heavy"])
    parser.add_argument("--phase", choices=["render", "embed", "all"], default="all")
    args = parser.parse_args()

    level = args.level
    src_dir = OUTPUT_DIR / f"hotpotqa_{level}"
    canvas_dir = src_dir / "canvases"

    if args.phase in ("render", "all"):
        n = build_canvases(level)

    if args.phase in ("embed", "all"):
        # Load compressed data for text embeddings
        compressed = pickle.load(open(src_dir / "compressed_texts.pkl", "rb"))
        n = len(compressed)

        # Image embeddings
        img_emb = compute_clip_embeddings(canvas_dir, n)

        # Text embeddings
        txt_emb_file = src_dir / "clip_txt_emb.npy"
        if txt_emb_file.exists():
            print(f"  Text embeddings exist: {txt_emb_file}")
        else:
            txt_emb = compute_text_embeddings(compressed, n)
            np.save(txt_emb_file, txt_emb)
            print(f"  Saved text embeddings: {txt_emb.shape}")

    print("\nDone!")
