#!/usr/bin/env python3
"""
Rebuild HotpotQA canvases using SmartCanvasLayout (CPU only).
Then recompute CLIP embeddings (needs GPU, but 3B CLIP fits in 10GB).
"""

import io
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/codex")
from smart_canvas_layout import (
    BlockType, ContentBlock, measure_text,
    choose_best_layout, render_layout,
)

DATA_DIR = Path("/home/cyf/codex/hotpotqa_data")
CANVAS_DIR = DATA_DIR / "canvases_smart"
ALPHA = 0.75


def render_hotpotqa_smart(sample: dict) -> bytes:
    """Render a HotpotQA sample using SmartCanvasLayout."""
    blocks = []

    # Header
    qtype = sample.get("type", "")
    level = sample.get("level", "")
    header = f"[HotpotQA] Type: {qtype} | Level: {level}"
    blocks.append(measure_text(header, font_size=12, ref_width=600))

    # Supporting context paragraphs
    sf_titles = set(t for t, _ in sample.get("supporting_facts", []))
    for para in sample.get("paragraphs", []):
        title = para["title"]
        if title in sf_titles:
            text = para["text"][:400]
            if len(para["text"]) > 400:
                text += "..."
            blocks.append(measure_text(f"[{title}]\n{text}", font_size=14, ref_width=600))

    # Question + Answer
    qa_text = f"Q: {sample['question']}\n✓ A: {sample['answer']}"
    blocks.append(measure_text(qa_text, font_size=16, ref_width=600))

    # Choose best layout and render
    layout = choose_best_layout(blocks)
    img = render_layout(layout)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build_smart_canvases():
    """Build all canvases with SmartCanvasLayout."""
    CANVAS_DIR.mkdir(exist_ok=True)

    with open(DATA_DIR / "hotpotqa_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    train_data = meta["train"]
    print(f"Building {len(train_data)} smart canvases...")

    done_marker = CANVAS_DIR / "done.txt"
    if done_marker.exists():
        n = int(done_marker.read_text().strip())
        print(f"Already built: {n} canvases")
        return n

    skipped = 0
    t0 = time.time()
    for i, sample in enumerate(tqdm(train_data, desc="Smart canvases")):
        out_path = CANVAS_DIR / f"{i:05d}.png"
        if out_path.exists():
            continue
        try:
            canvas_bytes = render_hotpotqa_smart(sample)
        except Exception as e:
            # Fallback: simple single-block
            blocks = [measure_text(
                f"Q: {sample['question']}\n✓ A: {sample['answer']}",
                font_size=16, ref_width=600
            )]
            layout = choose_best_layout(blocks)
            img = render_layout(layout)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            canvas_bytes = buf.getvalue()
            skipped += 1

        with open(out_path, "wb") as f:
            f.write(canvas_bytes)

    elapsed = time.time() - t0
    n = len(train_data)
    done_marker.write_text(str(n))
    print(f"Built {n} canvases in {elapsed:.0f}s ({skipped} fallbacks)")
    return n


def recompute_embeddings(n_canvases):
    """Recompute CLIP embeddings for smart canvases."""
    import torch
    from transformers import CLIPProcessor, CLIPModel

    canvas_emb_file = DATA_DIR / "canvas_embeddings_smart.npy"
    text_emb_file = DATA_DIR / "canvas_text_embeddings.npy"  # Reuse text embeddings
    query_emb_file = DATA_DIR / "query_embeddings.npy"       # Reuse query embeddings

    if canvas_emb_file.exists():
        print(f"Smart canvas embeddings already exist: {canvas_emb_file}")
        canvas_emb = np.load(canvas_emb_file)
        text_emb = np.load(text_emb_file)
        query_emb = np.load(query_emb_file)
        print(f"  img: {canvas_emb.shape}, txt: {text_emb.shape}, query: {query_emb.shape}")
        return canvas_emb, text_emb, query_emb

    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device).eval()
    print(f"  CLIP on {device}")

    print(f"Computing smart canvas image embeddings ({n_canvases})...")
    all_emb = []
    batch_size = 32
    for i in tqdm(range(0, n_canvases, batch_size), desc="Canvas img emb"):
        batch_imgs = []
        for j in range(i, min(i + batch_size, n_canvases)):
            path = CANVAS_DIR / f"{j:05d}.png"
            img = Image.open(path).convert("RGB")
            batch_imgs.append(img)
        inputs = clip_processor(images=batch_imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        all_emb.append(features.cpu().numpy())

    canvas_emb = np.concatenate(all_emb, axis=0)
    np.save(canvas_emb_file, canvas_emb)
    print(f"  Saved: {canvas_emb.shape}")

    del clip_model, clip_processor
    torch.cuda.empty_cache()

    text_emb = np.load(text_emb_file)
    query_emb = np.load(query_emb_file)
    return canvas_emb, text_emb, query_emb


def build_retrieval_map(canvas_emb, text_emb, query_emb):
    """Build retrieval map with alpha=0.75."""
    OUTPUT_DIR = Path("/home/cyf/codex/hotpotqa_experiment_v2")
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Building retrieval map (alpha={ALPHA}, K=2)...")
    key_emb = ALPHA * canvas_emb + (1 - ALPHA) * text_emb
    key_norm = key_emb / np.linalg.norm(key_emb, axis=1, keepdims=True).clip(min=1e-8)
    q_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True).clip(min=1e-8)

    sims = q_norm @ key_norm.T
    TOP_K = 2
    THRESHOLD = 0.1

    retrieval_map = {}
    for i in range(len(query_emb)):
        row = sims[i]
        top_indices = np.argsort(row)[::-1][:TOP_K + 5]
        results = []
        for idx in top_indices:
            if row[idx] < THRESHOLD:
                break
            results.append((int(idx), float(row[idx])))
            if len(results) >= TOP_K:
                break
        retrieval_map[i] = results

    has_mem = sum(1 for v in retrieval_map.values() if len(v) > 0)
    avg_sim = np.mean([r[1] for v in retrieval_map.values() for r in v if v])
    print(f"  {has_mem}/{len(query_emb)} have memories, avg_sim={avg_sim:.4f}")

    with open(OUTPUT_DIR / "retrieval_map_smart.pkl", "wb") as f:
        pickle.dump(retrieval_map, f)
    print(f"  Saved to {OUTPUT_DIR / 'retrieval_map_smart.pkl'}")

    # Save examples
    example_dir = OUTPUT_DIR / "canvas_examples_smart"
    example_dir.mkdir(exist_ok=True)

    with open(DATA_DIR / "hotpotqa_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    train_data = meta["train"]
    dev_data = meta["dev"]

    import shutil, json
    for qi in [0, 50, 100, 500, 1000]:
        if qi < len(dev_data) and retrieval_map.get(qi):
            for cidx, sim in retrieval_map[qi][:2]:
                src = CANVAS_DIR / f"{cidx:05d}.png"
                dst = example_dir / f"query{qi}_canvas{cidx}_sim{sim:.3f}.png"
                if src.exists():
                    shutil.copy(src, dst)
            info = {
                "query_idx": qi,
                "question": dev_data[qi]["question"],
                "answer": dev_data[qi]["answer"],
                "retrieved": [(idx, sim) for idx, sim in retrieval_map[qi]],
                "retrieved_questions": [train_data[idx]["question"] for idx, _ in retrieval_map[qi]],
                "retrieved_answers": [train_data[idx]["answer"] for idx, _ in retrieval_map[qi]],
            }
            with open(example_dir / f"query{qi}_info.json", "w") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

    return retrieval_map


if __name__ == "__main__":
    # Step 1: Build smart canvases (CPU only)
    n = build_smart_canvases()

    # Step 2: Recompute CLIP embeddings (GPU, CLIP fits in ~3GB)
    canvas_emb, text_emb, query_emb = recompute_embeddings(n)

    # Step 3: Build retrieval map
    build_retrieval_map(canvas_emb, text_emb, query_emb)

    print("\n=== All prep done. Run hotpotqa_experiment_v2.py --phase eval when GPU is free ===")
