#!/usr/bin/env python3
"""
InfographicVQA Experiment: Baseline vs MemCanvas.

Dataset: InfographicVQA (WACV 2022, Mathew et al.)
  - Train: 4,406 images / 23,946 QA pairs
  - Val:   500 images / 2,801 QA pairs
  - Test:  579 images / 3,288 QA pairs
Metric: ANLS (Average Normalized Levenshtein Similarity)

Phases:
  prep: Load data, build canvases, compute CLIP embeddings
  eval: Run VLM evaluation (baseline + memcanvas)

Usage:
  python -u /home/cyf/codex/infographicvqa_experiment.py --phase prep
  CUDA_VISIBLE_DEVICES=0 python -u /home/cyf/codex/infographicvqa_experiment.py --phase eval
  CUDA_VISIBLE_DEVICES=0 python -u /home/cyf/codex/infographicvqa_experiment.py --phase eval --skip-baseline
"""

import argparse
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/memory")
from memory_canvas.dynamic_canvas import DynamicCanvas, DynamicCanvasConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/cyf/codex/infographicvqa_data")
OUTPUT_DIR = Path("/home/cyf/codex/infographicvqa_experiment")
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
VLM_MODEL_PATH = "/home/cyf/Qwen2.5-VL-7B-Instruct"

ALPHA = 0.50
TOP_K = 2
SIMILARITY_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# ANLS Metric
# ---------------------------------------------------------------------------
def normalized_levenshtein(s1: str, s2: str) -> float:
    """Compute Normalized Levenshtein Similarity between two strings."""
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 and len2 == 0:
        return 1.0
    max_len = max(len1, len2)
    if max_len == 0:
        return 1.0

    # Levenshtein distance (dynamic programming)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    dist = dp[len1][len2]
    nls = 1.0 - dist / max_len
    return nls


def anls_score(prediction: str, ground_truths: List[str], threshold: float = 0.5) -> float:
    """
    ANLS (Average Normalized Levenshtein Similarity).
    For each GT answer, compute NLS; take max over all GTs.
    Apply threshold: if NLS < threshold, score is 0.
    """
    if not ground_truths:
        return 0.0
    max_nls = 0.0
    for gt in ground_truths:
        nls = normalized_levenshtein(prediction, gt)
        max_nls = max(max_nls, nls)
    return max_nls if max_nls >= threshold else 0.0


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def download_infographicvqa():
    """Download InfographicVQA from HuggingFace and cache locally.

    Images are saved to disk individually to avoid OOM.
    Metadata (question, answers, image path) is cached in a small pickle.
    """
    cache_file = DATA_DIR / "infographicvqa_meta.pkl"
    if cache_file.exists():
        print(f"Loading cached InfographicVQA metadata from {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        print(f"  Train: {len(data['train'])} samples")
        print(f"  Val: {len(data['val'])} samples")
        return data

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_img_dir = DATA_DIR / "images_train"
    val_img_dir = DATA_DIR / "images_val"
    train_img_dir.mkdir(exist_ok=True)
    val_img_dir.mkdir(exist_ok=True)

    print("Downloading InfographicVQA from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("Ahren09/InfoVQA")

    def process_split(split_data, img_dir, split_name):
        samples = []
        for i, item in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            img_path = img_dir / f"{i:05d}.jpg"
            if not img_path.exists():
                img = item["image"]
                if img is not None:
                    img.convert("RGB").save(str(img_path), "JPEG", quality=85)
            sample = {
                "question_id": item.get("id", ""),
                "question": item.get("question", ""),
                "answers": item.get("answer", []),
                "image_path": str(img_path),
                "deck_name": item.get("deck_name", ""),
            }
            samples.append(sample)
        return samples

    train_samples = process_split(ds["train"], train_img_dir, "train")
    val_samples = process_split(ds["val"], val_img_dir, "val")

    data = {"train": train_samples, "val": val_samples}
    with open(cache_file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached metadata: train={len(train_samples)}, val={len(val_samples)}")
    return data


def load_image(sample: dict) -> Optional[Image.Image]:
    """Load image from disk for a sample."""
    path = sample.get("image_path", "")
    if path and os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Canvas Building
# ---------------------------------------------------------------------------
def build_canvases(train_data: List[dict], rebuild: bool = False):
    """Build canvas images for each training sample. Saves to individual files."""
    canvas_dir = DATA_DIR / "canvases"
    done_marker = DATA_DIR / "canvases_done.txt"

    if rebuild:
        print("Rebuilding canvases: clearing old data...")
        if done_marker.exists():
            done_marker.unlink()
        # Remove old canvas files
        if canvas_dir.exists():
            import shutil
            shutil.rmtree(canvas_dir)
        # Remove old embedding caches (canvases changed)
        for emb_file in [
            DATA_DIR / "canvas_embeddings.npy",
            DATA_DIR / "canvas_text_embeddings.npy",
            DATA_DIR / "query_embeddings.npy",
            DATA_DIR / "retrieval_map.pkl",
        ]:
            if emb_file.exists():
                emb_file.unlink()
                print(f"  Removed: {emb_file.name}")

    if done_marker.exists():
        n = int(done_marker.read_text().strip())
        print(f"Canvases already built: {n} files in {canvas_dir}")
        return n

    canvas_dir.mkdir(exist_ok=True)
    print(f"Building {len(train_data)} canvases...")
    skipped = 0
    for i, sample in enumerate(tqdm(train_data, desc="Building canvases")):
        out_path = canvas_dir / f"{i:05d}.png"
        if out_path.exists():
            continue
        try:
            canvas_bytes = render_infographic_canvas(sample)
        except Exception:
            canvas_bytes = render_minimal_canvas(sample)
            skipped += 1
        with open(out_path, "wb") as f:
            f.write(canvas_bytes)

    if skipped:
        print(f"  Warning: {skipped} canvases used fallback rendering")

    n = len(train_data)
    done_marker.write_text(str(n))
    print(f"  Built {n} canvases in {canvas_dir}")
    return n


def load_canvas(idx: int) -> bytes:
    """Load a canvas by index from disk."""
    path = DATA_DIR / "canvases" / f"{idx:05d}.png"
    with open(path, "rb") as f:
        return f.read()


def render_infographic_canvas(sample: dict) -> bytes:
    """Render an InfographicVQA training sample to a canvas image.

    Uses add_tall_image for smart segmented cropping of tall infographics,
    and add_html for formatted Q&A text.
    """
    canvas = DynamicCanvas(DynamicCanvasConfig(
        patch_size=640,
        font_size=16,
        padding=20,
        content_gap=10,
        show_patch_boundary=False,
    ))

    # Header (HTML with formatting if available, fallback to text)
    canvas.add_html(
        "**[InfographicVQA]** Training Example",
        content_type="markdown",
    )
    canvas.add_separator()

    # Infographic: smart segmented cropping for tall images
    img = load_image(sample)
    if img is not None:
        canvas.add_tall_image(img, max_sections=3, overlap=50)

    canvas.add_separator()

    # Question (bold, formatted)
    q_md = f"**Q:** {sample['question']}"
    canvas.add_html(q_md, content_type="markdown")

    # Answer(s)
    answers = sample.get("answers", [])
    if answers:
        if isinstance(answers, list):
            ans_str = answers[0] if answers else ""
        else:
            ans_str = str(answers)
        a_md = f"\u2713 **A:** {ans_str}"
        canvas.add_html(a_md, content_type="markdown")

    # Combine patches compactly (no wasted whitespace)
    img_out = canvas.get_compact_image()

    buf = io.BytesIO()
    img_out.save(buf, format="PNG")
    return buf.getvalue()


def render_minimal_canvas(sample: dict) -> bytes:
    """Minimal fallback canvas with just question and answer."""
    canvas = DynamicCanvas(DynamicCanvasConfig(
        patch_size=640, font_size=16, padding=20,
        content_gap=10, show_patch_boundary=False,
    ))
    canvas.add_text(f"Q: {sample['question']}", font_size=16, bold=True)
    answers = sample.get("answers", [])
    if answers:
        ans_str = answers[0] if isinstance(answers, list) else str(answers)
        canvas.add_text(f"\u2713 A: {ans_str}", font_size=16)

    patches = canvas.get_images()
    img_out = patches[0]
    buf = io.BytesIO()
    img_out.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def compute_embeddings(train_data, val_data, n_canvases):
    """Compute CLIP embeddings for canvases and queries."""
    canvas_emb_file = DATA_DIR / "canvas_embeddings.npy"
    text_emb_file = DATA_DIR / "canvas_text_embeddings.npy"
    query_emb_file = DATA_DIR / "query_embeddings.npy"

    if canvas_emb_file.exists() and text_emb_file.exists() and query_emb_file.exists():
        print("Loading cached embeddings...")
        canvas_emb = np.load(canvas_emb_file)
        text_emb = np.load(text_emb_file)
        query_emb = np.load(query_emb_file)
        print(f"  Canvas img: {canvas_emb.shape}, txt: {text_emb.shape}, query: {query_emb.shape}")
        return canvas_emb, text_emb, query_emb

    print("Loading CLIP model...")
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device).eval()
    print(f"  CLIP on {device}")

    # Canvas image embeddings
    if not canvas_emb_file.exists():
        print(f"Computing canvas image embeddings ({n_canvases})...")
        all_emb = []
        batch_size = 32
        for i in tqdm(range(0, n_canvases, batch_size), desc="Canvas img emb"):
            batch_imgs = []
            for j in range(i, min(i + batch_size, n_canvases)):
                img = Image.open(io.BytesIO(load_canvas(j))).convert("RGB")
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
    else:
        canvas_emb = np.load(canvas_emb_file)

    # Canvas text embeddings (training question + OCR snippet)
    if not text_emb_file.exists():
        print(f"Computing canvas text embeddings ({len(train_data)})...")
        all_emb = []
        batch_size = 64
        for i in tqdm(range(0, len(train_data), batch_size), desc="Canvas txt emb"):
            batch_texts = []
            for j in range(i, min(i + batch_size, len(train_data))):
                s = train_data[j]
                text = s["question"]
                # Include first answer as context for better embeddings
                answers = s.get("answers", [])
                if answers:
                    ans = answers[0] if isinstance(answers, list) else str(answers)
                    text += " " + ans
                batch_texts.append(text)
            inputs = clip_processor(text=batch_texts, return_tensors="pt",
                                    padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                features = clip_model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            all_emb.append(features.cpu().numpy())
        text_emb = np.concatenate(all_emb, axis=0)
        np.save(text_emb_file, text_emb)
        print(f"  Saved: {text_emb.shape}")
    else:
        text_emb = np.load(text_emb_file)

    # Query embeddings (val question text)
    if not query_emb_file.exists():
        print(f"Computing query embeddings ({len(val_data)})...")
        all_emb = []
        batch_size = 64
        for i in tqdm(range(0, len(val_data), batch_size), desc="Query emb"):
            batch_texts = []
            for j in range(i, min(i + batch_size, len(val_data))):
                batch_texts.append(val_data[j]["question"])
            inputs = clip_processor(text=batch_texts, return_tensors="pt",
                                    padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                features = clip_model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            all_emb.append(features.cpu().numpy())
        query_emb = np.concatenate(all_emb, axis=0)
        np.save(query_emb_file, query_emb)
        print(f"  Saved: {query_emb.shape}")
    else:
        query_emb = np.load(query_emb_file)

    del clip_model, clip_processor
    torch.cuda.empty_cache()
    return canvas_emb, text_emb, query_emb


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def build_retrieval_map(query_emb, canvas_img_emb, canvas_txt_emb, alpha, top_k, threshold):
    """Build retrieval map: val_idx -> [(canvas_idx, similarity), ...]."""
    print(f"Building retrieval map (\u03b1={alpha}, K={top_k})...")
    key_emb = alpha * canvas_img_emb + (1 - alpha) * canvas_txt_emb
    k_n = np.linalg.norm(key_emb, axis=1, keepdims=True)
    k_n[k_n == 0] = 1.0
    key_norm = key_emb / k_n
    q_n = np.linalg.norm(query_emb, axis=1, keepdims=True)
    q_n[q_n == 0] = 1.0
    q_norm = query_emb / q_n

    sims = q_norm @ key_norm.T

    retrieval_map = {}
    for i in range(len(query_emb)):
        row = sims[i]
        top_indices = np.argsort(row)[::-1][:top_k + 5]
        results = []
        for idx in top_indices:
            if row[idx] < threshold:
                break
            results.append((int(idx), float(row[idx])))
            if len(results) >= top_k:
                break
        retrieval_map[i] = results

    has_mem = sum(1 for v in retrieval_map.values() if len(v) > 0)
    print(f"  {has_mem}/{len(query_emb)} val samples have memories")
    return retrieval_map


# ---------------------------------------------------------------------------
# VLM Evaluation
# ---------------------------------------------------------------------------
def load_vlm():
    print("Loading Qwen2.5-VL-7B...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(VLM_MODEL_PATH)
    print("  VLM loaded")
    return model, processor


def predict_baseline(model, processor, sample):
    """Baseline: infographic image + question -> VLM answer."""
    img = load_image(sample)

    prompt_parts = []
    prompt_parts.append("Look at the infographic image and answer the question.")
    prompt_parts.append(f"Question: {sample['question']}")
    prompt_parts.append("Answer concisely:")
    user_text = "\n".join(prompt_parts)

    content = []
    images = []
    if img is not None:
        content.append({"type": "image", "image": img})
        images.append(img)
    content.append({"type": "text", "text": user_text})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if images:
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return raw


def predict_memcanvas(model, processor, sample, retrieved):
    """MemCanvas: retrieved canvases + infographic + question -> VLM answer."""
    img = load_image(sample)

    prompt_parts = []
    memory_images = []

    if retrieved:
        prompt_parts.append(
            "Below are memory canvas images from similar infographic questions answered before. "
            "Each canvas shows: the infographic thumbnail, the question, and the correct answer. "
            "Study these canvases and use the knowledge to help answer the new question."
        )
        prompt_parts.append("")
        for i, (canvas_idx, sim) in enumerate(retrieved):
            canvas_img = Image.open(io.BytesIO(load_canvas(canvas_idx))).convert("RGB")
            memory_images.append(canvas_img)
            prompt_parts.append(f"[Canvas {i+1}]")
        prompt_parts.append("")
        prompt_parts.append("---")
        prompt_parts.append("")

    prompt_parts.append("Look at the infographic image and answer the question.")
    prompt_parts.append(f"Question: {sample['question']}")
    prompt_parts.append("Answer concisely:")
    user_text = "\n".join(prompt_parts)

    content = []
    all_images = []
    for mem_img in memory_images:
        content.append({"type": "image", "image": mem_img})
        all_images.append(mem_img)
    if img is not None:
        content.append({"type": "image", "image": img})
        all_images.append(img)
    content.append({"type": "text", "text": user_text})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if all_images:
        inputs = processor(text=[text], images=all_images, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return raw


def evaluate_condition(
    condition_name, model, processor, val_data,
    retrieval_map, output_dir
):
    """Evaluate a single condition (baseline or memcanvas)."""
    ckpt_file = output_dir / f"checkpoint_{condition_name}.json"

    results = {}
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            results = json.load(f)
        print(f"  Resumed {condition_name}: {len(results)} done")

    done = set(results.keys())
    remaining = [i for i in range(len(val_data)) if str(i) not in done]

    if not remaining:
        print(f"  {condition_name} already complete")
    else:
        print(f"  Running {condition_name}: {len(remaining)} remaining")
        for idx in tqdm(remaining, desc=condition_name):
            sample = val_data[idx]

            try:
                if condition_name == "baseline":
                    raw = predict_baseline(model, processor, sample)
                else:
                    retrieved = retrieval_map.get(idx, [])
                    raw = predict_memcanvas(
                        model, processor, sample, retrieved
                    )
            except Exception as e:
                raw = ""
                print(f"\n  Error on {idx}: {e}")

            # Compute ANLS
            gt_answers = sample.get("answers", [])
            if isinstance(gt_answers, str):
                gt_answers = [gt_answers]
            score = anls_score(raw, gt_answers)

            results[str(idx)] = {
                "raw": raw,
                "gt_answers": gt_answers,
                "anls": score,
            }

            if len(results) % 100 == 0:
                with open(ckpt_file, "w") as f:
                    json.dump(results, f)

        with open(ckpt_file, "w") as f:
            json.dump(results, f)

    # Compute overall ANLS
    all_anls = [v["anls"] for v in results.values()]
    avg_anls = np.mean(all_anls) * 100 if all_anls else 0
    n = len(all_anls)

    print(f"  {condition_name}: ANLS={avg_anls:.2f} ({n} samples)")
    return avg_anls, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prep", "eval", "all"], default="all")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--rebuild-canvases", action="store_true",
        help="Force rebuild all canvases and clear old embeddings",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    data = download_infographicvqa()
    train_data = data["train"]
    val_data = data["val"]

    if args.phase in ("prep", "all"):
        # Build canvases
        n_canvases = build_canvases(train_data, rebuild=args.rebuild_canvases)

        # Compute embeddings
        canvas_emb, text_emb, query_emb = compute_embeddings(train_data, val_data, n_canvases)

        # Build retrieval map
        retrieval_map = build_retrieval_map(
            query_emb, canvas_emb, text_emb, ALPHA, TOP_K, SIMILARITY_THRESHOLD
        )

        # Save retrieval map
        with open(DATA_DIR / "retrieval_map.pkl", "wb") as f:
            pickle.dump(retrieval_map, f)

        print("\n=== Preparation Complete ===")
        if args.phase == "prep":
            return

    if args.phase in ("eval", "all"):
        # Load retrieval map
        if args.phase == "eval":
            retrieval_map_file = DATA_DIR / "retrieval_map.pkl"
            print(f"Loading retrieval map...")
            with open(retrieval_map_file, "rb") as f:
                retrieval_map = pickle.load(f)

        # Load VLM
        model, processor = load_vlm()

        # Evaluate baseline
        if not args.skip_baseline:
            print("\n=== Evaluating BASELINE ===")
            bl_anls, bl_n = evaluate_condition(
                "baseline", model, processor, val_data,
                retrieval_map, OUTPUT_DIR
            )
        else:
            bl_ckpt = OUTPUT_DIR / "checkpoint_baseline.json"
            if bl_ckpt.exists():
                with open(bl_ckpt) as f:
                    bl_results = json.load(f)
                bl_anls = np.mean([v["anls"] for v in bl_results.values()]) * 100
                bl_n = len(bl_results)
                print(f"\n  Baseline (loaded): ANLS={bl_anls:.2f} ({bl_n} samples)")
            else:
                bl_anls, bl_n = 0, 0

        # Evaluate MemCanvas
        print("\n=== Evaluating MEMCANVAS ===")
        mc_anls, mc_n = evaluate_condition(
            "memcanvas", model, processor, val_data,
            retrieval_map, OUTPUT_DIR
        )

        # Summary
        print(f"\n{'='*60}")
        print(f"InfographicVQA Results (val set, {mc_n} samples)")
        print(f"{'='*60}")
        print(f"  Baseline:  ANLS={bl_anls:.2f}")
        print(f"  MemCanvas: ANLS={mc_anls:.2f}")
        print(f"  Delta:     ANLS=+{mc_anls-bl_anls:.2f}")
        print(f"{'='*60}")

        summary = {
            "dataset": "InfographicVQA",
            "split": "val",
            "n_samples": mc_n,
            "baseline": {"anls": bl_anls},
            "memcanvas": {"anls": mc_anls},
            "delta": {"anls": mc_anls - bl_anls},
            "config": {
                "alpha": ALPHA, "top_k": TOP_K, "encoder": "CLIP-L/14",
                "vlm": "Qwen2.5-VL-7B", "max_new_tokens": 128,
            },
        }
        with open(OUTPUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
