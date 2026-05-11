#!/usr/bin/env python3
"""
HotpotQA Experiment: Baseline vs MemCanvas (Text-only).

Dataset: HotpotQA (EMNLP 2018, Yang et al.) — Multi-hop QA
  - Distractor setting: train 90,447 / dev 7,405
  - Metrics: EM (Exact Match) and F1 (token-level)

This is a TEXT-ONLY benchmark (no images). MemCanvas creates text-based
canvases from training examples, uses CLIP text encoder for retrieval,
and provides canvas images as visual context to the VLM.

Phases:
  prep: Load data, build canvases, compute CLIP embeddings
  eval: Run VLM evaluation (baseline + memcanvas)

Usage:
  python -u /home/cyf/codex/hotpotqa_experiment.py --phase prep
  CUDA_VISIBLE_DEVICES=0 python -u /home/cyf/codex/hotpotqa_experiment.py --phase eval
  CUDA_VISIBLE_DEVICES=0 python -u /home/cyf/codex/hotpotqa_experiment.py --phase eval --skip-baseline
"""

import argparse
import io
import json
import os
import pickle
import re
import string
import sys
import time
from collections import Counter
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
DATA_DIR = Path("/home/cyf/codex/hotpotqa_data")
OUTPUT_DIR = Path("/home/cyf/codex/hotpotqa_experiment")
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
VLM_MODEL_PATH = "/home/cyf/Qwen2.5-VL-7B-Instruct"

ALPHA = 0.00  # Text-only keys (no image component for text-only benchmark)
TOP_K = 2
SIMILARITY_THRESHOLD = 0.1
MAX_TRAIN = 50000  # Use a subset of train for memory bank (90k is huge)


# ---------------------------------------------------------------------------
# Evaluation Metrics (DROP-style EM/F1)
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def compute_exact(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def download_hotpotqa():
    """Download HotpotQA from HuggingFace and cache locally."""
    cache_file = DATA_DIR / "hotpotqa_meta.pkl"
    if cache_file.exists():
        print(f"Loading cached HotpotQA metadata from {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        print(f"  Train: {len(data['train'])} samples")
        print(f"  Dev: {len(data['dev'])} samples")
        return data

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading HotpotQA from HuggingFace...")

    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")

    def process_split(split_data, split_name, max_samples=None):
        samples = []
        for i, item in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            if max_samples and i >= max_samples:
                break
            # Extract supporting context paragraphs
            context_titles = item["context"]["title"]
            context_sentences = item["context"]["sentences"]
            paragraphs = []
            for title, sents in zip(context_titles, context_sentences):
                para_text = " ".join(sents)
                paragraphs.append({"title": title, "text": para_text})

            # Supporting facts
            sf_titles = item["supporting_facts"]["title"]
            sf_sent_ids = item["supporting_facts"]["sent_id"]

            sample = {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "type": item["type"],  # "comparison" or "bridge"
                "level": item["level"],  # "easy", "medium", "hard"
                "paragraphs": paragraphs,
                "supporting_facts": list(zip(sf_titles, sf_sent_ids)),
            }
            samples.append(sample)
        return samples

    train_samples = process_split(ds["train"], "train", max_samples=MAX_TRAIN)
    dev_samples = process_split(ds["validation"], "dev")

    data = {"train": train_samples, "dev": dev_samples}
    with open(cache_file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached: train={len(train_samples)}, dev={len(dev_samples)}")
    return data


# ---------------------------------------------------------------------------
# Canvas Building
# ---------------------------------------------------------------------------
def build_canvases(train_data: List[dict]):
    """Build text-based canvas images for each training sample."""
    canvas_dir = DATA_DIR / "canvases"
    done_marker = DATA_DIR / "canvases_done.txt"

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
            canvas_bytes = render_hotpotqa_canvas(sample)
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


def render_hotpotqa_canvas(sample: dict) -> bytes:
    """Render a HotpotQA training sample to a text-based canvas image."""
    canvas = DynamicCanvas(DynamicCanvasConfig(
        patch_size=640,
        font_size=14,
        padding=20,
        content_gap=8,
        show_patch_boundary=False,
    ))

    # Header with question type and level
    qtype = sample.get("type", "")
    level = sample.get("level", "")
    canvas.add_text(f"[HotpotQA] Type: {qtype} | Level: {level}", font_size=11, bold=True)
    canvas.add_separator()

    # Supporting context (only supporting paragraphs, truncated)
    sf_titles = set(t for t, _ in sample.get("supporting_facts", []))
    for para in sample.get("paragraphs", []):
        title = para["title"]
        if title in sf_titles:
            text = para["text"][:300]
            if len(para["text"]) > 300:
                text += "..."
            canvas.add_text(f"[{title}]", font_size=13, bold=True)
            canvas.add_text(text, font_size=12)

    canvas.add_separator()

    # Question
    canvas.add_text(f"Q: {sample['question']}", font_size=15, bold=True)

    # Answer
    canvas.add_text(f"\u2713 A: {sample['answer']}", font_size=15)

    # Combine patches
    patches = canvas.get_images()
    if len(patches) == 1:
        img_out = patches[0]
    else:
        total_h = sum(p.height for p in patches)
        max_w = max(p.width for p in patches)
        img_out = Image.new("RGB", (max_w, total_h), (255, 255, 255))
        y = 0
        for p in patches:
            img_out.paste(p, (0, y))
            y += p.height

    buf = io.BytesIO()
    img_out.save(buf, format="PNG")
    return buf.getvalue()


def render_minimal_canvas(sample: dict) -> bytes:
    """Minimal fallback canvas."""
    canvas = DynamicCanvas(DynamicCanvasConfig(
        patch_size=640, font_size=14, padding=20,
        content_gap=8, show_patch_boundary=False,
    ))
    canvas.add_text(f"Q: {sample['question']}", font_size=15, bold=True)
    canvas.add_text(f"\u2713 A: {sample['answer']}", font_size=15)

    patches = canvas.get_images()
    img_out = patches[0]
    buf = io.BytesIO()
    img_out.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def compute_embeddings(train_data, dev_data, n_canvases):
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

    # Canvas text embeddings (question + answer)
    if not text_emb_file.exists():
        print(f"Computing canvas text embeddings ({len(train_data)})...")
        all_emb = []
        batch_size = 64
        for i in tqdm(range(0, len(train_data), batch_size), desc="Canvas txt emb"):
            batch_texts = []
            for j in range(i, min(i + batch_size, len(train_data))):
                s = train_data[j]
                text = s["question"] + " " + s["answer"]
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

    # Query embeddings (dev question text)
    if not query_emb_file.exists():
        print(f"Computing query embeddings ({len(dev_data)})...")
        all_emb = []
        batch_size = 64
        for i in tqdm(range(0, len(dev_data), batch_size), desc="Query emb"):
            batch_texts = []
            for j in range(i, min(i + batch_size, len(dev_data))):
                batch_texts.append(dev_data[j]["question"])
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
    """Build retrieval map: dev_idx -> [(canvas_idx, similarity), ...]."""
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
    print(f"  {has_mem}/{len(query_emb)} dev samples have memories")
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


def format_context(sample: dict) -> str:
    """Format context paragraphs for the prompt."""
    parts = []
    for para in sample.get("paragraphs", []):
        title = para["title"]
        text = para["text"][:500]
        parts.append(f"[{title}]\n{text}")
    return "\n\n".join(parts)


def predict_baseline(model, processor, sample):
    """Baseline: context paragraphs + question -> VLM answer."""
    context = format_context(sample)

    prompt_parts = []
    prompt_parts.append("Use the following context passages to answer the question.")
    prompt_parts.append("")
    prompt_parts.append(context)
    prompt_parts.append("")
    prompt_parts.append(f"Question: {sample['question']}")
    prompt_parts.append("Answer concisely:")
    user_text = "\n".join(prompt_parts)

    content = [{"type": "text", "text": user_text}]
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return raw


def predict_memcanvas(model, processor, sample, retrieved):
    """MemCanvas: retrieved canvases + context + question -> VLM answer."""
    context = format_context(sample)

    prompt_parts = []
    memory_images = []

    if retrieved:
        prompt_parts.append(
            "Below are memory canvas images from similar questions answered before. "
            "Each canvas shows: relevant context passages, the question, and the correct answer. "
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

    prompt_parts.append("Use the following context passages to answer the question.")
    prompt_parts.append("")
    prompt_parts.append(context)
    prompt_parts.append("")
    prompt_parts.append(f"Question: {sample['question']}")
    prompt_parts.append("Answer concisely:")
    user_text = "\n".join(prompt_parts)

    content = []
    for img in memory_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": user_text})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if memory_images:
        inputs = processor(text=[text], images=memory_images, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return raw


def evaluate_condition(
    condition_name, model, processor, dev_data,
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
    remaining = [i for i in range(len(dev_data)) if str(i) not in done]

    if not remaining:
        print(f"  {condition_name} already complete")
    else:
        print(f"  Running {condition_name}: {len(remaining)} remaining")
        for idx in tqdm(remaining, desc=condition_name):
            sample = dev_data[idx]

            try:
                if condition_name == "baseline":
                    raw = predict_baseline(model, processor, sample)
                else:
                    retrieved = retrieval_map.get(idx, [])
                    raw = predict_memcanvas(model, processor, sample, retrieved)
            except Exception as e:
                raw = ""
                print(f"\n  Error on {idx}: {e}")

            gt = sample["answer"]
            em = compute_exact(raw, gt)
            f1 = compute_f1(raw, gt)

            results[str(idx)] = {
                "raw": raw,
                "gt": gt,
                "em": em,
                "f1": f1,
            }

            if len(results) % 200 == 0:
                with open(ckpt_file, "w") as f:
                    json.dump(results, f)

        with open(ckpt_file, "w") as f:
            json.dump(results, f)

    # Compute overall metrics
    all_em = [v["em"] for v in results.values()]
    all_f1 = [v["f1"] for v in results.values()]
    em_avg = np.mean(all_em) * 100 if all_em else 0
    f1_avg = np.mean(all_f1) * 100 if all_f1 else 0
    n = len(all_em)

    print(f"  {condition_name}: EM={em_avg:.2f}%, F1={f1_avg:.2f}% ({n} samples)")
    return em_avg, f1_avg, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prep", "eval", "all"], default="all")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    data = download_hotpotqa()
    train_data = data["train"]
    dev_data = data["dev"]

    if args.phase in ("prep", "all"):
        # Build canvases
        n_canvases = build_canvases(train_data)

        # Compute embeddings
        canvas_emb, text_emb, query_emb = compute_embeddings(train_data, dev_data, n_canvases)

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
            bl_em, bl_f1, bl_n = evaluate_condition(
                "baseline", model, processor, dev_data,
                retrieval_map, OUTPUT_DIR
            )
        else:
            bl_ckpt = OUTPUT_DIR / "checkpoint_baseline.json"
            if bl_ckpt.exists():
                with open(bl_ckpt) as f:
                    bl_results = json.load(f)
                bl_em = np.mean([v["em"] for v in bl_results.values()]) * 100
                bl_f1 = np.mean([v["f1"] for v in bl_results.values()]) * 100
                bl_n = len(bl_results)
                print(f"\n  Baseline (loaded): EM={bl_em:.2f}%, F1={bl_f1:.2f}% ({bl_n} samples)")
            else:
                bl_em, bl_f1, bl_n = 0, 0, 0

        # Evaluate MemCanvas
        print("\n=== Evaluating MEMCANVAS ===")
        mc_em, mc_f1, mc_n = evaluate_condition(
            "memcanvas", model, processor, dev_data,
            retrieval_map, OUTPUT_DIR
        )

        # Summary
        print(f"\n{'='*60}")
        print(f"HotpotQA Results (dev set, {mc_n} samples)")
        print(f"{'='*60}")
        print(f"  Baseline:  EM={bl_em:.2f}%  F1={bl_f1:.2f}%")
        print(f"  MemCanvas: EM={mc_em:.2f}%  F1={mc_f1:.2f}%")
        print(f"  Delta:     EM=+{mc_em-bl_em:.2f}pp  F1=+{mc_f1-bl_f1:.2f}pp")
        print(f"{'='*60}")

        summary = {
            "dataset": "HotpotQA",
            "split": "dev",
            "n_samples": mc_n,
            "baseline": {"em": bl_em, "f1": bl_f1},
            "memcanvas": {"em": mc_em, "f1": mc_f1},
            "delta": {"em": mc_em - bl_em, "f1": mc_f1 - bl_f1},
            "config": {
                "alpha": ALPHA, "top_k": TOP_K, "encoder": "CLIP-L/14",
                "vlm": "Qwen2.5-VL-7B", "max_new_tokens": 64,
                "max_train": MAX_TRAIN,
            },
        }
        with open(OUTPUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
