#!/usr/bin/env python3
"""
OK-VQA Experiment: Baseline vs MemCanvas.

Pipeline:
  Phase 1: Download OK-VQA data and cache locally
  Phase 2: Build memory canvases from training set
  Phase 3: Compute CLIP embeddings (canvases + test queries)
  Phase 4: Evaluate Baseline (no memory) and MemCanvas (with memory)

Usage:
  CUDA_VISIBLE_DEVICES=1 python -u /home/cyf/codex/okvqa_experiment.py
  CUDA_VISIBLE_DEVICES=1 python -u /home/cyf/codex/okvqa_experiment.py --phase eval
  CUDA_VISIBLE_DEVICES=1 python -u /home/cyf/codex/okvqa_experiment.py --phase eval --skip-baseline
"""

import argparse
import io
import json
import os
import pickle
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
DATA_DIR = Path("/home/cyf/codex/okvqa_data")
OUTPUT_DIR = Path("/home/cyf/codex/okvqa_experiment")
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
VLM_MODEL_PATH = "/home/cyf/Qwen2.5-VL-7B-Instruct"

ALPHA = 0.50
TOP_K = 2
SIMILARITY_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# VQA Accuracy (standard soft accuracy)
# ---------------------------------------------------------------------------
def vqa_accuracy(prediction: str, gt_answers: List[str]) -> float:
    """Standard VQA accuracy: min(1, #matching / 3)."""
    pred = prediction.strip().lower()
    # Normalize common variations
    pred = pred.rstrip(".")
    matching = sum(1 for a in gt_answers if a.strip().lower() == pred)
    return min(1.0, matching / 3.0)


# ---------------------------------------------------------------------------
# Phase 1: Download and cache OK-VQA data
# ---------------------------------------------------------------------------
def download_okvqa():
    """Download OK-VQA from HuggingFace and cache locally."""
    cache_file = DATA_DIR / "okvqa_cached.pkl"
    if cache_file.exists():
        print(f"Loading cached OK-VQA data from {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        print(f"  Train: {len(data['train'])} samples")
        print(f"  Test: {len(data['test'])} samples")
        return data

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading OK-VQA from HuggingFace...")

    from datasets import load_dataset

    train_ds = load_dataset("Multimodal-Fatima/OK-VQA_train", split="train")
    test_ds = load_dataset("Multimodal-Fatima/OK-VQA_test", split="test")

    print(f"  Raw train: {len(train_ds)}, test: {len(test_ds)}")

    # Process and cache
    train_data = []
    img_dir = DATA_DIR / "images" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(tqdm(train_ds, desc="Caching train")):
        img_path = img_dir / f"{i:05d}.jpg"
        if not img_path.exists():
            sample["image"].save(img_path, format="JPEG", quality=95)
        train_data.append({
            "id": i,
            "question_id": sample.get("question_id", i),
            "question": sample["question"],
            "answers": sample["answers"],
            "image_path": str(img_path),
            "caption": sample.get("blip_caption_beam_5", ""),
        })

    test_data = []
    img_dir = DATA_DIR / "images" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(tqdm(test_ds, desc="Caching test")):
        img_path = img_dir / f"{i:05d}.jpg"
        if not img_path.exists():
            sample["image"].save(img_path, format="JPEG", quality=95)
        test_data.append({
            "id": i,
            "question_id": sample.get("question_id", i),
            "question": sample["question"],
            "answers": sample["answers"],
            "image_path": str(img_path),
            "caption": sample.get("blip_caption_beam_5", ""),
        })

    data = {"train": train_data, "test": test_data}
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    print(f"  Cached to {cache_file}")
    return data


# ---------------------------------------------------------------------------
# Phase 2: Build memory canvases from training set
# ---------------------------------------------------------------------------
def build_canvases(train_data: List[Dict]) -> List[bytes]:
    """Build canvas images for each training sample."""
    cache_file = DATA_DIR / "canvases.pkl"
    if cache_file.exists():
        print(f"Loading cached canvases from {cache_file}")
        with open(cache_file, "rb") as f:
            canvases = pickle.load(f)
        print(f"  {len(canvases)} canvases loaded")
        return canvases

    print(f"Building {len(train_data)} canvases...")
    canvases = []
    for sample in tqdm(train_data, desc="Building canvases"):
        canvas_bytes = render_okvqa_canvas(sample)
        canvases.append(canvas_bytes)

    with open(cache_file, "wb") as f:
        pickle.dump(canvases, f)
    print(f"  Cached to {cache_file}")
    return canvases


def render_okvqa_canvas(sample: Dict) -> bytes:
    """Render an OK-VQA training sample to a canvas image."""
    canvas = DynamicCanvas(DynamicCanvasConfig(
        patch_size=640,
        font_size=18,
        padding=25,
        content_gap=12,
        show_patch_boundary=False,
    ))

    # Image
    try:
        img = Image.open(sample["image_path"])
        canvas.add_image(img, max_height=400)
    except Exception:
        pass

    # Caption
    caption = sample.get("caption", "")
    if caption:
        canvas.add_text(f"[Caption] {caption}", font_size=14)

    canvas.add_separator()

    # Question
    canvas.add_text(f"Question: {sample['question']}", font_size=18)

    # Answers (show most common)
    answers = sample.get("answers", [])
    if answers:
        answer_counts = Counter(a.strip().lower() for a in answers)
        top_answer = answer_counts.most_common(1)[0][0]
        canvas.add_text(f"✓ Answer: {top_answer}", font_size=18)
        # Also show other frequent answers
        other = [a for a, c in answer_counts.most_common(3)[1:] if c >= 2]
        if other:
            canvas.add_text(f"  Also accepted: {', '.join(other)}", font_size=14)

    # Combine patches into single image
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


# ---------------------------------------------------------------------------
# Phase 3: Compute CLIP embeddings
# ---------------------------------------------------------------------------
def compute_embeddings(train_data, test_data, canvases):
    """Compute CLIP embeddings for canvases (image) and test queries (text)."""
    canvas_emb_file = DATA_DIR / "canvas_embeddings.npy"
    text_emb_file = DATA_DIR / "canvas_text_embeddings.npy"
    query_emb_file = DATA_DIR / "query_embeddings.npy"

    if canvas_emb_file.exists() and text_emb_file.exists() and query_emb_file.exists():
        print("Loading cached embeddings...")
        canvas_emb = np.load(canvas_emb_file)
        text_emb = np.load(text_emb_file)
        query_emb = np.load(query_emb_file)
        print(f"  Canvas img emb: {canvas_emb.shape}")
        print(f"  Canvas txt emb: {text_emb.shape}")
        print(f"  Query emb: {query_emb.shape}")
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
        print(f"Computing canvas image embeddings ({len(canvases)})...")
        all_emb = []
        batch_size = 32
        for i in tqdm(range(0, len(canvases), batch_size), desc="Canvas img emb"):
            batch_imgs = []
            for j in range(i, min(i + batch_size, len(canvases))):
                img = Image.open(io.BytesIO(canvases[j])).convert("RGB")
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

    # Canvas text embeddings (from training question text)
    if not text_emb_file.exists():
        print(f"Computing canvas text embeddings ({len(train_data)})...")
        all_emb = []
        batch_size = 64
        for i in tqdm(range(0, len(train_data), batch_size), desc="Canvas txt emb"):
            batch_texts = []
            for j in range(i, min(i + batch_size, len(train_data))):
                q = train_data[j]["question"]
                batch_texts.append(q)
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

    # Query embeddings (test question text)
    if not query_emb_file.exists():
        print(f"Computing query embeddings ({len(test_data)})...")
        all_emb = []
        batch_size = 64
        for i in tqdm(range(0, len(test_data), batch_size), desc="Query emb"):
            batch_texts = []
            for j in range(i, min(i + batch_size, len(test_data))):
                q = test_data[j]["question"]
                batch_texts.append(q)
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

    # Free CLIP
    del clip_model, clip_processor
    torch.cuda.empty_cache()

    return canvas_emb, text_emb, query_emb


# ---------------------------------------------------------------------------
# Phase 4: Retrieval
# ---------------------------------------------------------------------------
def build_retrieval_map(query_emb, canvas_img_emb, canvas_txt_emb, alpha, top_k, threshold):
    """Build retrieval map: test_idx -> [(canvas_idx, similarity), ...]."""
    print(f"Building retrieval map (α={alpha}, K={top_k})...")
    # Hybrid key
    key_emb = alpha * canvas_img_emb + (1 - alpha) * canvas_txt_emb
    # Normalize
    k_n = np.linalg.norm(key_emb, axis=1, keepdims=True)
    k_n[k_n == 0] = 1.0
    key_norm = key_emb / k_n

    q_n = np.linalg.norm(query_emb, axis=1, keepdims=True)
    q_n[q_n == 0] = 1.0
    q_norm = query_emb / q_n

    sims = q_norm @ key_norm.T  # (n_test, n_train)

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
    print(f"  {has_mem}/{len(query_emb)} test samples have memories")
    return retrieval_map


# ---------------------------------------------------------------------------
# Phase 5: VLM Evaluation
# ---------------------------------------------------------------------------
def load_vlm():
    """Load Qwen2.5-VL-7B."""
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
    """Baseline prediction: image + question only."""
    img = Image.open(sample["image_path"]).convert("RGB")

    user_text = (
        "Look at the image and answer the question with a short phrase.\n"
        f"Question: {sample['question']}\n"
        "Answer:"
    )
    content = [
        {"type": "image", "image": img},
        {"type": "text", "text": user_text},
    ]
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return raw


def predict_memcanvas(model, processor, sample, canvases, retrieved):
    """MemCanvas prediction: retrieved canvases + image + question."""
    img = Image.open(sample["image_path"]).convert("RGB")

    # Build prompt with memory canvases
    prompt_parts = []
    memory_images = []

    if retrieved:
        prompt_parts.append(
            "Below are memory canvas images from similar questions you answered before."
        )
        prompt_parts.append(
            "Each canvas shows: an image, the question, and the correct answer."
        )
        prompt_parts.append(
            "Study these canvases and use the knowledge to answer the new question."
        )
        prompt_parts.append("")
        for i, (canvas_idx, sim) in enumerate(retrieved):
            canvas_img = Image.open(io.BytesIO(canvases[canvas_idx])).convert("RGB")
            memory_images.append(canvas_img)
            prompt_parts.append(f"[Canvas {i+1}]")
        prompt_parts.append("")
        prompt_parts.append("---")

    prompt_parts.append("Now look at the new image and answer the question with a short phrase.")
    prompt_parts.append(f"Question: {sample['question']}")
    prompt_parts.append("Answer:")
    user_text = "\n".join(prompt_parts)

    content = []
    for mem_img in memory_images:
        content.append({"type": "image", "image": mem_img})
    content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": user_text})

    messages = [{"role": "user", "content": content}]
    all_images = memory_images + [img]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=all_images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return raw


def evaluate_condition(
    condition_name, model, processor, test_data, canvases, retrieval_map,
    output_dir, skip_existing=True
):
    """Evaluate a single condition (baseline or memcanvas)."""
    ckpt_file = output_dir / f"checkpoint_{condition_name}.json"

    results = {"predictions": {}, "total_acc": 0.0}
    if skip_existing and ckpt_file.exists():
        with open(ckpt_file) as f:
            results = json.load(f)
        print(f"  Resumed {condition_name}: {len(results['predictions'])} done")

    done = set(results["predictions"].keys())
    remaining = [i for i in range(len(test_data)) if str(i) not in done]

    if not remaining:
        print(f"  {condition_name} already complete")
    else:
        print(f"  Running {condition_name}: {len(remaining)} remaining")
        for idx in tqdm(remaining, desc=condition_name):
            sample = test_data[idx]
            if condition_name == "baseline":
                raw = predict_baseline(model, processor, sample)
            else:
                retrieved = retrieval_map.get(idx, [])
                raw = predict_memcanvas(model, processor, sample, canvases, retrieved)

            gt_answers = sample["answers"]
            acc = vqa_accuracy(raw, gt_answers)
            results["predictions"][str(idx)] = {
                "raw": raw,
                "gt_answers": gt_answers,
                "accuracy": acc,
            }

            if (len(results["predictions"]) % 200 == 0):
                # Save checkpoint
                with open(ckpt_file, "w") as f:
                    json.dump(results, f)

        with open(ckpt_file, "w") as f:
            json.dump(results, f)

    # Compute overall accuracy
    all_acc = [v["accuracy"] for v in results["predictions"].values()]
    overall = np.mean(all_acc) * 100 if all_acc else 0
    results["total_acc"] = overall
    results["n_samples"] = len(all_acc)

    with open(ckpt_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  {condition_name}: {overall:.2f}% ({len(all_acc)} samples)")
    return overall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["all", "prep", "eval"])
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")

    # Phase 1: Download data
    data = download_okvqa()
    train_data = data["train"]
    test_data = data["test"]

    if args.phase in ("all", "prep"):
        # Phase 2: Build canvases
        canvases = build_canvases(train_data)

        # Phase 3: Compute embeddings
        canvas_img_emb, canvas_txt_emb, query_emb = compute_embeddings(
            train_data, test_data, canvases
        )
    else:
        # Load existing
        canvases_file = DATA_DIR / "canvases.pkl"
        with open(canvases_file, "rb") as f:
            canvases = pickle.load(f)
        canvas_img_emb = np.load(DATA_DIR / "canvas_embeddings.npy")
        canvas_txt_emb = np.load(DATA_DIR / "canvas_text_embeddings.npy")
        query_emb = np.load(DATA_DIR / "query_embeddings.npy")

    if args.phase == "prep":
        print("\nPreparation complete. Run with --phase eval to evaluate.")
        return

    # Phase 4: Build retrieval map
    retrieval_map = build_retrieval_map(
        query_emb, canvas_img_emb, canvas_txt_emb,
        ALPHA, TOP_K, SIMILARITY_THRESHOLD
    )

    # Phase 5: Load VLM and evaluate
    model, processor = load_vlm()

    baseline_acc = None
    memcanvas_acc = None

    if not args.skip_baseline:
        baseline_acc = evaluate_condition(
            "baseline", model, processor, test_data, canvases, retrieval_map,
            OUTPUT_DIR
        )

    memcanvas_acc = evaluate_condition(
        "memcanvas", model, processor, test_data, canvases, retrieval_map,
        OUTPUT_DIR
    )

    # Summary
    print(f"\n{'='*50}")
    print("OK-VQA RESULTS:")
    if baseline_acc is not None:
        print(f"  Baseline (no memory):  {baseline_acc:.2f}%")
    print(f"  MemCanvas:             {memcanvas_acc:.2f}%")
    if baseline_acc is not None:
        print(f"  Improvement:           +{memcanvas_acc - baseline_acc:.2f}pp")
    print(f"{'='*50}")

    # Save final summary
    summary = {
        "dataset": "OK-VQA",
        "config": {
            "encoder": "CLIP-L/14",
            "vlm": "Qwen2.5-VL-7B",
            "alpha": ALPHA,
            "top_k": TOP_K,
            "n_train": len(train_data),
            "n_test": len(test_data),
        },
        "baseline_acc": baseline_acc,
        "memcanvas_acc": memcanvas_acc,
        "improvement": (memcanvas_acc - baseline_acc) if baseline_acc else None,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
