#!/usr/bin/env python3
"""
VQA-RAD Experiment: Baseline vs MemCanvas.

Dataset: VQA-RAD (medical radiology VQA)
  - Train: 1,793
  - Test:  451
Metric: exact-match accuracy after normalized answer comparison

Phases:
  prep: download data, cache images, build canvases, compute embeddings, build retrieval map
  eval: run baseline and MemCanvas evaluation with Qwen2.5-VL-7B

Usage:
  python -u /home/cyf/codex/vqarad_experiment.py --phase prep
  CUDA_VISIBLE_DEVICES=0 python -u /home/cyf/codex/vqarad_experiment.py --phase eval
  CUDA_VISIBLE_DEVICES=0 python -u /home/cyf/codex/vqarad_experiment.py --phase all
"""

import argparse
import io
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/memory")
from memory_canvas.dynamic_canvas import DynamicCanvas, DynamicCanvasConfig

DATA_DIR = Path("/home/cyf/codex/vqarad_data")
OUTPUT_DIR = Path("/home/cyf/codex/vqarad_experiment")
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
VLM_MODEL_PATH = "/home/cyf/Qwen2.5-VL-7B-Instruct"

ALPHA = 0.50
TOP_K = 2
SIMILARITY_THRESHOLD = 0.1


def normalize_answer(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, answers: List[str]) -> float:
    p = normalize_answer(prediction)
    return float(any(p == normalize_answer(a) for a in answers))


def download_vqarad():
    cache_file = DATA_DIR / "vqarad_meta.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded cached VQA-RAD metadata: train={len(data['train'])}, test={len(data['test'])}")
        return data

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_img_dir = DATA_DIR / "images_train"
    test_img_dir = DATA_DIR / "images_test"
    train_img_dir.mkdir(exist_ok=True)
    test_img_dir.mkdir(exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset("flaviagiammarino/vqa-rad")

    def process_split(split_data, img_dir, split_name):
        samples = []
        for i, item in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            img_path = img_dir / f"{i:05d}.jpg"
            if not img_path.exists():
                img = item["image"]
                if img is not None:
                    img.convert("RGB").save(str(img_path), "JPEG", quality=90)
            samples.append({
                "id": i,
                "question": item.get("question", ""),
                "answers": [item.get("answer", "")],
                "image_path": str(img_path),
            })
        return samples

    train_samples = process_split(ds["train"], train_img_dir, "train")
    test_samples = process_split(ds["test"], test_img_dir, "test")

    data = {"train": train_samples, "test": test_samples}
    with open(cache_file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached VQA-RAD metadata: train={len(train_samples)}, test={len(test_samples)}")
    return data


def load_image(sample: dict) -> Optional[Image.Image]:
    path = sample.get("image_path", "")
    if path and os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    return None


def render_vqarad_canvas(sample: dict) -> bytes:
    canvas = DynamicCanvas(DynamicCanvasConfig(
        patch_size=640,
        font_size=18,
        padding=20,
        content_gap=10,
        show_patch_boundary=False,
    ))

    img = load_image(sample)
    if img is not None:
        canvas.add_text("[Radiology Image]", font_size=14, bold=True)
        canvas.add_image(img, max_height=420)

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


def build_canvases(train_data: List[dict], rebuild: bool = False):
    canvas_dir = DATA_DIR / "canvases"
    done_marker = DATA_DIR / "canvases_done.txt"

    if rebuild:
        if done_marker.exists():
            done_marker.unlink()
        if canvas_dir.exists():
            import shutil
            shutil.rmtree(canvas_dir)
        for emb_file in [
            DATA_DIR / "canvas_embeddings.npy",
            DATA_DIR / "canvas_text_embeddings.npy",
            DATA_DIR / "query_embeddings.npy",
            DATA_DIR / "retrieval_map.pkl",
        ]:
            if emb_file.exists():
                emb_file.unlink()

    if done_marker.exists():
        n = int(done_marker.read_text().strip())
        print(f"Canvases already built: {n}")
        return n

    canvas_dir.mkdir(exist_ok=True)
    for i, sample in enumerate(tqdm(train_data, desc="Building canvases")):
        out_path = canvas_dir / f"{i:05d}.png"
        if out_path.exists():
            continue
        canvas_bytes = render_vqarad_canvas(sample)
        with open(out_path, "wb") as f:
            f.write(canvas_bytes)

    n = len(train_data)
    done_marker.write_text(str(n))
    print(f"Built {n} canvases")
    return n


def load_canvas(idx: int) -> bytes:
    with open(DATA_DIR / "canvases" / f"{idx:05d}.png", "rb") as f:
        return f.read()


def compute_embeddings(train_data, test_data, n_canvases):
    canvas_emb_file = DATA_DIR / "canvas_embeddings.npy"
    text_emb_file = DATA_DIR / "canvas_text_embeddings.npy"
    query_emb_file = DATA_DIR / "query_embeddings.npy"

    if canvas_emb_file.exists() and text_emb_file.exists() and query_emb_file.exists():
        canvas_emb = np.load(canvas_emb_file)
        text_emb = np.load(text_emb_file)
        query_emb = np.load(query_emb_file)
        print(f"Loaded cached embeddings: img={canvas_emb.shape}, txt={text_emb.shape}, query={query_emb.shape}")
        return canvas_emb, text_emb, query_emb

    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device).eval()
    print(f"CLIP on {device}")

    if not canvas_emb_file.exists():
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
                feats = clip_model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_emb.append(feats.cpu().numpy())
        canvas_emb = np.concatenate(all_emb, axis=0)
        np.save(canvas_emb_file, canvas_emb)
    else:
        canvas_emb = np.load(canvas_emb_file)

    if not text_emb_file.exists():
        texts = [f"Question: {s['question']} Answer: {s['answers'][0]}" for s in train_data]
        all_emb = []
        batch_size = 64
        for i in tqdm(range(0, len(texts), batch_size), desc="Canvas txt emb"):
            batch = texts[i:i+batch_size]
            inputs = clip_processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = clip_model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_emb.append(feats.cpu().numpy())
        text_emb = np.concatenate(all_emb, axis=0)
        np.save(text_emb_file, text_emb)
    else:
        text_emb = np.load(text_emb_file)

    if not query_emb_file.exists():
        texts = [s['question'] for s in test_data]
        all_emb = []
        batch_size = 64
        for i in tqdm(range(0, len(texts), batch_size), desc="Query emb"):
            batch = texts[i:i+batch_size]
            inputs = clip_processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = clip_model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_emb.append(feats.cpu().numpy())
        query_emb = np.concatenate(all_emb, axis=0)
        np.save(query_emb_file, query_emb)
    else:
        query_emb = np.load(query_emb_file)

    return canvas_emb, text_emb, query_emb


def build_retrieval_map(query_emb, canvas_img_emb, canvas_txt_emb, alpha, top_k, threshold):
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
    return retrieval_map


def load_vlm():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(VLM_MODEL_PATH)
    return model, processor


def predict_baseline(model, processor, sample):
    img = load_image(sample)
    user_text = f"Look at the radiology image and answer the question.\nQuestion: {sample['question']}\nAnswer concisely:"
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
        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    gen_ids = outputs[0][inputs['input_ids'].shape[1]:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def predict_memcanvas(model, processor, sample, retrieved):
    img = load_image(sample)
    prompt_parts = []
    memory_images = []
    if retrieved:
        prompt_parts.append(
            "Below are memory canvas images from similar radiology questions answered before. "
            "Each canvas shows the radiology image, question, and correct answer. "
            "Use them as reference memory for the new question."
        )
        for canvas_idx, _ in retrieved:
            canvas_img = Image.open(io.BytesIO(load_canvas(canvas_idx))).convert("RGB")
            memory_images.append(canvas_img)

    prompt_parts.append(f"Question: {sample['question']}")
    prompt_parts.append("Answer concisely:")
    user_text = "\n".join(prompt_parts)

    content = []
    images = []
    for m in memory_images:
        content.append({"type": "image", "image": m})
        images.append(m)
    if img is not None:
        content.append({"type": "image", "image": img})
        images.append(img)
    content.append({"type": "text", "text": user_text})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    gen_ids = outputs[0][inputs['input_ids'].shape[1]:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def evaluate_condition(condition_name, model, processor, test_data, retrieval_map, output_dir):
    ckpt_file = output_dir / f"checkpoint_{condition_name}.json"
    results = {}
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            results = json.load(f)
        print(f"Resumed {condition_name}: {len(results)} done")

    done = set(results.keys())
    remaining = [i for i in range(len(test_data)) if str(i) not in done]
    for idx in tqdm(remaining, desc=condition_name):
        sample = test_data[idx]
        try:
            if condition_name == "baseline":
                raw = predict_baseline(model, processor, sample)
            else:
                retrieved = retrieval_map.get(idx, [])
                raw = predict_memcanvas(model, processor, sample, retrieved)
        except Exception as e:
            raw = ""
            print(f"Error on {idx}: {e}")

        gts = sample.get('answers', [])
        score = exact_match(raw, gts)
        results[str(idx)] = {"raw": raw, "gt_answers": gts, "accuracy": score}
        if len(results) % 50 == 0:
            with open(ckpt_file, 'w') as f:
                json.dump(results, f)

    with open(ckpt_file, 'w') as f:
        json.dump(results, f)
    scores = [v['accuracy'] for v in results.values()]
    return float(np.mean(scores) * 100), len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prep", "eval", "all"], default="all")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--rebuild-canvases", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = download_vqarad()
    train_data = data['train']
    test_data = data['test']

    if args.phase in ("prep", "all"):
        n_canvases = build_canvases(train_data, rebuild=args.rebuild_canvases)
        canvas_emb, text_emb, query_emb = compute_embeddings(train_data, test_data, n_canvases)
        retrieval_map = build_retrieval_map(query_emb, canvas_emb, text_emb, ALPHA, TOP_K, SIMILARITY_THRESHOLD)
        with open(DATA_DIR / "retrieval_map.pkl", "wb") as f:
            pickle.dump(retrieval_map, f)
        print("=== Preparation Complete ===")
        if args.phase == "prep":
            return

    if args.phase in ("eval", "all"):
        if args.phase == "eval":
            with open(DATA_DIR / "retrieval_map.pkl", "rb") as f:
                retrieval_map = pickle.load(f)
        model, processor = load_vlm()
        summary = {
            "dataset": "VQA-RAD",
            "split": "test",
            "n_samples": len(test_data),
            "config": {"alpha": ALPHA, "top_k": TOP_K, "encoder": "CLIP-L/14", "vlm": "Qwen2.5-VL-7B"},
        }
        if not args.skip_baseline:
            print("=== Evaluating BASELINE ===")
            acc, n = evaluate_condition("baseline", model, processor, test_data, retrieval_map, OUTPUT_DIR)
            summary["baseline"] = {"accuracy": acc}
        print("=== Evaluating MEMCANVAS ===")
        acc, n = evaluate_condition("memcanvas", model, processor, test_data, retrieval_map, OUTPUT_DIR)
        summary["memcanvas"] = {"accuracy": acc}
        if "baseline" in summary:
            summary["delta"] = {"accuracy": summary["memcanvas"]["accuracy"] - summary["baseline"]["accuracy"]}
        with open(OUTPUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
