#!/usr/bin/env python3
"""
MultiModalQA (MMQA) Experiment: Baseline vs MemCanvas.

Dataset: MMQA (ICLR 2021) — Cross-modal QA over Text, Tables, Images
Split:   dev (2,441 examples with ground truth)
Metrics: Exact Match (EM) and F1 (DROP-style)

Phases:
  prep: Load data, download images, build canvases, compute embeddings
  eval: Run VLM evaluation (baseline + memcanvas)

Usage:
  python -u /home/cyf/codex/mmqa_experiment.py --phase prep
  CUDA_VISIBLE_DEVICES=0 python -u /home/cyf/codex/mmqa_experiment.py --phase eval
  CUDA_VISIBLE_DEVICES=0 python -u /home/cyf/codex/mmqa_experiment.py --phase eval --skip-baseline
"""

import argparse
import gzip
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
DATA_DIR = Path("/home/cyf/codex/mmqa_data")
OUTPUT_DIR = Path("/home/cyf/codex/mmqa_experiment")
IMAGES_DIR = DATA_DIR / "final_dataset_images"

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
VLM_MODEL_PATH = "/home/cyf/Qwen2.5-VL-7B-Instruct"

ALPHA = 0.50
TOP_K = 2
SIMILARITY_THRESHOLD = 0.1


# ===================================================================
# Evaluation Metrics (DROP-style EM/F1)
# ===================================================================
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


def get_metrics(predicted: List[str], gold: List[str]) -> Tuple[float, float]:
    """Compute EM and F1, handling list answers with greedy alignment."""
    if not predicted:
        predicted = [""]
    if not gold:
        return 0.0, 0.0

    # Single answer case (most common)
    if len(predicted) == 1 and len(gold) == 1:
        return compute_exact(predicted[0], gold[0]), compute_f1(predicted[0], gold[0])

    # List answer: greedy alignment
    pred_norm = [normalize_answer(p) for p in predicted]
    gold_norm = [normalize_answer(g) for g in gold]

    # EM: sets must match
    em = float(set(pred_norm) == set(gold_norm)) if len(pred_norm) == len(gold_norm) else 0.0

    # F1: greedy 1-to-1 alignment
    pred_bags = [set(p.split()) for p in pred_norm]
    gold_bags = [set(g.split()) for g in gold_norm]

    # Build score matrix and greedy align
    scores = []
    for g_bag in gold_bags:
        row = []
        for p_bag in pred_bags:
            if not g_bag and not p_bag:
                row.append(1.0)
            elif not g_bag or not p_bag:
                row.append(0.0)
            else:
                common = g_bag & p_bag
                n = len(common)
                p_val = n / len(p_bag) if p_bag else 0
                r_val = n / len(g_bag) if g_bag else 0
                row.append((2 * p_val * r_val) / (p_val + r_val) if (p_val + r_val) > 0 else 0.0)
        scores.append(row)

    used_g, used_p = set(), set()
    candidates = []
    for i, row in enumerate(scores):
        for j, s in enumerate(row):
            candidates.append((s, i, j))
    candidates.sort(reverse=True)

    f1_scores = []
    for s, i, j in candidates:
        if i not in used_g and j not in used_p:
            f1_scores.append(s)
            used_g.add(i)
            used_p.add(j)
    f1_scores.extend([0.0] * (len(gold_bags) - len(f1_scores)))
    f1 = sum(f1_scores) / max(len(gold_bags), 1)

    return em, f1


# ===================================================================
# Data Loading
# ===================================================================
def load_jsonl_gz(filepath: str) -> List[dict]:
    data = []
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_mmqa_data():
    """Load all MMQA data files."""
    cache_file = DATA_DIR / "mmqa_parsed.pkl"
    if cache_file.exists():
        print(f"Loading cached MMQA data from {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        print(f"  Train: {len(data['train'])}, Dev: {len(data['dev'])}")
        print(f"  Tables: {len(data['tables'])}, Texts: {len(data['texts'])}, Images: {len(data['images'])}")
        return data

    print("Loading MMQA data files...")
    train = load_jsonl_gz(str(DATA_DIR / "MMQA_train.jsonl.gz"))
    dev = load_jsonl_gz(str(DATA_DIR / "MMQA_dev.jsonl.gz"))

    tables_raw = load_jsonl_gz(str(DATA_DIR / "MMQA_tables.jsonl.gz"))
    texts_raw = load_jsonl_gz(str(DATA_DIR / "MMQA_texts.jsonl.gz"))
    images_raw = load_jsonl_gz(str(DATA_DIR / "MMQA_images.jsonl.gz"))

    tables = {t["id"]: t for t in tables_raw}
    texts = {t["id"]: t for t in texts_raw}
    images = {t["id"]: t for t in images_raw}

    print(f"  Train: {len(train)}, Dev: {len(dev)}")
    print(f"  Tables: {len(tables)}, Texts: {len(texts)}, Images: {len(images)}")

    data = {"train": train, "dev": dev, "tables": tables, "texts": texts, "images": images}
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    print(f"  Cached to {cache_file}")
    return data


def load_image_by_id(images_meta: dict, doc_id: str) -> Optional[Image.Image]:
    """Load an image by its doc_id from the extracted images directory."""
    if doc_id not in images_meta:
        return None
    path = IMAGES_DIR / images_meta[doc_id]["path"]
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def table_to_markdown(table_data: dict, max_rows: int = 10) -> str:
    """Convert MMQA table to markdown string."""
    header = [h["column_name"] for h in table_data["header"]]
    rows = [[cell["text"] for cell in row] for row in table_data["table_rows"]]

    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[:max_rows]:
        # Truncate cells
        cells = [c[:50] for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows)} rows total)")
    return "\n".join(lines)


def table_to_list(table_data: dict, max_rows: int = 8) -> Tuple[List[str], List[List[str]]]:
    """Convert MMQA table to headers and rows lists for DynamicCanvas.add_table."""
    headers = [h["column_name"] for h in table_data["header"]]
    rows = []
    for row in table_data["table_rows"][:max_rows]:
        cells = [cell["text"][:40] for cell in row]
        rows.append(cells)
    return headers, rows


# ===================================================================
# Canvas Building
# ===================================================================
def build_canvases(train_data: List[dict], tables: dict, texts: dict, images_meta: dict):
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
    skipped = 0
    for sample in tqdm(train_data, desc="Building canvases"):
        try:
            canvas_bytes = render_mmqa_canvas(sample, tables, texts, images_meta)
            canvases.append(canvas_bytes)
        except Exception as e:
            # Fallback: minimal canvas
            canvas_bytes = render_minimal_canvas(sample)
            canvases.append(canvas_bytes)
            skipped += 1

    if skipped:
        print(f"  Warning: {skipped} canvases used fallback rendering")

    with open(cache_file, "wb") as f:
        pickle.dump(canvases, f)
    total_mb = sum(len(c) for c in canvases) / (1024 * 1024)
    print(f"  Cached: {len(canvases)} canvases, {total_mb:.1f} MB")
    return canvases


def render_mmqa_canvas(sample: dict, tables: dict, texts: dict, images_meta: dict) -> bytes:
    """Render an MMQA training sample to a canvas image."""
    canvas = DynamicCanvas(DynamicCanvasConfig(
        patch_size=640,
        font_size=16,
        padding=20,
        content_gap=10,
        show_patch_boundary=False,
    ))

    modalities = sample["metadata"]["modalities"]
    qtype = sample["metadata"]["type"]

    # Header
    canvas.add_text(f"[{qtype}] Modalities: {', '.join(modalities)}", font_size=12, bold=True)
    canvas.add_separator()

    # Context sections based on supporting_context
    for ctx in sample.get("supporting_context", []):
        doc_id = ctx["doc_id"]
        doc_part = ctx["doc_part"]

        if doc_part == "text" and doc_id in texts:
            text_doc = texts[doc_id]
            title = text_doc.get("title", "")
            passage = text_doc.get("text", "")[:300]  # truncate
            canvas.add_text(f"[Text] {title}", font_size=14, bold=True)
            canvas.add_text(passage, font_size=13)

        elif doc_part == "table" and doc_id in tables:
            table_doc = tables[doc_id]
            title = table_doc.get("title", "")
            canvas.add_text(f"[Table] {title}", font_size=14, bold=True)
            try:
                headers, rows = table_to_list(table_doc["table"], max_rows=6)
                if headers and rows:
                    canvas.add_table(rows, headers=headers)
            except Exception:
                # Fallback: markdown text
                md = table_to_markdown(table_doc["table"], max_rows=5)
                canvas.add_text(md, font_size=12)

        elif doc_part == "image":
            img = load_image_by_id(images_meta, doc_id)
            if img is not None:
                title = images_meta.get(doc_id, {}).get("title", "")
                canvas.add_text(f"[Image] {title}", font_size=14, bold=True)
                canvas.add_image(img, max_height=300)
            else:
                title = images_meta.get(doc_id, {}).get("title", "Image unavailable")
                canvas.add_text(f"[Image] {title} (not available)", font_size=14)

    canvas.add_separator()

    # Question
    canvas.add_text(f"Q: {sample['question']}", font_size=16, bold=True)

    # Answer
    answers = sample.get("answers", [])
    if answers:
        ans_str = str(answers[0]["answer"])
        canvas.add_text(f"✓ A: {ans_str}", font_size=16)

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
    """Minimal fallback canvas with just question and answer."""
    canvas = DynamicCanvas(DynamicCanvasConfig(
        patch_size=640, font_size=16, padding=20,
        content_gap=10, show_patch_boundary=False,
    ))
    canvas.add_text(f"Q: {sample['question']}", font_size=16, bold=True)
    answers = sample.get("answers", [])
    if answers:
        canvas.add_text(f"✓ A: {str(answers[0]['answer'])}", font_size=16)

    patches = canvas.get_images()
    img_out = patches[0] if len(patches) == 1 else patches[0]
    buf = io.BytesIO()
    img_out.save(buf, format="PNG")
    return buf.getvalue()


# ===================================================================
# Embeddings
# ===================================================================
def compute_embeddings(train_data, dev_data, canvases):
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

    # Canvas text embeddings (training question text)
    if not text_emb_file.exists():
        print(f"Computing canvas text embeddings ({len(train_data)})...")
        all_emb = []
        batch_size = 64
        for i in tqdm(range(0, len(train_data), batch_size), desc="Canvas txt emb"):
            batch_texts = [train_data[j]["question"] for j in range(i, min(i + batch_size, len(train_data)))]
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
            batch_texts = [dev_data[j]["question"] for j in range(i, min(i + batch_size, len(dev_data)))]
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


# ===================================================================
# Retrieval
# ===================================================================
def build_retrieval_map(query_emb, canvas_img_emb, canvas_txt_emb, alpha, top_k, threshold):
    """Build retrieval map: dev_idx -> [(canvas_idx, similarity), ...]."""
    print(f"Building retrieval map (α={alpha}, K={top_k})...")
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


# ===================================================================
# VLM Evaluation
# ===================================================================
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


def format_context(sample: dict, tables: dict, texts: dict, images_meta: dict) -> Tuple[str, List[Image.Image]]:
    """Format the context for a question into text and images."""
    text_parts = []
    context_images = []

    for ctx in sample.get("supporting_context", []):
        doc_id = ctx["doc_id"]
        doc_part = ctx["doc_part"]

        if doc_part == "text" and doc_id in texts:
            text_doc = texts[doc_id]
            title = text_doc.get("title", "")
            passage = text_doc.get("text", "")[:500]
            text_parts.append(f"[Text: {title}]\n{passage}")

        elif doc_part == "table" and doc_id in tables:
            table_doc = tables[doc_id]
            title = table_doc.get("title", "")
            md = table_to_markdown(table_doc["table"], max_rows=10)
            text_parts.append(f"[Table: {title}]\n{md}")

        elif doc_part == "image":
            img = load_image_by_id(images_meta, doc_id)
            if img is not None:
                context_images.append(img)
                title = images_meta.get(doc_id, {}).get("title", "")
                text_parts.append(f"[Image: {title}] (shown above)")
            else:
                title = images_meta.get(doc_id, {}).get("title", "")
                text_parts.append(f"[Image: {title}] (image not available)")

    context_text = "\n\n".join(text_parts) if text_parts else ""
    return context_text, context_images


def predict_baseline(model, processor, sample, tables, texts, images_meta):
    """Baseline: question + context → VLM answer."""
    context_text, context_images = format_context(sample, tables, texts, images_meta)

    prompt_parts = []
    if context_text:
        prompt_parts.append("Use the following context to answer the question.\n")
        prompt_parts.append(context_text)
        prompt_parts.append("")
    prompt_parts.append(f"Question: {sample['question']}")
    prompt_parts.append("Answer concisely:")
    user_text = "\n".join(prompt_parts)

    content = []
    for img in context_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": user_text})

    messages = [{"role": "user", "content": content}]
    all_images = context_images if context_images else None

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if all_images:
        inputs = processor(text=[text], images=all_images, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return raw


def predict_memcanvas(model, processor, sample, tables, texts, images_meta, canvases, retrieved):
    """MemCanvas: retrieved canvases + context + question → VLM answer."""
    context_text, context_images = format_context(sample, tables, texts, images_meta)

    prompt_parts = []
    memory_images = []

    if retrieved:
        prompt_parts.append(
            "Below are memory canvas images from similar questions answered before. "
            "Each canvas shows context information (text, tables, images), the question, and the correct answer. "
            "Study these canvases and use the knowledge to answer the new question."
        )
        prompt_parts.append("")
        for i, (canvas_idx, sim) in enumerate(retrieved):
            canvas_img = Image.open(io.BytesIO(canvases[canvas_idx])).convert("RGB")
            memory_images.append(canvas_img)
            prompt_parts.append(f"[Canvas {i+1}]")
        prompt_parts.append("")
        prompt_parts.append("---")
        prompt_parts.append("")

    if context_text:
        prompt_parts.append("Context for the current question:")
        prompt_parts.append(context_text)
        prompt_parts.append("")

    prompt_parts.append(f"Question: {sample['question']}")
    prompt_parts.append("Answer concisely:")
    user_text = "\n".join(prompt_parts)

    content = []
    for img in memory_images:
        content.append({"type": "image", "image": img})
    for img in context_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": user_text})

    messages = [{"role": "user", "content": content}]
    all_images = memory_images + context_images

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if all_images:
        inputs = processor(text=[text], images=all_images, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return raw


def evaluate_condition(
    condition_name, model, processor, dev_data, tables, texts, images_meta,
    canvases, retrieval_map, output_dir
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
                    raw = predict_baseline(model, processor, sample, tables, texts, images_meta)
                else:
                    retrieved = retrieval_map.get(idx, [])
                    raw = predict_memcanvas(
                        model, processor, sample, tables, texts, images_meta,
                        canvases, retrieved
                    )
            except Exception as e:
                raw = ""
                print(f"\n  Error on {idx}: {e}")

            # Compute metrics
            gold_answers = [str(a["answer"]) for a in sample.get("answers", [])]
            pred_list = [raw]  # single answer prediction
            em, f1 = get_metrics(pred_list, gold_answers)

            results[str(idx)] = {
                "raw": raw,
                "gt_answers": gold_answers,
                "em": em,
                "f1": f1,
            }

            if len(results) % 100 == 0:
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


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prep", "eval", "all"], default="all")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    data = load_mmqa_data()
    train_data = data["train"]
    dev_data = data["dev"]
    tables = data["tables"]
    texts = data["texts"]
    images_meta = data["images"]

    if args.phase in ("prep", "all"):
        # Check images
        if IMAGES_DIR.exists():
            n_images = len(list(IMAGES_DIR.glob("*")))
            print(f"Images directory: {IMAGES_DIR} ({n_images} files)")
        else:
            print(f"WARNING: Images directory not found: {IMAGES_DIR}")
            print("  Canvas building will proceed without images.")
            print("  To download: wget https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip")

        # Build canvases
        canvases = build_canvases(train_data, tables, texts, images_meta)

        # Compute embeddings
        canvas_emb, text_emb, query_emb = compute_embeddings(train_data, dev_data, canvases)

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
        # Load canvases if not already loaded
        if args.phase == "eval":
            canvases_file = DATA_DIR / "canvases.pkl"
            print(f"Loading canvases from {canvases_file}...")
            with open(canvases_file, "rb") as f:
                canvases = pickle.load(f)

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
                "baseline", model, processor, dev_data, tables, texts, images_meta,
                canvases, retrieval_map, OUTPUT_DIR
            )
        else:
            # Load existing baseline results
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
            "memcanvas", model, processor, dev_data, tables, texts, images_meta,
            canvases, retrieval_map, OUTPUT_DIR
        )

        # Summary
        print(f"\n{'='*60}")
        print(f"MMQA Results (dev set, {mc_n} samples)")
        print(f"{'='*60}")
        print(f"  Baseline:  EM={bl_em:.2f}%  F1={bl_f1:.2f}%")
        print(f"  MemCanvas: EM={mc_em:.2f}%  F1={mc_f1:.2f}%")
        print(f"  Delta:     EM=+{mc_em-bl_em:.2f}pp  F1=+{mc_f1-bl_f1:.2f}pp")
        print(f"{'='*60}")

        summary = {
            "dataset": "MMQA",
            "split": "dev",
            "n_samples": mc_n,
            "baseline": {"em": bl_em, "f1": bl_f1},
            "memcanvas": {"em": mc_em, "f1": mc_f1},
            "delta": {"em": mc_em - bl_em, "f1": mc_f1 - bl_f1},
            "config": {
                "alpha": ALPHA, "top_k": TOP_K, "encoder": "CLIP-L/14",
                "vlm": "Qwen2.5-VL-7B", "max_new_tokens": 64,
            },
        }
        with open(OUTPUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
