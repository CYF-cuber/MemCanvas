#!/usr/bin/env python3
"""
Evaluate compressed-text MemCanvas on HotpotQA dev set.
Compares: baseline (no canvas), original canvas, light-compressed canvas, heavy-compressed canvas.
"""
import io, json, os, pickle, re, string, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/codex")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/cyf/codex/hotpotqa_data")
EXPERIMENT_DIR = Path("/home/cyf/memcanvas0402")
VLM_MODEL_PATH = "/home/cyf/Qwen2.5-VL-7B-Instruct"

ALPHA = 0.00
TOP_K = 2
SIMILARITY_THRESHOLD = 0.1
MAX_DEV = 500  # eval subset

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(str(s).lower())))

def compute_exact(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
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
# VLM
# ---------------------------------------------------------------------------
def load_vlm():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading Qwen2.5-VL-7B...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(VLM_MODEL_PATH)
    print("  VLM loaded")
    return model, processor

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def build_retrieval_map(query_emb, canvas_img_emb, canvas_txt_emb, alpha, top_k, threshold):
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

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def format_context(sample):
    parts = []
    for para in sample.get("paragraphs", []):
        title = para["title"]
        text = para["text"][:500]
        parts.append(f"[{title}]\n{text}")
    return "\n\n".join(parts)

def predict_baseline(model, processor, sample):
    context = format_context(sample)
    user_text = (
        "Use the following context passages to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {sample['question']}\n"
        "Answer concisely:"
    )
    content = [{"type": "text", "text": user_text}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()

def predict_memcanvas(model, processor, sample, retrieved, canvas_dir):
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
            canvas_path = Path(canvas_dir) / f"{canvas_idx:05d}.png"
            canvas_img = Image.open(canvas_path).convert("RGB")
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
    return processor.decode(gen_ids, skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_condition(condition_name, model, processor, dev_data,
                       retrieval_map, canvas_dir, output_dir, max_dev=MAX_DEV):
    ckpt_file = output_dir / f"checkpoint_{condition_name}.json"
    results = {}
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            results = json.load(f)
        print(f"  Resumed {condition_name}: {len(results)} done")

    done = set(results.keys())
    remaining = [i for i in range(min(len(dev_data), max_dev)) if str(i) not in done]

    if not remaining:
        print(f"  {condition_name} already complete ({len(results)} samples)")
    else:
        print(f"  Running {condition_name}: {len(remaining)} remaining")
        for idx in tqdm(remaining, desc=condition_name):
            sample = dev_data[idx]
            try:
                if condition_name == "baseline":
                    raw = predict_baseline(model, processor, sample)
                else:
                    retrieved = retrieval_map.get(idx, [])
                    raw = predict_memcanvas(model, processor, sample, retrieved, canvas_dir)
            except Exception as e:
                raw = ""
                print(f"\n  Error on {idx}: {e}")

            gt = sample["answer"]
            em = compute_exact(raw, gt)
            f1 = compute_f1(raw, gt)
            results[str(idx)] = {"raw": raw, "gt": gt, "em": em, "f1": f1}

            if len(results) % 100 == 0:
                with open(ckpt_file, "w") as f:
                    json.dump(results, f)

        with open(ckpt_file, "w") as f:
            json.dump(results, f)

    all_em = [v["em"] for v in results.values()]
    all_f1 = [v["f1"] for v in results.values()]
    em_avg = np.mean(all_em) * 100 if all_em else 0
    f1_avg = np.mean(all_f1) * 100 if all_f1 else 0
    print(f"  {condition_name}: EM={em_avg:.2f}%, F1={f1_avg:.2f}% ({len(all_em)} samples)")
    return em_avg, f1_avg, len(all_em)

# ---------------------------------------------------------------------------
# Query embeddings (for dev set)
# ---------------------------------------------------------------------------
def compute_query_embeddings(dev_data, max_dev):
    from transformers import CLIPProcessor, CLIPModel
    print(f"  Computing query embeddings for {max_dev} dev samples...")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    texts = [dev_data[i]["question"][:77] for i in range(min(len(dev_data), max_dev))]
    all_emb = []
    bs = 64
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        inp = proc(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            feat = clip.get_text_features(**inp)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        all_emb.append(feat.cpu().numpy())

    del clip, proc
    torch.cuda.empty_cache()
    return np.concatenate(all_emb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True, choices=["light", "heavy"])
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--max-dev", type=int, default=MAX_DEV)
    args = parser.parse_args()

    level = args.level
    max_dev = args.max_dev
    level_dir = EXPERIMENT_DIR / f"hotpotqa_{level}"
    canvas_dir = level_dir / "canvases"
    output_dir = EXPERIMENT_DIR / "results" / f"hotpotqa_{level}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading HotpotQA data...")
    meta = pickle.load(open(DATA_DIR / "hotpotqa_meta.pkl", "rb"))
    dev_data = meta["dev"]
    print(f"  Dev: {len(dev_data)} samples, evaluating {max_dev}")

    # Load embeddings
    img_emb_file = level_dir / "clip_img_emb.npy"
    txt_emb_file = level_dir / "clip_txt_emb.npy"
    query_emb_file = EXPERIMENT_DIR / "query_emb.npy"

    assert img_emb_file.exists(), f"Missing {img_emb_file}. Run build_compressed_canvases.py first."
    assert txt_emb_file.exists(), f"Missing {txt_emb_file}. Run build_compressed_canvases.py first."

    img_emb = np.load(img_emb_file)
    txt_emb = np.load(txt_emb_file)
    print(f"  Image embeddings: {img_emb.shape}")
    print(f"  Text embeddings: {txt_emb.shape}")

    # Query embeddings (compute or load cached)
    if query_emb_file.exists():
        query_emb = np.load(query_emb_file)
        if len(query_emb) < max_dev:
            query_emb = compute_query_embeddings(dev_data, max_dev)
            np.save(query_emb_file, query_emb)
    else:
        query_emb = compute_query_embeddings(dev_data, max_dev)
        np.save(query_emb_file, query_emb)

    print(f"  Query embeddings: {query_emb.shape}")

    # Build retrieval map
    retrieval_map = build_retrieval_map(
        query_emb, img_emb, txt_emb, ALPHA, TOP_K, SIMILARITY_THRESHOLD
    )

    # Load VLM
    model, processor = load_vlm()

    # Evaluate baseline (shared across levels)
    if not args.skip_baseline:
        print("\n=== Evaluating BASELINE ===")
        bl_em, bl_f1, bl_n = evaluate_condition(
            "baseline", model, processor, dev_data,
            retrieval_map, canvas_dir, output_dir, max_dev
        )
    else:
        bl_em, bl_f1, bl_n = 0, 0, 0
        # Try to load from existing results
        bl_ckpt = output_dir / "checkpoint_baseline.json"
        if bl_ckpt.exists():
            with open(bl_ckpt) as f:
                bl_results = json.load(f)
            bl_em = np.mean([v["em"] for v in bl_results.values()]) * 100
            bl_f1 = np.mean([v["f1"] for v in bl_results.values()]) * 100
            bl_n = len(bl_results)

    # Evaluate compressed MemCanvas
    print(f"\n=== Evaluating MEMCANVAS ({level} compression) ===")
    mc_em, mc_f1, mc_n = evaluate_condition(
        f"memcanvas_{level}", model, processor, dev_data,
        retrieval_map, canvas_dir, output_dir, max_dev
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"HotpotQA Results — {level} compression ({mc_n} dev samples)")
    print(f"{'='*60}")
    print(f"  Baseline:              EM={bl_em:.2f}%  F1={bl_f1:.2f}%")
    print(f"  MemCanvas ({level:5s}):  EM={mc_em:.2f}%  F1={mc_f1:.2f}%")
    print(f"  Delta:                 EM={mc_em-bl_em:+.2f}pp  F1={mc_f1-bl_f1:+.2f}pp")
    print(f"{'='*60}")

    summary = {
        "dataset": "HotpotQA",
        "compression": level,
        "split": "dev",
        "n_samples": mc_n,
        "baseline": {"em": bl_em, "f1": bl_f1},
        f"memcanvas_{level}": {"em": mc_em, "f1": mc_f1},
        "delta": {"em": mc_em - bl_em, "f1": mc_f1 - bl_f1},
        "config": {
            "alpha": ALPHA, "top_k": TOP_K, "encoder": "CLIP-L/14",
            "vlm": "Qwen2.5-VL-7B", "max_new_tokens": 64,
            "n_train_canvases": len(img_emb),
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
