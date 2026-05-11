#!/usr/bin/env python3
"""
Ablation: Embedding fusion methods — Concatenation vs Weighted Sum.

Current:  key = α·img + (1-α)·txt  (768-dim)
New:      key = [img; txt]          (1536-dim, concatenation)

For queries, also concatenate [query_img; query_txt].
Test questions without images use zero vectors for img part.

Usage:
  CUDA_VISIBLE_DEVICES=1 python -u /home/cyf/codex/ablation_concat_fusion.py
"""

import copy
import io
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/cyf/memory/memory_canvas/experiments")
from scienceqa_qwen25vl_full_experiment import (
    ExperimentConfig,
    MemoryEntry,
    Qwen25VLEvaluator,
    ScienceQADataLoader,
)

sys.path.insert(0, "/home/cyf/codex")
from prompt_improvement_eval import build_prompt_v2, extract_answer_last

# ---------------------------------------------------------------------------
MEMORY_INDEX = (
    "/home/cyf/memory/experiments/scienceqa_qwen25vl_full/"
    "memory_index_qwen25vl_full.pkl"
)
CLIP_CACHE_DIR = Path("/home/cyf/codex/retrieval_eval_20260224_174523")
OUTPUT_ROOT = Path("/home/cyf/codex")
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]

TOP_K = 2
THRESHOLD = 0.1


def compute_query_image_embeddings(test_data, test_pids):
    """Compute CLIP image embeddings for test question images."""
    from transformers import CLIPModel, CLIPProcessor

    cache_path = CLIP_CACHE_DIR / "clip-l14_query_image_embeddings.npy"
    if cache_path.exists():
        print("  Loading cached query image embeddings...")
        return np.load(cache_path)

    clip_model_name = "openai/clip-vit-large-patch14"
    print(f"  Computing query image embeddings with {clip_model_name}...")
    processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=True)
    model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True)
    model = model.eval().cuda()

    embeddings = np.zeros((len(test_pids), 768), dtype=np.float32)
    has_image_count = 0

    batch_size = 32
    # Collect images and their indices
    img_indices = []
    img_list = []
    for i, pid in enumerate(test_pids):
        problem = test_data[pid]
        img = problem.get("image")
        if img is not None and img != "":
            if isinstance(img, str):
                try:
                    pil_img = Image.open(img).convert("RGB")
                except:
                    continue
            elif isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            else:
                continue
            img_indices.append(i)
            img_list.append(pil_img)
            has_image_count += 1

    print(f"  {has_image_count}/{len(test_pids)} test questions have images")

    # Batch encode
    with torch.no_grad():
        for start in range(0, len(img_list), batch_size):
            batch_imgs = img_list[start:start + batch_size]
            inputs = processor(images=batch_imgs, return_tensors="pt", padding=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            for j, idx in enumerate(img_indices[start:start + batch_size]):
                embeddings[idx] = feats[j].cpu().numpy()

    np.save(cache_path, embeddings)
    print(f"  Saved: {cache_path}")

    del model, processor
    torch.cuda.empty_cache()
    return embeddings


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"ablation_concat_fusion_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # Load memory index
    print("\nLoading memory index...")
    with open(MEMORY_INDEX, "rb") as f:
        data = pickle.load(f)
    memories = data["memories"]
    img_emb = data["embeddings"]  # (12726, 768)
    n_memories = len(memories)
    print(f"  {n_memories} memories")

    # Load data
    loader = ScienceQADataLoader()
    test_data = loader.get_split("test")
    try:
        test_pids = sorted(test_data.keys(), key=lambda x: int(x))
    except:
        test_pids = sorted(test_data.keys())
    print(f"  {len(test_pids)} test samples")

    # Load cached embeddings
    print("\nLoading embeddings...")
    txt_emb = np.load(CLIP_CACHE_DIR / "clip-l14_memory_text_embeddings.npy")
    query_txt_emb = np.load(CLIP_CACHE_DIR / "clip-l14_query_embeddings.npy")

    # Compute query image embeddings
    query_img_emb = compute_query_image_embeddings(test_data, test_pids)

    # =====================================================================
    # Method 1: Weighted Sum (α=0.5) — baseline (known: 82.65% w/o forgetting)
    # =====================================================================
    print("\n--- Method 1: Weighted Sum (α=0.5) ---")
    alpha = 0.50
    key_ws = alpha * img_emb + (1 - alpha) * txt_emb  # (12726, 768)
    query_ws = query_txt_emb  # (4241, 768), text-only queries

    # =====================================================================
    # Method 2: Concatenation
    # =====================================================================
    print("--- Method 2: Concatenation [img; txt] ---")
    key_concat = np.concatenate([img_emb, txt_emb], axis=1)  # (12726, 1536)
    query_concat = np.concatenate([query_img_emb, query_txt_emb], axis=1)  # (4241, 1536)
    print(f"  key: {key_concat.shape}, query: {query_concat.shape}")

    # Count queries with non-zero image embeddings
    has_img = np.sum(np.linalg.norm(query_img_emb, axis=1) > 0.1)
    print(f"  {has_img}/{len(test_pids)} queries have image embeddings")

    # =====================================================================
    # Compute retrieval for both methods
    # =====================================================================
    mem_pids = [m.pid for m in memories]

    def retrieve_topk(key_emb, query_emb, method_name):
        """Retrieve top-K for each query."""
        # Normalize
        q_n = np.linalg.norm(query_emb, axis=1, keepdims=True)
        q_n[q_n == 0] = 1.0
        q_norm = query_emb / q_n
        k_n = np.linalg.norm(key_emb, axis=1, keepdims=True)
        k_n[k_n == 0] = 1.0
        k_norm = key_emb / k_n

        sims = q_norm @ k_norm.T

        retrieved_map = {}
        for i, pid in enumerate(test_pids):
            row = sims[i]
            top_indices = np.argsort(row)[::-1][:TOP_K + 10]
            results = []
            for idx in top_indices:
                if row[idx] < THRESHOLD:
                    break
                if mem_pids[idx] == pid:
                    continue
                results.append((int(idx), float(row[idx])))
                if len(results) >= TOP_K:
                    break
            retrieved_map[pid] = results

        has_mem = sum(1 for v in retrieved_map.values() if len(v) > 0)
        print(f"  [{method_name}] {has_mem}/{len(test_pids)} have memories")
        return retrieved_map

    retrieved_ws = retrieve_topk(key_ws, query_ws, "WeightedSum")
    retrieved_concat = retrieve_topk(key_concat, query_concat, "Concat")

    # Check overlap of retrieved memories
    same = 0
    for pid in test_pids:
        ws_set = set(idx for idx, _ in retrieved_ws.get(pid, []))
        cc_set = set(idx for idx, _ in retrieved_concat.get(pid, []))
        if ws_set == cc_set:
            same += 1
    print(f"  Same retrieval results: {same}/{len(test_pids)} ({same/len(test_pids)*100:.1f}%)")

    # =====================================================================
    # VLM evaluation
    # =====================================================================
    print("\nLoading VLM (Qwen2.5-VL-7B)...")
    config = ExperimentConfig()
    vlm = Qwen25VLEvaluator(config)
    print("  VLM loaded")

    def predict(problem, entries):
        if not entries:
            retrieved = None
        else:
            retrieved = []
            for mem_idx, sim in entries:
                mem = memories[mem_idx]
                retrieved.append((mem, sim))

        system_prompt, user_prompt, memory_images = build_prompt_v2(problem, retrieved)
        content = []
        for img in memory_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_prompt})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        text = vlm.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if memory_images:
            inputs = vlm.processor(
                text=[text], images=memory_images, return_tensors="pt", padding=True
            )
        else:
            inputs = vlm.processor(
                text=[text], return_tensors="pt", padding=True
            )
        inputs = {k: v.to(vlm.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = vlm.model.generate(
                **inputs, max_new_tokens=512, do_sample=False
            )
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw = vlm.processor.decode(gen_ids, skip_special_tokens=True).strip()
        answer = extract_answer_last(raw)
        return answer, raw

    # Evaluate both methods
    ckpt_path = out_dir / "checkpoint.json"
    results = {
        "weighted_sum": {"predictions": {}},
        "concat": {"predictions": {}},
    }
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            results = json.load(f)
        print(f"  Resumed: ws={len(results['weighted_sum']['predictions'])}, "
              f"concat={len(results['concat']['predictions'])}")

    for method_name, retrieved_map in [("weighted_sum", retrieved_ws), ("concat", retrieved_concat)]:
        print(f"\n=== Evaluating {method_name} ===")
        done = set(results[method_name]["predictions"].keys())
        remaining = [p for p in test_pids if p not in done]
        print(f"  {len(remaining)} remaining")

        if not remaining:
            print("  Already complete")
            continue

        for pid in tqdm(remaining, desc=method_name):
            problem = test_data[pid]
            answer_idx = problem.get("answer", 0)
            gt = CHOICE_LABELS[answer_idx] if isinstance(answer_idx, int) else str(answer_idx)
            entries = retrieved_map.get(pid, [])
            pred, raw = predict(problem, entries)
            correct = (pred == gt)
            results[method_name]["predictions"][pid] = {
                "pred": pred, "gt": gt, "correct": correct
            }
            if len(results[method_name]["predictions"]) % 200 == 0:
                with open(ckpt_path, "w") as f:
                    json.dump(results, f)

        # Save after each method
        with open(ckpt_path, "w") as f:
            json.dump(results, f)

    # Final stats
    print(f"\n{'='*60}")
    print("ABLATION: Embedding Fusion Methods")
    print(f"{'='*60}")
    for method_name in ["weighted_sum", "concat"]:
        preds = results[method_name]["predictions"]
        correct = sum(1 for v in preds.values() if v["correct"])
        total = len(preds)
        acc = correct / total * 100 if total > 0 else 0
        results[method_name]["accuracy"] = acc
        print(f"  {method_name:15s}: {correct}/{total} = {acc:.2f}%")

    results["config"] = {
        "top_k": TOP_K, "encoder": "clip-l14", "vlm": "qwen2.5-vl-7b",
        "forgetting": "NONE (all 12726)", "prompt": "v2",
        "weighted_sum_alpha": 0.50,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
