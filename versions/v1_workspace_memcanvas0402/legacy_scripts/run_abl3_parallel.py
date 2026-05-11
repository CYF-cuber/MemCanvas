#!/usr/bin/env python3
"""
Parallel T&S forgetting ablation — runs a subset of (T,S) configs on a single GPU.
Two instances can run simultaneously on different GPUs with different config subsets.

Usage:
  GPU0: CUDA_VISIBLE_DEVICES=0 python -u run_abl3_parallel.py --configs 250,0 250,1 250,2 500,2 750,0 750,1
  GPU1: CUDA_VISIBLE_DEVICES=1 python -u run_abl3_parallel.py --configs 750,2 1000,2
"""
import argparse, io, json, os, sys, time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/codex")

CANVAS_DIR = Path("/home/cyf/codex/scienceqa_smart_canvases")
VLM_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
CLIP_MODEL = "openai/clip-vit-large-patch14"
DEFAULT_ALPHA = 0.75
DEFAULT_TOP_K = 2
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]
OUTPUT_DIR = Path("/home/cyf/memcanvas0402/scienceqa_ablation")


def build_retrieval_map(img_emb, txt_emb, query_emb, alpha=DEFAULT_ALPHA, top_k=DEFAULT_TOP_K):
    keys = alpha * img_emb + (1 - alpha) * txt_emb
    norms = np.linalg.norm(keys, axis=1, keepdims=True).clip(1e-8)
    keys = keys / norms
    qn = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True).clip(1e-8)
    sims = qn @ keys.T
    rmap = {}
    for i in range(len(query_emb)):
        top = np.argsort(sims[i])[::-1][:top_k + 5]
        res = [(int(j), float(sims[i][j])) for j in top if sims[i][j] >= 0.1][:top_k]
        rmap[i] = res
    return rmap


def simulate_forgetting(rmap, n_memories, n_test, T, S):
    quality = [0] * n_memories
    retrieval_count = [0] * n_memories
    snapshots = []
    for qi in range(n_test):
        for mem_idx, sim in rmap.get(qi, []):
            if quality[mem_idx] < 4:
                retrieval_count[mem_idx] += 1
        if (qi + 1) % T == 0:
            degraded = 0
            for mi in range(n_memories):
                if quality[mi] >= 4:
                    continue
                if retrieval_count[mi] <= S:
                    quality[mi] += 1
                    degraded += 1
            # Do NOT reset counters — they accumulate across checkpoints
            surviving = sum(1 for q in quality if q < 4)
            dist_snap = dict(Counter(quality))
            snapshots.append({
                "query_idx": qi + 1,
                "surviving": surviving,
                "degraded_this_round": degraded,
                "quality_dist": dist_snap,
            })
    surviving = sum(1 for q in quality if q < 4)
    dist = Counter(quality)
    stats = {
        "T": T, "S": S, "surviving": surviving, "total": n_memories,
        "quality_distribution": {str(k): v for k, v in sorted(dist.items())},
        "snapshots": snapshots,
    }
    return quality, stats


def degrade_canvas_bytes(canvas_bytes, level):
    if not canvas_bytes or level >= 4:
        return b""
    if level == 0:
        return canvas_bytes
    scale = [1.0, 0.75, 0.5, 0.25][level]
    img = Image.open(io.BytesIO(canvas_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def extract_answer(raw):
    for c in raw.upper():
        if c in CHOICE_LABELS:
            return c
    return "A"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True,
                        help="T,S pairs like '250,0 250,1 500,2'")
    args = parser.parse_args()

    configs = []
    for c in args.configs:
        t, s = c.split(",")
        configs.append((int(t), int(s)))
    print(f"Configs to run: {configs}")

    # Load shared checkpoint (read-only for completed conditions)
    main_ckpt_path = OUTPUT_DIR / "checkpoint.json"
    main_ckpt = {}
    if main_ckpt_path.exists():
        main_ckpt = json.load(open(main_ckpt_path))

    # This worker's own checkpoint
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
    worker_ckpt_path = OUTPUT_DIR / f"checkpoint_worker_gpu{gpu_id}.json"
    worker_ckpt = {}
    if worker_ckpt_path.exists():
        worker_ckpt = json.load(open(worker_ckpt_path))

    # Merge: use main checkpoint for completed, worker for partial
    merged_ckpt = {**main_ckpt, **worker_ckpt}

    # Filter out already-completed configs
    todo = []
    for T, S in configs:
        label = f"abl3_T{T}_S{S}"
        if merged_ckpt.get(label, {}).get("complete", False):
            print(f"  [{label}] Already complete, skipping")
        else:
            todo.append((T, S))

    if not todo:
        print("All configs already complete!")
        return

    print(f"\nWill run {len(todo)} conditions: {todo}")

    # Load embeddings and build retrieval map
    print("Loading CLIP embeddings...")
    img_emb = np.load(CANVAS_DIR / "clip_img_emb.npy")
    txt_emb = np.load(CANVAS_DIR / "clip_txt_emb.npy")
    query_emb = np.load(CANVAS_DIR / "clip_query_emb.npy")
    n_memories = len(img_emb)
    rmap = build_retrieval_map(img_emb, txt_emb, query_emb)

    # Load test data
    print("Loading test data...")
    from datasets import load_dataset
    test_ds = load_dataset("derek-thomas/ScienceQA", split="test")
    n_test = len(test_ds)
    print(f"  {n_test} test samples, {n_memories} memories")

    # Load VLM
    print("Loading VLM...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)
    print("  VLM loaded")

    # Canvas bytes cache (shared across conditions)
    canvas_bytes_cache = {}

    for T, S in todo:
        label = f"abl3_T{T}_S{S}"
        print(f"\n{'='*50}")
        print(f"  {label}: T={T}, S={S}")
        print(f"{'='*50}")

        # Simulate forgetting
        quality_levels, stats = simulate_forgetting(rmap, n_memories, n_test, T, S)
        print(f"  Surviving: {stats['surviving']}/{n_memories}")

        # Degradation cache for this condition
        degrade_cache = {}

        def get_canvases(idx, _ql=quality_levels, _dc=degrade_cache):
            imgs = []
            for cidx, sim in rmap.get(idx, [])[:DEFAULT_TOP_K]:
                ql = _ql[cidx]
                if ql >= 4:
                    continue
                cache_key = (cidx, ql)
                if cache_key not in _dc:
                    if cidx not in canvas_bytes_cache:
                        canvas_bytes_cache[cidx] = (CANVAS_DIR / f"{cidx:05d}.png").read_bytes()
                    _dc[cache_key] = degrade_canvas_bytes(canvas_bytes_cache[cidx], ql)
                degraded = _dc[cache_key]
                if degraded:
                    imgs.append(Image.open(io.BytesIO(degraded)).convert("RGB"))
            return imgs

        # Resume from worker checkpoint
        partial = merged_ckpt.get(label, {})
        predictions = partial.get("predictions", [])
        start_idx = len(predictions)
        correct = partial.get("correct", 0)
        total = partial.get("total", 0)

        if start_idx > 0:
            print(f"  Resuming from {start_idx}/{n_test}")

        for idx in tqdm(range(start_idx, n_test), desc=label, initial=start_idx, total=n_test):
            item = test_ds[idx]
            gt = CHOICE_LABELS[item["answer"]] if item["answer"] < len(CHOICE_LABELS) else "A"
            subject = item.get("subject", "")

            canvas_images = get_canvases(idx)

            # Build VLM input
            content = [{"type": "image", "image": cimg} for cimg in canvas_images]
            hint = item.get("hint", "") or ""
            q = item["question"]
            choices = item["choices"]
            choice_txt = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))
            prompt = (
                "Study the reference canvases above. Each shows a solved example.\n"
                f"{hint}\n\nQuestion: {q}\n{choice_txt}\n"
                "Think step by step, then answer with just the letter:"
            )
            if item.get("image") is not None:
                content.append({"type": "image", "image": item["image"].convert("RGB")})
            content.append({"type": "text", "text": prompt})

            msgs = [{"role": "user", "content": content}]
            txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            all_imgs = canvas_images + ([item["image"].convert("RGB")] if item.get("image") else [])
            if all_imgs:
                inp = proc(text=[txt], images=all_imgs, return_tensors="pt", padding=True)
            else:
                inp = proc(text=[txt], return_tensors="pt", padding=True)
            inp = {k: v.to(vlm.device) for k, v in inp.items()}

            with torch.no_grad():
                out = vlm.generate(**inp, max_new_tokens=512, do_sample=False)
            raw = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            pred = extract_answer(raw)

            is_correct = pred == gt
            correct += int(is_correct)
            total += 1
            predictions.append({
                "idx": idx, "predicted": pred, "ground_truth": gt,
                "correct": is_correct, "subject": subject,
            })

            if total % 100 == 0:
                worker_ckpt[label] = {
                    "predictions": predictions, "correct": correct, "total": total,
                    "accuracy": correct / total * 100 if total else 0,
                    "complete": False,
                }
                json.dump(worker_ckpt, open(worker_ckpt_path, "w"))

        # Condition complete
        acc = correct / total * 100 if total else 0

        # Per-subject
        per_subject = {}
        for subj in ["natural science", "social science", "language science"]:
            subj_preds = [p for p in predictions if p.get("subject") == subj]
            if subj_preds:
                subj_correct = sum(p.get("correct", p.get("is_correct", False)) for p in subj_preds)
                per_subject[subj] = {
                    "n": len(subj_preds),
                    "correct": subj_correct,
                    "acc": subj_correct / len(subj_preds) * 100,
                }

        worker_ckpt[label] = {
            "predictions": predictions, "correct": correct, "total": total,
            "accuracy": acc, "per_subject": per_subject, "complete": True,
            "forgetting": stats,
        }
        json.dump(worker_ckpt, open(worker_ckpt_path, "w"))
        print(f"\n  [{label}] Done: {acc:.2f}% ({correct}/{total})")

    del vlm, proc
    torch.cuda.empty_cache()
    print("\nAll conditions complete for this worker!")


if __name__ == "__main__":
    main()
