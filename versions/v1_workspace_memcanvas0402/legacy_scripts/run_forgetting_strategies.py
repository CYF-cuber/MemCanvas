#!/usr/bin/env python3
"""
Forgetting strategy comparison experiment on ScienceQA.
Compares 4 strategies (Random, FIFO, LRU, Ebbinghaus) against our Freq-Adaptive.
All strategies target ~3374 surviving memories (26.5%) to match Freq-Adaptive T=1000,S=1.

Usage (2 GPUs in parallel):
  CUDA_VISIBLE_DEVICES=0 python -u run_forgetting_strategies.py --strategies random fifo
  CUDA_VISIBLE_DEVICES=1 python -u run_forgetting_strategies.py --strategies lru ebbinghaus
"""
import argparse, io, json, math, os, sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

PYTHON = sys.executable
VLM_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
CANVAS_DIR = Path("/home/cyf/codex/scienceqa_smart_canvases")
OUTPUT_DIR = Path("/home/cyf/memcanvas0402/forgetting_experiments")
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]
DEFAULT_ALPHA = 0.00
DEFAULT_TOP_K = 2
TARGET_KEEP = 3374  # match Freq-Adaptive T=1000,S=1


def build_retrieval_map(img_emb, txt_emb, query_emb, alpha=DEFAULT_ALPHA, top_k=DEFAULT_TOP_K):
    keys = alpha * img_emb + (1 - alpha) * txt_emb
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True).clip(1e-8)
    qn = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True).clip(1e-8)
    sims = qn @ keys.T
    rmap = {}
    for i in range(len(query_emb)):
        top = np.argsort(sims[i])[::-1][:top_k + 5]
        res = [(int(j), float(sims[i][j])) for j in top if sims[i][j] >= 0.1][:top_k]
        rmap[i] = res
    return rmap


# ============================================================
# Forgetting strategy implementations
# Each returns: quality array (0=keep, 4=deleted)
# ============================================================

def strategy_random(rmap, n_memories, n_test, keep_n, seed=42):
    """Random eviction: randomly keep keep_n memories."""
    rng = np.random.RandomState(seed)
    quality = [4] * n_memories
    keep_idx = rng.choice(n_memories, keep_n, replace=False)
    for i in keep_idx:
        quality[i] = 0
    return quality


def strategy_fifo(rmap, n_memories, n_test, keep_n):
    """FIFO: keep the most recent (highest index) memories."""
    quality = [4] * n_memories
    start = n_memories - keep_n
    for i in range(start, n_memories):
        quality[i] = 0
    return quality


def strategy_lru(rmap, n_memories, n_test, keep_n):
    """LRU: keep memories with most recent access time."""
    last_access = [-1] * n_memories
    for qi in range(n_test):
        for mem_idx, sim in rmap.get(qi, []):
            last_access[mem_idx] = qi
    # Sort by last_access descending, keep top keep_n
    ranked = sorted(range(n_memories), key=lambda i: last_access[i], reverse=True)
    quality = [4] * n_memories
    for i in ranked[:keep_n]:
        quality[i] = 0
    return quality


def strategy_ebbinghaus(rmap, n_memories, n_test, keep_n):
    """Ebbinghaus time-decay (MemoryBank): R = e^(-elapsed/strength).
    strength starts at 1, increments on each recall.
    Tune threshold to keep ~keep_n memories."""
    strength = [1.0] * n_memories
    last_access = [0] * n_memories  # query idx of last access

    for qi in range(n_test):
        for mem_idx, sim in rmap.get(qi, []):
            strength[mem_idx] += 1.0
            last_access[mem_idx] = qi

    # Compute retention at end of test
    retention = []
    for mi in range(n_memories):
        elapsed = n_test - last_access[mi]
        R = math.exp(-elapsed / max(strength[mi], 1e-8))
        retention.append(R)

    # Find threshold that keeps ~keep_n
    sorted_r = sorted(retention, reverse=True)
    threshold = sorted_r[min(keep_n - 1, len(sorted_r) - 1)]

    quality = [0 if retention[i] >= threshold else 4 for i in range(n_memories)]
    actual_keep = sum(1 for q in quality if q == 0)

    # Fine-tune: if we kept too many, randomly drop excess
    if actual_keep > keep_n:
        kept_indices = [i for i in range(n_memories) if quality[i] == 0]
        rng = np.random.RandomState(42)
        drop = rng.choice(kept_indices, actual_keep - keep_n, replace=False)
        for i in drop:
            quality[i] = 4

    return quality


def strategy_freq_adaptive(rmap, n_memories, n_test, keep_n, T=1000, S=1):
    """Freq-Adaptive (Ours): periodic review, low-frequency memories degraded.
    Multi-level: q0→q1→q2→q3→q4. Counts accumulate, never reset."""
    quality = [0] * n_memories
    retrieval_count = [0] * n_memories
    for qi in range(n_test):
        for mem_idx, sim in rmap.get(qi, []):
            if quality[mem_idx] < 4:
                retrieval_count[mem_idx] += 1
        if (qi + 1) % T == 0:
            for mi in range(n_memories):
                if quality[mi] >= 4:
                    continue
                if retrieval_count[mi] <= S:
                    quality[mi] += 1
    return quality


def strategy_no_forgetting(rmap, n_memories, n_test, keep_n):
    """No Forgetting: keep all memories at full quality."""
    return [0] * n_memories


STRATEGIES = {
    "random": strategy_random,
    "fifo": strategy_fifo,
    "lru": strategy_lru,
    "ebbinghaus": strategy_ebbinghaus,
    "freq_adaptive": strategy_freq_adaptive,
    "no_forgetting": strategy_no_forgetting,
}


def extract_answer(raw):
    for c in raw.upper():
        if c in CHOICE_LABELS:
            return c
    return "A"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", required=True,
                        choices=list(STRATEGIES.keys()),
                        help="Strategies to evaluate")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load CLIP embeddings
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
    print(f"  {n_test} test, {n_memories} memories, target keep={TARGET_KEEP}")

    # Simulate all strategies first (fast)
    print("\n--- Simulating forgetting strategies ---")
    strategy_qualities = {}
    for name in args.strategies:
        fn = STRATEGIES[name]
        quality = fn(rmap, n_memories, n_test, TARGET_KEEP)
        surviving = sum(1 for q in quality if q < 4)
        print(f"  {name}: {surviving} surviving ({surviving/n_memories*100:.1f}%)")
        strategy_qualities[name] = quality

    # Save simulation results
    sim_results = {}
    for name, quality in strategy_qualities.items():
        dist = Counter(quality)
        # Compute retrieval hit rate
        hits = 0
        total_retrievals = 0
        for qi in range(n_test):
            for mem_idx, sim in rmap.get(qi, [])[:DEFAULT_TOP_K]:
                total_retrievals += 1
                if quality[mem_idx] < 4:
                    hits += 1
        hit_rate = hits / total_retrievals * 100 if total_retrievals else 0

        # Compute frequency distribution of kept vs discarded
        access_counts = [0] * n_memories
        for qi in range(n_test):
            for mem_idx, sim in rmap.get(qi, []):
                access_counts[mem_idx] += 1

        kept_freqs = [access_counts[i] for i in range(n_memories) if quality[i] < 4]
        disc_freqs = [access_counts[i] for i in range(n_memories) if quality[i] >= 4]

        sim_results[name] = {
            "surviving": sum(1 for q in quality if q < 4),
            "quality_distribution": {str(k): v for k, v in sorted(dict(dist).items())},
            "hit_rate": hit_rate,
            "kept_freq_mean": float(np.mean(kept_freqs)) if kept_freqs else 0,
            "kept_freq_median": float(np.median(kept_freqs)) if kept_freqs else 0,
            "disc_freq_mean": float(np.mean(disc_freqs)) if disc_freqs else 0,
            "kept_freqs": kept_freqs,
            "disc_freqs": disc_freqs,
        }
        print(f"  {name} hit_rate={hit_rate:.1f}%, kept_freq_mean={sim_results[name]['kept_freq_mean']:.2f}")

    json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ('kept_freqs', 'disc_freqs')}
               for k, v in sim_results.items()},
              open(OUTPUT_DIR / "simulation_results.json", "w"), indent=2)

    # Load VLM
    print("\nLoading VLM...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)
    print("  VLM loaded")

    canvas_bytes_cache = {}

    for strat_name in args.strategies:
        quality = strategy_qualities[strat_name]
        label = f"strategy_{strat_name}"
        ckpt_path = OUTPUT_DIR / f"checkpoint_{strat_name}.json"

        print(f"\n{'='*50}")
        print(f"  Evaluating: {strat_name}")
        print(f"{'='*50}")

        # Resume
        if ckpt_path.exists():
            ckpt = json.load(open(ckpt_path))
            if ckpt.get("complete"):
                print(f"  Already complete: {ckpt['accuracy']:.2f}%")
                continue
            predictions = ckpt.get("predictions", [])
            correct = ckpt.get("correct", 0)
            total_eval = ckpt.get("total", 0)
        else:
            predictions = []
            correct = 0
            total_eval = 0

        start_idx = len(predictions)
        if start_idx > 0:
            print(f"  Resuming from {start_idx}/{n_test}")

        for idx in tqdm(range(start_idx, n_test), desc=strat_name, initial=start_idx, total=n_test):
            item = test_ds[idx]
            gt = CHOICE_LABELS[item["answer"]] if item["answer"] < len(CHOICE_LABELS) else "A"
            subject = item.get("subject", "")

            # Get canvases (only surviving ones, with degradation for freq_adaptive)
            canvas_images = []
            for cidx, sim in rmap.get(idx, [])[:DEFAULT_TOP_K]:
                if quality[cidx] >= 4:
                    continue
                if cidx not in canvas_bytes_cache:
                    canvas_bytes_cache[cidx] = (CANVAS_DIR / f"{cidx:05d}.png").read_bytes()
                img = Image.open(io.BytesIO(canvas_bytes_cache[cidx])).convert("RGB")
                # Apply resolution degradation for freq_adaptive
                if strat_name == "freq_adaptive" and quality[cidx] > 0:
                    scale = [1.0, 0.75, 0.5, 0.25][quality[cidx]]
                    w, h = img.size
                    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
                canvas_images.append(img)

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
            total_eval += 1
            predictions.append({
                "idx": idx, "predicted": pred, "ground_truth": gt,
                "correct": is_correct, "subject": subject,
            })

            if total_eval % 100 == 0:
                json.dump({"predictions": predictions, "correct": correct,
                           "total": total_eval, "accuracy": correct / total_eval * 100,
                           "complete": False},
                          open(ckpt_path, "w"))

        # Complete
        acc = correct / total_eval * 100 if total_eval else 0
        per_subject = {}
        for subj in ["natural science", "social science", "language science"]:
            subj_preds = [p for p in predictions if p.get("subject") == subj]
            if subj_preds:
                subj_correct = sum(p["correct"] for p in subj_preds)
                per_subject[subj] = {
                    "n": len(subj_preds), "correct": subj_correct,
                    "acc": subj_correct / len(subj_preds) * 100,
                }

        result = {
            "strategy": strat_name, "surviving": sum(1 for q in quality if q < 4),
            "total_memories": n_memories, "predictions": predictions,
            "correct": correct, "total": total_eval, "accuracy": acc,
            "per_subject": per_subject, "complete": True,
            "hit_rate": sim_results[strat_name]["hit_rate"],
        }
        json.dump(result, open(ckpt_path, "w"))
        print(f"\n  [{strat_name}] Done: {acc:.2f}% ({correct}/{total_eval})")

    del vlm, proc
    torch.cuda.empty_cache()
    print("\nAll strategies complete!")


if __name__ == "__main__":
    main()
