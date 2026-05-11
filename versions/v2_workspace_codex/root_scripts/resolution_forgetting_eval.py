#!/usr/bin/env python3
"""
Resolution-Based Progressive Forgetting Experiment.

Instead of degrading memory quality via format changes (PNG→WebP→AVIF→...),
this experiment keeps ALL canvases as PNG but progressively reduces resolution:

  640px → 480px → 320px → 160px → Deleted

This is more natural ("memories get blurry") and simpler (no multi-format codec).

Two-phase architecture (same as frequency forgetting):
  Phase 1 (Fast, no VLM): Simulate forgetting, compute convergence curves.
  Phase 2 (Slow, VLM):    Evaluate accuracy for each condition's final state.

Usage:
  # Phase 1 only (fast, ~10 min)
  python -u /home/cyf/codex/resolution_forgetting_eval.py --phase1-only

  # Full evaluation (Phase 1 + Phase 2 VLM, ~10+ hours)
  CUDA_VISIBLE_DEVICES=0 python -u /home/cyf/codex/resolution_forgetting_eval.py

  # Resume Phase 2 from checkpoint
  CUDA_VISIBLE_DEVICES=0 python -u /home/cyf/codex/resolution_forgetting_eval.py \
    --resume /home/cyf/codex/resolution_forgetting_eval_XXXXXXXX_XXXXXX
"""

import argparse
import copy
import io
import json
import os
import pickle
import random
import sys
import time
from collections import Counter
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Imports from existing experiment infrastructure
# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/cyf/memory/memory_canvas/experiments")
from scienceqa_qwen25vl_full_experiment import (  # noqa: E402
    MemoryIndex,
    MemoryEntry,  # noqa: F401
    CLIPLargeMemoryBuilder,
    Qwen25VLEvaluator,
    ExperimentConfig,
    ScienceQADataLoader,
)

sys.path.insert(0, "/home/cyf/codex")
from prompt_improvement_eval import build_prompt_v2, extract_answer_last  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MEMORY_INDEX = (
    "/home/cyf/memory/experiments/scienceqa_qwen25vl_full/"
    "memory_index_qwen25vl_full.pkl"
)
CLIP_CACHE_DIR = Path("/home/cyf/codex/retrieval_eval_20260224_174523")
DEFAULT_OUTPUT_ROOT = "/home/cyf/codex"
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]

ALPHA = 0.50
TOP_K = 2
THRESHOLD = 0.1
N_EPOCHS = 3

REVIEW_INTERVALS = [500, 1000, 2000]
FREQ_THRESHOLDS = [0, 1, 2]


# ---------------------------------------------------------------------------
# Resolution-based quality levels
# ---------------------------------------------------------------------------
class ResLevel(IntEnum):
    """Resolution-based degradation chain: 640 → 480 → 320 → 160 → Deleted."""
    W640 = 0     # original (canvas width = 640px)
    W480 = 1     # resize to 480px width
    W320 = 2     # resize to 320px width
    W160 = 3     # resize to 160px width
    DELETED = 4  # removed

RES_WIDTHS: Dict[int, Optional[int]] = {
    ResLevel.W640: None,   # keep original
    ResLevel.W480: 480,
    ResLevel.W320: 320,
    ResLevel.W160: 160,
    ResLevel.DELETED: None,
}

RES_LABELS: Dict[int, str] = {
    ResLevel.W640: "640px (original)",
    ResLevel.W480: "480px",
    ResLevel.W320: "320px",
    ResLevel.W160: "160px",
    ResLevel.DELETED: "Deleted",
}

# Pre-computed size ratios (width ratio squared approximation for PNG)
# Actual ratios will be measured from real data in Phase 1
RES_SIZE_RATIOS: Dict[int, float] = {
    ResLevel.W640: 1.0,
    ResLevel.W480: (480 / 640) ** 2,   # ~0.5625
    ResLevel.W320: (320 / 640) ** 2,   # ~0.25
    ResLevel.W160: (160 / 640) ** 2,   # ~0.0625
    ResLevel.DELETED: 0.0,
}


# ---------------------------------------------------------------------------
# Image resize (all stay PNG)
# ---------------------------------------------------------------------------
def resize_canvas_to_level(orig_bytes: bytes, level: int) -> bytes:
    """Resize canvas PNG to given resolution level. Output stays PNG."""
    if level == ResLevel.W640:
        return orig_bytes
    if level == ResLevel.DELETED:
        return b""

    target_width = RES_WIDTHS[level]
    if target_width is None:
        return orig_bytes

    img = Image.open(io.BytesIO(orig_bytes))
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    orig_w, orig_h = img.size
    if orig_w <= target_width:
        # Already smaller than target; keep as-is
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    scale = target_width / orig_w
    new_h = max(1, int(round(orig_h * scale)))
    resample = getattr(Image, "Resampling", Image).LANCZOS
    img = img.resize((target_width, new_h), resample=resample)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Measure actual size ratios from a sample of memories
# ---------------------------------------------------------------------------
def measure_size_ratios(memories: List, n_sample: int = 200) -> Dict[int, float]:
    """Measure actual PNG size ratios for each resolution level."""
    rng = random.Random(42)
    indices = rng.sample(range(len(memories)), min(n_sample, len(memories)))

    level_sizes: Dict[int, List[float]] = {lv: [] for lv in ResLevel if lv != ResLevel.DELETED}
    for idx in tqdm(indices, desc="Measuring size ratios"):
        orig = memories[idx].canvas_image_bytes
        orig_size = len(orig)
        if orig_size == 0:
            continue
        for lv in ResLevel:
            if lv == ResLevel.DELETED:
                continue
            resized = resize_canvas_to_level(orig, lv)
            level_sizes[lv].append(len(resized) / orig_size)

    ratios = {}
    for lv in ResLevel:
        if lv == ResLevel.DELETED:
            ratios[lv] = 0.0
        else:
            vals = level_sizes[lv]
            ratios[lv] = sum(vals) / len(vals) if vals else RES_SIZE_RATIOS[lv]

    return ratios


# ---------------------------------------------------------------------------
# Storage and distribution helpers
# ---------------------------------------------------------------------------
def compute_storage(png_sizes: List[int], qualities: List[int],
                    size_ratios: Dict[int, float]) -> int:
    total = 0
    for i, q in enumerate(qualities):
        if q != int(ResLevel.DELETED):
            total += int(png_sizes[i] * size_ratios[q])
    return total


def compute_quality_dist(qualities: List[int]) -> Dict[str, int]:
    dist = {}
    for lv in ResLevel:
        dist[RES_LABELS[lv]] = sum(1 for q in qualities if q == int(lv))
    return dist


# ---------------------------------------------------------------------------
# Phase 1: Forgetting simulation
# ---------------------------------------------------------------------------
def simulate_resolution_forgetting(
    test_pids: List[str],
    candidates: Dict[str, List[Tuple[int, float]]],
    png_sizes: List[int],
    size_ratios: Dict[int, float],
    review_interval: int,
    freq_threshold: int,
    n_epochs: int,
    top_k: int = 2,
) -> Dict[str, Any]:
    """
    Simulate frequency-based adaptive forgetting with resolution degradation.

    At each review point (every review_interval test samples):
      - Count how many times each memory was retrieved (cumulative, no reset)
      - Memories with count <= freq_threshold get demoted one resolution level
    """
    n_memories = len(png_sizes)
    n_test = len(test_pids)
    total_steps = n_test * n_epochs

    # Initialize all memories at full resolution
    qualities = [int(ResLevel.W640)] * n_memories
    retrieval_counts = [0] * n_memories

    storage_curve: List[Tuple[int, int]] = []
    review_snapshots: List[Dict] = []

    init_storage = compute_storage(png_sizes, qualities, size_ratios)
    storage_curve.append((0, init_storage))

    step = 0
    for epoch in range(n_epochs):
        for pid in test_pids:
            cands = candidates.get(pid, [])
            effective = []
            for mem_idx, sim in cands:
                if qualities[mem_idx] == int(ResLevel.DELETED):
                    continue
                effective.append(mem_idx)
                if len(effective) >= top_k:
                    break

            for mem_idx in effective:
                retrieval_counts[mem_idx] += 1

            step += 1

            if step % review_interval == 0:
                n_demoted = 0
                for i in range(n_memories):
                    if qualities[i] == int(ResLevel.DELETED):
                        continue
                    if retrieval_counts[i] <= freq_threshold:
                        qualities[i] = min(qualities[i] + 1, int(ResLevel.DELETED))
                        n_demoted += 1
                # Cumulative counts — NO reset

                storage = compute_storage(png_sizes, qualities, size_ratios)
                n_alive = sum(1 for q in qualities if q != int(ResLevel.DELETED))
                review_snapshots.append({
                    "step": step,
                    "epoch": epoch,
                    "storage_bytes": storage,
                    "storage_mb": storage / (1024 * 1024),
                    "n_alive": n_alive,
                    "n_demoted_this_review": n_demoted,
                    "quality_distribution": compute_quality_dist(qualities),
                })
                storage_curve.append((step, storage))

            elif step % 100 == 0:
                storage = compute_storage(png_sizes, qualities, size_ratios)
                storage_curve.append((step, storage))

    final_storage = compute_storage(png_sizes, qualities, size_ratios)
    storage_curve.append((step, final_storage))
    n_surviving = sum(1 for q in qualities if q != int(ResLevel.DELETED))

    return {
        "review_interval": review_interval,
        "freq_threshold": freq_threshold,
        "n_epochs": n_epochs,
        "total_steps": total_steps,
        "storage_curve": storage_curve,
        "review_snapshots": review_snapshots,
        "final_qualities": qualities,
        "final_storage_bytes": final_storage,
        "final_storage_mb": final_storage / (1024 * 1024),
        "n_surviving": n_surviving,
        "quality_distribution": compute_quality_dist(qualities),
    }


# ---------------------------------------------------------------------------
# Phase 1: Run all simulations
# ---------------------------------------------------------------------------
def run_phase1(
    memories: List,
    embeddings: np.ndarray,
    test_pids: List[str],
    test_data: Dict,
    output_dir: Path,
    n_epochs: int = 3,
) -> Dict:
    n = len(memories)
    png_sizes = [len(m.canvas_image_bytes) for m in memories]
    total_png_bytes = sum(png_sizes)

    print(f"\nPhase 1: Resolution-Based Forgetting Simulation")
    print(f"  Memories: {n}")
    print(f"  Total PNG size: {total_png_bytes:,} bytes ({total_png_bytes / 1024 / 1024:.1f} MB)")
    print(f"  Test samples: {len(test_pids)}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Degradation chain: 640px → 480px → 320px → 160px → Deleted")
    print(f"  Review intervals: {REVIEW_INTERVALS}")
    print(f"  Frequency thresholds: {FREQ_THRESHOLDS}")

    # Measure actual size ratios
    print("\nMeasuring actual size ratios from sample canvases...")
    size_ratios = measure_size_ratios(memories)
    for lv in ResLevel:
        print(f"  {RES_LABELS[lv]}: {size_ratios[lv]:.4f}")

    # Pre-compute retrieval using cached embeddings
    print("\nLoading cached embeddings...")
    img_emb = embeddings
    txt_emb = np.load(CLIP_CACHE_DIR / "clip-l14_memory_text_embeddings.npy")
    key_emb = ALPHA * img_emb + (1 - ALPHA) * txt_emb
    query_emb = np.load(CLIP_CACHE_DIR / "clip-l14_query_embeddings.npy")

    # Normalize
    k_n = np.linalg.norm(key_emb, axis=1, keepdims=True)
    k_n[k_n == 0] = 1.0
    k_norm = key_emb / k_n
    q_n = np.linalg.norm(query_emb, axis=1, keepdims=True)
    q_n[q_n == 0] = 1.0
    q_norm = query_emb / q_n

    sims = q_norm @ k_norm.T  # (4241, 12726)
    mem_pids = [m.pid for m in memories]

    print("Pre-computing retrieval candidates (top-30)...")
    candidates = {}
    for i, pid in enumerate(test_pids):
        row = sims[i]
        top_indices = np.argsort(row)[::-1][:30]
        cands = []
        for idx in top_indices:
            if row[idx] < THRESHOLD:
                break
            if mem_pids[idx] == pid:
                continue
            cands.append((int(idx), float(row[idx])))
        candidates[pid] = cands

    # Run simulations
    results = {}
    conditions = []
    for interval in REVIEW_INTERVALS:
        for threshold in FREQ_THRESHOLDS:
            label = f"res_i{interval}_t{threshold}"
            conditions.append(label)
            print(f"\n  Simulating: interval={interval}, threshold={threshold} ...")
            result = simulate_resolution_forgetting(
                test_pids, candidates, png_sizes, size_ratios,
                review_interval=interval,
                freq_threshold=threshold,
                n_epochs=n_epochs,
            )
            results[label] = result
            ratio = result['final_storage_mb'] / (total_png_bytes / 1024 / 1024) * 100
            print(f"    Final storage: {result['final_storage_mb']:.1f} MB ({ratio:.1f}%)")
            print(f"    Surviving: {result['n_surviving']}/{n}")
            print(f"    Distribution: {result['quality_distribution']}")

    phase1_data = {
        "num_memories": n,
        "total_png_bytes": total_png_bytes,
        "n_test": len(test_pids),
        "n_epochs": n_epochs,
        "review_intervals": REVIEW_INTERVALS,
        "freq_thresholds": FREQ_THRESHOLDS,
        "conditions": conditions,
        "results": results,
        "size_ratios": {str(k): v for k, v in size_ratios.items()},
        "degradation_chain": "640px → 480px → 320px → 160px → Deleted",
    }

    results_path = output_dir / "phase1_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(phase1_data, f, indent=2)
    print(f"\n  Phase 1 results saved: {results_path}")

    generate_phase1_plots(phase1_data, output_dir)

    return phase1_data


# ---------------------------------------------------------------------------
# Phase 1: Plots
# ---------------------------------------------------------------------------
def generate_phase1_plots(phase1_data: Dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = phase1_data["results"]
    total_png_bytes = phase1_data["total_png_bytes"]
    total_mb = total_png_bytes / (1024 * 1024)
    n_test = phase1_data["n_test"]
    n_epochs = phase1_data["n_epochs"]

    interval_colors = {500: "#e74c3c", 1000: "#3498db", 2000: "#2ecc71"}
    threshold_styles = {0: "-", 1: "--", 2: ":"}

    # Plot 1: Storage convergence
    fig, ax = plt.subplots(figsize=(14, 7))
    for interval in REVIEW_INTERVALS:
        for threshold in FREQ_THRESHOLDS:
            label = f"res_i{interval}_t{threshold}"
            res = results[label]
            curve = res["storage_curve"]
            steps = [s for s, _ in curve]
            storage_mb = [v / (1024 * 1024) for _, v in curve]
            ax.plot(
                steps, storage_mb,
                color=interval_colors[interval],
                linestyle=threshold_styles[threshold],
                linewidth=1.5, alpha=0.85,
                label=f"I={interval}, T={threshold}",
            )
    ax.axhline(y=total_mb, color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label=f"Original ({total_mb:.0f} MB)")
    for e in range(1, n_epochs + 1):
        ax.axvline(x=n_test * e, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.text(n_test * e, total_mb * 0.98, f"Epoch {e}", ha="center",
                fontsize=8, color="gray", alpha=0.6)
    ax.set_xlabel("Simulation Step", fontsize=12)
    ax.set_ylabel("Total Storage (MB)", fontsize=12)
    ax.set_title("Storage Convergence — Resolution-Based Forgetting (PNG only)", fontsize=14)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_dir / "storage_convergence.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: storage_convergence.png")

    # Plot 2: Final summary bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    conditions = phase1_data["conditions"]
    final_storage = [results[c]["final_storage_mb"] for c in conditions]
    n_surviving = [results[c]["n_surviving"] for c in conditions]
    short_labels = [f"I{c.split('_i')[1].split('_')[0]}_T{c.split('_t')[1]}" for c in conditions]
    colors = [interval_colors[int(c.split("_i")[1].split("_")[0])] for c in conditions]

    ax1.bar(range(len(conditions)), final_storage, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(conditions)))
    ax1.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Final Storage (MB)", fontsize=11)
    ax1.set_title("Converged Storage (Resolution Forgetting)", fontsize=13)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(range(len(conditions)), n_surviving, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Surviving Memories", fontsize=11)
    ax2.set_title("Memory Survival Count", fontsize=13)
    ax2.axhline(y=phase1_data["num_memories"], color="gray", linestyle="--",
                linewidth=1, alpha=0.5, label=f"Total ({phase1_data['num_memories']})")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "final_summary.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: final_summary.png")

    # Plot 3: Resolution distribution evolution for I=2000, T=0
    selected = "res_i2000_t0"
    if selected in results:
        snapshots = results[selected]["review_snapshots"]
        if snapshots:
            fig, ax = plt.subplots(figsize=(12, 6))
            steps_arr = [s["step"] for s in snapshots]
            level_names = [RES_LABELS[lv] for lv in ResLevel]
            level_colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c", "#95a5a6"]

            bottom = np.zeros(len(snapshots))
            for qi, qname in enumerate(level_names):
                counts = [s["quality_distribution"].get(qname, 0) for s in snapshots]
                ax.bar(range(len(snapshots)), counts, bottom=bottom,
                       label=qname, color=level_colors[qi], alpha=0.85, width=0.8)
                bottom += np.array(counts)

            ax.set_xticks(range(len(snapshots)))
            ax.set_xticklabels([str(s) for s in steps_arr], rotation=45, fontsize=8)
            ax.set_xlabel("Review Step", fontsize=12)
            ax.set_ylabel("Number of Memories", fontsize=12)
            ax.set_title(f"Resolution Distribution Over Time — {selected}", fontsize=13)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(output_dir / "resolution_evolution.png", dpi=150)
            plt.close(fig)
            print(f"  Plot saved: resolution_evolution.png")


# ---------------------------------------------------------------------------
# Phase 2: VLM Evaluation
# ---------------------------------------------------------------------------
def run_phase2(
    memories: List,
    embeddings: np.ndarray,
    test_pids: List[str],
    test_data: Dict,
    phase1_data: Dict,
    output_dir: Path,
) -> Dict:
    import torch

    print(f"\nPhase 2: VLM Accuracy Evaluation (Resolution Forgetting)")

    # Load cached embeddings & compute retrieval
    img_emb = embeddings
    txt_emb = np.load(CLIP_CACHE_DIR / "clip-l14_memory_text_embeddings.npy")
    key_emb = ALPHA * img_emb + (1 - ALPHA) * txt_emb
    query_emb = np.load(CLIP_CACHE_DIR / "clip-l14_query_embeddings.npy")

    k_n = np.linalg.norm(key_emb, axis=1, keepdims=True)
    k_n[k_n == 0] = 1.0
    k_norm = key_emb / k_n
    q_n = np.linalg.norm(query_emb, axis=1, keepdims=True)
    q_n[q_n == 0] = 1.0
    q_norm = query_emb / q_n

    sims = q_norm @ k_norm.T
    mem_pids = [m.pid for m in memories]

    # Load VLM
    print("Loading VLM (Qwen2.5-VL-7B)...")
    config = ExperimentConfig()
    vlm = Qwen25VLEvaluator(config)
    print("  VLM loaded")

    # Checkpoint
    ckpt_path = output_dir / "phase2_checkpoint.json"
    checkpoint = {}
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            checkpoint = json.load(f)
        print(f"  Resumed from checkpoint")

    # Compression cache: (mem_idx, level) -> bytes
    compress_cache: Dict[Tuple[int, int], bytes] = {}

    def predict(problem, entries):
        if not entries:
            retrieved = None
        else:
            retrieved = []
            for mem_idx, ql, sim in entries:
                mem = memories[mem_idx]
                cache_key = (mem_idx, ql)
                if cache_key not in compress_cache:
                    compress_cache[cache_key] = resize_canvas_to_level(
                        mem.canvas_image_bytes, ql
                    )
                temp_mem = copy.copy(mem)
                temp_mem.canvas_image_bytes = compress_cache[cache_key]
                retrieved.append((temp_mem, sim))

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

    def eval_condition(cond_label, final_qualities, existing_preds=None):
        """Evaluate one forgetting condition."""
        preds = dict(existing_preds) if existing_preds else {}
        remaining = [p for p in test_pids if p not in preds]
        if not remaining:
            return preds

        for pid in tqdm(remaining, desc=f"Eval {cond_label}"):
            problem = test_data[pid]
            answer_idx = problem.get("answer", 0)
            gt = CHOICE_LABELS[answer_idx] if isinstance(answer_idx, int) else str(answer_idx)

            # Retrieve top-K from surviving memories
            i = test_pids.index(pid)
            row = sims[i]
            top_indices = np.argsort(row)[::-1][:TOP_K + 20]
            entries = []
            for idx in top_indices:
                if row[idx] < THRESHOLD:
                    break
                if mem_pids[idx] == pid:
                    continue
                ql = final_qualities[idx]
                if ql == int(ResLevel.DELETED):
                    continue
                entries.append((int(idx), ql, float(row[idx])))
                if len(entries) >= TOP_K:
                    break

            pred, raw = predict(problem, entries)
            preds[pid] = {"pred": pred, "gt": gt, "correct": pred == gt}

            if len(preds) % 200 == 0:
                # Save intermediate checkpoint
                _save_phase2_checkpoint(checkpoint, ckpt_path)

        return preds

    # Determine which conditions to evaluate
    phase1_results = phase1_data["results"]
    conditions = phase1_data["conditions"]

    # Deduplicate conditions with identical final_qualities
    fingerprints: Dict[str, str] = {}
    for cond in conditions:
        quals = phase1_results[cond]["final_qualities"]
        fp = hash(tuple(quals))
        fingerprints[cond] = str(fp)

    fp_to_conds: Dict[str, List[str]] = {}
    for cond, fp in fingerprints.items():
        fp_to_conds.setdefault(fp, []).append(cond)

    n_unique = len(fp_to_conds)
    print(f"\n  Dedup: {len(conditions)} conditions → {n_unique} unique evaluations")

    # Evaluate
    cond_results = checkpoint.get("condition_results", {})
    for fp, cond_group in fp_to_conds.items():
        representative = cond_group[0]
        # Check if any in group already done
        done = None
        for c in cond_group:
            if c in cond_results and cond_results[c].get("complete"):
                done = c
                break
        if done:
            result = cond_results[done]
            for c in cond_group:
                cond_results[c] = result
            print(f"\n  Skipping {representative} (already done: {result['accuracy']:.2f}%)")
            continue

        # Load existing partial predictions
        existing_preds = {}
        if representative in cond_results:
            existing_preds = cond_results[representative].get("predictions", {})

        final_quals = phase1_results[representative]["final_qualities"]
        preds = eval_condition(representative, final_quals, existing_preds)

        correct = sum(1 for v in preds.values() if v["correct"])
        total = len(preds)
        acc = correct / total * 100 if total > 0 else 0
        result = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "complete": total >= len(test_pids),
            "predictions": preds,
        }

        for c in cond_group:
            cond_results[c] = result
        print(f"  {representative}: {acc:.2f}% ({correct}/{total})")

        checkpoint["condition_results"] = cond_results
        _save_phase2_checkpoint(checkpoint, ckpt_path)

    # Compile summary
    phase2_data = {
        "condition_accuracies": {},
        "dedup_stats": {
            "total_conditions": len(conditions),
            "unique_groups": n_unique,
        },
    }
    for cond in conditions:
        if cond in cond_results:
            phase2_data["condition_accuracies"][cond] = {
                "accuracy": cond_results[cond]["accuracy"],
                "correct": cond_results[cond]["correct"],
                "total": cond_results[cond]["total"],
            }

    results_path = output_dir / "phase2_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(phase2_data, f, indent=2)
    print(f"\n  Phase 2 results saved: {results_path}")

    # Generate accuracy plots
    generate_accuracy_plots(phase1_data, phase2_data, output_dir)

    return phase2_data


def _save_phase2_checkpoint(checkpoint: Dict, path: Path):
    # Strip predictions to save space in checkpoint (keep only result summaries)
    save = {}
    for k, v in checkpoint.items():
        if k == "condition_results":
            save[k] = {}
            for ck, cv in v.items():
                save[k][ck] = {
                    "accuracy": cv.get("accuracy", 0),
                    "correct": cv.get("correct", 0),
                    "total": cv.get("total", 0),
                    "complete": cv.get("complete", False),
                    "predictions": cv.get("predictions", {}),
                }
        else:
            save[k] = v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(save, f)


# ---------------------------------------------------------------------------
# Phase 2: Accuracy plots
# ---------------------------------------------------------------------------
def generate_accuracy_plots(
    phase1_data: Dict, phase2_data: Dict, output_dir: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cond_acc = phase2_data["condition_accuracies"]
    results = phase1_data["results"]

    # Reference values (known from previous experiments)
    BASELINE_ACC = 81.70   # v2 prompt, no memory
    NO_FORGETTING_ACC = 82.65  # all 12,726 memories
    WITH_FORMAT_FORGETTING_ACC = 82.57  # format-based forgetting (I=2000, T=0)

    interval_colors = {500: "#e74c3c", 1000: "#3498db", 2000: "#2ecc71"}

    # Scatter: Accuracy vs Storage
    fig, ax = plt.subplots(figsize=(10, 7))
    for cond, acc_data in cond_acc.items():
        interval = int(cond.split("_i")[1].split("_")[0])
        threshold = int(cond.split("_t")[1])
        acc = acc_data["accuracy"]
        storage_mb = results[cond]["final_storage_mb"]
        marker = "o" if threshold == 0 else ("s" if threshold == 1 else "D")
        ax.scatter(storage_mb, acc, color=interval_colors[interval],
                   marker=marker, s=120, zorder=5,
                   label=f"I={interval}, T={threshold}")
        ax.annotate(f"I{interval}_T{threshold}\n{acc:.1f}%",
                    (storage_mb, acc), textcoords="offset points",
                    xytext=(8, 5), fontsize=7, alpha=0.8)

    ax.axhline(y=NO_FORGETTING_ACC, color="green", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"No Forgetting: {NO_FORGETTING_ACC:.2f}%")
    ax.axhline(y=WITH_FORMAT_FORGETTING_ACC, color="orange", linestyle="-.", linewidth=1.5,
               alpha=0.7, label=f"Format Forgetting (I2000_T0): {WITH_FORMAT_FORGETTING_ACC:.2f}%")
    ax.axhline(y=BASELINE_ACC, color="gray", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Baseline (no memory): {BASELINE_ACC:.2f}%")

    ax.set_xlabel("Final Storage (MB)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Storage — Resolution-Based Forgetting", fontsize=14)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_storage.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: accuracy_vs_storage.png")

    # Bar chart: all conditions
    fig, ax = plt.subplots(figsize=(12, 6))
    conditions = phase1_data["conditions"]
    accs = [cond_acc[c]["accuracy"] if c in cond_acc else 0 for c in conditions]
    short_labels = [f"I{c.split('_i')[1].split('_')[0]}_T{c.split('_t')[1]}" for c in conditions]
    colors = [interval_colors[int(c.split("_i")[1].split("_")[0])] for c in conditions]

    bars = ax.bar(range(len(conditions)), accs, color=colors, alpha=0.8)
    ax.axhline(y=NO_FORGETTING_ACC, color="green", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"No Forgetting: {NO_FORGETTING_ACC:.2f}%")
    ax.axhline(y=WITH_FORMAT_FORGETTING_ACC, color="orange", linestyle="-.",
               linewidth=1.5, alpha=0.7,
               label=f"Format Forgetting: {WITH_FORMAT_FORGETTING_ACC:.2f}%")
    ax.axhline(y=BASELINE_ACC, color="gray", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Baseline: {BASELINE_ACC:.2f}%")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy — Resolution-Based Forgetting", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, acc in zip(bars, accs):
        if acc > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: accuracy_comparison.png")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    phase1_data: Dict,
    phase2_data: Optional[Dict],
    output_dir: Path,
) -> None:
    lines = []
    n = phase1_data["num_memories"]
    total_mb = phase1_data["total_png_bytes"] / (1024 * 1024)
    results = phase1_data["results"]
    conditions = phase1_data["conditions"]
    size_ratios = phase1_data.get("size_ratios", {})

    lines.append("# 分辨率退化遗忘实验报告")
    lines.append("")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## 1. 实验设计")
    lines.append("")
    lines.append("与先前的多格式遗忘方案（PNG→WebP→AVIF→AVIF@0.5x→删除）不同，")
    lines.append("本实验采用**统一 PNG 格式 + 分辨率退化**的方案：")
    lines.append("")
    lines.append("| 层级 | 分辨率 | 实测大小比 | 说明 |")
    lines.append("|------|--------|-----------|------|")
    for lv in ResLevel:
        ratio = float(size_ratios.get(str(int(lv)), RES_SIZE_RATIOS[lv]))
        lines.append(f"| {int(lv)} | {RES_LABELS[lv]} | {ratio:.4f} | "
                     f"{'原始画布' if lv == ResLevel.W640 else '缩放至' + RES_LABELS[lv] if lv != ResLevel.DELETED else '完全删除'} |")
    lines.append("")
    lines.append("**优势**：更符合「记忆逐渐模糊」的认知隐喻，实现简洁（仅 resize），无多格式编解码依赖。")
    lines.append("")

    lines.append("## 2. 遗忘模拟结果")
    lines.append("")
    lines.append("| 条件 | 最终存储(MB) | 占原始比例 | 存活记忆 | 存活率 |")
    lines.append("|------|-------------|-----------|---------|--------|")
    for cond in conditions:
        res = results[cond]
        ratio = res['final_storage_mb'] / total_mb * 100
        surv_rate = res['n_surviving'] / n * 100
        interval = cond.split("_i")[1].split("_")[0]
        thresh = cond.split("_t")[1]
        lines.append(f"| I={interval}, T={thresh} | {res['final_storage_mb']:.1f} "
                     f"| {ratio:.1f}% | {res['n_surviving']} | {surv_rate:.1f}% |")
    lines.append("")

    lines.append("### 分辨率分布（最终状态）")
    lines.append("")
    header = "| 条件 | " + " | ".join(RES_LABELS[lv] for lv in ResLevel) + " |"
    sep = "|------|" + "|".join("---:" for _ in ResLevel) + "|"
    lines.append(header)
    lines.append(sep)
    for cond in conditions:
        dist = results[cond]["quality_distribution"]
        interval = cond.split("_i")[1].split("_")[0]
        thresh = cond.split("_t")[1]
        vals = " | ".join(str(dist.get(RES_LABELS[lv], 0)) for lv in ResLevel)
        lines.append(f"| I={interval}, T={thresh} | {vals} |")
    lines.append("")

    if phase2_data:
        cond_acc = phase2_data["condition_accuracies"]
        lines.append("## 3. VLM 准确率")
        lines.append("")
        lines.append("| 条件 | 准确率 | 存储(MB) | 存活数 |")
        lines.append("|------|--------|----------|--------|")
        for cond in conditions:
            if cond in cond_acc:
                acc = cond_acc[cond]["accuracy"]
                interval = cond.split("_i")[1].split("_")[0]
                thresh = cond.split("_t")[1]
                lines.append(f"| I={interval}, T={thresh} | {acc:.2f}% "
                             f"| {results[cond]['final_storage_mb']:.1f} "
                             f"| {results[cond]['n_surviving']} |")
        lines.append("")
        lines.append("对比参考值：")
        lines.append("- Baseline (无记忆): 81.70%")
        lines.append("- 无遗忘 (全部 12,726 PNG): 82.65%")
        lines.append("- 多格式遗忘 (I=2000, T=0): 82.57%")
        lines.append("")

    report_path = output_dir / "resolution_forgetting_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Report saved: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolution-Based Progressive Forgetting Experiment"
    )
    parser.add_argument("--phase1-only", action="store_true",
                        help="Run only Phase 1 (simulation + plots, no VLM)")
    parser.add_argument("--n-epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="",
                        help="Resume Phase 2 from existing output directory")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.resume:
        output_dir = Path(args.resume)
        assert output_dir.exists(), f"Resume dir not found: {output_dir}"
        print(f"Resuming from: {output_dir}")
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(DEFAULT_OUTPUT_ROOT) / f"resolution_forgetting_eval_{ts}"
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config
    if not args.resume:
        config_path = output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    # Load memory index
    print(f"\nLoading memory index: {DEFAULT_MEMORY_INDEX}")
    with open(DEFAULT_MEMORY_INDEX, "rb") as f:
        data = pickle.load(f)
    memories = data["memories"]
    embeddings = data["embeddings"]
    print(f"  Loaded {len(memories)} memories")

    # Load test data
    loader = ScienceQADataLoader()
    test_data = loader.get_split("test")
    try:
        test_pids = sorted(test_data.keys(), key=lambda x: int(x))
    except Exception:
        test_pids = sorted(test_data.keys())
    print(f"  Test samples: {len(test_pids)}")

    # Phase 1
    if args.resume:
        phase1_path = output_dir / "phase1_results.json"
        assert phase1_path.exists(), f"Phase 1 not found: {phase1_path}"
        with open(phase1_path) as f:
            phase1_data = json.load(f)
        print(f"  Loaded Phase 1 from {phase1_path}")
    else:
        phase1_data = run_phase1(
            memories, embeddings, test_pids, test_data,
            output_dir, n_epochs=args.n_epochs,
        )

    # Phase 2
    phase2_data = None
    if not args.phase1_only:
        phase2_data = run_phase2(
            memories, embeddings, test_pids, test_data,
            phase1_data, output_dir,
        )

    # Report
    generate_report(phase1_data, phase2_data, output_dir)

    print(f"\nDone. Results in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
