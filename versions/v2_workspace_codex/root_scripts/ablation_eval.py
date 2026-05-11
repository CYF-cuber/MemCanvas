#!/usr/bin/env python3
"""
Ablation experiments for Memory Canvas retrieval on ScienceQA.

4 independent ablation studies (each varies ONE factor):
  1. Top-K:    K={1,2,3}        | alpha=0.5, Qwen, CLIP, forgetting
  2. Alpha:    {0,0.25,0.5,0.75,1.0} | K=2, Qwen, CLIP, forgetting
  3. VLM:      {Qwen, InternVL} | alpha=0.5, K=2, CLIP, forgetting
  4. Encoder:  {CLIP, SigLIP2}  | alpha=0.5, K=2, Qwen, forgetting

All experiments use frequency-based forgetting (i=2000, t=0).

Usage:
  CUDA_VISIBLE_DEVICES=1 python -u /home/cyf/codex/ablation_eval.py
  CUDA_VISIBLE_DEVICES=1 python -u /home/cyf/codex/ablation_eval.py --resume <dir>
  CUDA_VISIBLE_DEVICES=1 python -u /home/cyf/codex/ablation_eval.py --ablation 1 2
"""

import argparse
import copy
import gc
import io
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Imports from existing infrastructure
# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/cyf/memory/memory_canvas/experiments")
from scienceqa_qwen25vl_full_experiment import (
    ExperimentConfig,
    MemoryEntry,
    MemoryIndex,
    Qwen25VLEvaluator,
    ScienceQADataLoader,
)

sys.path.insert(0, "/home/cyf/codex")
from prompt_improvement_eval import build_prompt_v2, extract_answer_last
from retrieval_improvement_eval import (
    INTERNVL_MODEL_PATH,
    InternVL35Evaluator,
    _load_internvl_image,
    compute_hybrid_embeddings,
    predict_internvl,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MEMORY_INDEX = (
    "/home/cyf/memory/experiments/scienceqa_qwen25vl_full/"
    "memory_index_qwen25vl_full.pkl"
)
FORGETTING_RESULTS = (
    "/home/cyf/codex/freq_forgetting_eval_20260222_162923/phase1_results.json"
)
FORGETTING_CONFIG = "freq_i2000_t0"

# Cached embeddings from previous Phase 1 runs
CLIP_CACHE_DIR = Path("/home/cyf/codex/retrieval_eval_20260224_174523")
SIGLIP_CACHE_DIR = Path("/home/cyf/codex/retrieval_eval_20260224_192335")

DEFAULT_OUTPUT_ROOT = "/home/cyf/codex"
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]


# ---------------------------------------------------------------------------
# Quality levels (from memory_frequency_forgetting_eval.py)
# ---------------------------------------------------------------------------
class QualityLevel:
    PNG = 0
    WEBP_85 = 1
    AVIF_50 = 2
    AVIF_50_HALF = 3
    DELETED = 4


QUALITY_NAMES = {0: "PNG", 1: "WebP-85", 2: "AVIF-50", 3: "AVIF-50@0.5x", 4: "Deleted"}


def compress_to_quality(canvas_bytes: bytes, quality: int) -> bytes:
    """Compress canvas PNG bytes to the target quality level."""
    if not canvas_bytes or quality == QualityLevel.DELETED:
        return b""
    if quality == QualityLevel.PNG:
        return canvas_bytes

    img = Image.open(io.BytesIO(canvas_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")

    if quality == QualityLevel.AVIF_50_HALF:
        w, h = img.size
        img = img.resize((w // 2, h // 2), Image.LANCZOS)

    buf = io.BytesIO()
    if quality == QualityLevel.WEBP_85:
        img.save(buf, format="WEBP", quality=85, method=6)
    elif quality in (QualityLevel.AVIF_50, QualityLevel.AVIF_50_HALF):
        img.save(buf, format="AVIF", quality=50)
    else:
        img.save(buf, format="PNG")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_memory_index(path: str):
    """Load memory index: returns (memories, embeddings, dim)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["memories"], data["embeddings"], data.get("embedding_dim", 768)


def load_forgetting_qualities() -> List[int]:
    """Load final_qualities from best forgetting config (i=2000, t=0)."""
    with open(FORGETTING_RESULTS) as f:
        r = json.load(f)
    fq = r["results"][FORGETTING_CONFIG]["final_qualities"]
    dist = Counter(fq)
    surviving = sum(1 for q in fq if q != QualityLevel.DELETED)
    print(f"  Forgetting ({FORGETTING_CONFIG}): {surviving}/{len(fq)} surviving")
    for ql in sorted(dist.keys()):
        print(f"    {QUALITY_NAMES[ql]}: {dist[ql]}")
    return fq


def get_memory_text(pid: str, problems: Dict) -> str:
    """Extract key text from a problem for text-based key embedding."""
    prob = problems.get(pid, {})
    parts = []
    for field in ["subject", "topic", "question", "hint"]:
        v = prob.get(field, "")
        if v:
            parts.append(v)
    choices = prob.get("choices", [])
    if choices:
        parts.append(" ".join(choices))
    return " ".join(parts)


def get_query_text(problem: Dict) -> str:
    return (problem.get("question", "") + " " + problem.get("hint", "")).strip()


# ---------------------------------------------------------------------------
# Retrieval with forgetting
# ---------------------------------------------------------------------------
def retrieve_with_forgetting(
    query_embeddings: np.ndarray,
    key_embeddings: np.ndarray,
    test_pids: List[str],
    memory_pids: List[str],
    final_qualities: List[int],
    top_k: int = 2,
    threshold: float = 0.1,
) -> Dict[str, List[Tuple[int, int, float]]]:
    """
    Retrieve top-K memories, skipping DELETED ones.
    Returns: pid -> [(mem_idx, quality_level, similarity), ...]
    """
    # Normalize
    q_n = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    q_n[q_n == 0] = 1.0
    q_norm = query_embeddings / q_n

    k_n = np.linalg.norm(key_embeddings, axis=1, keepdims=True)
    k_n[k_n == 0] = 1.0
    k_norm = key_embeddings / k_n

    sims = q_norm @ k_norm.T  # (M, N)

    retrieved_map = {}
    for i, pid in enumerate(test_pids):
        row = sims[i]
        top_indices = np.argsort(row)[::-1][: top_k + 20]  # extra buffer for filtering

        results = []
        for idx in top_indices:
            if row[idx] < threshold:
                break
            if memory_pids[idx] == pid:
                continue
            ql = final_qualities[idx]
            if ql == QualityLevel.DELETED:
                continue
            results.append((int(idx), ql, float(row[idx])))
            if len(results) >= top_k:
                break

        retrieved_map[pid] = results

    return retrieved_map


# ---------------------------------------------------------------------------
# VLM prediction with quality degradation
# ---------------------------------------------------------------------------
_compress_cache: Dict[Tuple[int, int], bytes] = {}


def predict_qwen_with_forgetting(
    vlm, problem: Dict, entries: List[Tuple[int, int, float]], memories: List
) -> Tuple[str, str]:
    """Predict with Qwen2.5-VL using quality-degraded canvas images."""
    if not entries:
        retrieved = None
    else:
        retrieved = []
        for mem_idx, ql, sim in entries:
            mem = memories[mem_idx]
            cache_key = (mem_idx, ql)
            if cache_key not in _compress_cache:
                _compress_cache[cache_key] = compress_to_quality(
                    mem.canvas_image_bytes, ql
                )
            temp_mem = copy.copy(mem)
            temp_mem.canvas_image_bytes = _compress_cache[cache_key]
            retrieved.append((temp_mem, sim))

    # Use v2 prompt
    system_prompt, user_prompt, memory_images = build_prompt_v2(
        problem, retrieved
    )

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
        inputs = vlm.processor(text=[text], return_tensors="pt", padding=True)

    inputs = {k: v.to(vlm.model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = vlm.model.generate(**inputs, max_new_tokens=512, do_sample=False)

    gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
    raw = vlm.processor.decode(gen_ids, skip_special_tokens=True).strip()
    answer = extract_answer_last(raw)
    return answer, raw


def predict_internvl_with_forgetting(
    evaluator, problem: Dict, entries: List[Tuple[int, int, float]], memories: List
) -> Tuple[str, str]:
    """Predict with InternVL3.5 using quality-degraded canvas images."""
    if not entries:
        retrieved = None
    else:
        retrieved = []
        for mem_idx, ql, sim in entries:
            mem = memories[mem_idx]
            cache_key = (mem_idx, ql)
            if cache_key not in _compress_cache:
                _compress_cache[cache_key] = compress_to_quality(
                    mem.canvas_image_bytes, ql
                )
            temp_mem = copy.copy(mem)
            temp_mem.canvas_image_bytes = _compress_cache[cache_key]
            retrieved.append((temp_mem, sim))

    return predict_internvl(evaluator, problem, retrieved)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def evaluate_condition(
    label: str,
    retrieved_map: Dict[str, List[Tuple[int, int, float]]],
    memories: List,
    predict_fn,
    vlm,
    test_pids: List[str],
    test_data: Dict,
    checkpoint: Dict,
    checkpoint_path: Path,
    save_interval: int = 100,
) -> Dict:
    """Evaluate one condition. Returns result dict."""
    if label in checkpoint and checkpoint[label].get("complete", False):
        print(f"  [{label}] Cached: {checkpoint[label]['accuracy']:.2f}%")
        return checkpoint[label]

    partial = checkpoint.get(label, {})
    predictions = partial.get("predictions", [])
    start_idx = len(predictions)
    correct = partial.get("correct", 0)
    total = partial.get("total", 0)

    if start_idx > 0:
        print(f"  [{label}] Resuming from {start_idx}/{len(test_pids)}")

    for idx in tqdm(range(start_idx, len(test_pids)), desc=label,
                    initial=start_idx, total=len(test_pids)):
        pid = test_pids[idx]
        problem = test_data[pid]
        answer_idx = problem.get("answer", 0)
        correct_answer = CHOICE_LABELS[answer_idx] if answer_idx < len(CHOICE_LABELS) else "A"

        entries = retrieved_map.get(pid, [])

        try:
            pred, raw = predict_fn(vlm, problem, entries, memories)
        except Exception as e:
            print(f"\n  Warning: {pid} failed: {e}")
            pred, raw = "A", f"ERROR: {e}"

        is_correct = pred == correct_answer
        correct += int(is_correct)
        total += 1

        predictions.append({
            "pid": pid, "predicted": pred,
            "correct": correct_answer, "is_correct": is_correct,
        })

        if (idx + 1) % save_interval == 0 or idx == len(test_pids) - 1:
            acc = correct / total * 100 if total > 0 else 0
            checkpoint[label] = {
                "correct": correct, "total": total, "accuracy": acc,
                "predictions": predictions, "complete": False,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f, ensure_ascii=False)

    accuracy = correct / total * 100 if total > 0 else 0
    result = {
        "correct": correct, "total": total, "accuracy": accuracy,
        "predictions": predictions, "complete": True,
    }
    checkpoint[label] = result
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, ensure_ascii=False)

    print(f"  [{label}] Accuracy: {accuracy:.2f}%")
    return result


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------
def define_conditions():
    """
    Define all ablation conditions.
    Returns dict: label -> {ablation, encoder, alpha, top_k, vlm}
    """
    conditions = {}

    # Ablation 1: Top-K (K=1,2,3)
    for k in [1, 2, 3]:
        label = f"abl1_topk{k}"
        conditions[label] = dict(
            ablation=1, encoder="clip-l14", alpha=0.5, top_k=k, vlm="qwen25vl"
        )

    # Ablation 2: Alpha (0, 0.25, 0.5, 0.75, 1.0)
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        label = f"abl2_alpha{a:.2f}"
        conditions[label] = dict(
            ablation=2, encoder="clip-l14", alpha=a, top_k=2, vlm="qwen25vl"
        )

    # Ablation 3: VLM (Qwen, InternVL)
    conditions["abl3_qwen25vl"] = dict(
        ablation=3, encoder="clip-l14", alpha=0.5, top_k=2, vlm="qwen25vl"
    )
    conditions["abl3_internvl35"] = dict(
        ablation=3, encoder="clip-l14", alpha=0.5, top_k=2, vlm="internvl35"
    )

    # Ablation 4: Encoder (CLIP, SigLIP2)
    conditions["abl4_clip-l14"] = dict(
        ablation=4, encoder="clip-l14", alpha=0.5, top_k=2, vlm="qwen25vl"
    )
    conditions["abl4_siglip2"] = dict(
        ablation=4, encoder="siglip2-so400m-384", alpha=0.5, top_k=2, vlm="qwen25vl"
    )

    return conditions


def get_dedup_key(cond: Dict) -> str:
    """Unique key for retrieval + VLM evaluation (to avoid duplicate work)."""
    return f"{cond['encoder']}_a{cond['alpha']:.2f}_k{cond['top_k']}_{cond['vlm']}"


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------
def load_clip_embeddings(memories, test_pids, test_data, problems):
    """Load or compute CLIP-L/14 embeddings."""
    existing_embs = None
    mem_pids = [m.pid for m in memories]

    # Original CLIP image embeddings from memory index
    _, img_embs, _ = load_memory_index(DEFAULT_MEMORY_INDEX)

    # Cached text embeddings
    q_cache = CLIP_CACHE_DIR / "clip-l14_query_embeddings.npy"
    t_cache = CLIP_CACHE_DIR / "clip-l14_memory_text_embeddings.npy"

    if q_cache.exists() and t_cache.exists():
        query_embs = np.load(q_cache)
        mem_text_embs = np.load(t_cache)
        print(f"  CLIP embeddings loaded from cache")
    else:
        raise FileNotFoundError(f"CLIP embedding cache not found at {CLIP_CACHE_DIR}")

    return img_embs, mem_text_embs, query_embs


def load_siglip_embeddings():
    """Load SigLIP2-so400m embeddings from Phase 1 cache."""
    img_cache = SIGLIP_CACHE_DIR / "siglip2-so400m-384_image_embeddings.npy"
    q_cache = SIGLIP_CACHE_DIR / "siglip2-so400m-384_query_embeddings.npy"
    t_cache = SIGLIP_CACHE_DIR / "siglip2-so400m-384_memory_text_embeddings.npy"

    if img_cache.exists() and q_cache.exists() and t_cache.exists():
        img_embs = np.load(img_cache)
        query_embs = np.load(q_cache)
        mem_text_embs = np.load(t_cache)
        print(f"  SigLIP2 embeddings loaded from cache")
        return img_embs, mem_text_embs, query_embs
    else:
        raise FileNotFoundError(f"SigLIP2 embedding cache not found at {SIGLIP_CACHE_DIR}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(results: Dict, output_dir: Path):
    """Generate Chinese markdown report."""
    lines = [
        "# 检索消融实验报告",
        "",
        f"*生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        f"遗忘策略: 频率遗忘 (i=2000, t=0), 2677/12726 记忆存活",
        "",
        "---",
        "",
    ]

    # Group by ablation
    ablations = defaultdict(dict)
    for label, res in results.items():
        abl = res.get("ablation", 0)
        ablations[abl][label] = res

    # Ablation 1: Top-K
    if 1 in ablations:
        lines.append("## 消融1: 检索数量 Top-K")
        lines.append("")
        lines.append("固定条件: α=0.5, Qwen2.5-VL, CLIP-L/14, 频率遗忘")
        lines.append("")
        lines.append("| K | 准确率 | Δ vs K=2 |")
        lines.append("|:---:|:------:|:--------:|")
        base_acc = ablations[1].get("abl1_topk2", {}).get("accuracy", 0)
        for label in sorted(ablations[1].keys()):
            k = label.split("topk")[1]
            acc = ablations[1][label].get("accuracy", 0)
            diff = acc - base_acc
            lines.append(f"| {k} | {acc:.2f}% | {diff:+.2f}% |")
        lines.append("")

    # Ablation 2: Alpha
    if 2 in ablations:
        lines.append("## 消融2: 混合系数 α")
        lines.append("")
        lines.append("固定条件: K=2, Qwen2.5-VL, CLIP-L/14, 频率遗忘")
        lines.append("")
        lines.append("| α | 键类型 | 准确率 | Δ vs α=0.5 |")
        lines.append("|:---:|:------:|:------:|:----------:|")
        base_acc = ablations[2].get("abl2_alpha0.50", {}).get("accuracy", 0)
        for label in sorted(ablations[2].keys()):
            alpha_str = label.split("alpha")[1]
            alpha = float(alpha_str)
            acc = ablations[2][label].get("accuracy", 0)
            diff = acc - base_acc
            if alpha == 1.0:
                ktype = "纯图像"
            elif alpha == 0.0:
                ktype = "纯文本"
            else:
                ktype = f"混合({alpha:.2f})"
            lines.append(f"| {alpha:.2f} | {ktype} | {acc:.2f}% | {diff:+.2f}% |")
        lines.append("")

    # Ablation 3: VLM
    if 3 in ablations:
        lines.append("## 消融3: VLM 模型")
        lines.append("")
        lines.append("固定条件: α=0.5, K=2, CLIP-L/14, 频率遗忘")
        lines.append("")
        lines.append("| VLM | 准确率 |")
        lines.append("|-----|:------:|")
        for label in sorted(ablations[3].keys()):
            vlm_name = "Qwen2.5-VL-7B" if "qwen" in label else "InternVL3.5-8B"
            acc = ablations[3][label].get("accuracy", 0)
            lines.append(f"| {vlm_name} | {acc:.2f}% |")
        lines.append("")

    # Ablation 4: Encoder
    if 4 in ablations:
        lines.append("## 消融4: 编码器")
        lines.append("")
        lines.append("固定条件: α=0.5, K=2, Qwen2.5-VL, 频率遗忘")
        lines.append("")
        lines.append("| 编码器 | 维度 | 准确率 |")
        lines.append("|--------|:---:|:------:|")
        for label in sorted(ablations[4].keys()):
            if "clip" in label:
                enc_name, dim = "CLIP-L/14", 768
            else:
                enc_name, dim = "SigLIP2-So400M", 1152
            acc = ablations[4][label].get("accuracy", 0)
            lines.append(f"| {enc_name} | {dim} | {acc:.2f}% |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*实验环境: NVIDIA RTX A6000, ScienceQA 测试集 (4,241 样本)*")

    report_path = output_dir / "ablation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Report: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ablation experiments")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--ablation", nargs="+", type=int, default=[1, 2, 3, 4],
        help="Which ablations to run (1-4)",
    )
    parser.add_argument("--save-interval", type=int, default=100)
    args = parser.parse_args()

    # Output dir
    if args.resume:
        output_dir = Path(args.resume)
        assert output_dir.exists()
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_root) / f"ablation_eval_{ts}"
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({"ablations": args.ablation, "timestamp": time.strftime("%Y%m%d_%H%M%S")}, f)

    # Load data
    print("\nLoading memory index...")
    memories, clip_img_embs, emb_dim = load_memory_index(DEFAULT_MEMORY_INDEX)
    memory_pids = [m.pid for m in memories]
    print(f"  {len(memories)} memories, dim={emb_dim}")

    print("\nLoading forgetting qualities...")
    final_qualities = load_forgetting_qualities()

    print("\nLoading test data...")
    loader = ScienceQADataLoader()
    test_data = loader.get_split("test")
    all_problems = loader.problems
    try:
        test_pids = sorted(test_data.keys(), key=lambda x: int(x))
    except Exception:
        test_pids = sorted(test_data.keys())
    print(f"  {len(test_pids)} test samples")

    # Define conditions
    all_conditions = define_conditions()
    # Filter by requested ablations
    conditions = {
        k: v for k, v in all_conditions.items() if v["ablation"] in args.ablation
    }
    print(f"\nConditions to evaluate: {len(conditions)}")
    for label, c in sorted(conditions.items()):
        print(f"  {label}: enc={c['encoder']} α={c['alpha']} K={c['top_k']} vlm={c['vlm']}")

    # Load embeddings
    print("\nLoading CLIP-L/14 embeddings...")
    clip_img_embs_2, clip_mem_text_embs, clip_query_embs = load_clip_embeddings(
        memories, test_pids, test_data, all_problems
    )

    siglip_loaded = False
    if any(c["encoder"] == "siglip2-so400m-384" for c in conditions.values()):
        print("\nLoading SigLIP2 embeddings...")
        try:
            siglip_img_embs, siglip_mem_text_embs, siglip_query_embs = load_siglip_embeddings()
            siglip_loaded = True
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            print("  SigLIP2 conditions will be skipped")

    # Pre-compute retrieval maps for all unique conditions
    print("\nPre-computing retrieval maps...")
    retrieval_cache = {}  # dedup_key -> retrieved_map

    for label, cond in sorted(conditions.items()):
        enc = cond["encoder"]
        alpha = cond["alpha"]
        top_k = cond["top_k"]
        rkey = f"{enc}_a{alpha:.2f}_k{top_k}"

        if rkey in retrieval_cache:
            continue

        if enc == "clip-l14":
            img_embs = clip_img_embs
            txt_embs = clip_mem_text_embs
            q_embs = clip_query_embs
        elif enc == "siglip2-so400m-384":
            if not siglip_loaded:
                continue
            img_embs = siglip_img_embs
            txt_embs = siglip_mem_text_embs
            q_embs = siglip_query_embs
        else:
            continue

        # Compute hybrid key embeddings
        if alpha == 1.0:
            key_embs = img_embs
        elif alpha == 0.0:
            key_embs = txt_embs
        else:
            key_embs = compute_hybrid_embeddings(img_embs, txt_embs, alpha)

        # Retrieve with forgetting filter
        ret_map = retrieve_with_forgetting(
            q_embs, key_embs, test_pids, memory_pids,
            final_qualities, top_k=top_k, threshold=0.1,
        )

        # Stats
        n_with = sum(1 for v in ret_map.values() if v)
        avg_k = np.mean([len(v) for v in ret_map.values()])
        print(f"  {rkey}: {n_with}/{len(test_pids)} have retrieval, avg={avg_k:.2f}")

        retrieval_cache[rkey] = ret_map

    # Load checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        print(f"\nCheckpoint loaded: {sum(1 for v in checkpoint.values() if v.get('complete'))} done")

    # Group conditions by VLM to minimize model loading
    vlm_groups = defaultdict(list)
    for label, cond in sorted(conditions.items()):
        vlm_groups[cond["vlm"]].append((label, cond))

    results = {}

    # Evaluate each VLM group
    for vlm_type in ["qwen25vl", "internvl35"]:
        if vlm_type not in vlm_groups:
            continue

        group = vlm_groups[vlm_type]
        print(f"\n{'='*60}")
        print(f"Loading VLM: {vlm_type} ({len(group)} conditions)")
        print(f"{'='*60}")

        if vlm_type == "qwen25vl":
            exp_config = ExperimentConfig()
            vlm = Qwen25VLEvaluator(exp_config)
            predict_fn = predict_qwen_with_forgetting
        elif vlm_type == "internvl35":
            vlm = InternVL35Evaluator()
            predict_fn = predict_internvl_with_forgetting
        else:
            continue

        # Dedup: group conditions with identical eval key
        eval_done = {}  # dedup_key -> result
        for label, cond in group:
            enc = cond["encoder"]
            alpha = cond["alpha"]
            top_k = cond["top_k"]
            rkey = f"{enc}_a{alpha:.2f}_k{top_k}"
            dkey = get_dedup_key(cond)

            if rkey not in retrieval_cache:
                print(f"\n  [{label}] Skipped: no retrieval data")
                continue

            # If already evaluated with same dedup key, reuse result
            if dkey in eval_done:
                print(f"\n  [{label}] Reusing result from dedup key {dkey}")
                result = eval_done[dkey].copy()
                result["ablation"] = cond["ablation"]
                results[label] = result
                # Also save to checkpoint
                checkpoint[label] = result
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint, f, ensure_ascii=False)
                continue

            print(f"\n  --- {label} (dedup: {dkey}) ---")
            ret_map = retrieval_cache[rkey]

            result = evaluate_condition(
                label, ret_map, memories, predict_fn, vlm,
                test_pids, test_data, checkpoint, checkpoint_path,
                save_interval=args.save_interval,
            )
            result["ablation"] = cond["ablation"]
            results[label] = result
            eval_done[dkey] = result

        # Free VLM
        if vlm_type == "qwen25vl":
            del vlm
        else:
            vlm.free()
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\n  {vlm_type} freed")

    # Generate report
    # Merge checkpoint into results for complete picture
    for label in checkpoint:
        if label not in results and checkpoint[label].get("complete"):
            results[label] = checkpoint[label]
            # Try to recover ablation number
            for k, v in all_conditions.items():
                if k == label:
                    results[label]["ablation"] = v["ablation"]

    generate_report(results, output_dir)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for label in sorted(results.keys()):
        acc = results[label].get("accuracy", 0)
        abl = results[label].get("ablation", "?")
        print(f"  [abl{abl}] {label}: {acc:.2f}%")

    print(f"\nOutput: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
