#!/usr/bin/env python3
"""
ScienceQA ablation experiments under new SmartCanvas framework.

3 ablation studies:
  1. Summary extraction agent: with vs without lecture+solution on canvas
  2. Alpha: hybrid retrieval coefficient {0.0, 0.25, 0.5, 0.75, 1.0}
  3. T & S: frequency-based forgetting with resolution degradation

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u scienceqa_ablation_new.py --ablation 1 2 3
  CUDA_VISIBLE_DEVICES=0 python -u scienceqa_ablation_new.py --ablation 2 --resume <dir>
  CUDA_VISIBLE_DEVICES=0 python -u scienceqa_ablation_new.py --build-nosummary-only
"""

import argparse
import io
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/codex")
from smart_canvas_layout import (
    BlockType, ContentBlock, choose_best_layout,
    measure_image, measure_text, render_layout,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANVAS_DIR = Path("/home/cyf/codex/scienceqa_smart_canvases")
NOSUMMARY_DIR = Path("/home/cyf/memcanvas0402/scienceqa_smart_canvases_nosummary")
CACHED_DATA = "/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl"
VLM_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
CLIP_MODEL = "openai/clip-vit-large-patch14"

DEFAULT_ALPHA = 0.75
DEFAULT_TOP_K = 2
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]

SUBJECTS = ["natural science", "social science", "language science"]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def load_train_data():
    with open(CACHED_DATA, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data.get("train", data)
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        return data[0]
    return data


def load_test_data():
    from datasets import load_dataset
    return load_dataset("derek-thomas/ScienceQA", split="test")


def extract_answer(raw: str) -> str:
    for c in raw.upper():
        if c in "ABCDEF":
            return c
    return "A"


def per_subject_metrics(predictions: List[dict]) -> dict:
    """Compute per-subject accuracy from prediction list."""
    by_subj = defaultdict(lambda: {"correct": 0, "total": 0})
    for p in predictions:
        s = p.get("subject", "")
        by_subj[s]["total"] += 1
        by_subj[s]["correct"] += int(p["is_correct"])
    result = {}
    for s in SUBJECTS:
        d = by_subj.get(s, {"correct": 0, "total": 0})
        if d["total"] > 0:
            result[s] = {
                "n": d["total"],
                "correct": d["correct"],
                "acc": d["correct"] / d["total"] * 100,
            }
    return result


# ---------------------------------------------------------------------------
# CLIP Embeddings
# ---------------------------------------------------------------------------
def clip_embed_images(canvas_dir: Path, n: int, output_file: Path):
    if output_file.exists():
        print(f"  CLIP img embeddings exist: {output_file}")
        return np.load(output_file)
    from transformers import CLIPProcessor, CLIPModel
    clip = CLIPModel.from_pretrained(CLIP_MODEL).cuda().eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    all_emb = []
    for i in tqdm(range(0, n, 32), desc="CLIP img"):
        imgs = [
            Image.open(canvas_dir / f"{j:05d}.png").convert("RGB")
            for j in range(i, min(i + 32, n))
        ]
        inp = proc(images=imgs, return_tensors="pt", padding=True)
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            f = clip.get_image_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
        all_emb.append(f.cpu().numpy())
    emb = np.concatenate(all_emb)
    np.save(output_file, emb)
    del clip, proc
    torch.cuda.empty_cache()
    return emb


def clip_embed_texts(texts: List[str], output_file: Path):
    if output_file.exists():
        return np.load(output_file)
    from transformers import CLIPProcessor, CLIPModel
    clip = CLIPModel.from_pretrained(CLIP_MODEL).cuda().eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    all_emb = []
    for i in tqdm(range(0, len(texts), 64), desc="CLIP txt"):
        batch = texts[i : i + 64]
        inp = proc(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            f = clip.get_text_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
        all_emb.append(f.cpu().numpy())
    emb = np.concatenate(all_emb)
    np.save(output_file, emb)
    del clip, proc
    torch.cuda.empty_cache()
    return emb


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def build_retrieval_map(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    query_emb: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = 0.1,
) -> dict:
    keys = alpha * img_emb + (1 - alpha) * txt_emb
    kn = np.linalg.norm(keys, axis=1, keepdims=True)
    kn[kn == 0] = 1.0
    keys = keys / kn
    qn = np.linalg.norm(query_emb, axis=1, keepdims=True)
    qn[qn == 0] = 1.0
    qnorm = query_emb / qn
    sims = qnorm @ keys.T
    rmap = {}
    for i in range(len(query_emb)):
        top = np.argsort(sims[i])[::-1][: top_k + 5]
        res = [(int(j), float(sims[i][j])) for j in top if sims[i][j] >= threshold][:top_k]
        rmap[i] = res
    return rmap


def retrieval_stats(rmap: dict, test_ds, n_test: int) -> dict:
    """Compute retrieval quality stats."""
    hit = sum(1 for v in rmap.values() if v)
    all_sims = [s for entries in rmap.values() for _, s in entries]
    per_subj_hit = defaultdict(lambda: {"hit": 0, "total": 0})
    for i in range(n_test):
        subj = test_ds[i].get("subject", "")
        per_subj_hit[subj]["total"] += 1
        if rmap.get(i, []):
            per_subj_hit[subj]["hit"] += 1
    subj_rates = {}
    for s in SUBJECTS:
        d = per_subj_hit.get(s, {"hit": 0, "total": 0})
        if d["total"] > 0:
            subj_rates[s] = d["hit"] / d["total"]
    return {
        "hit_rate": hit / n_test if n_test else 0,
        "avg_sim": float(np.mean(all_sims)) if all_sims else 0,
        "per_subject_hit": subj_rates,
    }


# ---------------------------------------------------------------------------
# Build no-summary canvases (Ablation 1)
# ---------------------------------------------------------------------------
def build_nosummary_canvases():
    """Rebuild canvases without lecture and solution blocks."""
    NOSUMMARY_DIR.mkdir(exist_ok=True)
    done_marker = NOSUMMARY_DIR / "done.txt"
    if done_marker.exists():
        n = int(done_marker.read_text().strip())
        print(f"  No-summary canvases already built: {n}")
        return n

    train = load_train_data()
    from datasets import load_dataset
    hf_ds = load_dataset("derek-thomas/ScienceQA", split="train")

    n = len(train)
    print(f"  Building {n} no-summary canvases...")
    for i in tqdm(range(n), desc="NoSummary"):
        out = NOSUMMARY_DIR / f"{i:05d}.png"
        if out.exists():
            continue
        p = train[i]
        blocks = []

        # Header
        subj = p.get("subject", "")
        topic = p.get("topic", "")
        if subj or topic:
            blocks.append(measure_text(f"[{subj}] {topic}", font_size=12, ref_width=600))

        # Hint/context
        hint = p.get("hint", "")
        if hint:
            blocks.append(measure_text(hint, font_size=14, ref_width=600))

        # Image
        if i < len(hf_ds) and hf_ds[i].get("image") is not None:
            img = hf_ds[i]["image"].convert("RGB")
            blocks.append(measure_image(img, max_dim=400))

        # NO lecture, NO solution — that's the ablation

        # Question + choices + answer
        q = p.get("question", "")
        choices = p.get("choices", [])
        answer_idx = p.get("answer", 0)
        choice_text = "\n".join(
            f"{'✓ ' if j == answer_idx else '  '}{chr(65+j)}. {c}"
            for j, c in enumerate(choices)
        )
        blocks.append(measure_text(f"Q: {q}\n{choice_text}", font_size=15, ref_width=600))

        if not blocks:
            blocks.append(measure_text("(empty)", font_size=14, ref_width=600))

        layout = choose_best_layout(blocks)
        img_out = render_layout(layout)
        buf = io.BytesIO()
        img_out.save(buf, format="PNG", optimize=True)
        out.write_bytes(buf.getvalue())

    done_marker.write_text(str(n))
    print(f"  Done: {n} no-summary canvases")
    return n


# ---------------------------------------------------------------------------
# Forgetting simulation (Ablation 3)
# ---------------------------------------------------------------------------
def simulate_forgetting(
    rmap: dict,
    n_memories: int,
    n_test: int,
    T: int,
    S: int,
) -> Tuple[List[int], dict]:
    """
    Simulate frequency-based forgetting.
    Every T queries, degrade memories with retrieval count < S.
    Degradation chain: 0(original) -> 1(0.75x) -> 2(0.5x) -> 3(0.25x) -> 4(deleted)

    Returns: (quality_levels, stats_dict)
    """
    quality = [0] * n_memories  # all start at level 0 (original)
    retrieval_count = [0] * n_memories
    snapshots = []

    for qi in range(n_test):
        # Record retrievals for this query
        for mem_idx, sim in rmap.get(qi, []):
            if quality[mem_idx] < 4:  # not deleted
                retrieval_count[mem_idx] += 1

        # Review at every T queries
        if (qi + 1) % T == 0:
            degraded = 0
            for mi in range(n_memories):
                if quality[mi] >= 4:
                    continue
                if retrieval_count[mi] <= S:
                    quality[mi] += 1
                    degraded += 1
            # Do NOT reset counts — they accumulate across checkpoints

            surviving = sum(1 for q in quality if q < 4)
            snapshots.append({
                "query_idx": qi + 1,
                "surviving": surviving,
                "degraded_this_round": degraded,
                "quality_dist": dict(Counter(quality)),
            })

    surviving = sum(1 for q in quality if q < 4)
    dist = Counter(quality)
    stats = {
        "T": T,
        "S": S,
        "surviving": surviving,
        "total": n_memories,
        "quality_distribution": {str(k): v for k, v in sorted(dist.items())},
        "snapshots": snapshots,
    }
    return quality, stats


def degrade_canvas_bytes(canvas_bytes: bytes, level: int) -> bytes:
    """Degrade canvas resolution by level."""
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


# ---------------------------------------------------------------------------
# VLM evaluation
# ---------------------------------------------------------------------------
def load_vlm():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    proc = AutoProcessor.from_pretrained(VLM_MODEL)
    return vlm, proc


def predict_one(
    vlm, proc, test_item: dict,
    canvas_images: List[Image.Image],
) -> Tuple[str, str]:
    """Run VLM prediction for one test item with retrieved canvas images."""
    content = []
    for cimg in canvas_images:
        content.append({"type": "image", "image": cimg})

    hint = test_item.get("hint", "") or ""
    q = test_item["question"]
    choices = test_item["choices"]
    choice_txt = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))

    prompt = (
        "Study the reference canvases above. Each shows a solved example.\n"
        f"{hint}\n\nQuestion: {q}\n{choice_txt}\n"
        "Think step by step, then answer with just the letter:"
    )

    if test_item.get("image") is not None:
        content.append({"type": "image", "image": test_item["image"].convert("RGB")})
    content.append({"type": "text", "text": prompt})

    msgs = [{"role": "user", "content": content}]
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    all_imgs = canvas_images + ([test_item["image"].convert("RGB")] if test_item.get("image") else [])
    if all_imgs:
        inp = proc(text=[txt], images=all_imgs, return_tensors="pt", padding=True)
    else:
        inp = proc(text=[txt], return_tensors="pt", padding=True)
    inp = {k: v.to(vlm.device) for k, v in inp.items()}

    with torch.no_grad():
        out = vlm.generate(**inp, max_new_tokens=512, do_sample=False)
    raw = proc.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    pred = extract_answer(raw)
    return pred, raw


def evaluate_condition(
    label: str,
    vlm, proc,
    test_ds,
    get_canvas_images_fn,  # callable(test_idx) -> List[Image]
    checkpoint: dict,
    checkpoint_path: Path,
    save_interval: int = 100,
) -> dict:
    """Evaluate one condition with checkpoint resumability."""
    if label in checkpoint and checkpoint[label].get("complete", False):
        acc = checkpoint[label]["accuracy"]
        print(f"  [{label}] Cached: {acc:.2f}%")
        return checkpoint[label]

    partial = checkpoint.get(label, {})
    predictions = partial.get("predictions", [])
    start_idx = len(predictions)
    correct = partial.get("correct", 0)
    total = partial.get("total", 0)

    n_test = len(test_ds)
    if start_idx > 0:
        print(f"  [{label}] Resuming from {start_idx}/{n_test}")

    for idx in tqdm(range(start_idx, n_test), desc=label, initial=start_idx, total=n_test):
        item = test_ds[idx]
        gt = CHOICE_LABELS[item["answer"]] if item["answer"] < len(CHOICE_LABELS) else "A"
        subject = item.get("subject", "")

        canvas_images = get_canvas_images_fn(idx)

        try:
            pred, raw = predict_one(vlm, proc, item, canvas_images)
        except Exception as e:
            print(f"\n  Warning: idx={idx} failed: {e}")
            pred, raw = "A", f"ERROR: {e}"

        is_correct = pred == gt
        correct += int(is_correct)
        total += 1

        predictions.append({
            "idx": idx,
            "predicted": pred,
            "correct_answer": gt,
            "is_correct": is_correct,
            "subject": subject,
        })

        if (idx + 1) % save_interval == 0 or idx == n_test - 1:
            acc = correct / total * 100 if total > 0 else 0
            checkpoint[label] = {
                "correct": correct,
                "total": total,
                "accuracy": acc,
                "predictions": predictions,
                "complete": False,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f, ensure_ascii=False)

    accuracy = correct / total * 100 if total > 0 else 0
    result = {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "per_subject": per_subject_metrics(predictions),
        "predictions": predictions,
        "complete": True,
    }
    checkpoint[label] = result
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, ensure_ascii=False)

    print(f"  [{label}] Accuracy: {accuracy:.2f}%")
    return result


# ---------------------------------------------------------------------------
# Ablation runners
# ---------------------------------------------------------------------------
def run_ablation1(vlm, proc, test_ds, checkpoint, checkpoint_path, output_dir):
    """Ablation 1: Summary extraction agent (with vs without)."""
    print("\n" + "=" * 60)
    print("Ablation 1: Summary Extraction Agent")
    print("=" * 60)

    abl_dir = output_dir / "abl1_summary"
    abl_dir.mkdir(exist_ok=True)

    # Build no-summary canvases if needed
    n_nosummary = build_nosummary_canvases()

    # Load embeddings for both conditions
    # With-summary: existing embeddings
    img_emb = np.load(CANVAS_DIR / "clip_img_emb.npy")
    txt_emb = np.load(CANVAS_DIR / "clip_txt_emb.npy")
    query_emb = np.load(CANVAS_DIR / "clip_query_emb.npy")

    # No-summary: compute new embeddings
    ns_img_emb = clip_embed_images(
        NOSUMMARY_DIR, n_nosummary, NOSUMMARY_DIR / "clip_img_emb.npy"
    )
    train = load_train_data()
    train_texts = [f"{s.get('question', '')} {s.get('hint', '')}" for s in train]
    ns_txt_emb = clip_embed_texts(train_texts, NOSUMMARY_DIR / "clip_txt_emb.npy")
    # Query embeddings are the same (test side doesn't change)
    ns_query_emb = query_emb

    # Retrieval maps
    rmap_with = build_retrieval_map(img_emb, txt_emb, query_emb)
    rmap_without = build_retrieval_map(ns_img_emb, ns_txt_emb, ns_query_emb)

    results = {}

    # Condition: with_summary
    def get_canvases_with(idx):
        imgs = []
        for cidx, sim in rmap_with.get(idx, [])[:DEFAULT_TOP_K]:
            imgs.append(Image.open(CANVAS_DIR / f"{cidx:05d}.png").convert("RGB"))
        return imgs

    r = evaluate_condition(
        "abl1_with_summary", vlm, proc, test_ds,
        get_canvases_with, checkpoint, checkpoint_path,
    )
    r["retrieval"] = retrieval_stats(rmap_with, test_ds, len(test_ds))
    results["with_summary"] = r

    # Condition: without_summary
    def get_canvases_without(idx):
        imgs = []
        for cidx, sim in rmap_without.get(idx, [])[:DEFAULT_TOP_K]:
            imgs.append(Image.open(NOSUMMARY_DIR / f"{cidx:05d}.png").convert("RGB"))
        return imgs

    r = evaluate_condition(
        "abl1_without_summary", vlm, proc, test_ds,
        get_canvases_without, checkpoint, checkpoint_path,
    )
    r["retrieval"] = retrieval_stats(rmap_without, test_ds, len(test_ds))
    results["without_summary"] = r

    with open(abl_dir / "results.json", "w") as f:
        # Save without predictions for readability
        clean = {}
        for k, v in results.items():
            clean[k] = {kk: vv for kk, vv in v.items() if kk != "predictions"}
        json.dump(clean, f, indent=2, ensure_ascii=False)

    return results


def run_ablation2(vlm, proc, test_ds, checkpoint, checkpoint_path, output_dir):
    """Ablation 2: Alpha (retrieval mixing coefficient)."""
    print("\n" + "=" * 60)
    print("Ablation 2: Alpha")
    print("=" * 60)

    abl_dir = output_dir / "abl2_alpha"
    abl_dir.mkdir(exist_ok=True)

    img_emb = np.load(CANVAS_DIR / "clip_img_emb.npy")
    txt_emb = np.load(CANVAS_DIR / "clip_txt_emb.npy")
    query_emb = np.load(CANVAS_DIR / "clip_query_emb.npy")

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {}

    for alpha in alphas:
        label = f"abl2_alpha_{alpha:.2f}"
        print(f"\n  --- α={alpha:.2f} ---")

        rmap = build_retrieval_map(img_emb, txt_emb, query_emb, alpha=alpha)

        def get_canvases(idx, _rmap=rmap):
            imgs = []
            for cidx, sim in _rmap.get(idx, [])[:DEFAULT_TOP_K]:
                imgs.append(Image.open(CANVAS_DIR / f"{cidx:05d}.png").convert("RGB"))
            return imgs

        r = evaluate_condition(
            label, vlm, proc, test_ds,
            get_canvases, checkpoint, checkpoint_path,
        )
        r["retrieval"] = retrieval_stats(rmap, test_ds, len(test_ds))
        r["alpha"] = alpha
        results[f"alpha_{alpha:.2f}"] = r

    with open(abl_dir / "results.json", "w") as f:
        clean = {}
        for k, v in results.items():
            clean[k] = {kk: vv for kk, vv in v.items() if kk != "predictions"}
        json.dump(clean, f, indent=2, ensure_ascii=False)

    return results


def run_ablation3(vlm, proc, test_ds, checkpoint, checkpoint_path, output_dir):
    """Ablation 3: T & S (frequency-based forgetting)."""
    print("\n" + "=" * 60)
    print("Ablation 3: T & S Forgetting")
    print("=" * 60)

    abl_dir = output_dir / "abl3_forgetting"
    abl_dir.mkdir(exist_ok=True)

    img_emb = np.load(CANVAS_DIR / "clip_img_emb.npy")
    txt_emb = np.load(CANVAS_DIR / "clip_txt_emb.npy")
    query_emb = np.load(CANVAS_DIR / "clip_query_emb.npy")

    n_memories = len(img_emb)
    n_test = len(test_ds)

    # Build retrieval map for forgetting simulation
    rmap = build_retrieval_map(img_emb, txt_emb, query_emb)

    # Load all canvas bytes for degradation
    print("  Loading canvas bytes...")
    canvas_bytes_cache = {}

    # Forgetting configurations
    configs = [
        (250, 0), (250, 1), (250, 2),
        (500, 0), (500, 1), (500, 2),
        (750, 0), (750, 1), (750, 2),
        (1000, 0), (1000, 1), (1000, 2),
    ]

    # Also add no-forgetting baseline
    forgetting_results = {}

    for T, S in configs:
        label = f"abl3_T{T}_S{S}"
        print(f"\n  --- T={T}, S={S} ---")

        # Simulate forgetting
        quality_levels, stats = simulate_forgetting(rmap, n_memories, n_test, T, S)
        forgetting_results[label] = stats
        print(f"    Surviving: {stats['surviving']}/{n_memories}")
        print(f"    Quality dist: {stats['quality_distribution']}")

        # Build degradation cache
        degrade_cache = {}

        def get_canvases_degraded(idx, _rmap=rmap, _ql=quality_levels, _cache=degrade_cache):
            imgs = []
            for cidx, sim in _rmap.get(idx, [])[:DEFAULT_TOP_K]:
                ql = _ql[cidx]
                if ql >= 4:
                    continue  # deleted
                cache_key = (cidx, ql)
                if cache_key not in _cache:
                    if cidx not in canvas_bytes_cache:
                        canvas_bytes_cache[cidx] = (
                            CANVAS_DIR / f"{cidx:05d}.png"
                        ).read_bytes()
                    _cache[cache_key] = degrade_canvas_bytes(canvas_bytes_cache[cidx], ql)
                degraded_bytes = _cache[cache_key]
                if degraded_bytes:
                    imgs.append(Image.open(io.BytesIO(degraded_bytes)).convert("RGB"))
            return imgs

        r = evaluate_condition(
            label, vlm, proc, test_ds,
            get_canvases_degraded, checkpoint, checkpoint_path,
        )
        r["forgetting"] = stats
        forgetting_results[label]["accuracy"] = r["accuracy"]
        forgetting_results[label]["per_subject"] = r.get("per_subject", {})

    # Save forgetting curves
    with open(abl_dir / "forgetting_curves.json", "w") as f:
        json.dump(forgetting_results, f, indent=2, ensure_ascii=False)

    # Save clean results
    with open(abl_dir / "results.json", "w") as f:
        clean = {}
        for k, v in forgetting_results.items():
            clean[k] = {kk: vv for kk, vv in v.items() if kk != "snapshots"}
        json.dump(clean, f, indent=2, ensure_ascii=False)

    return forgetting_results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(output_dir: Path, all_results: dict):
    """Generate Chinese markdown report."""
    from datetime import datetime
    lines = [
        "# ScienceQA 消融实验报告 (SmartCanvas 新框架)",
        "",
        f"*生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        f"基础配置: Qwen2.5-VL-7B, CLIP-L/14, α=0.75, K=2, 12726 训练样本, 4241 测试样本",
        "",
        "---",
        "",
    ]

    # Ablation 1: Summary
    if "abl1" in all_results:
        abl1 = all_results["abl1"]
        lines.append("## 消融1: 摘要提取Agent的有无")
        lines.append("")
        lines.append("测试画布上是否包含 lecture（背景知识）和 solution（解题过程）信息。")
        lines.append("")
        lines.append("| 条件 | 总准确率 | 自然科学 | 社会科学 | 语言科学 |")
        lines.append("|------|:-------:|:-------:|:-------:|:-------:|")
        for key, name in [("with_summary", "含摘要"), ("without_summary", "无摘要")]:
            r = abl1.get(key, {})
            acc = r.get("accuracy", 0)
            ps = r.get("per_subject", {})
            nat = ps.get("natural science", {}).get("acc", 0)
            soc = ps.get("social science", {}).get("acc", 0)
            lang = ps.get("language science", {}).get("acc", 0)
            lines.append(f"| {name} | {acc:.2f}% | {nat:.2f}% | {soc:.2f}% | {lang:.2f}% |")
        lines.append("")

    # Ablation 2: Alpha
    if "abl2" in all_results:
        abl2 = all_results["abl2"]
        lines.append("## 消融2: 检索混合系数 α")
        lines.append("")
        lines.append("混合键 = α × 图像嵌入 + (1-α) × 文本嵌入")
        lines.append("")
        lines.append("| α | 键类型 | 总准确率 | 自然科学 | 社会科学 | 语言科学 | 命中率 | 平均相似度 |")
        lines.append("|:---:|:------:|:-------:|:-------:|:-------:|:-------:|:------:|:---------:|")
        base_acc = 0
        for key in sorted(abl2.keys()):
            r = abl2[key]
            alpha = r.get("alpha", 0)
            acc = r.get("accuracy", 0)
            if abs(alpha - 0.75) < 0.01:
                base_acc = acc
            ps = r.get("per_subject", {})
            nat = ps.get("natural science", {}).get("acc", 0)
            soc = ps.get("social science", {}).get("acc", 0)
            lang = ps.get("language science", {}).get("acc", 0)
            ret = r.get("retrieval", {})
            hr = ret.get("hit_rate", 0)
            avg_s = ret.get("avg_sim", 0)
            if alpha == 1.0:
                ktype = "纯图像"
            elif alpha == 0.0:
                ktype = "纯文本"
            else:
                ktype = f"混合"
            lines.append(
                f"| {alpha:.2f} | {ktype} | {acc:.2f}% | "
                f"{nat:.2f}% | {soc:.2f}% | {lang:.2f}% | "
                f"{hr:.3f} | {avg_s:.4f} |"
            )
        lines.append("")

    # Ablation 3: T&S
    if "abl3" in all_results:
        abl3 = all_results["abl3"]
        lines.append("## 消融3: 频率遗忘 T & S")
        lines.append("")
        lines.append("每经过T次query，对被检索次数<S的画布进行分辨率降级。")
        lines.append("降级链: 原始 → 0.75× → 0.5× → 0.25× → 删除")
        lines.append("")
        lines.append("| T | S | 存活记忆 | 存活率 | 总准确率 | 自然科学 | 社会科学 | 语言科学 |")
        lines.append("|:---:|:---:|:-------:|:------:|:-------:|:-------:|:-------:|:-------:|")
        # Add no-forgetting baseline
        lines.append("| - | - | 12726 | 100% | 88.82% | 89.30% | 90.21% | 86.73% |")
        for key in sorted(abl3.keys()):
            r = abl3[key]
            T = r.get("T", 0)
            S = r.get("S", 0)
            surv = r.get("surviving", 0)
            total = r.get("total", 12726)
            surv_rate = surv / total * 100 if total else 0
            acc = r.get("accuracy", 0)
            ps = r.get("per_subject", {})
            nat = ps.get("natural science", {}).get("acc", 0)
            soc = ps.get("social science", {}).get("acc", 0)
            lang = ps.get("language science", {}).get("acc", 0)
            lines.append(
                f"| {T} | {S} | {surv} | {surv_rate:.1f}% | {acc:.2f}% | "
                f"{nat:.2f}% | {soc:.2f}% | {lang:.2f}% |"
            )
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
    parser = argparse.ArgumentParser(description="ScienceQA ablation experiments (new framework)")
    parser.add_argument("--ablation", nargs="+", type=int, default=[1, 2, 3],
                        help="Which ablations to run (1=summary, 2=alpha, 3=T&S)")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--output-root", default="/home/cyf/memcanvas0402")
    parser.add_argument("--build-nosummary-only", action="store_true",
                        help="Only build no-summary canvases, don't run eval")
    parser.add_argument("--save-interval", type=int, default=100)
    args = parser.parse_args()

    if args.build_nosummary_only:
        build_nosummary_canvases()
        return 0

    # Output dir
    if args.resume:
        output_dir = Path(args.resume)
        assert output_dir.exists()
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_root) / f"scienceqa_ablation_{ts}"
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "ablations": args.ablation,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "alpha": DEFAULT_ALPHA, "top_k": DEFAULT_TOP_K,
            "vlm": VLM_MODEL,
        }, f, indent=2)

    # Checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        n_done = sum(1 for v in checkpoint.values() if v.get("complete"))
        print(f"Checkpoint: {n_done} conditions completed")

    # Load test data
    print("\nLoading test data...")
    test_ds = load_test_data()
    print(f"  {len(test_ds)} test samples")

    # Load VLM
    print("\nLoading VLM...")
    vlm, proc = load_vlm()
    print("  VLM loaded")

    all_results = {}

    # Run ablations in order: 2 (alpha) -> 1 (summary) -> 3 (forgetting)
    if 2 in args.ablation:
        all_results["abl2"] = run_ablation2(
            vlm, proc, test_ds, checkpoint, checkpoint_path, output_dir
        )

    if 1 in args.ablation:
        all_results["abl1"] = run_ablation1(
            vlm, proc, test_ds, checkpoint, checkpoint_path, output_dir
        )

    if 3 in args.ablation:
        all_results["abl3"] = run_ablation3(
            vlm, proc, test_ds, checkpoint, checkpoint_path, output_dir
        )

    # Free VLM
    del vlm, proc
    torch.cuda.empty_cache()

    # Generate report
    generate_report(output_dir, all_results)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for abl_key in sorted(all_results.keys()):
        abl = all_results[abl_key]
        for label in sorted(abl.keys()):
            r = abl[label]
            acc = r.get("accuracy", 0)
            print(f"  [{abl_key}] {label}: {acc:.2f}%")

    print(f"\nOutput: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
