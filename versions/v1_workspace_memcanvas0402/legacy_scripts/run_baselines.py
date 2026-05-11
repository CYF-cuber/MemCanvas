#!/usr/bin/env python3
"""
No-retrieval baselines for all 4 benchmarks.
Evaluates Qwen2.5-VL-7B WITHOUT any canvas retrieval.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u run_baselines.py --benchmark scienceqa
    CUDA_VISIBLE_DEVICES=1 python -u run_baselines.py --benchmark okvqa
    CUDA_VISIBLE_DEVICES=0 python -u run_baselines.py --benchmark hotpotqa mmqa
    CUDA_VISIBLE_DEVICES=0 python -u run_baselines.py --benchmark all
"""
import argparse, io, json, os, pickle, re, string, sys
from collections import Counter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch

VLM_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
OUTPUT_ROOT = Path("/home/cyf/memcanvas0402/baselines")
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]


def normalize_answer(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())

def compute_em(p, g): return float(normalize_answer(p) == normalize_answer(g))
def compute_f1(p, g):
    pt, gt = normalize_answer(p).split(), normalize_answer(g).split()
    c = Counter(pt) & Counter(gt); n = sum(c.values())
    if n == 0: return 0.0
    pr = n/len(pt) if pt else 0; rc = n/len(gt) if gt else 0
    return 2*pr*rc/(pr+rc) if pr+rc > 0 else 0.0


def extract_answer(raw):
    for c in raw.upper():
        if c in CHOICE_LABELS:
            return c
    return "A"


def load_vlm():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)
    return vlm, proc


def run_vlm(vlm, proc, content, max_tokens=64):
    from PIL import Image
    msgs = [{"role": "user", "content": content}]
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs = [c["image"] for c in content if c.get("type") == "image"]
    if imgs:
        inp = proc(text=[txt], images=imgs, return_tensors="pt", padding=True)
    else:
        inp = proc(text=[txt], return_tensors="pt", padding=True)
    inp = {k: v.to(vlm.device) for k, v in inp.items()}
    with torch.no_grad():
        out = vlm.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
    return proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ============================================================
# ScienceQA baseline
# ============================================================
def baseline_scienceqa(vlm, proc):
    print("\n=== ScienceQA Baseline (no retrieval) ===")
    from datasets import load_dataset
    from PIL import Image
    OUT = OUTPUT_ROOT / "scienceqa"
    OUT.mkdir(parents=True, exist_ok=True)
    test_ds = load_dataset("derek-thomas/ScienceQA", split="test")
    n = len(test_ds)

    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())
    correct = sum(1 for v in results.values() if v.get("correct"))
    print(f"  {n} test samples, resuming from {len(done)}")

    for idx in tqdm(range(n), desc="ScienceQA baseline"):
        if str(idx) in done: continue
        item = test_ds[idx]
        gt = CHOICE_LABELS[item["answer"]] if item["answer"] < len(CHOICE_LABELS) else "A"

        content = []
        hint = item.get("hint", "") or ""
        q = item["question"]
        choices = item["choices"]
        choice_txt = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))
        prompt = f"{hint}\n\nQuestion: {q}\n{choice_txt}\nThink step by step, then answer with just the letter:"

        if item.get("image") is not None:
            content.append({"type": "image", "image": item["image"].convert("RGB")})
        content.append({"type": "text", "text": prompt})

        raw = run_vlm(vlm, proc, content, max_tokens=512)
        pred = extract_answer(raw)
        is_correct = pred == gt
        correct += int(is_correct)
        subject = item.get("subject", "")

        results[str(idx)] = {
            "idx": idx, "predicted": pred, "ground_truth": gt,
            "correct": is_correct, "subject": subject,
        }
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    total = len(results)
    acc = sum(1 for v in results.values() if v["correct"]) / total * 100

    per_subject = {}
    for subj in ["natural science", "social science", "language science"]:
        sp = [v for v in results.values() if v.get("subject") == subj]
        if sp:
            sc = sum(1 for v in sp if v["correct"])
            per_subject[subj] = {"n": len(sp), "correct": sc, "acc": sc/len(sp)*100}

    summary = {"n": total, "accuracy": acc, "per_subject": per_subject,
               "config": {"retrieval": "none", "model": "Qwen2.5-VL-7B"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  ScienceQA baseline: {acc:.2f}%")
    return summary


# ============================================================
# HotpotQA baseline
# ============================================================
def baseline_hotpotqa(vlm, proc):
    print("\n=== HotpotQA Baseline (no retrieval) ===")
    OUT = OUTPUT_ROOT / "hotpotqa"
    OUT.mkdir(parents=True, exist_ok=True)

    # Load data
    meta_path = Path("/home/cyf/codex/hotpotqa_data/hotpotqa_meta.pkl")
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        dev = meta.get("dev", meta.get("validation", []))
    else:
        from datasets import load_dataset
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        dev = list(ds)

    n = len(dev)
    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())
    print(f"  {n} dev samples, resuming from {len(done)}")

    for idx in tqdm(range(n), desc="HotpotQA baseline"):
        if str(idx) in done: continue
        item = dev[idx]
        q = item["question"]
        gt = item["answer"]

        content = [{"type": "text", "text": f"Question: {q}\nAnswer concisely:"}]
        pred = run_vlm(vlm, proc, content, max_tokens=64)

        em = compute_em(pred, gt)
        f1 = compute_f1(pred, gt)
        results[str(idx)] = {"gt": gt, "pred": pred, "em": em, "f1": f1}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    total = len(results)
    avg_em = np.mean([v["em"] for v in results.values()]) * 100
    avg_f1 = np.mean([v["f1"] for v in results.values()]) * 100

    summary = {"n": total, "em": avg_em, "f1": avg_f1,
               "config": {"retrieval": "none", "model": "Qwen2.5-VL-7B"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  HotpotQA baseline: EM={avg_em:.2f}%, F1={avg_f1:.2f}%")
    return summary


# ============================================================
# OK-VQA baseline
# ============================================================
def baseline_okvqa(vlm, proc):
    print("\n=== OK-VQA Baseline (no retrieval) ===")
    from PIL import Image
    OUT = OUTPUT_ROOT / "okvqa"
    OUT.mkdir(parents=True, exist_ok=True)

    DATA = Path("/home/cyf/codex/okvqa_data")
    with open(DATA / "okvqa_cached.pkl", "rb") as f:
        data = pickle.load(f)
    test = data["test"]
    n = len(test)

    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())
    print(f"  {n} test samples, resuming from {len(done)}")

    for i in tqdm(range(n), desc="OK-VQA baseline"):
        if str(i) in done: continue
        s = test[i]
        q = s["question"]
        answers = s.get("answers", [])

        content = []
        img_path = s.get("image_path", "")
        if img_path and os.path.exists(img_path):
            content.append({"type": "image", "image": Image.open(img_path).convert("RGB")})

        content.append({"type": "text", "text": f"Question: {q}\nAnswer concisely:"})
        pred = run_vlm(vlm, proc, content, max_tokens=32)

        correct = float(any(normalize_answer(pred) == normalize_answer(a) for a in answers))
        results[str(i)] = {"gt": answers, "pred": pred, "correct": correct}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    acc = np.mean([v["correct"] for v in results.values()]) * 100

    summary = {"n": len(results), "accuracy": acc,
               "config": {"retrieval": "none", "model": "Qwen2.5-VL-7B"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  OK-VQA baseline: {acc:.2f}%")
    return summary


# ============================================================
# MMQA baseline
# ============================================================
def baseline_mmqa(vlm, proc):
    print("\n=== MMQA Baseline (no retrieval) ===")
    OUT = OUTPUT_ROOT / "mmqa"
    OUT.mkdir(parents=True, exist_ok=True)

    DATA = Path("/home/cyf/codex/mmqa_data")
    with open(DATA / "mmqa_parsed.pkl", "rb") as f:
        mmqa = pickle.load(f)
    dev = mmqa["dev"]
    tables = mmqa["tables"]
    texts = mmqa["texts"]
    n = len(dev)

    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())
    print(f"  {n} dev samples, resuming from {len(done)}")

    for i in tqdm(range(n), desc="MMQA baseline"):
        if str(i) in done: continue
        item = dev[i]
        q = item["question"]
        gt_answers = item["answers"]
        gt = gt_answers[0]["answer"] if gt_answers and isinstance(gt_answers[0], dict) else (gt_answers[0] if gt_answers else "")

        # Include supporting context (text/table) but NO canvases
        parts = []
        for ctx in item.get("supporting_context", [])[:3]:
            doc_id = ctx["doc_id"]
            doc_part = ctx["doc_part"]
            if doc_part == "text" and doc_id in texts:
                doc = texts[doc_id]
                parts.append(f"[Text: {doc.get('title','')}]\n{doc.get('text','')[:400]}")
            elif doc_part == "table" and doc_id in tables:
                doc = tables[doc_id]
                tbl = doc.get("table", {})
                headers = [h["column_name"] for h in tbl.get("header", [])]
                rows = [[c["text"][:30] for c in row] for row in tbl.get("table_rows", [])[:6]]
                lines = [" | ".join(headers)]
                for row in rows:
                    lines.append(" | ".join(row))
                parts.append(f"[Table: {doc.get('title','')}]\n" + "\n".join(lines))

        ctx_text = "\n\n".join(parts)
        prompt = ""
        if ctx_text:
            prompt += f"Context:\n{ctx_text}\n\n"
        prompt += f"Question: {q}\nAnswer with ONLY the answer:"
        content = [{"type": "text", "text": prompt}]
        pred = run_vlm(vlm, proc, content, max_tokens=64)

        em = compute_em(pred, gt)
        f1 = compute_f1(pred, gt)
        qtype = item.get("metadata", {}).get("type", "unknown")

        results[str(i)] = {"gt": gt, "pred": pred, "em": em, "f1": f1, "type": qtype}
        if len(results) % 50 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    avg_em = np.mean([v["em"] for v in results.values()]) * 100
    avg_f1 = np.mean([v["f1"] for v in results.values()]) * 100

    summary = {"n": len(results), "em": avg_em, "f1": avg_f1,
               "config": {"retrieval": "none", "model": "Qwen2.5-VL-7B"}}
    for qt in set(v.get("type", "unknown") for v in results.values()):
        vals = [v for v in results.values() if v.get("type") == qt]
        if vals:
            summary[f"type_{qt}"] = {
                "n": len(vals),
                "em": float(np.mean([v["em"] for v in vals]) * 100),
                "f1": float(np.mean([v["f1"] for v in vals]) * 100),
            }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  MMQA baseline: EM={avg_em:.2f}%, F1={avg_f1:.2f}%")
    return summary


BENCHMARKS = {
    "scienceqa": baseline_scienceqa,
    "hotpotqa": baseline_hotpotqa,
    "okvqa": baseline_okvqa,
    "mmqa": baseline_mmqa,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", nargs="+", required=True,
                        help="Benchmarks to run: scienceqa hotpotqa okvqa mmqa all")
    args = parser.parse_args()

    benchmarks = args.benchmark
    if "all" in benchmarks:
        benchmarks = list(BENCHMARKS.keys())

    print(f"Running baselines for: {benchmarks}")
    vlm, proc = load_vlm()

    for bm in benchmarks:
        if bm in BENCHMARKS:
            BENCHMARKS[bm](vlm, proc)
        else:
            print(f"Unknown benchmark: {bm}")

    del vlm, proc
    torch.cuda.empty_cache()
    print("\nAll baselines complete!")


if __name__ == "__main__":
    main()
