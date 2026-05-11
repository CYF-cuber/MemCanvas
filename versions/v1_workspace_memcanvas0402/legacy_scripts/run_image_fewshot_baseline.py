#!/usr/bin/env python3
"""
Image Few-shot (Visual RAG) baselines for all 4 benchmarks.
Uses same CLIP retrieval as MemCanvas, but provides original training IMAGES
(not canvas images) as visual few-shot examples alongside text Q&A context.
Tests whether Canvas *rendering* adds value over raw image retrieval.

For HotpotQA (text-only), this degrades to Text-RAG since there are no images.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -u run_image_fewshot_baseline.py --benchmark scienceqa okvqa mmqa
"""
import argparse, json, os, pickle, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

sys.path.insert(0, "/home/cyf/codex")

VLM_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
CLIP_MODEL = "openai/clip-vit-large-patch14"
TOP_K = 2
ALPHA = 0.00
OUTPUT_ROOT = Path("/home/cyf/memcanvas0402/image_fewshot_baselines")
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
        if c in CHOICE_LABELS: return c
    return "A"

def build_retrieval_map(img_emb, txt_emb, query_emb, alpha=ALPHA, top_k=TOP_K):
    keys = alpha * img_emb + (1-alpha) * txt_emb
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True).clip(1e-8)
    qn = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True).clip(1e-8)
    sims = qn @ keys.T
    rmap = {}
    for i in range(len(query_emb)):
        top = np.argsort(sims[i])[::-1][:top_k+5]
        res = [(int(j), float(sims[i][j])) for j in top if sims[i][j] >= 0.1][:top_k]
        rmap[i] = res
    return rmap


def load_vlm():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)
    return vlm, proc


def run_vlm(vlm, proc, content, max_tokens=64):
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
# ScienceQA Image Few-shot
# ============================================================
def imgfs_scienceqa(vlm, proc):
    print("\n=== ScienceQA Image Few-shot ===")
    OUT = OUTPUT_ROOT / "scienceqa"; OUT.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    test_ds = load_dataset("derek-thomas/ScienceQA", split="test")
    train_ds = load_dataset("derek-thomas/ScienceQA", split="train")

    CANVAS_DIR = Path("/home/cyf/codex/scienceqa_smart_canvases")
    img_emb = np.load(CANVAS_DIR / "clip_img_emb.npy")
    txt_emb = np.load(CANVAS_DIR / "clip_txt_emb.npy")
    q_emb = np.load(CANVAS_DIR / "clip_query_emb.npy")
    rmap = build_retrieval_map(img_emb, txt_emb, q_emb)
    print(f"  {len(test_ds)} test, {len(train_ds)} train")

    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())
    print(f"  Resuming from {len(done)}")

    for idx in tqdm(range(len(test_ds)), desc="ScienceQA img-fewshot"):
        if str(idx) in done: continue
        item = test_ds[idx]
        gt = CHOICE_LABELS[item["answer"]] if item["answer"] < len(CHOICE_LABELS) else "A"

        content = []
        # Add retrieved training sample ORIGINAL images + text
        ref_texts = []
        for cidx, sim in rmap.get(idx, [])[:TOP_K]:
            tr = train_ds[cidx]
            # Add original training image if available
            if tr.get("image") is not None:
                content.append({"type": "image", "image": tr["image"].convert("RGB")})
            parts = [f"Subject: {tr.get('subject','')}, Topic: {tr.get('topic','')}"]
            parts.append(f"Q: {tr['question']}")
            choices = tr['choices']
            parts.append("Choices: " + ", ".join(f"{chr(65+j)}.{c}" for j, c in enumerate(choices)))
            parts.append(f"Answer: {CHOICE_LABELS[tr['answer']]}")
            if tr.get('lecture'): parts.append(f"Lecture: {tr['lecture'][:400]}")
            if tr.get('solution'): parts.append(f"Solution: {tr['solution'][:300]}")
            ref_texts.append("\n".join(parts))

        context = "\n---\n".join(ref_texts)
        hint = item.get("hint", "") or ""
        q = item["question"]
        choices = item["choices"]
        choice_txt = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))
        prompt = f"Reference examples (with their images shown above):\n{context}\n\n{hint}\n\nQuestion: {q}\n{choice_txt}\nThink step by step, then answer with just the letter:"

        if item.get("image") is not None:
            content.append({"type": "image", "image": item["image"].convert("RGB")})
        content.append({"type": "text", "text": prompt})

        raw = run_vlm(vlm, proc, content, max_tokens=512)
        pred = extract_answer(raw)
        results[str(idx)] = {"idx": idx, "predicted": pred, "ground_truth": gt,
                             "correct": pred == gt, "subject": item.get("subject", "")}
        if len(results) % 100 == 0: json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    acc = sum(1 for v in results.values() if v["correct"]) / len(results) * 100
    summary = {"n": len(results), "accuracy": acc,
               "config": {"method": "image-fewshot", "alpha": ALPHA, "top_k": TOP_K, "model": "Qwen2.5-VL-7B"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  ScienceQA image-fewshot: {acc:.2f}%")
    return summary


# ============================================================
# OK-VQA Image Few-shot
# ============================================================
def imgfs_okvqa(vlm, proc):
    print("\n=== OK-VQA Image Few-shot ===")
    OUT = OUTPUT_ROOT / "okvqa"; OUT.mkdir(parents=True, exist_ok=True)
    DATA = Path("/home/cyf/codex/okvqa_data")
    CANVAS_DIR = DATA / "canvases_smart"
    with open(DATA / "okvqa_cached.pkl", "rb") as f:
        d = pickle.load(f)
    train, test = d["train"], d["test"]

    img_emb = np.load(str(CANVAS_DIR / "clip_img_emb.npy"))
    txt_emb = np.load(str(CANVAS_DIR / "clip_txt_emb.npy"))
    q_emb = np.load(str(CANVAS_DIR / "clip_query_emb.npy"))
    rmap = build_retrieval_map(img_emb, txt_emb, q_emb)
    print(f"  {len(test)} test, {len(train)} train")

    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())
    print(f"  Resuming from {len(done)}")

    for i in tqdm(range(len(test)), desc="OK-VQA img-fewshot"):
        if str(i) in done: continue
        s = test[i]
        q = s["question"]
        answers = s.get("answers", [])

        content = []
        ref_texts = []
        for cidx, sim in rmap.get(i, [])[:TOP_K]:
            tr = train[cidx]
            # Add original training image
            tr_img_path = tr.get("image_path", "")
            if tr_img_path and os.path.exists(tr_img_path):
                content.append({"type": "image", "image": Image.open(tr_img_path).convert("RGB")})
            parts = [f"Q: {tr['question']}"]
            if tr.get('caption'): parts.append(f"Image description: {tr['caption'][:300]}")
            parts.append(f"Answers: {', '.join(tr.get('answers', [])[:5])}")
            ref_texts.append("\n".join(parts))

        # Test image
        img_path = s.get("image_path", "")
        if img_path and os.path.exists(img_path):
            content.append({"type": "image", "image": Image.open(img_path).convert("RGB")})

        context = "\n---\n".join(ref_texts)
        prompt = f"Reference examples (with their images shown above):\n{context}\n\nQuestion: {q}\nAnswer concisely:"
        content.append({"type": "text", "text": prompt})

        pred = run_vlm(vlm, proc, content, max_tokens=32)
        correct = float(any(normalize_answer(pred) == normalize_answer(a) for a in answers))
        results[str(i)] = {"gt": answers, "pred": pred, "correct": correct}
        if len(results) % 100 == 0: json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    acc = np.mean([v["correct"] for v in results.values()]) * 100
    summary = {"n": len(results), "accuracy": acc,
               "config": {"method": "image-fewshot", "alpha": ALPHA, "top_k": TOP_K, "model": "Qwen2.5-VL-7B"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  OK-VQA image-fewshot: {acc:.2f}%")
    return summary


# ============================================================
# HotpotQA — skip (text-only, no images to provide)
# ============================================================
def imgfs_hotpotqa(vlm, proc):
    print("\n=== HotpotQA Image Few-shot: SKIPPED (text-only dataset, same as Text-RAG) ===")
    return None


# ============================================================
# MMQA Image Few-shot
# ============================================================
def imgfs_mmqa(vlm, proc):
    print("\n=== MMQA Image Few-shot ===")
    OUT = OUTPUT_ROOT / "mmqa"; OUT.mkdir(parents=True, exist_ok=True)
    DATA = Path("/home/cyf/codex/mmqa_data")
    CANVAS_DIR = DATA / "canvases_smart"
    with open(DATA / "mmqa_parsed.pkl", "rb") as f:
        mmqa = pickle.load(f)
    train_list, dev = mmqa["train"], mmqa["dev"]
    tables, texts, images = mmqa["tables"], mmqa["texts"], mmqa.get("images", {})

    img_emb = np.load(str(CANVAS_DIR / "clip_img_emb.npy"))
    txt_emb = np.load(str(CANVAS_DIR / "clip_txt_emb.npy"))
    q_emb = np.load(str(CANVAS_DIR / "clip_query_emb.npy"))
    rmap = build_retrieval_map(img_emb, txt_emb, q_emb)
    print(f"  {len(dev)} dev, {len(train_list)} train, {len(images)} images")

    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())
    print(f"  Resuming from {len(done)}")

    for i in tqdm(range(len(dev)), desc="MMQA img-fewshot"):
        if str(i) in done: continue
        item = dev[i]
        q = item["question"]
        gt_answers = item["answers"]
        gt = gt_answers[0]["answer"] if gt_answers and isinstance(gt_answers[0], dict) else (gt_answers[0] if gt_answers else "")

        content = []
        ref_texts = []
        for cidx, sim in rmap.get(i, [])[:TOP_K]:
            tr = train_list[cidx]
            parts = [f"Q: {tr['question']}"]
            tr_ans = tr.get("answers", [])
            if tr_ans:
                a = tr_ans[0]["answer"] if isinstance(tr_ans[0], dict) else tr_ans[0]
                parts.append(f"A: {a}")
            # Add original images from supporting context
            for ctx in tr.get("supporting_context", [])[:3]:
                doc_id, doc_part = ctx["doc_id"], ctx["doc_part"]
                if doc_part == "image" and doc_id in images:
                    img_info = images[doc_id]
                    img_path = img_info.get("path", "")
                    if img_path and os.path.exists(img_path):
                        try:
                            content.append({"type": "image", "image": Image.open(img_path).convert("RGB")})
                        except: pass
                elif doc_part == "text" and doc_id in texts:
                    doc = texts[doc_id]
                    parts.append(f"[Text: {doc.get('title','')}] {doc.get('text','')[:300]}")
                elif doc_part == "table" and doc_id in tables:
                    doc = tables[doc_id]
                    tbl = doc.get("table", {})
                    headers = [h["column_name"] for h in tbl.get("header", [])]
                    rows = [[c["text"][:30] for c in row] for row in tbl.get("table_rows", [])[:4]]
                    lines = [" | ".join(headers)] + [" | ".join(r) for r in rows]
                    parts.append(f"[Table: {doc.get('title','')}]\n" + "\n".join(lines))
            ref_texts.append("\n".join(parts))

        # Test item context
        test_ctx_parts = []
        for ctx in item.get("supporting_context", [])[:3]:
            doc_id, doc_part = ctx["doc_id"], ctx["doc_part"]
            if doc_part == "image" and doc_id in images:
                img_info = images[doc_id]
                img_path = img_info.get("path", "")
                if img_path and os.path.exists(img_path):
                    try:
                        content.append({"type": "image", "image": Image.open(img_path).convert("RGB")})
                    except: pass
            elif doc_part == "text" and doc_id in texts:
                doc = texts[doc_id]
                test_ctx_parts.append(f"[Text: {doc.get('title','')}]\n{doc.get('text','')[:400]}")
            elif doc_part == "table" and doc_id in tables:
                doc = tables[doc_id]
                tbl = doc.get("table", {})
                headers = [h["column_name"] for h in tbl.get("header", [])]
                rows = [[c["text"][:30] for c in row] for row in tbl.get("table_rows", [])[:6]]
                lines = [" | ".join(headers)] + [" | ".join(r) for r in rows]
                test_ctx_parts.append(f"[Table: {doc.get('title','')}]\n" + "\n".join(lines))

        context = "\n---\n".join(ref_texts)
        test_ctx = "\n\n".join(test_ctx_parts)
        prompt = f"Reference examples (with images shown above):\n{context}\n\n"
        if test_ctx: prompt += f"Context:\n{test_ctx}\n\n"
        prompt += f"Question: {q}\nAnswer with ONLY the answer:"
        content.append({"type": "text", "text": prompt})
        pred = run_vlm(vlm, proc, content, max_tokens=64)

        results[str(i)] = {"gt": gt, "pred": pred, "em": compute_em(pred, gt), "f1": compute_f1(pred, gt),
                           "type": item.get("metadata", {}).get("type", "unknown")}
        if len(results) % 50 == 0: json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    avg_em = np.mean([v["em"] for v in results.values()]) * 100
    avg_f1 = np.mean([v["f1"] for v in results.values()]) * 100
    summary = {"n": len(results), "em": avg_em, "f1": avg_f1,
               "config": {"method": "image-fewshot", "alpha": ALPHA, "top_k": TOP_K, "model": "Qwen2.5-VL-7B"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  MMQA image-fewshot: EM={avg_em:.2f}%, F1={avg_f1:.2f}%")
    return summary


BENCHMARKS = {
    "scienceqa": imgfs_scienceqa, "hotpotqa": imgfs_hotpotqa,
    "okvqa": imgfs_okvqa, "mmqa": imgfs_mmqa,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", nargs="+", required=True)
    args = parser.parse_args()
    benchmarks = list(BENCHMARKS.keys()) if "all" in args.benchmark else args.benchmark
    print(f"Running image-fewshot baselines for: {benchmarks}")
    vlm, proc = load_vlm()
    for bm in benchmarks:
        if bm in BENCHMARKS: BENCHMARKS[bm](vlm, proc)
    del vlm, proc; torch.cuda.empty_cache()
    print("\nAll image-fewshot baselines complete!")

if __name__ == "__main__":
    main()
