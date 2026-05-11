#!/usr/bin/env python3
"""
Text RAG baseline for all benchmarks.
Uses BM25 retrieval over text + Qwen2.5-VL-7B as reader.
Usage:
  CUDA_VISIBLE_DEVICES=X python -u text_rag_baseline.py --benchmark hotpotqa
  CUDA_VISIBLE_DEVICES=X python -u text_rag_baseline.py --benchmark scienceqa
  CUDA_VISIBLE_DEVICES=X python -u text_rag_baseline.py --benchmark mmqa
  CUDA_VISIBLE_DEVICES=X python -u text_rag_baseline.py --benchmark infographicvqa
"""
import argparse, json, os, pickle, re, string, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch

VLM_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
OUTPUT_ROOT = Path("/home/cyf/codex/text_rag_results")

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

def compute_anls(p, g):
    from difflib import SequenceMatcher
    p, g = normalize_answer(p), normalize_answer(g)
    if not p or not g: return 0.0
    ratio = SequenceMatcher(None, p, g).ratio()
    return ratio if ratio >= 0.5 else 0.0

def bm25_retrieve(corpus_texts, query, top_k=2):
    """Simple BM25-like TF-IDF retrieval."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not hasattr(bm25_retrieve, '_cache'):
        bm25_retrieve._cache = {}
    cache_key = id(corpus_texts)
    if cache_key not in bm25_retrieve._cache:
        vectorizer = TfidfVectorizer(max_features=50000, stop_words='english')
        tfidf = vectorizer.fit_transform([str(t) for t in corpus_texts])
        bm25_retrieve._cache[cache_key] = (vectorizer, tfidf)
    vectorizer, tfidf = bm25_retrieve._cache[cache_key]
    q_vec = vectorizer.transform([query])
    scores = (tfidf @ q_vec.T).toarray().flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

def vlm_answer(vlm, proc, prompt, max_tokens=64):
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[txt], return_tensors="pt", padding=True)
    inp = {k: v.to(vlm.device) for k, v in inp.items()}
    with torch.no_grad():
        out = vlm.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
    return proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ============================================================
# HotpotQA Text RAG
# ============================================================
def eval_hotpotqa():
    print("=== HotpotQA Text RAG ===")
    DATA = Path("/home/cyf/codex/hotpotqa_data")
    with open(DATA / "hotpotqa_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    train, dev = meta["train"], meta["dev"]

    # Build corpus: concatenate paragraphs for each train sample
    corpus = []
    for s in train:
        text = f"Q: {s['question']} A: {s['answer']}\n"
        for p in s.get("paragraphs", []):
            text += f"{p.get('title','')}: {p.get('text','')[:300]}\n"
        corpus.append(text)

    out_dir = OUTPUT_ROOT / "hotpotqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)

    for i in tqdm(range(len(dev)), desc="HotpotQA TextRAG"):
        if str(i) in results: continue
        s = dev[i]
        q = s["question"]
        gt = s["answer"]
        ctx = "\n\n".join(f"[{p['title']}]\n{p['text'][:500]}" for p in s.get("paragraphs", []))

        # Retrieve similar training examples as text
        retrieved = bm25_retrieve(corpus, q, top_k=2)
        rag_ctx = "\n\n---\n\n".join(corpus[idx][:500] for idx, _ in retrieved)

        prompt = (
            f"Reference examples:\n{rag_ctx}\n\n"
            f"Now answer using context:\n{ctx}\n\n"
            f"Question: {q}\nAnswer concisely:"
        )
        pred = vlm_answer(vlm, proc, prompt)
        results[str(i)] = {"gt": gt, "pred": pred, "em": compute_em(pred, gt), "f1": compute_f1(pred, gt)}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    em = np.mean([v["em"] for v in results.values()]) * 100
    f1 = np.mean([v["f1"] for v in results.values()]) * 100
    summary = {"n": len(results), "em": em, "f1": f1}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"  HotpotQA TextRAG: EM={em:.2f}%, F1={f1:.2f}%")
    del vlm, proc; torch.cuda.empty_cache()

# ============================================================
# ScienceQA Text RAG
# ============================================================
def eval_scienceqa():
    print("=== ScienceQA Text RAG ===")
    from datasets import load_dataset
    train_ds = load_dataset("derek-thomas/ScienceQA", split="train")
    test_ds = load_dataset("derek-thomas/ScienceQA", split="test")

    # Build text corpus from training examples
    corpus = []
    train_answers = []
    for item in train_ds:
        text = f"Q: {item['question']}\n"
        if item.get("hint"): text += f"Hint: {item['hint']}\n"
        if item.get("lecture"): text += f"Lecture: {item['lecture'][:300]}\n"
        text += f"Choices: {', '.join(item['choices'])}\n"
        text += f"Answer: {chr(65 + item['answer'])}"
        corpus.append(text)
        train_answers.append(chr(65 + item["answer"]))

    out_dir = OUTPUT_ROOT / "scienceqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)

    for i in tqdm(range(len(test_ds)), desc="ScienceQA TextRAG"):
        if str(i) in results: continue
        item = test_ds[i]
        q = item["question"]
        gt = chr(65 + item["answer"])
        choices = item["choices"]
        hint = item.get("hint", "") or ""
        choice_txt = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))

        retrieved = bm25_retrieve(corpus, f"{q} {hint}", top_k=2)
        rag_ctx = "\n\n---\n\n".join(corpus[idx][:400] for idx, _ in retrieved)

        prompt = (
            f"Reference examples:\n{rag_ctx}\n\n"
            f"{hint}\nQuestion: {q}\n{choice_txt}\n"
            "Answer with just the letter:"
        )

        # Include image if available
        content = []
        imgs = []
        if item.get("image") is not None:
            from PIL import Image
            img = item["image"].convert("RGB")
            imgs.append(img)
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if imgs:
            inp = proc(text=[txt], images=imgs, return_tensors="pt", padding=True)
        else:
            inp = proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=512, do_sample=False)
        raw = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        pred = ""
        for c in raw.upper():
            if c in "ABCDEF": pred = c; break

        results[str(i)] = {
            "gt": gt, "pred": pred, "correct": float(pred == gt),
            "subject": item.get("subject", ""),
        }
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    acc = np.mean([v["correct"] for v in results.values()]) * 100
    summary = {"n": len(results), "accuracy": acc}
    for subj in ["natural science", "social science", "language science"]:
        vals = [v["correct"] for v in results.values() if v.get("subject","") == subj]
        if vals: summary[subj] = np.mean(vals) * 100
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"  ScienceQA TextRAG: {acc:.2f}%")
    del vlm, proc; torch.cuda.empty_cache()

# ============================================================
# MMQA Text RAG
# ============================================================
def eval_mmqa():
    print("=== MMQA Text RAG ===")
    import pickle
    DATA = Path("/home/cyf/codex/mmqa_data")
    mmqa_parsed = pickle.load(open(DATA / "mmqa_parsed.pkl", "rb"))
    train = mmqa_parsed["train"]
    dev = mmqa_parsed["dev"]

    # Build corpus from train split
    corpus = []
    for item in train:
        text = f"Q: {item['question']}\nA: {item['answers'][0]['answer'] if item['answers'] and isinstance(item['answers'][0], dict) else item['answers'][0] if item['answers'] else ''}"
        corpus.append(text)

    out_dir = OUTPUT_ROOT / "mmqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)

    for i in tqdm(range(len(dev)), desc="MMQA TextRAG"):
        if str(i) in results: continue
        item = dev[i]
        q = item["question"]
        gt_answers = item["answers"]
        gt = gt_answers[0]["answer"] if gt_answers and isinstance(gt_answers[0], dict) else (gt_answers[0] if gt_answers else "")

        retrieved = bm25_retrieve(corpus, q, top_k=2)
        rag_ctx = "\n\n---\n\n".join(corpus[idx][:400] for idx, _ in retrieved)

        prompt = f"Reference:\n{rag_ctx}\n\nQuestion: {q}\nAnswer concisely:"
        pred = vlm_answer(vlm, proc, prompt)
        results[str(i)] = {"gt": gt, "pred": pred, "em": compute_em(pred, gt), "f1": compute_f1(pred, gt)}
        if len(results) % 50 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    em = np.mean([v["em"] for v in results.values()]) * 100
    f1 = np.mean([v["f1"] for v in results.values()]) * 100
    summary = {"n": len(results), "em": em, "f1": f1}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"  MMQA TextRAG: EM={em:.2f}%, F1={f1:.2f}%")
    del vlm, proc; torch.cuda.empty_cache()

# ============================================================
# InfographicVQA Text RAG
# ============================================================
def eval_infographicvqa():
    print("=== InfographicVQA Text RAG ===")
    DATA = Path("/home/cyf/codex/infographicvqa_data")
    with open(DATA / "infographicvqa_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    train, val = meta["train"], meta["val"]

    # Build corpus from training QA pairs
    corpus = []
    for s in train:
        text = f"Q: {s['question']}\nA: {', '.join(s['answers'][:3])}"
        corpus.append(text)

    out_dir = OUTPUT_ROOT / "infographicvqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)

    for i in tqdm(range(len(val)), desc="InfoVQA TextRAG"):
        if str(i) in results: continue
        s = val[i]
        q = s["question"]
        gt_answers = s["answers"]

        retrieved = bm25_retrieve(corpus, q, top_k=2)
        rag_ctx = "\n\n---\n\n".join(corpus[idx][:400] for idx, _ in retrieved)

        # Include the infographic image
        from PIL import Image
        content = []
        imgs = []
        img_path = s.get("image_path", "")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            imgs.append(img)
            content.append({"type": "image", "image": img})
        prompt = f"Reference QA:\n{rag_ctx}\n\nQuestion about the infographic: {q}\nAnswer concisely:"
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if imgs:
            inp = proc(text=[txt], images=imgs, return_tensors="pt", padding=True)
        else:
            inp = proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=128, do_sample=False)
        pred = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        anls = max(compute_anls(pred, a) for a in gt_answers)
        results[str(i)] = {"gt": gt_answers, "pred": pred, "anls": anls}
        if len(results) % 50 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    avg_anls = np.mean([v["anls"] for v in results.values()]) * 100
    summary = {"n": len(results), "anls": avg_anls}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"  InfographicVQA TextRAG: ANLS={avg_anls:.2f}%")
    del vlm, proc; torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True,
                        choices=["hotpotqa", "scienceqa", "mmqa", "infographicvqa", "all"])
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.benchmark == "all":
        for b in ["hotpotqa", "scienceqa", "mmqa", "infographicvqa"]:
            eval_fn = {"hotpotqa": eval_hotpotqa, "scienceqa": eval_scienceqa,
                       "mmqa": eval_mmqa, "infographicvqa": eval_infographicvqa}[b]
            eval_fn()
    else:
        {"hotpotqa": eval_hotpotqa, "scienceqa": eval_scienceqa,
         "mmqa": eval_mmqa, "infographicvqa": eval_infographicvqa}[args.benchmark]()

if __name__ == "__main__":
    main()
