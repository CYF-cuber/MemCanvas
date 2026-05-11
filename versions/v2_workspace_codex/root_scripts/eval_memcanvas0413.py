#!/usr/bin/env python3
"""
MemCanvas 0413 Evaluation — Unified SmartCanvas eval for all 4 benchmarks.

Pipeline: SmartCanvas canvases + CLIP hybrid retrieval + VLM (no Read Agent).
All SmartCanvas canvases and CLIP embeddings are pre-computed.

Usage:
  CUDA_VISIBLE_DEVICES=1 python -u eval_memcanvas0413.py --benchmark scienceqa --alpha 0.00
  CUDA_VISIBLE_DEVICES=1 python -u eval_memcanvas0413.py --benchmark okvqa --alpha 0.75
  CUDA_VISIBLE_DEVICES=1 python -u eval_memcanvas0413.py --benchmark mmqa --alpha 0.50
  CUDA_VISIBLE_DEVICES=1 python -u eval_memcanvas0413.py --benchmark hotpotqa --alpha 0.75
  CUDA_VISIBLE_DEVICES=1 python -u eval_memcanvas0413.py --all
"""
import argparse, io, json, os, pickle, re, string, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

VLM_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
TOP_K = 2
SIM_THRESHOLD = 0.1
OUT_ROOT = Path("/home/cyf/codex/memcanvas0413_eval")

# ============================================================
# Metrics
# ============================================================
def normalize_answer(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())

def compute_em(p, g):
    return float(normalize_answer(p) == normalize_answer(g))

def compute_f1(p, g):
    pt, gt = normalize_answer(p).split(), normalize_answer(g).split()
    c = Counter(pt) & Counter(gt); n = sum(c.values())
    if n == 0: return 0.0
    pr = n / len(pt) if pt else 0; rc = n / len(gt) if gt else 0
    return 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0

# ============================================================
# Retrieval (shared)
# ============================================================
def build_retrieval_map(img_emb, txt_emb, query_emb, alpha, top_k=TOP_K):
    keys = alpha * img_emb + (1 - alpha) * txt_emb
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True).clip(1e-8)
    qn = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True).clip(1e-8)
    sims = qn @ keys.T
    rmap = {}
    for i in range(len(query_emb)):
        top = np.argsort(sims[i])[::-1][:top_k + 5]
        res = [(int(j), float(sims[i][j])) for j in top if sims[i][j] >= SIM_THRESHOLD][:top_k]
        rmap[i] = res
    has = sum(1 for v in rmap.values() if v)
    print(f"  Retrieval map: {has}/{len(query_emb)} have memories (α={alpha}, K={top_k})")
    return rmap

def load_vlm():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading VLM...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)
    return model, proc

# ============================================================
# ScienceQA
# ============================================================
def run_scienceqa(alpha):
    print(f"\n{'='*60}\n  ScienceQA (α={alpha})\n{'='*60}")
    CANVAS_DIR = Path("/home/cyf/codex/scienceqa_smart_canvases")
    OUT = OUT_ROOT / f"scienceqa_alpha{alpha:.2f}"
    OUT.mkdir(parents=True, exist_ok=True)

    # Load data
    with open("/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl", "rb") as f:
        cache = pickle.load(f)
    train = cache["train"] if isinstance(cache, dict) else cache[0]
    from datasets import load_dataset
    test_ds = load_dataset("derek-thomas/ScienceQA", split="test")

    # Load pre-computed SmartCanvas CLIP embeddings
    img_emb = np.load(CANVAS_DIR / "clip_img_emb.npy")
    txt_emb = np.load(CANVAS_DIR / "clip_txt_emb.npy")
    q_emb = np.load(CANVAS_DIR / "clip_query_emb.npy")
    rmap = build_retrieval_map(img_emb, txt_emb, q_emb, alpha)

    # VLM eval
    vlm, proc = load_vlm()
    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())

    for i in tqdm(range(len(test_ds)), desc="ScienceQA"):
        if str(i) in done: continue
        item = test_ds[i]
        q = item["question"]
        choices = item["choices"]
        gt = chr(65 + item["answer"])
        hint = item.get("hint", "") or ""
        choice_txt = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))

        content = []
        canvas_imgs = []
        for cidx, sim in rmap.get(i, [])[:TOP_K]:
            img = Image.open(CANVAS_DIR / f"{cidx:05d}.png").convert("RGB")
            canvas_imgs.append(img)
            content.append({"type": "image", "image": img})

        prompt = (f"Study the reference canvases above. Each shows a solved example.\n"
                  f"{hint}\n\nQuestion: {q}\n{choice_txt}\n"
                  f"Think step by step, then answer with just the letter:")
        if item.get("image") is not None:
            content.append({"type": "image", "image": item["image"].convert("RGB")})
        content.append({"type": "text", "text": prompt})

        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        all_imgs = canvas_imgs + ([item["image"].convert("RGB")] if item.get("image") else [])
        if all_imgs:
            inp = proc(text=[txt], images=all_imgs, return_tensors="pt", padding=True)
        else:
            inp = proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=512, do_sample=False)
        raw = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        pred = ""
        for c in raw.upper():
            if c in "ABCDEF": pred = c; break

        results[str(i)] = {"gt": gt, "pred": pred, "correct": float(pred == gt),
                           "subject": item.get("subject", "")}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    acc = np.mean([v["correct"] for v in results.values()]) * 100
    summary = {"n": len(results), "accuracy": acc,
               "config": {"alpha": alpha, "top_k": TOP_K, "method": "SmartCanvas (no Read Agent)"}}
    for subj in ["natural science", "social science", "language science"]:
        vals = [v["correct"] for v in results.values() if v.get("subject", "") == subj]
        if vals: summary[subj] = np.mean(vals) * 100
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  ScienceQA α={alpha}: {acc:.2f}%")
    del vlm, proc; torch.cuda.empty_cache()
    return summary

# ============================================================
# OK-VQA
# ============================================================
def run_okvqa(alpha):
    print(f"\n{'='*60}\n  OK-VQA (α={alpha})\n{'='*60}")
    DATA = Path("/home/cyf/codex/okvqa_data")
    CANVAS_DIR = DATA / "canvases_smart"
    OUT = OUT_ROOT / f"okvqa_alpha{alpha:.2f}"
    OUT.mkdir(parents=True, exist_ok=True)

    with open(DATA / "okvqa_cached.pkl", "rb") as f:
        data = pickle.load(f)
    train, test = data["train"], data["test"]

    img_emb = np.load(CANVAS_DIR / "clip_img_emb.npy")
    txt_emb = np.load(CANVAS_DIR / "clip_txt_emb.npy")
    q_emb = np.load(CANVAS_DIR / "clip_query_emb.npy")
    rmap = build_retrieval_map(img_emb, txt_emb, q_emb, alpha)

    vlm, proc = load_vlm()
    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())

    for i in tqdm(range(len(test)), desc="OK-VQA"):
        if str(i) in done: continue
        s = test[i]
        q = s["question"]
        answers = s.get("answers", [])

        content = []
        canvas_imgs = []
        for cidx, sim in rmap.get(i, [])[:TOP_K]:
            img = Image.open(CANVAS_DIR / f"{cidx:05d}.png").convert("RGB")
            canvas_imgs.append(img)
            content.append({"type": "image", "image": img})

        img_path = s.get("image_path", "")
        test_img = None
        if img_path and os.path.exists(img_path):
            test_img = Image.open(img_path).convert("RGB")
            content.append({"type": "image", "image": test_img})

        prompt = ("Study the reference canvases. Answer the question about the last image.\n"
                  f"Question: {q}\nAnswer concisely:")
        content.append({"type": "text", "text": prompt})

        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        all_imgs = canvas_imgs + ([test_img] if test_img else [])
        if all_imgs:
            inp = proc(text=[txt], images=all_imgs, return_tensors="pt", padding=True)
        else:
            inp = proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=32, do_sample=False)
        pred = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        correct = float(any(normalize_answer(pred) == normalize_answer(a) for a in answers))
        results[str(i)] = {"gt": answers, "pred": pred, "correct": correct}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    acc = np.mean([v["correct"] for v in results.values()]) * 100
    summary = {"n": len(results), "accuracy": acc,
               "config": {"alpha": alpha, "top_k": TOP_K, "method": "SmartCanvas (no Read Agent)"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  OK-VQA α={alpha}: {acc:.2f}%")
    del vlm, proc; torch.cuda.empty_cache()
    return summary

# ============================================================
# MMQA
# ============================================================
def run_mmqa(alpha):
    print(f"\n{'='*60}\n  MMQA (α={alpha})\n{'='*60}")
    DATA = Path("/home/cyf/codex/mmqa_data")
    CANVAS_DIR = DATA / "canvases_smart"
    OUT = OUT_ROOT / f"mmqa_alpha{alpha:.2f}"
    OUT.mkdir(parents=True, exist_ok=True)

    # Load parsed data
    with open(DATA / "mmqa_parsed.pkl", "rb") as f:
        parsed = pickle.load(f)
    dev_data = list(parsed["dev"].values()) if isinstance(parsed["dev"], dict) else parsed["dev"]
    tables = parsed.get("tables", {})
    texts = parsed.get("texts", {})
    images_meta = parsed.get("images", {})

    # Load canvases (byte blobs for MMQA)
    with open(DATA / "canvases.pkl", "rb") as f:
        canvases = pickle.load(f)

    img_emb = np.load(CANVAS_DIR / "clip_img_emb.npy")
    txt_emb = np.load(CANVAS_DIR / "clip_txt_emb.npy")
    q_emb = np.load(CANVAS_DIR / "clip_query_emb.npy")
    rmap = build_retrieval_map(img_emb, txt_emb, q_emb, alpha)

    vlm, proc = load_vlm()
    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())

    # Helper: table to markdown
    def table_to_md(table_doc, max_rows=10):
        header = table_doc.get("header", [])
        rows = table_doc.get("rows", table_doc.get("data", []))
        if not header and rows:
            header = [f"Col{i}" for i in range(len(rows[0]))]
        lines = ["| " + " | ".join(str(h) for h in header) + " |",
                 "| " + " | ".join("---" for _ in header) + " |"]
        for row in rows[:max_rows]:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)

    # Helper: load image by id
    IMG_DIR = DATA / "final_dataset_images"
    def load_img(doc_id):
        for ext in [".jpg", ".png", ".jpeg"]:
            p = IMG_DIR / f"{doc_id}{ext}"
            if p.exists():
                return Image.open(p).convert("RGB")
        return None

    for i in tqdm(range(len(dev_data)), desc="MMQA"):
        if str(i) in done: continue
        sample = dev_data[i]
        q = sample["question"]
        gold_answers = [str(a["answer"]) for a in sample.get("answers", [])]

        # Build context
        text_parts = []
        ctx_images = []
        for ctx in sample.get("supporting_context", []):
            doc_id = ctx["doc_id"]
            doc_part = ctx["doc_part"]
            if doc_part == "text" and doc_id in texts:
                td = texts[doc_id]
                text_parts.append(f"[Text: {td.get('title','')}]\n{td.get('text','')[:500]}")
            elif doc_part == "table" and doc_id in tables:
                td = tables[doc_id]
                md = table_to_md(td.get("table", td), max_rows=10)
                text_parts.append(f"[Table: {td.get('title','')}]\n{md}")
            elif doc_part == "image":
                img = load_img(doc_id)
                if img:
                    ctx_images.append(img)
                    text_parts.append(f"[Image: {images_meta.get(doc_id, {}).get('title', '')}]")
        ctx_text = "\n\n".join(text_parts)

        # MemCanvas prompt
        content = []
        mem_imgs = []
        retrieved = rmap.get(i, [])
        if retrieved:
            for cidx, sim in retrieved[:TOP_K]:
                cpath = CANVAS_DIR / f"{cidx:05d}.png"
                if cpath.exists():
                    cimg = Image.open(cpath).convert("RGB")
                else:
                    cimg = Image.open(io.BytesIO(canvases[cidx])).convert("RGB")
                mem_imgs.append(cimg)
                content.append({"type": "image", "image": cimg})

        for img in ctx_images:
            content.append({"type": "image", "image": img})

        prompt_parts = []
        if mem_imgs:
            prompt_parts.append("Below are memory canvases from similar questions. Study them.\n---\n")
        if ctx_text:
            prompt_parts.append(f"Context:\n{ctx_text}\n")
        prompt_parts.append(f"Question: {q}\nAnswer concisely:")
        content.append({"type": "text", "text": "\n".join(prompt_parts)})

        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        all_imgs = mem_imgs + ctx_images
        if all_imgs:
            inp = proc(text=[txt], images=all_imgs, return_tensors="pt", padding=True)
        else:
            inp = proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}

        try:
            with torch.no_grad():
                out = vlm.generate(**inp, max_new_tokens=64, do_sample=False)
            raw = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            raw = ""
            print(f"\n  Error {i}: {e}")

        best_em = max(compute_em(raw, ga) for ga in gold_answers) if gold_answers else 0.0
        best_f1 = max(compute_f1(raw, ga) for ga in gold_answers) if gold_answers else 0.0
        results[str(i)] = {"gt": gold_answers, "pred": raw, "em": best_em, "f1": best_f1}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    em = np.mean([v["em"] for v in results.values()]) * 100
    f1 = np.mean([v["f1"] for v in results.values()]) * 100
    summary = {"n": len(results), "em": em, "f1": f1,
               "config": {"alpha": alpha, "top_k": TOP_K, "method": "SmartCanvas (no Read Agent)"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  MMQA α={alpha}: EM={em:.2f}, F1={f1:.2f}")
    del vlm, proc; torch.cuda.empty_cache()
    return summary

# ============================================================
# HotpotQA
# ============================================================
def run_hotpotqa(alpha):
    print(f"\n{'='*60}\n  HotpotQA (α={alpha})\n{'='*60}")
    DATA = Path("/home/cyf/codex/hotpotqa_data")
    CANVAS_DIR = DATA / "canvases_smart"
    OUT = OUT_ROOT / f"hotpotqa_alpha{alpha:.2f}"
    OUT.mkdir(parents=True, exist_ok=True)

    with open(DATA / "hotpotqa_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dev_data = meta["dev"]

    # Use SmartCanvas embeddings
    img_emb = np.load(DATA / "canvas_embeddings_smart.npy")
    txt_emb = np.load(DATA / "canvas_text_embeddings.npy")
    q_emb = np.load(DATA / "query_embeddings.npy")
    rmap = build_retrieval_map(img_emb, txt_emb, q_emb, alpha)

    vlm, proc = load_vlm()
    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())

    def format_context(sample):
        parts = []
        for para in sample.get("paragraphs", []):
            parts.append(f"[{para['title']}]\n{para['text'][:500]}")
        return "\n\n".join(parts)

    for i in tqdm(range(len(dev_data)), desc="HotpotQA"):
        if str(i) in done: continue
        sample = dev_data[i]
        q = sample["question"]
        gt = sample["answer"]
        ctx = format_context(sample)

        content = []
        mem_imgs = []
        retrieved = rmap.get(i, [])
        if retrieved:
            for cidx, sim in retrieved[:TOP_K]:
                cpath = CANVAS_DIR / f"{cidx:05d}.png"
                if cpath.exists():
                    cimg = Image.open(cpath).convert("RGB")
                    mem_imgs.append(cimg)
                    content.append({"type": "image", "image": cimg})

        prompt_parts = []
        if mem_imgs:
            prompt_parts.append(
                "Below are memory canvases from previously solved similar questions. "
                "Each canvas shows: relevant context passages, the question, and the "
                "correct answer (marked with ✓). Study these canvases carefully."
            )
            prompt_parts.append("")
            for j in range(len(mem_imgs)):
                prompt_parts.append(f"[Memory Canvas {j+1}]")
            prompt_parts.extend(["", "---", ""])

        prompt_parts.append("Now answer the following new question using the context below.")
        prompt_parts.append("")
        prompt_parts.append(ctx)
        prompt_parts.append("")
        prompt_parts.append(f"Question: {q}")
        prompt_parts.append("Answer concisely:")
        content.append({"type": "text", "text": "\n".join(prompt_parts)})

        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if mem_imgs:
            inp = proc(text=[txt], images=mem_imgs, return_tensors="pt", padding=True)
        else:
            inp = proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}

        try:
            with torch.no_grad():
                out = vlm.generate(**inp, max_new_tokens=64, do_sample=False)
            raw = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            raw = ""
            print(f"\n  Error {i}: {e}")

        em_val = compute_em(raw, gt)
        f1_val = compute_f1(raw, gt)
        results[str(i)] = {"gt": gt, "pred": raw, "em": em_val, "f1": f1_val}
        if len(results) % 200 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    em = np.mean([v["em"] for v in results.values()]) * 100
    f1 = np.mean([v["f1"] for v in results.values()]) * 100
    summary = {"n": len(results), "em": em, "f1": f1,
               "config": {"alpha": alpha, "top_k": TOP_K, "method": "SmartCanvas (no Read Agent)"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  HotpotQA α={alpha}: EM={em:.2f}, F1={f1:.2f}")
    del vlm, proc; torch.cuda.empty_cache()
    return summary

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["scienceqa", "okvqa", "mmqa", "hotpotqa"])
    parser.add_argument("--alpha", type=float, default=0.00,
                        help="Hybrid key mixing: 0.0=text only, 1.0=image only")
    parser.add_argument("--all", action="store_true", help="Run all 4 benchmarks sequentially")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = {}

    if args.all:
        # Run all benchmarks with their best alpha
        configs = [
            ("scienceqa", 0.00),
            ("okvqa", 0.75),
            ("mmqa", 0.75),
            ("hotpotqa", 0.75),
        ]
        for bm, alpha in configs:
            print(f"\n{'#'*60}")
            print(f"  Running {bm} with α={alpha}")
            print(f"{'#'*60}")
            fn = {"scienceqa": run_scienceqa, "okvqa": run_okvqa,
                  "mmqa": run_mmqa, "hotpotqa": run_hotpotqa}[bm]
            s = fn(alpha)
            all_results[bm] = s
        # Save combined summary
        json.dump(all_results, open(OUT_ROOT / "all_results.json", "w"), indent=2)
        print(f"\n{'='*60}\n  ALL RESULTS\n{'='*60}")
        for bm, s in all_results.items():
            if "accuracy" in s:
                print(f"  {bm}: Acc={s['accuracy']:.2f}%")
            else:
                print(f"  {bm}: EM={s['em']:.2f}, F1={s['f1']:.2f}")
    elif args.benchmark:
        fn = {"scienceqa": run_scienceqa, "okvqa": run_okvqa,
              "mmqa": run_mmqa, "hotpotqa": run_hotpotqa}[args.benchmark]
        fn(args.alpha)
    else:
        parser.print_help()
