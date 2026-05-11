#!/usr/bin/env python3
"""
Evaluate MemCanvas with canvas0415 (single-column, large font, large image).

Usage:
  CUDA_VISIBLE_DEVICES=1 python -u eval_canvas0415.py --benchmark scienceqa --alpha 0.00
"""
import argparse, json, os, pickle, re, string, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

VLM_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
TOP_K = 2
SIM_THRESHOLD = 0.1
CANVAS_BASE = Path("/home/cyf/codex/canvas0415")
OUT_ROOT = Path("/home/cyf/codex/canvas0415_eval")


def normalize_answer(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())

def compute_f1(p, g):
    pt, gt = normalize_answer(p).split(), normalize_answer(g).split()
    c = Counter(pt) & Counter(gt); n = sum(c.values())
    if n == 0: return 0.0
    pr = n / len(pt) if pt else 0; rc = n / len(gt) if gt else 0
    return 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0

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
    print(f"  Retrieval map: {has}/{len(query_emb)} have memories (alpha={alpha}, K={top_k})")
    return rmap

def load_vlm():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    print("Loading VLM...")
    quant = os.environ.get("VLM_QUANT", "")
    if quant == "4bit":
        print("  Using 4-bit quantization")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM_MODEL, quantization_config=bnb_config, device_map="auto")
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)
    return model, proc


def run_scienceqa(alpha):
    print(f"\n{'='*60}\n  ScienceQA Canvas0415 (alpha={alpha})\n{'='*60}")
    CANVAS_DIR = CANVAS_BASE / "scienceqa"
    OUT = OUT_ROOT / f"scienceqa_alpha{alpha:.2f}"
    OUT.mkdir(parents=True, exist_ok=True)

    with open("/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl", "rb") as f:
        cache = pickle.load(f)
    train = cache["train"] if isinstance(cache, dict) else cache[0]
    from datasets import load_dataset
    test_ds = load_dataset("derek-thomas/ScienceQA", split="test")

    img_emb = np.load(CANVAS_DIR / "clip_img_emb.npy")
    txt_emb = np.load(CANVAS_DIR / "clip_txt_emb.npy")
    q_emb = np.load(CANVAS_DIR / "clip_query_emb.npy")
    rmap = build_retrieval_map(img_emb, txt_emb, q_emb, alpha)

    vlm, proc = load_vlm()
    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())

    for i in tqdm(range(len(test_ds)), desc="ScienceQA Canvas0415"):
        if str(i) in done:
            continue
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
            if c in "ABCDEF":
                pred = c
                break

        results[str(i)] = {"gt": gt, "pred": pred, "correct": float(pred == gt),
                           "subject": item.get("subject", "")}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    acc = np.mean([v["correct"] for v in results.values()]) * 100
    summary = {"n": len(results), "accuracy": acc,
               "config": {"alpha": alpha, "top_k": TOP_K, "method": "Canvas0415 (single-col, large font)"}}
    for subj in ["natural science", "social science", "language science"]:
        vals = [v["correct"] for v in results.values() if v.get("subject", "") == subj]
        if vals:
            summary[subj] = np.mean(vals) * 100
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  ScienceQA Canvas0415 alpha={alpha}: {acc:.2f}%")
    del vlm, proc
    torch.cuda.empty_cache()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["scienceqa", "okvqa", "mmqa", "hotpotqa", "all"], default="scienceqa")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--canvas_dir", type=str, default=None, help="Override canvas directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    if args.canvas_dir:
        _cd = Path(args.canvas_dir)
        if (_cd / "clip_img_emb.npy").exists():
            # Direct canvas dir (e.g. canvas0415_compressed/scienceqa)
            _cd = _cd.parent
        CANVAS_BASE = _cd
    if args.output_dir:
        OUT_ROOT = Path(args.output_dir)

    DEFAULTS = {"scienceqa": 0.00, "okvqa": 0.75, "mmqa": 0.50, "hotpotqa": 0.75}

    if args.benchmark == "all":
        for bm in DEFAULTS:
            if bm == "scienceqa":
                run_scienceqa(DEFAULTS[bm])
            else:
                print(f"  {bm}: not yet implemented for canvas0415")
    elif args.benchmark == "scienceqa":
        alpha = args.alpha if args.alpha is not None else DEFAULTS["scienceqa"]
        run_scienceqa(alpha)
    else:
        print(f"Canvas0415 eval for {args.benchmark} not yet implemented")
