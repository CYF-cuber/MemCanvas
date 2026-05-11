#!/usr/bin/env python3
"""
MMQA SmartCanvas evaluation with different alpha values.
Reuses cached CLIP embeddings, only rebuilds retrieval map per alpha.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u run_mmqa_alpha.py --alphas 0.50 0.75
"""
import argparse, io, json, os, pickle, re, string, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

sys.path.insert(0, "/home/cyf/codex")

CLIP_MODEL = "openai/clip-vit-large-patch14"
VLM_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
TOP_K = 2
OUTPUT_ROOT = Path("/home/cyf/memcanvas0402")


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


def build_retrieval_map(img_emb, txt_emb, query_emb, alpha, top_k=TOP_K):
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


def _resolve_mmqa_context(item, tables, texts, max_ctx=3):
    parts = []
    for ctx in item.get("supporting_context", [])[:max_ctx]:
        doc_id = ctx["doc_id"]
        doc_part = ctx["doc_part"]
        if doc_part == "text" and doc_id in texts:
            doc = texts[doc_id]
            title = doc.get("title", "")
            passage = doc.get("text", "")[:400]
            parts.append(f"[Text: {title}]\n{passage}")
        elif doc_part == "table" and doc_id in tables:
            doc = tables[doc_id]
            title = doc.get("title", "")
            tbl = doc.get("table", {})
            headers = [h["column_name"] for h in tbl.get("header", [])]
            rows = [[c["text"][:30] for c in row] for row in tbl.get("table_rows", [])[:6]]
            lines = [" | ".join(headers)]
            for row in rows:
                lines.append(" | ".join(row))
            parts.append(f"[Table: {title}]\n" + "\n".join(lines))
        elif doc_part == "image":
            title = texts.get(doc_id, {}).get("title", doc_id[:8])
            parts.append(f"[Image: {title}]")
    return "\n\n".join(parts)


def run_mmqa_alpha(alpha, vlm, proc):
    print(f"\n{'='*50}")
    print(f"  MMQA α={alpha:.2f}")
    print(f"{'='*50}")

    DATA = Path("/home/cyf/codex/mmqa_data")
    CANVAS_DIR = DATA / "canvases_smart"
    EMB_DIR = CANVAS_DIR
    OUT = OUTPUT_ROOT / f"mmqa_smart_eval_alpha{alpha:.2f}"
    OUT.mkdir(parents=True, exist_ok=True)

    with open(DATA / "mmqa_parsed.pkl", "rb") as f:
        mmqa = pickle.load(f)
    dev = mmqa["dev"]
    tables = mmqa["tables"]
    texts = mmqa["texts"]
    n = int((CANVAS_DIR / "done.txt").read_text().strip())

    # Load cached CLIP embeddings
    img_emb = np.load(str(EMB_DIR / "clip_img_emb.npy"))
    txt_emb = np.load(str(EMB_DIR / "clip_txt_emb.npy"))
    q_emb = np.load(str(EMB_DIR / "clip_query_emb.npy"))
    rmap = build_retrieval_map(img_emb, txt_emb, q_emb, alpha)
    print(f"  Retrieval map built: {len(rmap)} queries, {n} canvases")

    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())
    print(f"  Resuming from {len(done)} completed samples")

    if len(done) >= len(dev):
        print(f"  Already complete!")
        return

    for i in tqdm(range(len(dev)), desc=f"MMQA α={alpha:.2f}"):
        if str(i) in done: continue
        item = dev[i]
        q = item["question"]
        gt_answers = item["answers"]
        gt = gt_answers[0]["answer"] if gt_answers and isinstance(gt_answers[0], dict) else (gt_answers[0] if gt_answers else "")

        ctx = _resolve_mmqa_context(item, tables, texts)
        content = []
        canvas_imgs = []
        for cidx, sim in rmap.get(i, [])[:TOP_K]:
            img = Image.open(CANVAS_DIR / f"{cidx:05d}.png").convert("RGB")
            canvas_imgs.append(img)
            content.append({"type": "image", "image": img})

        prompt = "Above are memory canvases from similar solved questions.\n\n"
        if ctx:
            prompt += f"Context:\n{ctx}\n\n"
        prompt += f"Question: {q}\nAnswer with ONLY the answer:"
        content.append({"type": "text", "text": prompt})

        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if canvas_imgs:
            inp = proc(text=[txt], images=canvas_imgs, return_tensors="pt", padding=True)
        else:
            inp = proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=64, do_sample=False)
        pred = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        em = compute_em(pred, gt)
        f1 = compute_f1(pred, gt)
        qtype = item.get("metadata", {}).get("type", "unknown")
        modalities = item.get("metadata", {}).get("modalities", [])

        results[str(i)] = {
            "gt": gt, "pred": pred, "em": em, "f1": f1,
            "type": qtype, "modalities": modalities
        }
        if len(results) % 50 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))

    # Summary
    ems = [v["em"] for v in results.values()]
    f1s = [v["f1"] for v in results.values()]
    summary = {
        "n": len(results),
        "em": float(np.mean(ems) * 100),
        "f1": float(np.mean(f1s) * 100),
        "config": {"alpha": alpha, "top_k": TOP_K, "model": "Qwen2.5-VL-7B"}
    }
    for qt in set(v.get("type", "unknown") for v in results.values()):
        vals = [v for v in results.values() if v.get("type") == qt]
        if vals:
            summary[f"type_{qt}"] = {
                "n": len(vals),
                "em": float(np.mean([v["em"] for v in vals]) * 100),
                "f1": float(np.mean([v["f1"] for v in vals]) * 100),
            }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  MMQA α={alpha:.2f}: EM={summary['em']:.2f}%, F1={summary['f1']:.2f}%")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", nargs="+", type=float, required=True,
                        help="Alpha values to test, e.g. 0.50 0.75")
    args = parser.parse_args()

    print(f"MMQA α-tuning: {args.alphas}")

    # Load VLM once, reuse across alphas
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)

    for alpha in args.alphas:
        run_mmqa_alpha(alpha, vlm, proc)

    del vlm, proc
    torch.cuda.empty_cache()
    print("\nAll alpha conditions complete!")


if __name__ == "__main__":
    main()
