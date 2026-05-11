#!/usr/bin/env python3
"""
SmartCanvas VLM evaluation for OK-VQA and MMQA.
Computes SmartCanvas CLIP embeddings, builds retrieval maps, evaluates with Qwen2.5-VL-7B.
Outputs to /home/cyf/memcanvas0402/{okvqa,mmqa}_smart_eval/.

Usage:
    python eval_okvqa_mmqa.py --benchmark okvqa
    python eval_okvqa_mmqa.py --benchmark mmqa
    python eval_okvqa_mmqa.py --benchmark all
"""
import argparse, io, json, os, pickle, re, string, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

CLIP_MODEL = "openai/clip-vit-large-patch14"
VLM_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
ALPHA = 0.00       # Pure text retrieval (optimal from ScienceQA ablation)
TOP_K = 2
OUTPUT_ROOT = Path("/home/cyf/memcanvas0402")

# ============================================================
# Utilities
# ============================================================
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

# ============================================================
# CLIP embedding
# ============================================================
def clip_embed_images(canvas_dir, n, output_file):
    if Path(output_file).exists():
        print(f"  CLIP img embeddings cached: {output_file}")
        return np.load(output_file)
    from transformers import CLIPProcessor, CLIPModel
    clip = CLIPModel.from_pretrained(CLIP_MODEL).cuda().eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    all_emb = []
    for i in tqdm(range(0, n, 32), desc="CLIP img"):
        imgs = [Image.open(Path(canvas_dir)/f"{j:05d}.png").convert("RGB")
                for j in range(i, min(i+32, n))]
        inp = proc(images=imgs, return_tensors="pt", padding=True)
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            f = clip.get_image_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
        all_emb.append(f.cpu().numpy())
    emb = np.concatenate(all_emb)
    np.save(output_file, emb)
    print(f"  Saved CLIP img embeddings: {emb.shape}")
    del clip, proc; torch.cuda.empty_cache()
    return emb

def clip_embed_texts(texts, output_file):
    if Path(output_file).exists():
        print(f"  CLIP txt embeddings cached: {output_file}")
        return np.load(output_file)
    from transformers import CLIPProcessor, CLIPModel
    clip = CLIPModel.from_pretrained(CLIP_MODEL).cuda().eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    all_emb = []
    for i in tqdm(range(0, len(texts), 64), desc="CLIP txt"):
        batch = texts[i:i+64]
        inp = proc(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            f = clip.get_text_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
        all_emb.append(f.cpu().numpy())
    emb = np.concatenate(all_emb)
    np.save(output_file, emb)
    print(f"  Saved CLIP txt embeddings: {emb.shape}")
    del clip, proc; torch.cuda.empty_cache()
    return emb

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

# ============================================================
# OK-VQA
# ============================================================
def run_okvqa():
    print("\n=== OK-VQA SmartCanvas Eval ===")
    DATA = Path("/home/cyf/codex/okvqa_data")
    CANVAS_DIR = DATA / "canvases_smart"
    EMB_DIR = CANVAS_DIR  # Store SmartCanvas embeddings alongside canvases
    OUT = OUTPUT_ROOT / "okvqa_smart_eval"
    OUT.mkdir(parents=True, exist_ok=True)

    with open(DATA / "okvqa_cached.pkl", "rb") as f:
        data = pickle.load(f)
    train, test = data["train"], data["test"]
    n = int((CANVAS_DIR / "done.txt").read_text().strip())
    print(f"  Train: {len(train)}, Test: {len(test)}, Canvases: {n}")

    # SmartCanvas CLIP embeddings
    img_emb = clip_embed_images(CANVAS_DIR, n, str(EMB_DIR / "clip_img_emb.npy"))
    train_texts = [f"{s['question']} {' '.join(s.get('answers',[])[:3])}" for s in train]
    txt_emb = clip_embed_texts(train_texts, str(EMB_DIR / "clip_txt_emb.npy"))
    test_texts = [s["question"] for s in test]
    q_emb = clip_embed_texts(test_texts, str(EMB_DIR / "clip_query_emb.npy"))
    rmap = build_retrieval_map(img_emb, txt_emb, q_emb)

    # VLM eval
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)

    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())
    print(f"  Resuming from {len(done)} completed samples")

    for i in tqdm(range(len(test)), desc="OK-VQA eval"):
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

        # Test image
        img_path = s.get("image_path", "")
        test_img = None
        if img_path and os.path.exists(img_path):
            test_img = Image.open(img_path).convert("RGB")
            content.append({"type": "image", "image": test_img})

        prompt = "Study the reference canvases. Answer the question about the last image.\n"
        prompt += f"Question: {q}\nAnswer concisely:"
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

        # VQA accuracy: match any ground truth answer
        correct = float(any(normalize_answer(pred) == normalize_answer(a) for a in answers))
        results[str(i)] = {"gt": answers, "pred": pred, "correct": correct}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    acc = np.mean([v["correct"] for v in results.values()]) * 100
    summary = {"n": len(results), "accuracy": acc,
               "config": {"alpha": ALPHA, "top_k": TOP_K, "model": "Qwen2.5-VL-7B"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  OK-VQA SmartCanvas: {acc:.2f}% ({len(results)} samples)")
    del vlm, proc; torch.cuda.empty_cache()
    return summary

# ============================================================
# MMQA
# ============================================================
def _resolve_mmqa_context(item, tables, texts, max_ctx=3):
    """Resolve supporting_context doc_ids to text for the VLM prompt."""
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

def run_mmqa():
    print("\n=== MMQA SmartCanvas Eval ===")
    DATA = Path("/home/cyf/codex/mmqa_data")
    CANVAS_DIR = DATA / "canvases_smart"
    EMB_DIR = CANVAS_DIR
    OUT = OUTPUT_ROOT / "mmqa_smart_eval"
    OUT.mkdir(parents=True, exist_ok=True)

    with open(DATA / "mmqa_parsed.pkl", "rb") as f:
        mmqa = pickle.load(f)
    train_list = mmqa["train"]
    dev = mmqa["dev"]
    tables = mmqa["tables"]
    texts = mmqa["texts"]
    n = int((CANVAS_DIR / "done.txt").read_text().strip())
    print(f"  Train: {len(train_list)}, Dev: {len(dev)}, Canvases: {n}")

    # SmartCanvas CLIP embeddings
    img_emb = clip_embed_images(CANVAS_DIR, n, str(EMB_DIR / "clip_img_emb.npy"))
    train_texts_list = [s["question"] for s in train_list]
    txt_emb = clip_embed_texts(train_texts_list, str(EMB_DIR / "clip_txt_emb.npy"))
    dev_texts = [s["question"] for s in dev]
    q_emb = clip_embed_texts(dev_texts, str(EMB_DIR / "clip_query_emb.npy"))
    rmap = build_retrieval_map(img_emb, txt_emb, q_emb)

    # VLM eval
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)

    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())
    print(f"  Resuming from {len(done)} completed samples")

    for i in tqdm(range(len(dev)), desc="MMQA eval"):
        if str(i) in done: continue
        item = dev[i]
        q = item["question"]
        gt_answers = item["answers"]
        gt = gt_answers[0]["answer"] if gt_answers and isinstance(gt_answers[0], dict) else (gt_answers[0] if gt_answers else "")

        # Resolve supporting context to text
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
        "config": {"alpha": ALPHA, "top_k": TOP_K, "model": "Qwen2.5-VL-7B"}
    }

    # Per question type
    for qt in set(v.get("type", "unknown") for v in results.values()):
        vals = [v for v in results.values() if v.get("type") == qt]
        if vals:
            summary[f"type_{qt}"] = {
                "n": len(vals),
                "em": float(np.mean([v["em"] for v in vals]) * 100),
                "f1": float(np.mean([v["f1"] for v in vals]) * 100),
            }

    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  MMQA SmartCanvas: EM={summary['em']:.2f}%, F1={summary['f1']:.2f}% ({len(results)} samples)")
    del vlm, proc; torch.cuda.empty_cache()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["okvqa", "mmqa", "all"])
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    ALPHA = args.alpha
    TOP_K = args.top_k

    if args.benchmark in ("okvqa", "all"):
        run_okvqa()
    if args.benchmark in ("mmqa", "all"):
        run_mmqa()
