#!/usr/bin/env python3
"""
Unified SmartCanvas VLM eval for ScienceQA/OK-VQA/MMQA/InfographicVQA.
Phases: embed (CLIP) -> retrieval_map -> eval (VLM)
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
ALPHA = 0.75
TOP_K = 2

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

# ============================================================
def clip_embed(canvas_dir, n, output_prefix):
    img_file = Path(f"{output_prefix}_img_emb.npy")
    if img_file.exists():
        print(f"  CLIP img embeddings exist: {img_file}")
        return np.load(img_file)
    from transformers import CLIPProcessor, CLIPModel
    clip = CLIPModel.from_pretrained(CLIP_MODEL).cuda().eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    all_emb = []
    for i in tqdm(range(0, n, 32), desc="CLIP img"):
        imgs = [Image.open(Path(canvas_dir)/f"{j:05d}.png").convert("RGB") for j in range(i, min(i+32, n))]
        inp = proc(images=imgs, return_tensors="pt", padding=True)
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            f = clip.get_image_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
        all_emb.append(f.cpu().numpy())
    emb = np.concatenate(all_emb)
    np.save(img_file, emb)
    del clip, proc; torch.cuda.empty_cache()
    return emb

def clip_text_embed(texts, output_file):
    if Path(output_file).exists():
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
# ScienceQA
# ============================================================
def run_scienceqa():
    print("\n=== ScienceQA SmartCanvas Eval ===")
    CANVAS_DIR = Path("/home/cyf/codex/scienceqa_smart_canvases")
    OUT = Path("/home/cyf/codex/scienceqa_smart_eval")
    OUT.mkdir(exist_ok=True)

    with open("/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl", "rb") as f:
        cache = pickle.load(f)
    train = cache["train"] if isinstance(cache, dict) else cache[0]
    from datasets import load_dataset
    test_ds = load_dataset("derek-thomas/ScienceQA", split="test")

    n = int((CANVAS_DIR / "done.txt").read_text().strip())

    # Embeddings
    img_emb = clip_embed(CANVAS_DIR, n, str(CANVAS_DIR / "clip"))

    train_texts = [f"{s.get('question','')} {s.get('hint','')}" for s in train]
    txt_emb = clip_text_embed(train_texts, str(CANVAS_DIR / "clip_txt_emb.npy"))

    test_texts = [f"{item['question']} {item.get('hint','')}" for item in test_ds]
    q_emb = clip_text_embed(test_texts, str(CANVAS_DIR / "clip_query_emb.npy"))

    rmap = build_retrieval_map(img_emb, txt_emb, q_emb)

    # VLM eval
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)

    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())

    for i in tqdm(range(len(test_ds)), desc="ScienceQA eval"):
        if str(i) in done: continue
        item = test_ds[i]
        q = item["question"]
        choices = item["choices"]
        gt = chr(65 + item["answer"])
        hint = item.get("hint", "") or ""
        choice_txt = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))

        # MemCanvas
        retrieved = rmap.get(i, [])
        content = []
        canvas_imgs = []
        for cidx, sim in retrieved[:TOP_K]:
            img = Image.open(CANVAS_DIR / f"{cidx:05d}.png").convert("RGB")
            canvas_imgs.append(img)
            content.append({"type": "image", "image": img})

        prompt = (
            "Study the reference canvases above. Each shows a solved example.\n"
            f"{hint}\n\nQuestion: {q}\n{choice_txt}\n"
            "Think step by step, then answer with just the letter:"
        )
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

        results[str(i)] = {"gt": gt, "pred": pred, "correct": float(pred == gt), "subject": item.get("subject","")}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    acc = np.mean([v["correct"] for v in results.values()]) * 100
    summary = {"n": len(results), "accuracy": acc, "config": {"alpha": ALPHA, "top_k": TOP_K}}

    # Per-subject
    for subj in ["natural science", "social science", "language science"]:
        vals = [v["correct"] for v in results.values() if v.get("subject","") == subj]
        if vals: summary[subj] = np.mean(vals) * 100

    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"  ScienceQA: {acc:.2f}%")
    del vlm, proc; torch.cuda.empty_cache()


# ============================================================
# OK-VQA
# ============================================================
def run_okvqa():
    print("\n=== OK-VQA SmartCanvas Eval ===")
    DATA = Path("/home/cyf/codex/okvqa_data")
    CANVAS_DIR = DATA / "canvases_smart"
    OUT = Path("/home/cyf/codex/okvqa_smart_eval")
    OUT.mkdir(exist_ok=True)

    with open(DATA / "okvqa_cached.pkl", "rb") as f:
        data = pickle.load(f)
    train, test = data["train"], data["test"]
    n = int((CANVAS_DIR / "done.txt").read_text().strip())

    img_emb = clip_embed(CANVAS_DIR, n, str(CANVAS_DIR / "clip"))
    train_texts = [f"{s['question']} {' '.join(s.get('answers',[][:3]))}" for s in train]
    txt_emb = clip_text_embed(train_texts, str(CANVAS_DIR / "clip_txt_emb.npy"))
    test_texts = [s["question"] for s in test]
    q_emb = clip_text_embed(test_texts, str(CANVAS_DIR / "clip_query_emb.npy"))
    rmap = build_retrieval_map(img_emb, txt_emb, q_emb)

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)

    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}
    done = set(results.keys())

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

        # VQA accuracy: check if prediction matches any answer
        correct = float(any(normalize_answer(pred) == normalize_answer(a) for a in answers))
        results[str(i)] = {"gt": answers, "pred": pred, "correct": correct}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    acc = np.mean([v["correct"] for v in results.values()]) * 100
    json.dump({"n": len(results), "accuracy": acc}, open(OUT / "summary.json", "w"), indent=2)
    print(f"  OK-VQA: {acc:.2f}%")
    del vlm, proc; torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["scienceqa", "okvqa"])
    args = parser.parse_args()
    if args.benchmark == "scienceqa": run_scienceqa()
    elif args.benchmark == "okvqa": run_okvqa()
