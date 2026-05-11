#!/usr/bin/env python3
"""
LoCoMo-10 QA — MemCanvas v2.
Metrics: EM, F1, BLEU-1 (aligned with Memory-R1/MAGMA).
Pipeline: SmartCanvasLayout canvases + hybrid CLIP retrieval (alpha=0.75) + Qwen2.5-VL-7B.
"""
import io, json, os, re, string, time, base64
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

import sys
sys.path.insert(0, "/home/cyf/codex")
from smart_canvas_layout import measure_text, choose_best_layout, render_layout

DATA_PATH = "/home/cyf/codex/datasets/locomo/locomo10.json"
VLM_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
OUTPUT_DIR = Path("/home/cyf/codex/locomo_experiment_v2")
ALPHA = 0.75
TOP_K = 6

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def normalize_answer(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())

def compute_em(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))

def compute_f1(pred, gold):
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0: return 0.0
    prec = n / len(p) if p else 0.0
    rec = n / len(g) if g else 0.0
    return (2*prec*rec) / (prec+rec) if (prec+rec) > 0 else 0.0

def compute_bleu1(pred, gold):
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g: return 0.0
    g_cnt = Counter(g)
    clipped = sum(min(Counter(p)[w], g_cnt[w]) for w in Counter(p))
    return clipped / len(p)

# ---------------------------------------------------------------------------
# Canvas rendering
# ---------------------------------------------------------------------------
def render_turn_canvas(sample_id, msg):
    speaker = msg.get("speaker", "")
    text = msg.get("text", "")
    dia_id = msg.get("dia_id", "")
    blocks = [
        measure_text(f"[{sample_id}] {speaker} | {dia_id}", font_size=12, ref_width=500),
        measure_text(text, font_size=15, ref_width=500),
    ]
    layout = choose_best_layout(blocks)
    img = render_layout(layout)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def parse_conversation(item):
    conv = item.get("conversation", {})
    keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
        key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0
    )
    turns = []
    for sk in keys:
        for msg in conv.get(sk, []):
            turns.append(msg)
    return turns

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas_dir = OUTPUT_DIR / "canvases"
    canvas_dir.mkdir(exist_ok=True)

    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} conversations")

    # --- Phase 1: Build canvases + ALL embeddings (canvas img, canvas txt, query) ---
    print("Loading CLIP...")
    from transformers import CLIPProcessor, CLIPModel
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).cuda().eval()

    # Build per-conversation memory
    all_memory = {}
    total_turns = 0
    for item in tqdm(data, desc="Building canvases + embeddings"):
        sid = item["sample_id"]
        turns = parse_conversation(item)
        img_embs, txt_embs, cfiles = [], [], []

        for i, msg in enumerate(turns):
            cfile = canvas_dir / f"{sid}_{i:04d}.png"
            if not cfile.exists():
                cfile.write_bytes(render_turn_canvas(sid, msg))

            # Image embedding
            img = Image.open(cfile).convert("RGB")
            inp = clip_proc(images=img, return_tensors="pt")
            inp = {k: v.cuda() for k, v in inp.items()}
            with torch.no_grad():
                ie = clip_model.get_image_features(**inp)
                ie = (ie / ie.norm(dim=-1, keepdim=True)).cpu().numpy().squeeze()
            img_embs.append(ie)

            # Text embedding (turn text)
            text = msg.get("text", "")[:300]
            inp_t = clip_proc(text=text, return_tensors="pt", truncation=True, max_length=77)
            inp_t = {k: v.cuda() for k, v in inp_t.items()}
            with torch.no_grad():
                te = clip_model.get_text_features(**inp_t)
                te = (te / te.norm(dim=-1, keepdim=True)).cpu().numpy().squeeze()
            txt_embs.append(te)
            cfiles.append(str(cfile))

        all_memory[sid] = {
            "turns": turns,
            "img_emb": np.stack(img_embs),
            "txt_emb": np.stack(txt_embs),
            "canvas_files": cfiles,
        }
        total_turns += len(turns)

    print(f"  Total turns: {total_turns}")

    # Pre-compute ALL query embeddings before freeing CLIP
    print("Pre-computing query embeddings...")
    query_list = []
    for item in data:
        sid = item["sample_id"]
        for qi, qa in enumerate(item.get("qa", [])):
            query_list.append((sid, qi, qa["question"]))

    query_embs = {}
    for sid, qi, question in tqdm(query_list, desc="Query embeddings"):
        inp_q = clip_proc(text=question, return_tensors="pt", truncation=True, max_length=77)
        inp_q = {k: v.cuda() for k, v in inp_q.items()}
        with torch.no_grad():
            qe = clip_model.get_text_features(**inp_q)
            qe = (qe / qe.norm(dim=-1, keepdim=True)).cpu().numpy().squeeze()
        query_embs[f"{sid}_{qi}"] = qe

    print(f"  {len(query_embs)} queries encoded")

    # Free CLIP
    del clip_model, clip_proc
    torch.cuda.empty_cache()

    # --- Phase 2: VLM evaluation ---
    print("Loading VLM...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    vlm_proc = AutoProcessor.from_pretrained(VLM_MODEL)
    print("  VLM loaded")

    ckpt_file = OUTPUT_DIR / "checkpoint.json"
    results = {}
    if ckpt_file.exists():
        results = json.load(open(ckpt_file))
        print(f"  Resumed: {len(results)} done")

    done_keys = set(results.keys())

    for item in tqdm(data, desc="Evaluating"):
        sid = item["sample_id"]
        mem = all_memory[sid]

        # Build hybrid keys for this conversation
        hybrid_keys = ALPHA * mem["img_emb"] + (1 - ALPHA) * mem["txt_emb"]
        hybrid_keys = hybrid_keys / np.linalg.norm(hybrid_keys, axis=1, keepdims=True).clip(1e-8)

        for qi, qa in enumerate(item.get("qa", [])):
            key = f"{sid}_{qi}"
            if key in done_keys:
                continue

            question = qa["question"]
            answer = str(qa.get("answer", qa.get("adversarial_answer", "")))

            # Retrieve top-K via hybrid key
            q_emb = query_embs[key]
            sims = hybrid_keys @ q_emb
            top_idx = np.argsort(-sims)[:TOP_K]

            # --- Baseline (no memory) ---
            content_b = [{"type": "text", "text": f"Answer briefly.\nQuestion: {question}"}]
            msgs_b = [{"role": "user", "content": content_b}]
            txt_b = vlm_proc.apply_chat_template(msgs_b, tokenize=False, add_generation_prompt=True)
            inp_b = vlm_proc(text=[txt_b], return_tensors="pt", padding=True)
            inp_b = {k: v.to(vlm.device) for k, v in inp_b.items()}
            with torch.no_grad():
                out_b = vlm.generate(**inp_b, max_new_tokens=32, do_sample=False)
            pred_base = vlm_proc.decode(out_b[0][inp_b["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # --- MemCanvas (top-K canvases) ---
            memory_images = []
            for idx in top_idx:
                img = Image.open(mem["canvas_files"][idx]).convert("RGB")
                memory_images.append(img)

            prompt = (
                "Below are memory canvases from past conversations.\n"
                "Each canvas shows a dialogue turn with speaker identity and message.\n"
                "Use these memories to answer the question.\n"
                "---\n"
                f"Question: {question}\n"
                "Answer briefly and directly:"
            )
            content_m = [{"type": "image", "image": img} for img in memory_images]
            content_m.append({"type": "text", "text": prompt})
            msgs_m = [{"role": "user", "content": content_m}]
            txt_m = vlm_proc.apply_chat_template(msgs_m, tokenize=False, add_generation_prompt=True)
            inp_m = vlm_proc(text=[txt_m], images=memory_images, return_tensors="pt", padding=True)
            inp_m = {k: v.to(vlm.device) for k, v in inp_m.items()}
            with torch.no_grad():
                out_m = vlm.generate(**inp_m, max_new_tokens=32, do_sample=False)
            pred_mem = vlm_proc.decode(out_m[0][inp_m["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            results[key] = {
                "question": question, "answer": answer,
                "pred_base": pred_base, "pred_mem": pred_mem,
                "em_base": compute_em(pred_base, answer),
                "em_mem": compute_em(pred_mem, answer),
                "f1_base": compute_f1(pred_base, answer),
                "f1_mem": compute_f1(pred_mem, answer),
                "b1_base": compute_bleu1(pred_base, answer),
                "b1_mem": compute_bleu1(pred_mem, answer),
            }

            if len(results) % 50 == 0:
                json.dump(results, open(ckpt_file, "w"))

    json.dump(results, open(ckpt_file, "w"))

    # Summary
    n = len(results)
    summary = {
        "n": n,
        "config": {"alpha": ALPHA, "top_k": TOP_K, "vlm": "Qwen2.5-VL-7B"},
        "baseline": {
            "f1": np.mean([v["f1_base"] for v in results.values()]) * 100,
            "bleu1": np.mean([v["b1_base"] for v in results.values()]) * 100,
            "em": np.mean([v["em_base"] for v in results.values()]) * 100,
        },
        "memcanvas": {
            "f1": np.mean([v["f1_mem"] for v in results.values()]) * 100,
            "bleu1": np.mean([v["b1_mem"] for v in results.values()]) * 100,
            "em": np.mean([v["em_mem"] for v in results.values()]) * 100,
        },
    }
    json.dump(summary, open(OUTPUT_DIR / "summary.json", "w"), indent=2)

    print(f"\n{'='*60}")
    print(f"LoCoMo Results ({n} questions)")
    print(f"Config: alpha={ALPHA}, K={TOP_K}")
    print(f"{'='*60}")
    for cond in ["baseline", "memcanvas"]:
        s = summary[cond]
        print(f"  {cond:12s}: F1={s['f1']:.2f}% B1={s['bleu1']:.2f}% EM={s['em']:.2f}%")


if __name__ == "__main__":
    main()
