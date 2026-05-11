#!/usr/bin/env python3
"""
LoCoMo Session-level Canvas Builder — renders each conversation session
as a single rich canvas (multiple turns per canvas) instead of one turn per canvas.

Output: /home/cyf/codex/locomo_experiment_v3/canvases/
Then runs CLIP embedding + retrieval + VLM eval.
"""
import io, json, os, re, string, time
from collections import Counter
from pathlib import Path
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
OUTPUT_DIR = Path("/home/cyf/codex/locomo_experiment_v3")
ALPHA = 0.75
TOP_K = 6

# Metrics
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

def compute_bleu1(p, g):
    pt, gt = normalize_answer(p).split(), normalize_answer(g).split()
    if not pt or not gt: return 0.0
    gc = Counter(gt)
    return sum(min(Counter(pt)[w], gc[w]) for w in Counter(pt)) / len(pt)

def render_session_canvas(sample_id, session_name, messages, session_date=""):
    """Render a full session (multiple turns) into one rich canvas."""
    blocks = []

    # Header: conversation ID + session info
    header = f"[{sample_id}] {session_name}"
    if session_date:
        header += f" | {session_date}"
    blocks.append(measure_text(header, font_size=14, ref_width=600))
    blocks.append(measure_text("─" * 60, font_size=10, ref_width=600))

    # Add each message as a block
    for msg in messages:
        speaker = msg.get("speaker", "Unknown")
        text = msg.get("text", "")
        # Truncate very long messages
        if len(text) > 300:
            text = text[:297] + "..."
        line = f"[{speaker}] {text}"
        blocks.append(measure_text(line, font_size=13, ref_width=600))

    layout = choose_best_layout(blocks)
    img = render_layout(layout)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas_dir = OUTPUT_DIR / "canvases"
    canvas_dir.mkdir(exist_ok=True)

    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} conversations")

    # --- Phase 1: Build session-level canvases + embeddings ---
    print("Loading CLIP...")
    from transformers import CLIPProcessor, CLIPModel
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).cuda().eval()

    all_memory = {}
    total_canvases = 0

    for item in tqdm(data, desc="Building session canvases"):
        sid = item["sample_id"]
        conv = item.get("conversation", {})
        sessions = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
            key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0
        )

        img_embs, txt_embs, cfiles = [], [], []

        for si, session_key in enumerate(sessions):
            messages = conv[session_key]
            date_key = f"{session_key}_date_time"
            session_date = conv.get(date_key, "")

            cfile = canvas_dir / f"{sid}_s{si:03d}.png"
            if not cfile.exists():
                canvas_bytes = render_session_canvas(sid, session_key, messages, session_date)
                cfile.write_bytes(canvas_bytes)

            # Image embedding
            img = Image.open(cfile).convert("RGB")
            inp = clip_proc(images=img, return_tensors="pt")
            inp = {k: v.cuda() for k, v in inp.items()}
            with torch.no_grad():
                ie = clip_model.get_image_features(**inp)
                ie = (ie / ie.norm(dim=-1, keepdim=True)).cpu().numpy().squeeze()
            img_embs.append(ie)

            # Text embedding (concatenate all messages in session)
            full_text = " ".join(m.get("text", "")[:100] for m in messages)[:500]
            inp_t = clip_proc(text=full_text, return_tensors="pt", truncation=True, max_length=77)
            inp_t = {k: v.cuda() for k, v in inp_t.items()}
            with torch.no_grad():
                te = clip_model.get_text_features(**inp_t)
                te = (te / te.norm(dim=-1, keepdim=True)).cpu().numpy().squeeze()
            txt_embs.append(te)
            cfiles.append(str(cfile))
            total_canvases += 1

        all_memory[sid] = {
            "img_emb": np.stack(img_embs),
            "txt_emb": np.stack(txt_embs),
            "canvas_files": cfiles,
        }

    print(f"  Total session canvases: {total_canvases}")

    # --- Phase 2: Pre-compute query embeddings ---
    print("Computing query embeddings...")
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
        query_embs[(sid, qi)] = qe

    del clip_model, clip_proc
    torch.cuda.empty_cache()

    # --- Phase 3: VLM evaluation ---
    print("Loading VLM...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    proc = AutoProcessor.from_pretrained(VLM_MODEL)

    ckpt_file = OUTPUT_DIR / "checkpoint.json"
    results = json.load(open(ckpt_file)) if ckpt_file.exists() else {}
    done = set(results.keys())

    for item in tqdm(data, desc="Evaluating"):
        sid = item["sample_id"]
        mem = all_memory[sid]
        img_emb = mem["img_emb"]
        txt_emb = mem["txt_emb"]
        canvas_files = mem["canvas_files"]

        for qi, qa in enumerate(item.get("qa", [])):
            key = f"{sid}_{qi}"
            if key in done:
                continue

            question = qa["question"]
            answer = qa.get("answer", qa.get("adversarial_answer", ""))
            if not answer:
                continue
            q_emb = query_embs[(sid, qi)]

            # Hybrid retrieval
            keys = ALPHA * img_emb + (1 - ALPHA) * txt_emb
            keys = keys / np.linalg.norm(keys, axis=1, keepdims=True).clip(1e-8)
            qn = q_emb / np.linalg.norm(q_emb).clip(1e-8)
            sims = keys @ qn
            top_idx = np.argsort(sims)[::-1][:TOP_K]
            retrieved = [(int(j), float(sims[j])) for j in top_idx if sims[j] >= 0.1]

            # Baseline (no memory)
            msgs_base = [{"role": "user", "content": [
                {"type": "text", "text": f"Question: {question}\nAnswer concisely:"}
            ]}]
            txt_b = proc.apply_chat_template(msgs_base, tokenize=False, add_generation_prompt=True)
            inp_b = proc(text=[txt_b], return_tensors="pt", padding=True)
            inp_b = {k: v.to(vlm.device) for k, v in inp_b.items()}
            with torch.no_grad():
                out_b = vlm.generate(**inp_b, max_new_tokens=128, do_sample=False)
            pred_base = proc.decode(out_b[0][inp_b["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # MemCanvas
            content = []
            canvas_imgs = []
            for cidx, sim in retrieved[:TOP_K]:
                img = Image.open(canvas_files[cidx]).convert("RGB")
                canvas_imgs.append(img)
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": (
                "Above are memory canvases from past conversation sessions.\n"
                f"Question: {question}\nAnswer concisely based on the conversation history:"
            )})

            msgs_mem = [{"role": "user", "content": content}]
            txt_m = proc.apply_chat_template(msgs_mem, tokenize=False, add_generation_prompt=True)
            inp_m = proc(text=[txt_m], images=canvas_imgs, return_tensors="pt", padding=True)
            inp_m = {k: v.to(vlm.device) for k, v in inp_m.items()}
            with torch.no_grad():
                out_m = vlm.generate(**inp_m, max_new_tokens=128, do_sample=False)
            pred_mem = proc.decode(out_m[0][inp_m["input_ids"].shape[1]:], skip_special_tokens=True).strip()

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
        "config": {"alpha": ALPHA, "top_k": TOP_K, "vlm": "Qwen2.5-VL-7B", "canvas_level": "session"},
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
    print(f"\n=== LoCoMo Session-Level Results ===")
    print(f"  Baseline: F1={summary['baseline']['f1']:.2f}% BLEU1={summary['baseline']['bleu1']:.2f}%")
    print(f"  MemCanvas: F1={summary['memcanvas']['f1']:.2f}% BLEU1={summary['memcanvas']['bleu1']:.2f}%")

    del vlm, proc
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
