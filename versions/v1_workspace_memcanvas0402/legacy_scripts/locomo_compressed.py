#!/usr/bin/env python3
"""
LoCoMo text compression experiment.
Compress dialogue sessions with Qwen2.5-VL-3B before rendering canvases.
Then CLIP embed + retrieve + evaluate with Qwen2.5-VL-7B.
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
COMPRESSOR_MODEL = "/home/cyf/Qwen2.5-VL-3B-Instruct"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
OUTPUT_DIR = Path("/home/cyf/memcanvas0402/locomo")
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

# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------
COMPRESS_PROMPTS = {
    "light": (
        "Summarize this conversation session. Keep all key facts, names, dates, "
        "events, and opinions. Remove greetings, filler, and repetition. "
        "Output the summary as a list of bullet points."
    ),
    "heavy": (
        "Compress this conversation into the shortest possible form. "
        "Use 'Speaker: key fact' format. Keep only facts that could answer questions. "
        "Maximum 5 lines."
    ),
}

def load_compressor(device="cuda:0"):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print(f"Loading compressor on {device}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        COMPRESSOR_MODEL, torch_dtype=torch.float16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(COMPRESSOR_MODEL)
    model.eval()
    return model, processor

def compress_session(model, processor, messages, level):
    """Compress a session's dialogue text."""
    # Build dialogue text
    lines = []
    for msg in messages:
        speaker = msg.get("speaker", "Unknown")
        text = msg.get("text", "")[:300]
        lines.append(f"[{speaker}] {text}")
    dialogue = "\n".join(lines)

    if len(dialogue) < 50:
        return dialogue  # Too short to compress

    prompt = COMPRESS_PROMPTS[level]
    content = [{"type": "text", "text": f"{prompt}\n\nConversation:\n{dialogue}"}]
    messages_fmt = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages_fmt, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=1.0)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    compressed = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return compressed

# ---------------------------------------------------------------------------
# Canvas rendering
# ---------------------------------------------------------------------------
def render_session_canvas(sample_id, session_name, text_content, session_date=""):
    blocks = []
    header = f"[{sample_id}] {session_name}"
    if session_date:
        header += f" | {session_date}"
    blocks.append(measure_text(header, font_size=14, ref_width=600))
    blocks.append(measure_text("─" * 60, font_size=10, ref_width=600))
    blocks.append(measure_text(text_content, font_size=13, ref_width=600))

    layout = choose_best_layout(blocks)
    img = render_layout(layout)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True, choices=["light", "heavy"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    level = args.level
    device = f"cuda:{args.gpu}"
    level_dir = OUTPUT_DIR / level
    canvas_dir = level_dir / "canvases"
    canvas_dir.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} conversations")

    # --- Phase 1: Compress session text ---
    compressed_cache = level_dir / "compressed_sessions.json"
    if compressed_cache.exists():
        compressed_data = json.load(open(compressed_cache))
        print(f"Loaded {sum(len(v) for v in compressed_data.values())} compressed sessions from cache")
    else:
        comp_model, comp_proc = load_compressor(device)
        compressed_data = {}
        total_in, total_out = 0, 0

        for item in tqdm(data, desc=f"Compressing ({level})"):
            sid = item["sample_id"]
            conv = item.get("conversation", {})
            sessions = sorted(
                [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
                key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0
            )
            compressed_data[sid] = {}
            for session_key in sessions:
                messages = conv[session_key]
                original = "\n".join(f"[{m.get('speaker','?')}] {m.get('text','')[:300]}" for m in messages)
                total_in += len(original)

                compressed = compress_session(comp_model, comp_proc, messages, level)
                total_out += len(compressed)
                compressed_data[sid][session_key] = compressed

        json.dump(compressed_data, open(compressed_cache, "w"), indent=2, ensure_ascii=False)
        ratio = total_out / total_in * 100 if total_in > 0 else 0
        print(f"\n=== Compression Stats ({level}) ===")
        print(f"  Total sessions: {sum(len(v) for v in compressed_data.values())}")
        print(f"  Avg input: {total_in / sum(len(v) for v in compressed_data.values()):.0f} chars")
        print(f"  Avg output: {total_out / sum(len(v) for v in compressed_data.values()):.0f} chars")
        print(f"  Compression ratio: {ratio:.1f}%")

        del comp_model, comp_proc
        torch.cuda.empty_cache()

    # --- Phase 2: Build canvases + CLIP embeddings ---
    print("\nBuilding canvases + CLIP embeddings...")
    from transformers import CLIPProcessor, CLIPModel
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).cuda().eval()

    all_memory = {}
    total_canvases = 0

    for item in tqdm(data, desc="Building canvases"):
        sid = item["sample_id"]
        conv = item.get("conversation", {})
        sessions = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
            key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0
        )

        img_embs, txt_embs, cfiles = [], [], []

        for si, session_key in enumerate(sessions):
            date_key = f"{session_key}_date_time"
            session_date = conv.get(date_key, "")

            # Use compressed text
            compressed_text = compressed_data[sid][session_key]

            cfile = canvas_dir / f"{sid}_s{si:03d}.png"
            if not cfile.exists():
                canvas_bytes = render_session_canvas(sid, session_key, compressed_text, session_date)
                cfile.write_bytes(canvas_bytes)

            # Image embedding
            img = Image.open(cfile).convert("RGB")
            inp = clip_proc(images=img, return_tensors="pt")
            inp = {k: v.cuda() for k, v in inp.items()}
            with torch.no_grad():
                ie = clip_model.get_image_features(**inp)
                ie = (ie / ie.norm(dim=-1, keepdim=True)).cpu().numpy().squeeze()
            img_embs.append(ie)

            # Text embedding
            txt_clip = compressed_text[:500]
            inp_t = clip_proc(text=txt_clip, return_tensors="pt", truncation=True, max_length=77)
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

    print(f"  Total canvases: {total_canvases}")

    # Pre-compute query embeddings
    print("Computing query embeddings...")
    query_embs = {}
    for item in data:
        sid = item["sample_id"]
        for qi, qa in enumerate(item.get("qa", [])):
            inp_q = clip_proc(text=qa["question"], return_tensors="pt", truncation=True, max_length=77)
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

    ckpt_file = level_dir / "checkpoint.json"
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

            # MemCanvas only (baseline already computed in v3)
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
                "pred_mem": pred_mem,
                "em_mem": compute_em(pred_mem, answer),
                "f1_mem": compute_f1(pred_mem, answer),
                "b1_mem": compute_bleu1(pred_mem, answer),
            }

            if len(results) % 50 == 0:
                json.dump(results, open(ckpt_file, "w"))

    json.dump(results, open(ckpt_file, "w"))

    # Summary
    n = len(results)
    summary = {
        "n": n,
        "compression": level,
        "config": {"alpha": ALPHA, "top_k": TOP_K, "vlm": "Qwen2.5-VL-7B", "canvas_level": "session"},
        f"memcanvas_{level}": {
            "f1": np.mean([v["f1_mem"] for v in results.values()]) * 100,
            "bleu1": np.mean([v["b1_mem"] for v in results.values()]) * 100,
            "em": np.mean([v["em_mem"] for v in results.values()]) * 100,
        },
        "reference_original": {
            "baseline": {"f1": 6.21, "bleu1": 4.68, "em": 0.30},
            "memcanvas": {"f1": 32.59, "bleu1": 35.63, "em": 15.56},
        },
    }
    json.dump(summary, open(level_dir / "summary.json", "w"), indent=2)
    mc = summary[f"memcanvas_{level}"]
    print(f"\n{'='*60}")
    print(f"LoCoMo Results — {level} compression ({n} QA pairs)")
    print(f"{'='*60}")
    print(f"  Original Baseline:    F1=6.21%   BLEU1=4.68%   EM=0.30%")
    print(f"  Original MemCanvas:   F1=32.59%  BLEU1=35.63%  EM=15.56%")
    print(f"  Compressed ({level:5s}):  F1={mc['f1']:.2f}%  BLEU1={mc['bleu1']:.2f}%  EM={mc['em']:.2f}%")
    print(f"{'='*60}")

    del vlm, proc
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
