#!/usr/bin/env python3
"""
MemCanvas evaluation on LoCoMo benchmark (Long Conversation Memory).

Pipeline:
1. Parse LoCoMo conversations into sessions
2. Render each session as a SmartCanvas image
3. Compute CLIP embeddings (image + text) per canvas
4. For each QA pair: retrieve top-K canvases → VLM inference
5. Score with LLM-as-Judge (Qwen2.5-7B-Instruct, same protocol as Mem0 paper)

Usage:
    python eval_locomo.py --phase render     # Step 1-2: render canvases
    python eval_locomo.py --phase embed      # Step 3: CLIP embeddings
    python eval_locomo.py --phase eval       # Step 4: VLM QA
    python eval_locomo.py --phase judge      # Step 5: LLM-as-Judge scoring
    python eval_locomo.py --phase all        # All steps sequentially
"""

import argparse, io, json, os, re, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/codex")
from smart_canvas_layout import (
    measure_text, measure_image, choose_best_layout, render_layout,
    layout_single_column, ContentBlock, BlockType,
)

# ============================================================
# Config
# ============================================================
LOCOMO_JSON = Path("/home/cyf/codex/locomo_repo/data/locomo10.json")
OUTPUT_DIR = Path("/home/cyf/codex/locomo_eval")
CANVAS_DIR = OUTPUT_DIR / "canvases"
VLM_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
TOP_K = 2
SIM_THRESHOLD = 0.1
ALPHA = 0.0  # text-only retrieval (LoCoMo is text conversation)

# Category names
CAT_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-domain", 5: "adversarial"}

# ============================================================
# Step 1-2: Parse & Render
# ============================================================
def load_locomo():
    """Load LoCoMo conversations and QA pairs."""
    with open(LOCOMO_JSON) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} conversations")
    return data


def parse_sessions(conv):
    """Extract ordered sessions from a conversation dict."""
    conversation = conv["conversation"]
    sessions = []
    for key in sorted(conversation.keys()):
        if key.startswith("session_") and "date" not in key:
            session_num = int(key.split("_")[1])
            date_key = f"{key}_date_time"
            date_str = conversation.get(date_key, "")
            turns = conversation[key]
            sessions.append({
                "session_num": session_num,
                "key": key,
                "date": date_str,
                "turns": turns,
            })
    return sessions


SESSIONS_PER_CANVAS = 3  # Aggregate N sessions per canvas for denser information


def extract_session_facts(conv, session_key):
    """Extract structured facts from a session: observations + summary + events."""
    parts = []

    # Session summary (concise paragraph)
    summary = conv.get("session_summary", {}).get(f"{session_key}_summary", "")
    if summary:
        parts.append(summary.strip())

    # Key observations per speaker (factual assertions)
    obs = conv.get("observation", {}).get(f"{session_key}_observation", {})
    if isinstance(obs, dict):
        for speaker, facts in obs.items():
            if isinstance(facts, list):
                for fact_item in facts:
                    fact_text = fact_item[0] if isinstance(fact_item, list) else str(fact_item)
                    parts.append(f"• {fact_text}")

    # Key events per speaker
    events = conv.get("event_summary", {}).get(f"events_{session_key}", {})
    if isinstance(events, dict):
        for speaker, evts in events.items():
            if speaker == "date":
                continue
            if isinstance(evts, list):
                for e in evts:
                    if e:
                        parts.append(f"Event ({speaker}): {e}")

    return "\n".join(parts) if parts else ""


def render_multi_session_canvas(sessions_group, conv_id, conv):
    """Render multiple sessions into one dense canvas using structured facts."""
    blocks = []

    # Header: conversation + session range + date range
    s_nums = [s["session_num"] for s in sessions_group]
    dates = [s["date"] for s in sessions_group if s["date"]]
    header = f"[Conv {conv_id}] Sessions {s_nums[0]}-{s_nums[-1]}"
    if dates:
        header += f" ({dates[0]} — {dates[-1]})"
    blocks.append(measure_text(header, font_size=12, ref_width=600))

    for sess in sessions_group:
        # Session sub-header
        sub_header = f"— Session {sess['session_num']}"
        if sess["date"]:
            sub_header += f" ({sess['date']})"
        blocks.append(measure_text(sub_header, font_size=13, ref_width=600))

        # Structured facts (summary + observations + events)
        facts = extract_session_facts(conv, sess["key"])
        if facts:
            # Truncate if very long
            if len(facts) > 800:
                facts = facts[:800] + "..."
            blocks.append(measure_text(facts, font_size=14, ref_width=600))
        else:
            # Fallback: use raw dialog (compressed)
            dialog_lines = []
            for turn in sess["turns"]:
                text = turn.get("text", "")
                speaker = turn.get("speaker", "?")
                if text:
                    dialog_lines.append(f"{speaker}: {text}")
            dialog = "\n".join(dialog_lines)
            if len(dialog) > 600:
                dialog = dialog[:600] + "..."
            if dialog:
                blocks.append(measure_text(dialog, font_size=14, ref_width=600))

    if len(blocks) <= 1:
        blocks.append(measure_text("(no content)", font_size=14, ref_width=600))

    layout = layout_single_column(blocks)
    img = render_layout(layout)
    return img


def render_all_canvases(data):
    """Render aggregated multi-session canvases for all conversations."""
    CANVAS_DIR.mkdir(parents=True, exist_ok=True)
    done_marker = CANVAS_DIR / "done.json"
    if done_marker.exists():
        meta = json.loads(done_marker.read_text())
        print(f"Canvases already rendered: {meta['total']} canvases")
        return meta

    meta = {"conversations": {}, "total": 0}
    canvas_idx = 0

    for conv_idx, conv in enumerate(tqdm(data, desc="Rendering canvases")):
        conv_id = conv.get("sample_id", str(conv_idx))
        sessions = parse_sessions(conv)

        # Group sessions into chunks of SESSIONS_PER_CANVAS
        conv_canvases = []
        for g_start in range(0, len(sessions), SESSIONS_PER_CANVAS):
            group = sessions[g_start:g_start + SESSIONS_PER_CANVAS]

            img = render_multi_session_canvas(group, conv_id, conv)
            out_path = CANVAS_DIR / f"{canvas_idx:05d}.png"
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            out_path.write_bytes(buf.getvalue())

            conv_canvases.append({
                "canvas_idx": canvas_idx,
                "session_range": [group[0]["session_num"], group[-1]["session_num"]],
                "n_sessions": len(group),
                "n_turns": sum(len(s["turns"]) for s in group),
            })
            canvas_idx += 1

        meta["conversations"][conv_id] = {
            "conv_idx": conv_idx,
            "canvases": conv_canvases,
            "canvas_range": [conv_canvases[0]["canvas_idx"], conv_canvases[-1]["canvas_idx"]],
        }

    meta["total"] = canvas_idx
    done_marker.write_text(json.dumps(meta, indent=2))
    print(f"Rendered {canvas_idx} canvases ({SESSIONS_PER_CANVAS} sessions/canvas) across {len(data)} conversations")
    return meta


# ============================================================
# Step 3: CLIP Embeddings
# ============================================================
def compute_clip_embeddings(data, meta):
    """Compute CLIP image + text embeddings for all canvases and queries."""
    import torch
    from transformers import CLIPProcessor, CLIPModel

    n = meta["total"]
    img_emb_path = OUTPUT_DIR / "clip_img_emb.npy"
    txt_emb_path = OUTPUT_DIR / "clip_txt_emb.npy"
    query_emb_path = OUTPUT_DIR / "clip_query_emb.npy"

    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Image embeddings (from canvas PNGs)
    if not img_emb_path.exists():
        print(f"Computing CLIP image embeddings for {n} canvases...")
        all_emb = []
        bs = 32
        for i in tqdm(range(0, n, bs), desc="CLIP img"):
            imgs = []
            for j in range(i, min(i + bs, n)):
                imgs.append(Image.open(CANVAS_DIR / f"{j:05d}.png").convert("RGB"))
            inp = proc(images=imgs, return_tensors="pt", padding=True)
            inp = {k: v.cuda() for k, v in inp.items()}
            with torch.no_grad():
                feat = clip.get_image_features(**inp)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            all_emb.append(feat.cpu().numpy())
        emb = np.concatenate(all_emb)
        np.save(img_emb_path, emb)
        print(f"  Saved image embeddings: {emb.shape}")
    else:
        print(f"  Image embeddings exist: {img_emb_path}")

    # Text embeddings (from session dialog text — serves as text key for each canvas)
    if not txt_emb_path.exists():
        print(f"Computing CLIP text embeddings for {n} canvases...")
        all_emb = []
        bs = 32
        canvas_texts = build_canvas_texts(data, meta)
        for i in tqdm(range(0, n, bs), desc="CLIP txt"):
            batch = canvas_texts[i:min(i + bs, n)]
            # Truncate to CLIP max length
            batch = [t[:300] for t in batch]
            inp = proc(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inp = {k: v.cuda() for k, v in inp.items()}
            with torch.no_grad():
                feat = clip.get_text_features(**inp)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            all_emb.append(feat.cpu().numpy())
        emb = np.concatenate(all_emb)
        np.save(txt_emb_path, emb)
        print(f"  Saved text embeddings: {emb.shape}")
    else:
        print(f"  Text embeddings exist: {txt_emb_path}")

    # Query embeddings (from QA questions)
    if not query_emb_path.exists():
        print("Computing CLIP query embeddings...")
        questions = []
        for conv in data:
            for qa in conv["qa"]:
                if qa["category"] != 5:  # Skip adversarial
                    questions.append(qa["question"][:300])
        all_emb = []
        bs = 64
        for i in tqdm(range(0, len(questions), bs), desc="CLIP query"):
            batch = questions[i:i + bs]
            inp = proc(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inp = {k: v.cuda() for k, v in inp.items()}
            with torch.no_grad():
                feat = clip.get_text_features(**inp)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            all_emb.append(feat.cpu().numpy())
        emb = np.concatenate(all_emb)
        np.save(query_emb_path, emb)
        print(f"  Saved query embeddings: {emb.shape}")
    else:
        print(f"  Query embeddings exist: {query_emb_path}")

    del clip, proc
    torch.cuda.empty_cache()


def build_canvas_texts(data, meta):
    """Build text summaries for each canvas (used for CLIP text embedding)."""
    texts = [""] * meta["total"]
    for conv_idx, conv in enumerate(data):
        conv_id = conv.get("sample_id", str(conv_idx))
        conv_meta = meta["conversations"][conv_id]
        sessions = parse_sessions(conv)

        for ci, canvas_info in enumerate(conv_meta["canvases"]):
            canvas_idx = canvas_info["canvas_idx"]
            s_start, s_end = canvas_info["session_range"]
            # Collect facts from all sessions in this canvas
            parts = []
            for sess in sessions:
                if s_start <= sess["session_num"] <= s_end:
                    facts = extract_session_facts(conv, sess["key"])
                    if facts:
                        parts.append(facts)
                    else:
                        for turn in sess["turns"]:
                            text = turn.get("text", "")
                            if text:
                                parts.append(text)
            texts[canvas_idx] = " ".join(parts)[:300]
    return texts


# ============================================================
# Step 4: VLM Evaluation
# ============================================================
def build_retrieval_map(conv_meta, img_emb, txt_emb, query_emb, alpha=ALPHA, top_k=TOP_K):
    """Build per-query retrieval map, restricted to same conversation."""
    keys = alpha * img_emb + (1 - alpha) * txt_emb
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True).clip(1e-8)
    qn = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True).clip(1e-8)
    return keys, qn


def eval_vlm(data, meta):
    """Run VLM inference on LoCoMo QA pairs."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    # Load embeddings
    img_emb = np.load(OUTPUT_DIR / "clip_img_emb.npy")
    txt_emb = np.load(OUTPUT_DIR / "clip_txt_emb.npy")
    query_emb = np.load(OUTPUT_DIR / "clip_query_emb.npy")

    # Build global keys
    keys = ALPHA * img_emb + (1 - ALPHA) * txt_emb
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True).clip(1e-8)
    qn = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True).clip(1e-8)

    # Load VLM
    print("Loading VLM...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)

    # Checkpoint
    ckpt_path = OUTPUT_DIR / "eval_checkpoint.json"
    results = []
    start_idx = 0
    if ckpt_path.exists():
        results = json.loads(ckpt_path.read_text())
        start_idx = len(results)
        print(f"Resuming from checkpoint: {start_idx} done")

    # Build flat QA list with conversation mapping
    qa_list = []
    for conv_idx, conv in enumerate(data):
        conv_id = conv.get("sample_id", str(conv_idx))
        conv_info = meta["conversations"][conv_id]
        canvas_start, canvas_end = conv_info["canvas_range"]

        for qa in conv["qa"]:
            if qa["category"] == 5:  # Skip adversarial
                continue
            qa_list.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "category": qa["category"],
                "evidence": qa.get("evidence", []),
                "conv_id": conv_id,
                "canvas_start": canvas_start,
                "canvas_end": canvas_end,
            })

    print(f"Total QA pairs: {len(qa_list)} (non-adversarial)")

    for qi in tqdm(range(start_idx, len(qa_list)), desc="VLM eval"):
        qa = qa_list[qi]

        # Retrieve within same conversation only
        cs, ce = qa["canvas_start"], qa["canvas_end"] + 1
        conv_keys = keys[cs:ce]
        q_vec = qn[qi:qi + 1]
        sims = (q_vec @ conv_keys.T)[0]
        top_local = np.argsort(sims)[::-1][:TOP_K]
        retrieved = [(cs + int(j), float(sims[j])) for j in top_local if sims[j] >= SIM_THRESHOLD]

        # Build prompt with canvas images
        content = []
        for (cidx, sim) in retrieved:
            img_path = str(CANVAS_DIR / f"{cidx:05d}.png")
            content.append({"type": "image", "image": f"file://{img_path}"})

        content.append({"type": "text", "text": (
            "The images above are memory canvases from a long-term conversation. "
            "Each canvas captures one session of dialog between two speakers, "
            "including key facts and events discussed.\n\n"
            f"Question: {qa['question']}\n"
            "Answer concisely based on the conversation memories:"
        )})

        messages = [{"role": "user", "content": content}]
        text_input = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = proc(text=[text_input], images=image_inputs, videos=video_inputs,
                      padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        pred = proc.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        results.append({
            "qi": qi,
            "question": qa["question"],
            "gold": qa["answer"],
            "pred": pred,
            "category": qa["category"],
            "conv_id": qa["conv_id"],
            "retrieved": retrieved,
        })

        if (qi + 1) % 50 == 0:
            ckpt_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))
            print(f"  Checkpoint saved: {qi + 1}/{len(qa_list)}")

    ckpt_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"Evaluation complete: {len(results)} QA pairs")
    return results


# ============================================================
# Step 5: LLM-as-Judge Scoring
# ============================================================
JUDGE_PROMPT = """You are evaluating a question-answering system's response about a long-term conversation.

Question: {question}
Ground Truth Answer: {gold}
System Answer: {pred}

Is the system's answer correct? Be generous — if the system answer touches on the same key facts as the ground truth, even if phrased differently, mark it as CORRECT.

Respond with exactly one word: CORRECT or WRONG"""


def score_with_judge(results):
    """Score results using LLM-as-Judge via OpenAI-compatible API server."""
    import requests

    api_url = os.environ.get("JUDGE_API_URL", "http://localhost:8100/v1/chat/completions")
    print(f"Using judge API: {api_url}")

    scored = []
    errors = 0
    for r in tqdm(results, desc="Judging"):
        prompt = JUDGE_PROMPT.format(
            question=r["question"], gold=r["gold"], pred=r["pred"])
        try:
            resp = requests.post(api_url, json={
                "model": "qwen2.5-vl-7b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.0,
            }, timeout=30)
            verdict = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            verdict = "ERROR"
            errors += 1
            if errors <= 3:
                print(f"\n  Judge API error: {e}")
        is_correct = "CORRECT" in verdict.upper()
        r["verdict"] = verdict
        r["correct"] = is_correct
        scored.append(r)
    if errors:
        print(f"\n  Total judge errors: {errors}")

    # Compute metrics
    by_cat = {}
    total_correct = 0
    for r in scored:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"correct": 0, "total": 0}
        by_cat[cat]["total"] += 1
        if r["correct"]:
            by_cat[cat]["correct"] += 1
            total_correct += 1

    overall = total_correct / len(scored) * 100 if scored else 0
    print(f"\n{'='*50}")
    print(f"LoCoMo Results (MemCanvas, α={ALPHA}, K={TOP_K})")
    print(f"{'='*50}")
    print(f"Overall: {overall:.1f}% ({total_correct}/{len(scored)})")
    for cat in sorted(by_cat):
        c = by_cat[cat]
        acc = c["correct"] / c["total"] * 100 if c["total"] else 0
        print(f"  {CAT_NAMES.get(cat, cat)}: {acc:.1f}% ({c['correct']}/{c['total']})")

    summary = {
        "overall": overall,
        "total": len(scored),
        "categories": {
            CAT_NAMES.get(k, str(k)): {
                "accuracy": v["correct"] / v["total"] * 100,
                "correct": v["correct"],
                "total": v["total"],
            }
            for k, v in by_cat.items()
        },
        "config": {"alpha": ALPHA, "top_k": TOP_K, "method": "MemCanvas"},
    }

    (OUTPUT_DIR / "results.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT_DIR / "scored_results.json").write_text(json.dumps(scored, ensure_ascii=False, indent=1))
    print(f"\nSaved to {OUTPUT_DIR}/results.json")
    return summary


# ============================================================
# Main
# ============================================================
def main():
    global ALPHA, TOP_K

    parser = argparse.ArgumentParser(description="MemCanvas LoCoMo evaluation")
    parser.add_argument("--phase", choices=["render", "embed", "eval", "judge", "all"], default="all")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    ALPHA = args.alpha
    TOP_K = args.top_k

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_locomo()

    if args.phase in ("render", "all"):
        meta = render_all_canvases(data)
    else:
        meta = json.loads((CANVAS_DIR / "done.json").read_text())

    if args.phase in ("embed", "all"):
        compute_clip_embeddings(data, meta)

    if args.phase in ("eval", "all"):
        results = eval_vlm(data, meta)
    else:
        results = None

    if args.phase in ("judge", "all"):
        if results is None:
            results = json.loads((OUTPUT_DIR / "eval_checkpoint.json").read_text())
        score_with_judge(results)


if __name__ == "__main__":
    main()
