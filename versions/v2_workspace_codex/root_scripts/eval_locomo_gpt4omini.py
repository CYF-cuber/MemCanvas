#!/usr/bin/env python3
"""
LoCoMo eval with GPT-4o-mini as the VLM backbone + judge.
Reuses existing canvases and CLIP embeddings from eval_locomo.py.

Usage:
    python eval_locomo_gpt4omini.py
"""

import base64, io, json, os, sys, time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

# ============================================================
# Config
# ============================================================
LOCOMO_JSON = Path("/home/cyf/codex/locomo_repo/data/locomo10.json")
OUTPUT_DIR = Path("/home/cyf/codex/locomo_eval")
CANVAS_DIR = OUTPUT_DIR / "canvases"
GPT_OUTPUT_DIR = OUTPUT_DIR / "gpt4omini"
TOP_K = 2
ALPHA = 0.0
SIM_THRESHOLD = 0.1
CAT_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-domain", 5: "adversarial"}

client = OpenAI(
    api_key="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",
)


def img_to_b64_url(img_path, max_dim=768):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def gpt_call(messages, max_tokens=128):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** (attempt + 1))
    return ""


def load_locomo():
    with open(LOCOMO_JSON) as f:
        data = json.load(f)
    return data


def load_meta():
    meta_path = OUTPUT_DIR / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    # Rebuild meta from data
    data = load_locomo()
    meta = {"conversations": {}}
    canvas_idx = 0
    for conv_idx, conv in enumerate(data):
        conv_id = conv.get("sample_id", str(conv_idx))
        sessions = []
        conversation = conv["conversation"]
        for key in sorted(conversation.keys()):
            if key.startswith("session_") and "date" not in key:
                sessions.append(int(key.split("_")[1]))
        n_sessions = len(sessions)
        n_canvases = (n_sessions + 2) // 3  # 3 sessions per canvas
        meta["conversations"][conv_id] = {
            "canvas_range": [canvas_idx, canvas_idx + n_canvases - 1],
            "n_sessions": n_sessions,
            "canvases": [
                {"canvas_idx": canvas_idx + i, "session_range": [i*3+1, min((i+1)*3, n_sessions)]}
                for i in range(n_canvases)
            ]
        }
        canvas_idx += n_canvases
    return meta


def main():
    GPT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_locomo()
    meta = load_meta()

    # Load CLIP embeddings
    img_emb = np.load(OUTPUT_DIR / "clip_img_emb.npy")
    txt_emb = np.load(OUTPUT_DIR / "clip_txt_emb.npy")
    query_emb = np.load(OUTPUT_DIR / "clip_query_emb.npy")

    keys = ALPHA * img_emb + (1 - ALPHA) * txt_emb
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True).clip(1e-8)
    qn = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True).clip(1e-8)

    # Build QA list
    qa_list = []
    for conv_idx, conv in enumerate(data):
        conv_id = conv.get("sample_id", str(conv_idx))
        conv_info = meta["conversations"][conv_id]
        cs, ce = conv_info["canvas_range"]
        for qa in conv["qa"]:
            if qa["category"] == 5:  # Skip adversarial
                continue
            qa_list.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "category": qa["category"],
                "conv_id": conv_id,
                "canvas_start": cs,
                "canvas_end": ce,
            })

    print(f"Total QA pairs: {len(qa_list)}")

    # Checkpoint
    ckpt_path = GPT_OUTPUT_DIR / "eval_checkpoint.json"
    results = []
    start_idx = 0
    if ckpt_path.exists():
        results = json.loads(ckpt_path.read_text())
        start_idx = len(results)
        print(f"Resuming from checkpoint: {start_idx}")

    # Eval loop
    for qi in tqdm(range(start_idx, len(qa_list)), desc="GPT-4o-mini LoCoMo"):
        qa = qa_list[qi]

        # Retrieve canvases
        cs, ce = qa["canvas_start"], qa["canvas_end"] + 1
        conv_keys = keys[cs:ce]
        q_vec = qn[qi:qi + 1]
        sims = (q_vec @ conv_keys.T)[0]
        top_local = np.argsort(sims)[::-1][:TOP_K]
        retrieved = [(cs + int(j), float(sims[j])) for j in top_local if sims[j] >= SIM_THRESHOLD]

        # Build GPT-4o-mini message with canvas images
        content = []
        for (cidx, sim) in retrieved:
            img_path = CANVAS_DIR / f"{cidx:05d}.png"
            b64_url = img_to_b64_url(img_path)
            content.append({"type": "image_url", "image_url": {"url": b64_url}})

        content.append({"type": "text", "text": (
            "The images above are memory canvases from a long-term conversation. "
            "Each canvas captures one session of dialog between two speakers, "
            "including key facts and events discussed.\n\n"
            f"Question: {qa['question']}\n"
            "Answer concisely based on the conversation memories:"
        )})

        messages = [{"role": "user", "content": content}]
        pred = gpt_call(messages, max_tokens=128)

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

    ckpt_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\nEval complete: {len(results)} QA pairs")

    # ============================================================
    # Judge phase (also GPT-4o-mini)
    # ============================================================
    print("\n--- Scoring with GPT-4o-mini judge ---")
    JUDGE_PROMPT = """You are evaluating a question-answering system's response about a long-term conversation.

Question: {question}
Ground Truth Answer: {gold}
System Answer: {pred}

Is the system's answer correct? Be generous — if the system answer touches on the same key facts as the ground truth, even if phrased differently, mark it as CORRECT.

Respond with exactly one word: CORRECT or WRONG"""

    for r in tqdm(results, desc="Judging"):
        if "verdict" in r:
            continue
        prompt = JUDGE_PROMPT.format(question=r["question"], gold=r["gold"], pred=r["pred"])
        verdict = gpt_call([{"role": "user", "content": prompt}], max_tokens=10)
        r["verdict"] = verdict
        r["correct"] = "CORRECT" in verdict.upper()

    # Compute metrics
    by_cat = {}
    total_correct = 0
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"correct": 0, "total": 0}
        by_cat[cat]["total"] += 1
        if r.get("correct", False):
            by_cat[cat]["correct"] += 1
            total_correct += 1

    overall = total_correct / len(results) * 100 if results else 0
    print(f"\n{'='*50}")
    print(f"LoCoMo Results (MemCanvas + GPT-4o-mini)")
    print(f"{'='*50}")
    print(f"Overall: {overall:.1f}% ({total_correct}/{len(results)})")
    for cat in sorted(by_cat):
        c = by_cat[cat]
        acc = c["correct"] / c["total"] * 100 if c["total"] else 0
        print(f"  {CAT_NAMES.get(cat, cat)}: {acc:.1f}% ({c['correct']}/{c['total']})")

    summary = {
        "overall": overall,
        "total": len(results),
        "categories": {
            CAT_NAMES.get(k, str(k)): {
                "accuracy": v["correct"] / v["total"] * 100,
                "correct": v["correct"],
                "total": v["total"],
            }
            for k, v in by_cat.items()
        },
        "config": {"alpha": ALPHA, "top_k": TOP_K, "method": "MemCanvas+GPT-4o-mini"},
    }

    results_path = GPT_OUTPUT_DIR / "results.json"
    results_path.write_text(json.dumps(summary, indent=2))
    scored_path = GPT_OUTPUT_DIR / "scored_results.json"
    scored_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
