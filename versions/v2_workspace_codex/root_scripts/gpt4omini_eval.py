#!/usr/bin/env python3
"""
GPT-4o-mini evaluation for MemCanvas — all 6 benchmarks.
Uses OpenRouter API. Canvas images sent as base64.
Evaluates both zero-shot baseline AND MemCanvas-augmented.

v2: Fixed issues:
  - Added system prompt to force short answers
  - Added extract_short_answer() post-processing
  - OK-VQA: fixed metric to standard VQA soft accuracy min(1, count/3)
  - MMQA: fixed context resolution via tables/texts/images dicts
  - MMQA MemCanvas: added supporting context (was missing)
  - All prompts: "Answer with ONLY the answer, no explanation"

Usage:
  python gpt4omini_eval.py --benchmark hotpotqa
  python gpt4omini_eval.py --benchmark scienceqa
  python gpt4omini_eval.py --benchmark okvqa
  python gpt4omini_eval.py --benchmark mmqa
  python gpt4omini_eval.py --benchmark locomo
  python gpt4omini_eval.py --benchmark all
"""
import argparse, base64, io, json, os, pickle, re, string, time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

OUTPUT_ROOT = Path("/home/cyf/codex/gpt4omini_eval_v2")

SYSTEM_MSG = (
    "You are a QA assistant. Always respond with ONLY the answer itself — "
    "a single word, number, or short phrase. Never include explanations, "
    "full sentences, or reasoning. Examples:\n"
    "Q: What color is the sky? → blue\n"
    "Q: Who painted the Mona Lisa? → Leonardo da Vinci\n"
    "Q: What year was it founded? → 1998"
)

client = OpenAI(
    api_key="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",
)

def gpt_call(messages, max_tokens=64):
    # Prepend system message
    full_msgs = [{"role": "system", "content": SYSTEM_MSG}] + messages
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=full_msgs,
                max_tokens=max_tokens,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** (attempt + 1))
    return ""

def img_to_b64_url(img):
    if isinstance(img, (str, Path)):
        img = Image.open(img).convert("RGB")
    max_dim = 768  # Increased from 512 for better detail
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def extract_short_answer(text):
    """Post-process GPT response to extract short answer."""
    if not text:
        return ""
    # Remove common prefixes
    text = re.sub(r"^(the answer is|answer:|a:)\s*", "", text, flags=re.IGNORECASE)
    # If it's already short, return as-is
    if len(text.split()) <= 5:
        return text
    # Take first sentence
    first = re.split(r"[.\n]", text)[0].strip()
    # Remove trailing period
    first = first.rstrip(".")
    return first


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
    gc = Counter(gt); return sum(min(Counter(pt)[w], gc[w]) for w in Counter(pt)) / len(pt)
def compute_anls(p, g):
    p, g = normalize_answer(p), normalize_answer(g)
    if not p or not g: return 0.0
    ratio = SequenceMatcher(None, p, g).ratio()
    return ratio if ratio >= 0.5 else 0.0

def compute_vqa_accuracy(pred, answers):
    """Standard VQA soft accuracy: min(1, num_matching / 3)."""
    pred_norm = normalize_answer(pred)
    count = sum(1 for a in answers if normalize_answer(a) == pred_norm)
    return min(1.0, count / 3.0)


def build_retrieval_map(img_emb, txt_emb, query_emb, alpha=0.75, top_k=2):
    keys = alpha * img_emb + (1 - alpha) * txt_emb
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True).clip(1e-8)
    qn = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True).clip(1e-8)
    sims = qn @ keys.T
    rmap = {}
    for i in range(len(query_emb)):
        top = np.argsort(sims[i])[::-1][:top_k + 5]
        res = [(int(j), float(sims[i][j])) for j in top if sims[i][j] >= 0.1][:top_k]
        rmap[i] = res
    return rmap

def save_summary_dual(results, out_dir, metrics):
    n = len(results)
    summary = {"n": n}
    for cond in ["b", "m"]:
        label = "baseline" if cond == "b" else "memcanvas"
        summary[label] = {}
        for m in metrics:
            vals = [v[f"{m}_{cond}"] for v in results.values()]
            summary[label][m] = np.mean(vals) * 100
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"  Results ({n} samples):")
    for label in ["baseline", "memcanvas"]:
        parts = " ".join(f"{m}={summary[label][m]:.2f}%" for m in metrics)
        print(f"    {label}: {parts}")

# ============================================================
# HotpotQA
# ============================================================
def eval_hotpotqa():
    print("=== HotpotQA GPT-4o-mini ===")
    DATA = Path("/home/cyf/codex/hotpotqa_data")
    with open(DATA / "hotpotqa_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dev = meta["dev"]
    rmap = pickle.load(open("/home/cyf/codex/hotpotqa_experiment_v2/retrieval_map_smart.pkl", "rb"))
    canvas_dir = DATA / "canvases_smart"

    out_dir = OUTPUT_ROOT / "hotpotqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    MAX_SAMPLES = 500
    for i in tqdm(range(len(dev)), desc="HotpotQA"):
        if len(results) >= MAX_SAMPLES: break
        if str(i) in results: continue
        s = dev[i]
        q, gt = s["question"], s["answer"]
        ctx = "\n\n".join(f"[{p['title']}]\n{p['text'][:500]}" for p in s.get("paragraphs", []))

        # Baseline
        pred_b = gpt_call([{"role": "user", "content":
            f"Use the context to answer.\n\n{ctx}\n\nQuestion: {q}\nAnswer with ONLY the answer:"}])
        pred_b = extract_short_answer(pred_b)

        # MemCanvas
        content_m = []
        for cidx, sim in rmap.get(i, [])[:2]:
            cfile = canvas_dir / f"{cidx:05d}.png"
            if cfile.exists():
                content_m.append({"type": "image_url", "image_url": {"url": img_to_b64_url(cfile)}})
        content_m.append({"type": "text", "text":
            f"Above are memory canvases from similar solved questions.\n\n{ctx}\n\nQuestion: {q}\nAnswer with ONLY the answer:"})
        pred_m = gpt_call([{"role": "user", "content": content_m}])
        pred_m = extract_short_answer(pred_m)

        results[str(i)] = {
            "gt": gt, "pred_b": pred_b, "pred_m": pred_m,
            "em_b": compute_em(pred_b, gt), "em_m": compute_em(pred_m, gt),
            "f1_b": compute_f1(pred_b, gt), "f1_m": compute_f1(pred_m, gt),
        }
        if len(results) % 20 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    save_summary_dual(results, out_dir, ["em", "f1"])

# ============================================================
# ScienceQA (zero-shot + MemCanvas)
# ============================================================
def eval_scienceqa():
    print("=== ScienceQA GPT-4o-mini ===")
    from datasets import load_dataset
    ds = load_dataset("derek-thomas/ScienceQA", split="test")

    CANVAS_DIR = Path("/home/cyf/codex/scienceqa_smart_canvases")
    img_emb = np.load(CANVAS_DIR / "clip_img_emb.npy")
    txt_emb = np.load(CANVAS_DIR / "clip_txt_emb.npy")
    q_emb = np.load(CANVAS_DIR / "clip_query_emb.npy")
    rmap = build_retrieval_map(img_emb, txt_emb, q_emb, alpha=0.75, top_k=2)

    out_dir = OUTPUT_ROOT / "scienceqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    MAX_SAMPLES = 500
    for i in tqdm(range(len(ds)), desc="ScienceQA"):
        if len(results) >= MAX_SAMPLES: break
        if str(i) in results: continue
        item = ds[i]
        q = item["question"]
        gt = chr(65 + item["answer"])
        choices = item["choices"]
        hint = item.get("hint", "") or ""
        choice_txt = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))

        # Zero-shot baseline
        content_b = []
        if item.get("image") is not None:
            content_b.append({"type": "image_url", "image_url": {"url": img_to_b64_url(item["image"])}})
        content_b.append({"type": "text", "text": f"{hint}\n\nQuestion: {q}\n{choice_txt}\nAnswer with just the letter:"})
        pred_b = gpt_call([{"role": "user", "content": content_b}], max_tokens=8)

        # MemCanvas
        content_m = []
        for cidx, sim in rmap.get(i, [])[:2]:
            cfile = CANVAS_DIR / f"{cidx:05d}.png"
            if cfile.exists():
                content_m.append({"type": "image_url", "image_url": {"url": img_to_b64_url(cfile)}})
        if item.get("image") is not None:
            content_m.append({"type": "image_url", "image_url": {"url": img_to_b64_url(item["image"])}})
        content_m.append({"type": "text", "text": f"Study the reference canvases above.\n{hint}\n\nQuestion: {q}\n{choice_txt}\nAnswer with just the letter:"})
        pred_m = gpt_call([{"role": "user", "content": content_m}], max_tokens=8)

        pred_b_letter = next((c for c in pred_b.upper() if c in "ABCDEF"), "")
        pred_m_letter = next((c for c in pred_m.upper() if c in "ABCDEF"), "")

        results[str(i)] = {
            "gt": gt, "pred_b": pred_b_letter, "pred_m": pred_m_letter,
            "correct_b": float(pred_b_letter == gt), "correct_m": float(pred_m_letter == gt),
            "subject": item.get("subject", ""),
        }
        if len(results) % 50 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    n = len(results)
    acc_b = np.mean([v["correct_b"] for v in results.values()]) * 100
    acc_m = np.mean([v["correct_m"] for v in results.values()]) * 100
    summary = {"n": n, "baseline": {"accuracy": acc_b}, "memcanvas": {"accuracy": acc_m}}
    for subj in ["natural science", "social science", "language science"]:
        vals_b = [v["correct_b"] for v in results.values() if v.get("subject","") == subj]
        vals_m = [v["correct_m"] for v in results.values() if v.get("subject","") == subj]
        if vals_b:
            summary["baseline"][subj] = np.mean(vals_b) * 100
            summary["memcanvas"][subj] = np.mean(vals_m) * 100
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"  ScienceQA GPT-4o-mini: baseline={acc_b:.2f}%, memcanvas={acc_m:.2f}%")

# ============================================================
# OK-VQA — uses standard VQA soft accuracy: min(1, count/3)
# ============================================================
def eval_okvqa():
    print("=== OK-VQA GPT-4o-mini ===")
    DATA = Path("/home/cyf/codex/okvqa_data")
    with open(DATA / "okvqa_cached.pkl", "rb") as f:
        data = pickle.load(f)
    test = data["test"]
    canvas_dir = DATA / "canvases_smart"

    img_emb = np.load(DATA / "canvas_embeddings.npy")
    txt_emb = np.load(DATA / "canvas_text_embeddings.npy")
    q_emb = np.load(DATA / "query_embeddings.npy")
    rmap = build_retrieval_map(img_emb, txt_emb, q_emb, alpha=0.5, top_k=2)

    out_dir = OUTPUT_ROOT / "okvqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    MAX_SAMPLES = 500
    for i in tqdm(range(len(test)), desc="OK-VQA"):
        if len(results) >= MAX_SAMPLES: break
        if str(i) in results: continue
        s = test[i]
        q = s["question"]
        answers = s.get("answers", [])

        # Zero-shot baseline
        content_b = []
        if s.get("image_path") and os.path.exists(s["image_path"]):
            content_b.append({"type": "image_url", "image_url": {"url": img_to_b64_url(s["image_path"])}})
        content_b.append({"type": "text", "text": f"Question: {q}\nAnswer with ONLY the answer:"})
        pred_b = extract_short_answer(gpt_call([{"role": "user", "content": content_b}]))

        # MemCanvas
        content_m = []
        for cidx, sim in rmap.get(i, [])[:2]:
            cfile = canvas_dir / f"{cidx:05d}.png"
            if cfile.exists():
                content_m.append({"type": "image_url", "image_url": {"url": img_to_b64_url(cfile)}})
        if s.get("image_path") and os.path.exists(s["image_path"]):
            content_m.append({"type": "image_url", "image_url": {"url": img_to_b64_url(s["image_path"])}})
        content_m.append({"type": "text", "text":
            f"Above are memory canvases from similar visual questions with answers.\nQuestion: {q}\nAnswer with ONLY the answer:"})
        pred_m = extract_short_answer(gpt_call([{"role": "user", "content": content_m}]))

        # Standard VQA soft accuracy: min(1, matching_count / 3)
        acc_b = compute_vqa_accuracy(pred_b, answers)
        acc_m = compute_vqa_accuracy(pred_m, answers)
        results[str(i)] = {
            "gt": answers, "pred_b": pred_b, "pred_m": pred_m,
            "accuracy_b": acc_b, "accuracy_m": acc_m,
        }
        if len(results) % 20 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    save_summary_dual(results, out_dir, ["accuracy"])

# ============================================================
# MMQA — fixed context resolution
# ============================================================
def _resolve_mmqa_context(item, tables, texts, max_ctx=3):
    """Resolve supporting_context doc_ids to actual text."""
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
            # Simple markdown table
            tbl = doc.get("table", {})
            headers = [h["column_name"] for h in tbl.get("header", [])]
            rows = [[c["text"][:30] for c in row] for row in tbl.get("table_rows", [])[:6]]
            lines = [" | ".join(headers)]
            for row in rows:
                lines.append(" | ".join(row))
            parts.append(f"[Table: {title}]\n" + "\n".join(lines))

        elif doc_part == "image":
            # Image context noted as text (actual image not sent in text-only baseline)
            title = ""
            if doc_id in texts:
                title = texts[doc_id].get("title", "")
            parts.append(f"[Image: {title or doc_id[:8]}]")

    return "\n\n".join(parts)


def eval_mmqa():
    print("=== MMQA GPT-4o-mini ===")
    DATA = Path("/home/cyf/codex/mmqa_data")
    mmqa_parsed = pickle.load(open(DATA / "mmqa_parsed.pkl", "rb"))
    dev = mmqa_parsed["dev"]
    tables = mmqa_parsed["tables"]
    texts = mmqa_parsed["texts"]
    rmap = pickle.load(open(DATA / "retrieval_map.pkl", "rb"))
    canvas_dir = DATA / "canvases_smart"

    out_dir = OUTPUT_ROOT / "mmqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    MAX_SAMPLES = 500
    for i in tqdm(range(len(dev)), desc="MMQA"):
        if len(results) >= MAX_SAMPLES: break
        if str(i) in results: continue

        item = dev[i]
        q = item["question"]
        gt_answers = item["answers"]
        gt = gt_answers[0]["answer"] if gt_answers and isinstance(gt_answers[0], dict) else (gt_answers[0] if gt_answers else "")

        # Resolve supporting context properly
        ctx = _resolve_mmqa_context(item, tables, texts)

        # Baseline (with resolved context)
        prompt_b = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer with ONLY the answer:" if ctx else f"Question: {q}\nAnswer with ONLY the answer:"
        pred_b = extract_short_answer(gpt_call([{"role": "user", "content": prompt_b}]))

        # MemCanvas (canvases + context)
        content_m = []
        for cidx, sim in rmap.get(i, [])[:2]:
            cfile = canvas_dir / f"{cidx:05d}.png"
            if cfile.exists():
                content_m.append({"type": "image_url", "image_url": {"url": img_to_b64_url(cfile)}})
        # Include context in MemCanvas condition too (matching Qwen eval)
        prompt_m = f"Above are memory canvases from similar solved questions.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer with ONLY the answer:" if ctx else f"Above are memory canvases from similar solved questions.\nQuestion: {q}\nAnswer with ONLY the answer:"
        content_m.append({"type": "text", "text": prompt_m})
        pred_m = extract_short_answer(gpt_call([{"role": "user", "content": content_m}]))

        results[str(i)] = {
            "gt": gt, "pred_b": pred_b, "pred_m": pred_m,
            "em_b": compute_em(pred_b, gt), "em_m": compute_em(pred_m, gt),
            "f1_b": compute_f1(pred_b, gt), "f1_m": compute_f1(pred_m, gt),
        }
        if len(results) % 20 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    save_summary_dual(results, out_dir, ["em", "f1"])

# ============================================================
# LoCoMo
# ============================================================
def eval_locomo():
    print("=== LoCoMo GPT-4o-mini ===")
    LOCO_DIR = Path("/home/cyf/codex/locomo_experiment_v3")
    canvas_dir = LOCO_DIR / "canvases"

    DATA_PATH = "/home/cyf/codex/datasets/locomo/locomo10.json"
    with open(DATA_PATH) as f:
        data = json.load(f)

    out_dir = OUTPUT_ROOT / "locomo"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    # Build per-conversation canvas file lists
    conv_canvases = {}
    for item in data:
        sid = item["sample_id"]
        cfiles = sorted(canvas_dir.glob(f"{sid}_*.png"))
        conv_canvases[sid] = cfiles

    MAX_SAMPLES = 500
    for item in tqdm(data, desc="LoCoMo conversations"):
        if len(results) >= MAX_SAMPLES: break
        sid = item["sample_id"]
        cfiles = conv_canvases.get(sid, [])

        for qi, qa in enumerate(item.get("qa", [])):
            key = f"{sid}_{qi}"
            if key in results: continue

            q = qa["question"]
            gt = qa.get("answer", qa.get("adversarial_answer", ""))
            if not gt: continue

            # Zero-shot baseline
            pred_b = extract_short_answer(gpt_call([{"role": "user", "content":
                f"Question: {q}\nAnswer with ONLY the answer:"}], max_tokens=64))

            # MemCanvas — use canvases from this conversation
            content_m = []
            selected = cfiles[:6] if len(cfiles) <= 6 else [cfiles[j] for j in np.linspace(0, len(cfiles)-1, 6, dtype=int)]
            for cf in selected:
                content_m.append({"type": "image_url", "image_url": {"url": img_to_b64_url(cf)}})
            content_m.append({"type": "text", "text":
                f"Above are memory canvases summarizing prior conversation sessions.\nQuestion: {q}\nAnswer with ONLY the answer:"})
            pred_m = extract_short_answer(gpt_call([{"role": "user", "content": content_m}], max_tokens=128))

            results[key] = {
                "gt": gt, "pred_b": pred_b, "pred_m": pred_m,
                "f1_b": compute_f1(pred_b, gt), "f1_m": compute_f1(pred_m, gt),
                "bleu1_b": compute_bleu1(pred_b, gt), "bleu1_m": compute_bleu1(pred_m, gt),
                "em_b": compute_em(pred_b, gt), "em_m": compute_em(pred_m, gt),
            }
            if len(results) % 20 == 0:
                json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    save_summary_dual(results, out_dir, ["f1", "bleu1"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True,
                        choices=["hotpotqa", "scienceqa", "okvqa", "mmqa", "locomo", "all"])
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    benchmarks = ["hotpotqa", "scienceqa", "okvqa", "mmqa", "locomo"] if args.benchmark == "all" else [args.benchmark]
    fns = {
        "hotpotqa": eval_hotpotqa, "scienceqa": eval_scienceqa,
        "okvqa": eval_okvqa, "mmqa": eval_mmqa,
        "locomo": eval_locomo,
    }
    for b in benchmarks:
        fns[b]()

if __name__ == "__main__":
    main()
