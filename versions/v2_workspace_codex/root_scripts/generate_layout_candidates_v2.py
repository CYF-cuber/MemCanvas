#!/usr/bin/env python3
"""
Generate clean layout candidates for all benchmarks.
No annotation bars — pure canvas images only.
3 samples per benchmark × multiple candidate layouts.
"""
import io, json, os, pickle, sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, "/home/cyf/codex")
from smart_canvas_layout import (
    measure_text, measure_image, measure_table,
    layout_single_column, layout_two_column, layout_side_by_side,
    render_layout, choose_best_layout, ContentBlock, BlockType,
)

# Canvas0415 params
FONT_HEADER = 14
FONT_BODY = 16
FONT_QA = 18
FONT_HINT = 16
FONT_SOLUTION = 15
FONT_TABLE = 14
REF_WIDTH = 800
CANVAS_WIDTH = 830
IMG_MAX_DIM = 800

OUT_DIR = Path("/home/cyf/codex/layout_candidates_v2")


def generate_candidates(blocks):
    """Generate multiple layout candidates."""
    candidates = []
    for w in [600, 700, 830, 1000, 1200]:
        try:
            layout = layout_single_column(blocks, target_width=w)
            candidates.append((f"Single_w{w}", layout))
        except Exception:
            pass
    for w in [800, 1000, 1200]:
        try:
            layout = layout_two_column(blocks, target_width=w)
            candidates.append((f"TwoCol_w{w}", layout))
        except Exception:
            pass
    for w in [800, 1000, 1200]:
        try:
            layout = layout_side_by_side(blocks, target_width=w)
            candidates.append((f"SideBySide_w{w}", layout))
        except Exception:
            pass
    try:
        layout = choose_best_layout(blocks)
        candidates.append(("AutoBest", layout))
    except Exception:
        pass
    return candidates


def save_candidates(candidates, sample_dir, label):
    """Save candidate images (no annotation bars)."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    for j, (name, layout) in enumerate(candidates):
        img = render_layout(layout)
        path = sample_dir / f"{j:02d}_{name}.png"
        img.save(path)
        print(f"  [{j}] {name}: {layout.width}x{layout.height}")


# ============================================================
# MMQA
# ============================================================
def run_mmqa():
    print("\n=== MMQA ===")
    cache_file = Path("/home/cyf/codex/mmqa_data/mmqa_parsed.pkl")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    train = data["train"]
    tables = data["tables"]
    texts = data["texts"]
    images_meta = data["images"]
    IMAGES_DIR = Path("/home/cyf/codex/mmqa_data/final_dataset_images")

    # Find samples with image+table
    indices = []
    for i, s in enumerate(train):
        ctx_types = {ctx["doc_part"] for ctx in s.get("supporting_context", [])}
        if "image" in ctx_types and "table" in ctx_types:
            indices.append(i)
            if len(indices) >= 3:
                break
    print(f"  Samples: {indices}")

    for idx in indices:
        s = train[idx]
        blocks = []

        modalities = s.get("metadata", {}).get("modalities", [])
        qtype = s.get("metadata", {}).get("type", "")
        if qtype or modalities:
            blocks.append(measure_text(f"[{qtype}] Modalities: {', '.join(modalities)}", font_size=FONT_HEADER, ref_width=REF_WIDTH))

        for ctx in s.get("supporting_context", [])[:3]:
            doc_id, doc_part = ctx["doc_id"], ctx["doc_part"]
            if doc_part == "text" and doc_id in texts:
                td = texts[doc_id]
                blocks.append(measure_text(f"[Text] {td.get('title','')}\n{td.get('text','')[:300]}", font_size=FONT_BODY, ref_width=REF_WIDTH))
            elif doc_part == "table" and doc_id in tables:
                td = tables[doc_id]
                if td.get("title"):
                    blocks.append(measure_text(f"[Table] {td['title']}", font_size=FONT_HEADER, ref_width=REF_WIDTH))
                headers = [h["column_name"] for h in td["table"]["header"]]
                rows = [[cell["text"][:40] for cell in row] for row in td["table"]["table_rows"][:8]]
                table_data = [headers] + rows
                if len(td["table"]["table_rows"]) > 8:
                    table_data.append([f"...({len(td['table']['table_rows'])} rows)"] + [""] * (len(headers) - 1))
                blocks.append(measure_table(table_data, font_size=FONT_TABLE))
            elif doc_part == "image" and doc_id in images_meta:
                img_info = images_meta[doc_id]
                img_path = IMAGES_DIR / img_info["path"]
                if img_path.exists():
                    try:
                        img = Image.open(img_path).convert("RGB")
                        blocks.append(measure_text(f"[Image] {img_info.get('title','')}", font_size=FONT_HEADER, ref_width=REF_WIDTH))
                        blocks.append(measure_image(img, max_dim=IMG_MAX_DIM))
                    except Exception:
                        pass

        q = s.get("question", "")
        answers = s.get("answers", [])
        ans = str(answers[0]["answer"]) if answers else ""
        blocks.append(measure_text(f"Q: {q}\n>>> A: {ans}", font_size=FONT_QA, ref_width=REF_WIDTH))

        candidates = generate_candidates(blocks)
        save_candidates(candidates, OUT_DIR / f"mmqa_{idx}", f"MMQA sample {idx}")


# ============================================================
# ScienceQA
# ============================================================
def run_scienceqa():
    print("\n=== ScienceQA ===")
    cache = Path("/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl")
    with open(cache, "rb") as f:
        data = pickle.load(f)
    train = data.get("train", data) if isinstance(data, dict) else data[0] if isinstance(data, (list, tuple)) else data

    from datasets import load_dataset
    hf_ds = load_dataset("derek-thomas/ScienceQA", split="train")

    # Find 3 samples with images + lecture
    indices = []
    for i, p in enumerate(train):
        has_img = i < len(hf_ds) and hf_ds[i].get("image") is not None
        has_lecture = bool((p.get("lecture", "") or "").strip())
        if has_img and has_lecture:
            indices.append(i)
            if len(indices) >= 3:
                break
    print(f"  Samples: {indices}")

    for idx in indices:
        p = train[idx]
        blocks = []

        subj = p.get("subject", "")
        topic = p.get("topic", "")
        if subj or topic:
            blocks.append(measure_text(f"[{subj}] {topic}", font_size=FONT_HEADER, ref_width=REF_WIDTH))

        hint = p.get("hint", "")
        if hint:
            blocks.append(measure_text(hint, font_size=FONT_HINT, ref_width=REF_WIDTH))

        if idx < len(hf_ds) and hf_ds[idx].get("image") is not None:
            img = hf_ds[idx]["image"].convert("RGB")
            blocks.append(measure_image(img, max_dim=IMG_MAX_DIM))

        lecture = p.get("lecture", "")
        if lecture:
            blocks.append(measure_text(lecture[:500], font_size=FONT_BODY, ref_width=REF_WIDTH))

        q = p.get("question", "")
        choices = p.get("choices", [])
        answer_idx = p.get("answer", 0)
        choice_text = "\n".join(
            f"{'>>> ' if j == answer_idx else '    '}{chr(65+j)}. {c}"
            for j, c in enumerate(choices)
        )
        blocks.append(measure_text(f"Q: {q}\n{choice_text}", font_size=FONT_QA, ref_width=REF_WIDTH))

        solution = p.get("solution", "")
        if solution:
            blocks.append(measure_text(f"Solution: {solution[:300]}", font_size=FONT_SOLUTION, ref_width=REF_WIDTH))

        candidates = generate_candidates(blocks)
        save_candidates(candidates, OUT_DIR / f"scienceqa_{idx}", f"ScienceQA sample {idx}")


# ============================================================
# OK-VQA
# ============================================================
def run_okvqa():
    print("\n=== OK-VQA ===")
    cache_file = Path("/home/cyf/codex/okvqa_data/okvqa_cached.pkl")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    train = data["train"]

    # Find 3 samples with images
    indices = []
    for i, s in enumerate(train):
        img_path = s.get("image_path", "")
        if img_path and os.path.exists(img_path):
            indices.append(i)
            if len(indices) >= 3:
                break
    print(f"  Samples: {indices}")

    for idx in indices:
        s = train[idx]
        blocks = []

        img_path = s.get("image_path", "")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            blocks.append(measure_image(img, max_dim=IMG_MAX_DIM))

        q = s.get("question", "")
        answers = s.get("answers", [])
        ans_text = ", ".join(answers[:5]) if answers else ""
        blocks.append(measure_text(f"Q: {q}\n>>> A: {ans_text}", font_size=FONT_QA, ref_width=REF_WIDTH))

        cap = s.get("caption", "")
        if cap:
            blocks.append(measure_text(f"Caption: {cap[:200]}", font_size=FONT_SOLUTION, ref_width=REF_WIDTH))

        candidates = generate_candidates(blocks)
        save_candidates(candidates, OUT_DIR / f"okvqa_{idx}", f"OK-VQA sample {idx}")


# ============================================================
# HotpotQA
# ============================================================
def run_hotpotqa():
    print("\n=== HotpotQA ===")
    cache_file = Path("/home/cyf/codex/hotpotqa_data/hotpotqa_meta.pkl")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    train = data["train"] if isinstance(data, dict) else data

    # Pick 3 samples with multiple paragraphs
    indices = []
    for i, s in enumerate(train):
        paras = s.get("paragraphs", [])
        if len(paras) >= 2:
            indices.append(i)
            if len(indices) >= 3:
                break
    print(f"  Samples: {indices}")

    for idx in indices:
        s = train[idx]
        blocks = []

        qtype = s.get("type", "")
        level = s.get("level", "")
        if qtype or level:
            blocks.append(measure_text(f"[{qtype}] Level: {level}", font_size=FONT_HEADER, ref_width=REF_WIDTH))

        for para in s.get("paragraphs", [])[:4]:
            title = para["title"]
            text = para["text"][:400]
            blocks.append(measure_text(f"[{title}]\n{text}", font_size=FONT_BODY, ref_width=REF_WIDTH))

        q = s.get("question", "")
        ans = s.get("answer", "")
        blocks.append(measure_text(f"Q: {q}\n>>> A: {ans}", font_size=FONT_QA, ref_width=REF_WIDTH))

        candidates = generate_candidates(blocks)
        save_candidates(candidates, OUT_DIR / f"hotpotqa_{idx}", f"HotpotQA sample {idx}")


# ============================================================
# LoCoMo
# ============================================================
def run_locomo():
    print("\n=== LoCoMo ===")
    data_path = Path("/home/cyf/codex/datasets/locomo/locomo10.json")
    with open(data_path) as f:
        data = json.load(f)

    # Pick 3 conversations, use session_1 of each
    indices = [0, 1, 2]
    print(f"  Samples: {indices}")

    for idx in indices:
        s = data[idx]
        conv = s["conversation"]
        sample_id = s.get("sample_id", f"conv-{idx}")
        session = conv.get("session_1", [])
        session_date = conv.get("session_1_date_time", "")
        speaker_a = conv.get("speaker_a", "A")
        speaker_b = conv.get("speaker_b", "B")

        blocks = []
        header = f"[{sample_id}] Session 1"
        if session_date:
            header += f" | {session_date}"
        blocks.append(measure_text(header, font_size=FONT_HEADER, ref_width=REF_WIDTH))

        for msg in session[:12]:  # first 12 turns
            speaker = msg.get("speaker", "Unknown")
            text = msg.get("text", "")
            if len(text) > 300:
                text = text[:297] + "..."
            blocks.append(measure_text(f"[{speaker}] {text}", font_size=FONT_BODY, ref_width=REF_WIDTH))

        candidates = generate_candidates(blocks)
        save_candidates(candidates, OUT_DIR / f"locomo_{idx}", f"LoCoMo sample {idx}")


# ============================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_mmqa()
    run_scienceqa()
    run_okvqa()
    run_hotpotqa()
    run_locomo()
    print(f"\nAll candidates saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
