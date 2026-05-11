#!/usr/bin/env python3
"""
Generate compressed layout candidates for the same samples as layout_candidates_v2.
Uses Qwen3-4B to compress text, then generates 12 layout candidates per sample.
"""
import io, json, os, pickle, sys, torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, "/home/cyf/codex")
from smart_canvas_layout import (
    measure_text, measure_image, measure_table,
    layout_single_column, layout_two_column, layout_side_by_side,
    render_layout, choose_best_layout,
)

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

COMPRESS_PROMPT = (
    "You are a text compressor. Compress the following text into the shortest "
    "possible form while preserving ALL key factual information needed to answer "
    "questions. Use concise notation: 'entity: key facts' format. Remove filler, "
    "background, and redundant info. Output only the compressed text, nothing else."
)


class Compressor:
    def __init__(self, model_name="Qwen/Qwen3-4B", device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=device, trust_remote_code=True
        )
        self.model.eval()
        print(f"  Loaded, {sum(p.numel() for p in self.model.parameters())/1e9:.1f}B params")

    @torch.no_grad()
    def compress(self, text, max_new_tokens=256):
        messages = [
            {"role": "system", "content": COMPRESS_PROMPT},
            {"role": "user", "content": f"Compress the following text:\n\n{text[:800]}\n\nCompressed:"},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, top_p=0.9, do_sample=True,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_candidates(blocks):
    candidates = []
    for w in [600, 700, 830, 1000, 1200]:
        try:
            candidates.append((f"Single_w{w}", layout_single_column(blocks, target_width=w)))
        except Exception:
            pass
    for w in [800, 1000, 1200]:
        try:
            candidates.append((f"TwoCol_w{w}", layout_two_column(blocks, target_width=w)))
        except Exception:
            pass
    for w in [800, 1000, 1200]:
        try:
            candidates.append((f"SideBySide_w{w}", layout_side_by_side(blocks, target_width=w)))
        except Exception:
            pass
    try:
        candidates.append(("AutoBest", choose_best_layout(blocks)))
    except Exception:
        pass
    return candidates


def save_candidates(candidates, sample_dir):
    sample_dir.mkdir(parents=True, exist_ok=True)
    for j, (name, layout) in enumerate(candidates):
        img = render_layout(layout)
        img.save(sample_dir / f"{j:02d}_{name}.png")
        print(f"    [{j}] {name}: {layout.width}x{layout.height}")


# ============================================================
# ScienceQA — same indices [0,1,2]
# ============================================================
def run_scienceqa(comp):
    print("\n=== ScienceQA (compressed candidates) ===")
    cache = Path("/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl")
    with open(cache, "rb") as f:
        data = pickle.load(f)
    train = data.get("train", data) if isinstance(data, dict) else data[0] if isinstance(data, (list, tuple)) else data

    from datasets import load_dataset
    hf_ds = load_dataset("derek-thomas/ScienceQA", split="train")

    indices = []
    for i, p in enumerate(train):
        has_img = i < len(hf_ds) and hf_ds[i].get("image") is not None
        has_lecture = bool((p.get("lecture", "") or "").strip())
        if has_img and has_lecture:
            indices.append(i)
            if len(indices) >= 3:
                break

    for idx in indices:
        p = train[idx]
        lecture = (p.get("lecture", "") or "").strip()
        solution = (p.get("solution", "") or "").strip()
        raw = ""
        if lecture: raw += f"Background: {lecture}\n"
        if solution: raw += f"Solution: {solution}"
        compressed = comp.compress(raw)
        print(f"  [{idx}] {len(raw)} -> {len(compressed)} chars ({len(compressed)/max(len(raw),1):.0%})")

        blocks = []
        subj, topic = p.get("subject", ""), p.get("topic", "")
        if subj or topic:
            blocks.append(measure_text(f"[{subj}] {topic}", font_size=FONT_HEADER, ref_width=REF_WIDTH))
        hint = p.get("hint", "")
        if hint:
            blocks.append(measure_text(hint, font_size=FONT_HINT, ref_width=REF_WIDTH))
        if idx < len(hf_ds) and hf_ds[idx].get("image") is not None:
            blocks.append(measure_image(hf_ds[idx]["image"].convert("RGB"), max_dim=IMG_MAX_DIM))
        if compressed:
            blocks.append(measure_text(compressed, font_size=FONT_BODY, ref_width=REF_WIDTH))
        q, choices, answer_idx = p.get("question", ""), p.get("choices", []), p.get("answer", 0)
        choice_text = "\n".join(f"{'>>> ' if j == answer_idx else '    '}{chr(65+j)}. {c}" for j, c in enumerate(choices))
        blocks.append(measure_text(f"Q: {q}\n{choice_text}", font_size=FONT_QA, ref_width=REF_WIDTH))

        candidates = generate_candidates(blocks)
        save_candidates(candidates, OUT_DIR / f"scienceqa_{idx}_compressed")


# ============================================================
# MMQA — same indices [17,27,29] (image+table samples)
# ============================================================
def run_mmqa(comp):
    print("\n=== MMQA (compressed candidates) ===")
    cache_file = Path("/home/cyf/codex/mmqa_data/mmqa_parsed.pkl")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    train, tables, texts, images_meta = data["train"], data["tables"], data["texts"], data["images"]
    IMAGES_DIR = Path("/home/cyf/codex/mmqa_data/final_dataset_images")

    indices = []
    for i, s in enumerate(train):
        ctx_types = {ctx["doc_part"] for ctx in s.get("supporting_context", [])}
        if "image" in ctx_types and "table" in ctx_types:
            indices.append(i)
            if len(indices) >= 3:
                break

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
                passage = td.get("text", "")[:300]
                compressed = comp.compress(passage)
                print(f"  [{idx}] text {len(passage)} -> {len(compressed)} ({len(compressed)/max(len(passage),1):.0%})")
                blocks.append(measure_text(f"[Text] {td.get('title','')}\n{compressed}", font_size=FONT_BODY, ref_width=REF_WIDTH))
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
        save_candidates(candidates, OUT_DIR / f"mmqa_{idx}_compressed")


# ============================================================
# HotpotQA — same indices [0,1,2]
# ============================================================
def run_hotpotqa(comp):
    print("\n=== HotpotQA (compressed candidates) ===")
    cache_file = Path("/home/cyf/codex/hotpotqa_data/hotpotqa_meta.pkl")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    train = data["train"] if isinstance(data, dict) else data

    indices = []
    for i, s in enumerate(train):
        if len(s.get("paragraphs", [])) >= 2:
            indices.append(i)
            if len(indices) >= 3:
                break

    for idx in indices:
        s = train[idx]
        blocks = []

        qtype, level = s.get("type", ""), s.get("level", "")
        if qtype or level:
            blocks.append(measure_text(f"[{qtype}] Level: {level}", font_size=FONT_HEADER, ref_width=REF_WIDTH))

        for para in s.get("paragraphs", [])[:4]:
            title, text = para["title"], para["text"][:400]
            compressed = comp.compress(text)
            print(f"  [{idx}] '{title[:20]}' {len(text)} -> {len(compressed)} ({len(compressed)/max(len(text),1):.0%})")
            blocks.append(measure_text(f"[{title}]\n{compressed}", font_size=FONT_BODY, ref_width=REF_WIDTH))

        q, ans = s.get("question", ""), s.get("answer", "")
        blocks.append(measure_text(f"Q: {q}\n>>> A: {ans}", font_size=FONT_QA, ref_width=REF_WIDTH))

        candidates = generate_candidates(blocks)
        save_candidates(candidates, OUT_DIR / f"hotpotqa_{idx}_compressed")


# ============================================================
# LoCoMo — same indices [0,1,2]
# ============================================================
def run_locomo(comp):
    print("\n=== LoCoMo (compressed candidates) ===")
    data_path = Path("/home/cyf/codex/datasets/locomo/locomo10.json")
    with open(data_path) as f:
        data = json.load(f)

    for idx in range(3):
        s = data[idx]
        conv = s["conversation"]
        sample_id = s.get("sample_id", f"conv-{idx}")
        session = conv.get("session_1", [])
        session_date = conv.get("session_1_date_time", "")

        conv_lines = []
        for msg in session[:12]:
            speaker = msg.get("speaker", "Unknown")
            text = msg.get("text", "")
            if len(text) > 300:
                text = text[:297] + "..."
            conv_lines.append(f"[{speaker}] {text}")
        full_text = "\n".join(conv_lines)

        compressed = comp.compress(full_text)
        print(f"  [{idx}] conv {len(full_text)} -> {len(compressed)} ({len(compressed)/max(len(full_text),1):.0%})")

        blocks = []
        header = f"[{sample_id}] Session 1"
        if session_date:
            header += f" | {session_date}"
        blocks.append(measure_text(header, font_size=FONT_HEADER, ref_width=REF_WIDTH))
        blocks.append(measure_text(compressed, font_size=FONT_BODY, ref_width=REF_WIDTH))

        candidates = generate_candidates(blocks)
        save_candidates(candidates, OUT_DIR / f"locomo_{idx}_compressed")


def main():
    comp = Compressor("Qwen/Qwen3-4B", device="cuda")
    run_scienceqa(comp)
    run_mmqa(comp)
    run_hotpotqa(comp)
    run_locomo(comp)
    print(f"\nDone! Compressed candidates in {OUT_DIR}/")


if __name__ == "__main__":
    main()
