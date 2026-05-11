#!/usr/bin/env python3
"""
Demo: Use Qwen3-4B to compress text, then render canvases.
Processes a few samples from each benchmark to show compression effect.
"""
import io, json, os, pickle, sys, torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, "/home/cyf/codex")
from smart_canvas_layout import (
    measure_text, measure_image, measure_table,
    layout_single_column, choose_best_layout, render_layout,
)

# Canvas params
FONT_HEADER = 14
FONT_BODY = 16
FONT_QA = 18
FONT_HINT = 16
FONT_SOLUTION = 15
FONT_TABLE = 14
REF_WIDTH = 800
CANVAS_WIDTH = 830
IMG_MAX_DIM = 800

OUT_DIR = Path("/home/cyf/codex/compressed_canvas_demo")

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
            model_name, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
        )
        self.model.eval()
        print(f"  Loaded on {device}, {sum(p.numel() for p in self.model.parameters())/1e9:.1f}B params")

    @torch.no_grad()
    def compress(self, text, max_new_tokens=256):
        messages = [
            {"role": "system", "content": COMPRESS_PROMPT},
            {"role": "user", "content": f"Compress the following text:\n\n{text[:800]}\n\nCompressed:"},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, top_p=0.9, do_sample=True,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return result


def render_canvas(blocks, out_path):
    layout = layout_single_column(blocks, target_width=CANVAS_WIDTH)
    img = render_layout(layout)
    img.save(out_path)
    return layout.width, layout.height


# ============================================================
# ScienceQA
# ============================================================
def run_scienceqa(compressor, n=3):
    print("\n=== ScienceQA (compressed) ===")
    cache = Path("/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl")
    with open(cache, "rb") as f:
        data = pickle.load(f)
    train = data.get("train", data) if isinstance(data, dict) else data[0] if isinstance(data, (list, tuple)) else data

    from datasets import load_dataset
    hf_ds = load_dataset("derek-thomas/ScienceQA", split="train")

    # Find samples with image + lecture
    indices = []
    for i, p in enumerate(train):
        has_img = i < len(hf_ds) and hf_ds[i].get("image") is not None
        has_lecture = bool((p.get("lecture", "") or "").strip())
        if has_img and has_lecture:
            indices.append(i)
            if len(indices) >= n:
                break

    out_dir = OUT_DIR / "scienceqa"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        p = train[idx]
        lecture = (p.get("lecture", "") or "").strip()
        solution = (p.get("solution", "") or "").strip()

        # Compress lecture + solution together
        raw_text = ""
        if lecture:
            raw_text += f"Background: {lecture}\n"
        if solution:
            raw_text += f"Solution: {solution}"
        compressed = compressor.compress(raw_text)
        ratio = len(compressed) / max(len(raw_text), 1)
        print(f"  [{idx}] {len(raw_text)} -> {len(compressed)} chars ({ratio:.0%})")
        print(f"    Original: {raw_text[:120]}...")
        print(f"    Compressed: {compressed[:120]}...")

        # --- Original canvas ---
        blocks_orig = []
        subj, topic = p.get("subject", ""), p.get("topic", "")
        if subj or topic:
            blocks_orig.append(measure_text(f"[{subj}] {topic}", font_size=FONT_HEADER, ref_width=REF_WIDTH))
        hint = p.get("hint", "")
        if hint:
            blocks_orig.append(measure_text(hint, font_size=FONT_HINT, ref_width=REF_WIDTH))
        if idx < len(hf_ds) and hf_ds[idx].get("image") is not None:
            img = hf_ds[idx]["image"].convert("RGB")
            blocks_orig.append(measure_image(img, max_dim=IMG_MAX_DIM))
        if lecture:
            blocks_orig.append(measure_text(lecture[:500], font_size=FONT_BODY, ref_width=REF_WIDTH))
        q = p.get("question", "")
        choices = p.get("choices", [])
        answer_idx = p.get("answer", 0)
        choice_text = "\n".join(f"{'>>> ' if j == answer_idx else '    '}{chr(65+j)}. {c}" for j, c in enumerate(choices))
        blocks_orig.append(measure_text(f"Q: {q}\n{choice_text}", font_size=FONT_QA, ref_width=REF_WIDTH))
        if solution:
            blocks_orig.append(measure_text(f"Solution: {solution[:300]}", font_size=FONT_SOLUTION, ref_width=REF_WIDTH))
        w1, h1 = render_canvas(blocks_orig, out_dir / f"{idx:05d}_original.png")

        # --- Compressed canvas ---
        blocks_comp = []
        if subj or topic:
            blocks_comp.append(measure_text(f"[{subj}] {topic}", font_size=FONT_HEADER, ref_width=REF_WIDTH))
        if hint:
            blocks_comp.append(measure_text(hint, font_size=FONT_HINT, ref_width=REF_WIDTH))
        if idx < len(hf_ds) and hf_ds[idx].get("image") is not None:
            img = hf_ds[idx]["image"].convert("RGB")
            blocks_comp.append(measure_image(img, max_dim=IMG_MAX_DIM))
        if compressed:
            blocks_comp.append(measure_text(compressed, font_size=FONT_BODY, ref_width=REF_WIDTH))
        blocks_comp.append(measure_text(f"Q: {q}\n{choice_text}", font_size=FONT_QA, ref_width=REF_WIDTH))
        w2, h2 = render_canvas(blocks_comp, out_dir / f"{idx:05d}_compressed.png")

        print(f"    Canvas: {w1}x{h1} -> {w2}x{h2}")


# ============================================================
# MMQA
# ============================================================
def run_mmqa(compressor, n=3):
    print("\n=== MMQA (compressed) ===")
    cache_file = Path("/home/cyf/codex/mmqa_data/mmqa_parsed.pkl")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    train, tables, texts, images_meta = data["train"], data["tables"], data["texts"], data["images"]
    IMAGES_DIR = Path("/home/cyf/codex/mmqa_data/final_dataset_images")

    # Find samples with text context
    indices = []
    for i, s in enumerate(train):
        has_text = any(ctx["doc_part"] == "text" for ctx in s.get("supporting_context", []))
        if has_text:
            indices.append(i)
            if len(indices) >= n:
                break

    out_dir = OUT_DIR / "mmqa"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        s = train[idx]
        blocks_orig, blocks_comp = [], []

        modalities = s.get("metadata", {}).get("modalities", [])
        qtype = s.get("metadata", {}).get("type", "")
        header = f"[{qtype}] Modalities: {', '.join(modalities)}" if (qtype or modalities) else ""
        if header:
            blocks_orig.append(measure_text(header, font_size=FONT_HEADER, ref_width=REF_WIDTH))
            blocks_comp.append(measure_text(header, font_size=FONT_HEADER, ref_width=REF_WIDTH))

        for ctx in s.get("supporting_context", [])[:3]:
            doc_id, doc_part = ctx["doc_id"], ctx["doc_part"]

            if doc_part == "text" and doc_id in texts:
                td = texts[doc_id]
                title = td.get("title", "")
                passage = td.get("text", "")[:300]
                raw = f"[Text] {title}\n{passage}"
                blocks_orig.append(measure_text(raw, font_size=FONT_BODY, ref_width=REF_WIDTH))
                # Compress
                compressed = compressor.compress(passage)
                ratio = len(compressed) / max(len(passage), 1)
                print(f"  [{idx}] text {len(passage)} -> {len(compressed)} ({ratio:.0%})")
                blocks_comp.append(measure_text(f"[Text] {title}\n{compressed}", font_size=FONT_BODY, ref_width=REF_WIDTH))

            elif doc_part == "table" and doc_id in tables:
                td = tables[doc_id]
                if td.get("title"):
                    blocks_orig.append(measure_text(f"[Table] {td['title']}", font_size=FONT_HEADER, ref_width=REF_WIDTH))
                    blocks_comp.append(measure_text(f"[Table] {td['title']}", font_size=FONT_HEADER, ref_width=REF_WIDTH))
                headers = [h["column_name"] for h in td["table"]["header"]]
                rows = [[cell["text"][:40] for cell in row] for row in td["table"]["table_rows"][:8]]
                table_data = [headers] + rows
                if len(td["table"]["table_rows"]) > 8:
                    table_data.append([f"...({len(td['table']['table_rows'])} rows)"] + [""] * (len(headers) - 1))
                blocks_orig.append(measure_table(table_data, font_size=FONT_TABLE))
                blocks_comp.append(measure_table(table_data, font_size=FONT_TABLE))

            elif doc_part == "image" and doc_id in images_meta:
                img_info = images_meta[doc_id]
                img_path = IMAGES_DIR / img_info["path"]
                if img_path.exists():
                    try:
                        img = Image.open(img_path).convert("RGB")
                        blocks_orig.append(measure_text(f"[Image] {img_info.get('title','')}", font_size=FONT_HEADER, ref_width=REF_WIDTH))
                        blocks_orig.append(measure_image(img, max_dim=IMG_MAX_DIM))
                        blocks_comp.append(measure_text(f"[Image] {img_info.get('title','')}", font_size=FONT_HEADER, ref_width=REF_WIDTH))
                        blocks_comp.append(measure_image(img, max_dim=IMG_MAX_DIM))
                    except Exception:
                        pass

        q = s.get("question", "")
        answers = s.get("answers", [])
        ans = str(answers[0]["answer"]) if answers else ""
        qa_block = measure_text(f"Q: {q}\n>>> A: {ans}", font_size=FONT_QA, ref_width=REF_WIDTH)
        blocks_orig.append(qa_block)
        blocks_comp.append(measure_text(f"Q: {q}\n>>> A: {ans}", font_size=FONT_QA, ref_width=REF_WIDTH))

        w1, h1 = render_canvas(blocks_orig, out_dir / f"{idx:05d}_original.png")
        w2, h2 = render_canvas(blocks_comp, out_dir / f"{idx:05d}_compressed.png")
        print(f"    Canvas: {w1}x{h1} -> {w2}x{h2}")


# ============================================================
# HotpotQA
# ============================================================
def run_hotpotqa(compressor, n=3):
    print("\n=== HotpotQA (compressed) ===")
    cache_file = Path("/home/cyf/codex/hotpotqa_data/hotpotqa_meta.pkl")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    train = data["train"] if isinstance(data, dict) else data

    indices = []
    for i, s in enumerate(train):
        if len(s.get("paragraphs", [])) >= 2:
            indices.append(i)
            if len(indices) >= n:
                break

    out_dir = OUT_DIR / "hotpotqa"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        s = train[idx]
        blocks_orig, blocks_comp = [], []

        qtype, level = s.get("type", ""), s.get("level", "")
        if qtype or level:
            hdr = measure_text(f"[{qtype}] Level: {level}", font_size=FONT_HEADER, ref_width=REF_WIDTH)
            blocks_orig.append(hdr)
            blocks_comp.append(measure_text(f"[{qtype}] Level: {level}", font_size=FONT_HEADER, ref_width=REF_WIDTH))

        for para in s.get("paragraphs", [])[:4]:
            title = para["title"]
            text = para["text"][:400]
            blocks_orig.append(measure_text(f"[{title}]\n{text}", font_size=FONT_BODY, ref_width=REF_WIDTH))
            # Compress each paragraph
            compressed = compressor.compress(text)
            ratio = len(compressed) / max(len(text), 1)
            print(f"  [{idx}] para '{title[:20]}' {len(text)} -> {len(compressed)} ({ratio:.0%})")
            blocks_comp.append(measure_text(f"[{title}]\n{compressed}", font_size=FONT_BODY, ref_width=REF_WIDTH))

        q, ans = s.get("question", ""), s.get("answer", "")
        blocks_orig.append(measure_text(f"Q: {q}\n>>> A: {ans}", font_size=FONT_QA, ref_width=REF_WIDTH))
        blocks_comp.append(measure_text(f"Q: {q}\n>>> A: {ans}", font_size=FONT_QA, ref_width=REF_WIDTH))

        w1, h1 = render_canvas(blocks_orig, out_dir / f"{idx:05d}_original.png")
        w2, h2 = render_canvas(blocks_comp, out_dir / f"{idx:05d}_compressed.png")
        print(f"    Canvas: {w1}x{h1} -> {w2}x{h2}")


# ============================================================
# LoCoMo
# ============================================================
def run_locomo(compressor, n=3):
    print("\n=== LoCoMo (compressed) ===")
    data_path = Path("/home/cyf/codex/datasets/locomo/locomo10.json")
    with open(data_path) as f:
        data = json.load(f)

    out_dir = OUT_DIR / "locomo"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(min(n, len(data))):
        s = data[idx]
        conv = s["conversation"]
        sample_id = s.get("sample_id", f"conv-{idx}")
        session = conv.get("session_1", [])
        session_date = conv.get("session_1_date_time", "")

        # Build full conversation text
        conv_lines = []
        for msg in session[:12]:
            speaker = msg.get("speaker", "Unknown")
            text = msg.get("text", "")
            if len(text) > 300:
                text = text[:297] + "..."
            conv_lines.append(f"[{speaker}] {text}")

        full_text = "\n".join(conv_lines)

        # --- Original canvas ---
        blocks_orig = []
        header = f"[{sample_id}] Session 1"
        if session_date:
            header += f" | {session_date}"
        blocks_orig.append(measure_text(header, font_size=FONT_HEADER, ref_width=REF_WIDTH))
        for line in conv_lines:
            blocks_orig.append(measure_text(line, font_size=FONT_BODY, ref_width=REF_WIDTH))

        # --- Compressed canvas ---
        compressed = compressor.compress(full_text)
        ratio = len(compressed) / max(len(full_text), 1)
        print(f"  [{idx}] conv {len(full_text)} -> {len(compressed)} ({ratio:.0%})")
        print(f"    Compressed: {compressed[:150]}...")

        blocks_comp = []
        blocks_comp.append(measure_text(header, font_size=FONT_HEADER, ref_width=REF_WIDTH))
        blocks_comp.append(measure_text(compressed, font_size=FONT_BODY, ref_width=REF_WIDTH))

        w1, h1 = render_canvas(blocks_orig, out_dir / f"{idx:05d}_original.png")
        w2, h2 = render_canvas(blocks_comp, out_dir / f"{idx:05d}_compressed.png")
        print(f"    Canvas: {w1}x{h1} -> {w2}x{h2}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    compressor = Compressor("Qwen/Qwen3-4B", device="cuda")

    run_scienceqa(compressor, n=3)
    run_mmqa(compressor, n=3)
    run_hotpotqa(compressor, n=3)
    run_locomo(compressor, n=3)

    print(f"\nAll done! Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
