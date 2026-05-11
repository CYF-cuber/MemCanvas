#!/usr/bin/env python3
"""
Compress text with GRPO-trained LoRA compressor, then render canvas0415 canvases.

Uses Qwen3-4B base + LoRA adapter (GRPO v2 step 300, val=0.6745) to compress
lecture+solution text in ScienceQA, then renders single-column canvases and
computes CLIP image embeddings.

Usage:
    CUDA_VISIBLE_DEVICES=1 python compress_and_render_canvas0415.py
    CUDA_VISIBLE_DEVICES=1 python compress_and_render_canvas0415.py --embed
"""
import argparse, io, json, os, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/codex")
from smart_canvas_layout import (
    measure_text, measure_image, measure_table, layout_single_column, render_layout,
)

# Canvas0415 parameters
FONT_HEADER = 14
FONT_HINT = 16
FONT_BODY = 16
FONT_QA = 18
FONT_SOLUTION = 15
FONT_TABLE = 14
REF_WIDTH = 800
CANVAS_WIDTH = 830
IMG_MAX_DIM = 800

# Paths
BASE_MODEL = "/home/cyf/codex/agent_sft_merged"
LORA_ADAPTER = "/home/cyf/codex/checkpoints/memcanvas_compress_grpo_v2/qwen3_4b_compress_grpo_v2_vlm_reward/global_step_300/actor/lora_adapter"
TOKENIZER_PATH = "/home/cyf/codex/checkpoints/memcanvas_compress_grpo_v2/qwen3_4b_compress_grpo_v2_vlm_reward/global_step_300/actor/huggingface"

OUTPUT_DIR = Path("/home/cyf/codex/canvas0415_compressed/scienceqa")

COMPRESS_SYSTEM_PROMPT = (
    "You are a text compressor. Given a passage of text, compress it into "
    "the shortest possible form while preserving ALL key factual information "
    "needed to answer questions. Use concise notation: "
    "'entity: key facts' format. Remove filler, background, and redundant info. "
    "IMPORTANT: The compressed text will be rendered on a canvas image and read by a VLM. "
    "Keep text clear and well-structured so it remains readable when rendered. "
    "Output only the compressed text, nothing else."
)


class LoRACompressor:
    """Qwen3-4B + GRPO-trained LoRA adapter for text compression."""

    def __init__(self, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print(f"Loading base model: {BASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.float16, device_map=device, trust_remote_code=True
        )
        print(f"Loading LoRA adapter: {LORA_ADAPTER}")
        self.model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"  Loaded: {n_params:.1f}B params (base + LoRA)")

    @torch.no_grad()
    def compress(self, text, max_new_tokens=256):
        if not text or not text.strip():
            return ""
        messages = [
            {"role": "system", "content": COMPRESS_SYSTEM_PROMPT},
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


def render_scienceqa_compressed(comp: LoRACompressor):
    """Render all ScienceQA canvases with compressed lecture+solution."""
    print("\n=== ScienceQA Compressed Canvas0415 ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    cache = Path("/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl")
    with open(cache, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        train = data.get("train", data)
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        train, _ = data
    else:
        train = data

    from datasets import load_dataset
    hf_ds = load_dataset("derek-thomas/ScienceQA", split="train")

    n = len(train)
    print(f"  Rendering {n} compressed canvases...")

    # Track compression stats
    compress_count = 0
    total_orig_len = 0
    total_comp_len = 0

    # Checkpoint: save progress every 500 samples
    progress_file = OUTPUT_DIR / "progress.json"
    start_idx = 0
    if progress_file.exists():
        with open(progress_file) as f:
            prog = json.load(f)
        start_idx = prog.get("last_completed", 0) + 1
        compress_count = prog.get("compress_count", 0)
        total_orig_len = prog.get("total_orig_len", 0)
        total_comp_len = prog.get("total_comp_len", 0)
        print(f"  Resuming from index {start_idx} ({start_idx}/{n} done)")

    t0 = time.time()
    for i in tqdm(range(start_idx, n), initial=start_idx, total=n, desc="Compress+Render"):
        out = OUTPUT_DIR / f"{i:05d}.png"

        p = train[i]
        blocks = []

        # Header
        subj = p.get("subject", "")
        topic = p.get("topic", "")
        if subj or topic:
            blocks.append(measure_text(f"[{subj}] {topic}", font_size=FONT_HEADER, ref_width=REF_WIDTH))

        # Hint
        hint = p.get("hint", "")
        if hint:
            blocks.append(measure_text(hint, font_size=FONT_HINT, ref_width=REF_WIDTH))

        # Image
        if i < len(hf_ds) and hf_ds[i].get("image") is not None:
            img = hf_ds[i]["image"].convert("RGB")
            blocks.append(measure_image(img, max_dim=IMG_MAX_DIM))

        # Compress lecture + solution together
        lecture = (p.get("lecture", "") or "").strip()
        solution = (p.get("solution", "") or "").strip()
        raw_text = ""
        if lecture:
            raw_text += f"Background: {lecture}\n"
        if solution:
            raw_text += f"Solution: {solution}"
        raw_text = raw_text.strip()

        if raw_text:
            compressed = comp.compress(raw_text)
            total_orig_len += len(raw_text)
            total_comp_len += len(compressed)
            compress_count += 1
            if compressed:
                blocks.append(measure_text(compressed, font_size=FONT_BODY, ref_width=REF_WIDTH))
        else:
            # No text to compress — same as original canvas0415
            pass

        # Question + choices + answer
        q = p.get("question", "")
        choices = p.get("choices", [])
        answer_idx = p.get("answer", 0)
        choice_text = "\n".join(
            f"{'>>> ' if j == answer_idx else '    '}{chr(65+j)}. {c}"
            for j, c in enumerate(choices)
        )
        blocks.append(measure_text(f"Q: {q}\n{choice_text}", font_size=FONT_QA, ref_width=REF_WIDTH))

        if not blocks:
            blocks.append(measure_text("(empty)", font_size=FONT_BODY, ref_width=REF_WIDTH))

        # Render single-column canvas
        layout = layout_single_column(blocks, target_width=CANVAS_WIDTH)
        img_out = render_layout(layout)
        buf = io.BytesIO()
        img_out.save(buf, format="PNG", optimize=True)
        out.write_bytes(buf.getvalue())

        # Checkpoint every 500
        if (i + 1) % 500 == 0:
            with open(progress_file, "w") as f:
                json.dump({
                    "last_completed": i,
                    "compress_count": compress_count,
                    "total_orig_len": total_orig_len,
                    "total_comp_len": total_comp_len,
                }, f)
            elapsed = time.time() - t0
            rate = (i - start_idx + 1) / elapsed
            remaining = (n - i - 1) / rate if rate > 0 else 0
            avg_ratio = total_comp_len / max(total_orig_len, 1)
            print(f"\n  [{i+1}/{n}] {rate:.1f} samples/s, ETA {remaining/60:.0f}min, "
                  f"avg compression: {avg_ratio:.0%}")

    # Final stats
    elapsed = time.time() - t0
    avg_ratio = total_comp_len / max(total_orig_len, 1)
    print(f"\n  Done: {n} canvases in {elapsed/60:.1f}min")
    print(f"  Compressed {compress_count} texts, avg ratio: {avg_ratio:.0%}")
    print(f"  Output: {OUTPUT_DIR}")

    # Save final progress
    with open(progress_file, "w") as f:
        json.dump({
            "last_completed": n - 1,
            "compress_count": compress_count,
            "total_orig_len": total_orig_len,
            "total_comp_len": total_comp_len,
            "elapsed_seconds": elapsed,
        }, f, indent=2)

    # Write done marker
    (OUTPUT_DIR / "done.txt").write_text(str(n))
    return n


def render_okvqa_compressed(comp: LoRACompressor):
    """Render all OK-VQA canvases with compressed captions."""
    print("\n=== OK-VQA Compressed Canvas0415 ===")
    out_dir = Path("/home/cyf/codex/canvas0415_compressed/okvqa")
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = Path("/home/cyf/codex/okvqa_data/okvqa_cached.pkl")
    with open(cache, "rb") as f:
        data = pickle.load(f)
    train = data["train"]

    n = len(train)
    print(f"  Rendering {n} compressed canvases...")

    compress_count = 0
    total_orig_len = 0
    total_comp_len = 0

    progress_file = out_dir / "progress.json"
    start_idx = 0
    if progress_file.exists():
        with open(progress_file) as f:
            prog = json.load(f)
        start_idx = prog.get("last_completed", 0) + 1
        compress_count = prog.get("compress_count", 0)
        total_orig_len = prog.get("total_orig_len", 0)
        total_comp_len = prog.get("total_comp_len", 0)
        print(f"  Resuming from index {start_idx} ({start_idx}/{n} done)")

    t0 = time.time()
    for i in tqdm(range(start_idx, n), initial=start_idx, total=n, desc="OK-VQA Compress+Render"):
        out = out_dir / f"{i:05d}.png"
        s = train[i]
        blocks = []

        # Image
        img_path = s.get("image_path", "")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            blocks.append(measure_image(img, max_dim=IMG_MAX_DIM))

        # Question + answers
        q = s.get("question", "")
        answers = s.get("answers", [])
        ans_text = ", ".join(answers[:5]) if answers else ""
        blocks.append(measure_text(f"Q: {q}\n>>> A: {ans_text}", font_size=FONT_QA, ref_width=REF_WIDTH))

        # Compress caption
        cap = (s.get("caption", "") or "").strip()
        if cap:
            compressed = comp.compress(cap)
            total_orig_len += len(cap)
            total_comp_len += len(compressed)
            compress_count += 1
            if compressed:
                blocks.append(measure_text(f"Caption: {compressed}", font_size=FONT_SOLUTION, ref_width=REF_WIDTH))

        if not blocks:
            blocks.append(measure_text("(empty)", font_size=FONT_BODY, ref_width=REF_WIDTH))

        layout = layout_single_column(blocks, target_width=CANVAS_WIDTH)
        img_out = render_layout(layout)
        buf = io.BytesIO()
        img_out.save(buf, format="PNG", optimize=True)
        out.write_bytes(buf.getvalue())

        if (i + 1) % 500 == 0:
            with open(progress_file, "w") as f:
                json.dump({"last_completed": i, "compress_count": compress_count,
                           "total_orig_len": total_orig_len, "total_comp_len": total_comp_len}, f)
            elapsed = time.time() - t0
            rate = (i - start_idx + 1) / elapsed
            remaining = (n - i - 1) / rate if rate > 0 else 0
            avg_ratio = total_comp_len / max(total_orig_len, 1)
            print(f"\n  [{i+1}/{n}] {rate:.1f} samples/s, ETA {remaining/60:.0f}min, "
                  f"avg compression: {avg_ratio:.0%}")

    elapsed = time.time() - t0
    avg_ratio = total_comp_len / max(total_orig_len, 1)
    print(f"\n  Done: {n} canvases in {elapsed/60:.1f}min")
    print(f"  Compressed {compress_count} texts, avg ratio: {avg_ratio:.0%}")
    with open(progress_file, "w") as f:
        json.dump({"last_completed": n - 1, "compress_count": compress_count,
                   "total_orig_len": total_orig_len, "total_comp_len": total_comp_len,
                   "elapsed_seconds": elapsed}, f, indent=2)
    (out_dir / "done.txt").write_text(str(n))
    return n


def render_mmqa_compressed(comp: LoRACompressor):
    """Render all MMQA canvases with compressed text passages."""
    print("\n=== MMQA Compressed Canvas0415 ===")
    out_dir = Path("/home/cyf/codex/canvas0415_compressed/mmqa")
    out_dir.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR = Path("/home/cyf/codex/mmqa_data/final_dataset_images")

    cache = Path("/home/cyf/codex/mmqa_data/mmqa_parsed.pkl")
    with open(cache, "rb") as f:
        data = pickle.load(f)
    train, tables, texts, images_meta = data["train"], data["tables"], data["texts"], data["images"]

    n = len(train)
    print(f"  Rendering {n} compressed canvases...")

    compress_count = 0
    total_orig_len = 0
    total_comp_len = 0

    progress_file = out_dir / "progress.json"
    start_idx = 0
    if progress_file.exists():
        with open(progress_file) as f:
            prog = json.load(f)
        start_idx = prog.get("last_completed", 0) + 1
        compress_count = prog.get("compress_count", 0)
        total_orig_len = prog.get("total_orig_len", 0)
        total_comp_len = prog.get("total_comp_len", 0)
        print(f"  Resuming from index {start_idx} ({start_idx}/{n} done)")

    t0 = time.time()
    for i in tqdm(range(start_idx, n), initial=start_idx, total=n, desc="MMQA Compress+Render"):
        out = out_dir / f"{i:05d}.png"
        s = train[i]
        blocks = []

        # Header
        modalities = s.get("metadata", {}).get("modalities", [])
        qtype = s.get("metadata", {}).get("type", "")
        if qtype or modalities:
            blocks.append(measure_text(f"[{qtype}] Modalities: {', '.join(modalities)}",
                                       font_size=FONT_HEADER, ref_width=REF_WIDTH))

        # Supporting context
        for ctx in s.get("supporting_context", [])[:3]:
            doc_id, doc_part = ctx["doc_id"], ctx["doc_part"]

            if doc_part == "text" and doc_id in texts:
                text_doc = texts[doc_id]
                title = text_doc.get("title", "")
                passage = (text_doc.get("text", "") or "")[:300].strip()
                if passage:
                    compressed = comp.compress(passage)
                    total_orig_len += len(passage)
                    total_comp_len += len(compressed)
                    compress_count += 1
                    blocks.append(measure_text(f"[Text] {title}\n{compressed}",
                                               font_size=FONT_BODY, ref_width=REF_WIDTH))
                elif title:
                    blocks.append(measure_text(f"[Text] {title}", font_size=FONT_BODY, ref_width=REF_WIDTH))

            elif doc_part == "table" and doc_id in tables:
                table_doc = tables[doc_id]
                title = table_doc.get("title", "")
                if title:
                    blocks.append(measure_text(f"[Table] {title}", font_size=FONT_HEADER, ref_width=REF_WIDTH))
                headers = [h["column_name"] for h in table_doc["table"]["header"]]
                rows = [[cell["text"][:40] for cell in row] for row in table_doc["table"]["table_rows"][:8]]
                table_data = [headers] + rows
                if len(table_doc["table"]["table_rows"]) > 8:
                    table_data.append([f"...({len(table_doc['table']['table_rows'])} rows)"] + [""] * (len(headers) - 1))
                blocks.append(measure_table(table_data, font_size=FONT_TABLE))

            elif doc_part == "image" and doc_id in images_meta:
                img_info = images_meta[doc_id]
                img_path = IMAGES_DIR / img_info["path"]
                if img_path.exists():
                    try:
                        img = Image.open(img_path).convert("RGB")
                        blocks.append(measure_text(f"[Image] {img_info.get('title', '')}",
                                                   font_size=FONT_HEADER, ref_width=REF_WIDTH))
                        blocks.append(measure_image(img, max_dim=IMG_MAX_DIM))
                    except Exception:
                        pass

        # Question + answer
        q = s.get("question", "")
        answers = s.get("answers", [])
        ans = str(answers[0]["answer"]) if answers else ""
        blocks.append(measure_text(f"Q: {q}\n>>> A: {ans}", font_size=FONT_QA, ref_width=REF_WIDTH))

        if not blocks:
            blocks.append(measure_text("(empty)", font_size=FONT_BODY, ref_width=REF_WIDTH))

        layout = layout_single_column(blocks, target_width=CANVAS_WIDTH)
        img_out = render_layout(layout)
        buf = io.BytesIO()
        img_out.save(buf, format="PNG", optimize=True)
        out.write_bytes(buf.getvalue())

        if (i + 1) % 500 == 0:
            with open(progress_file, "w") as f:
                json.dump({"last_completed": i, "compress_count": compress_count,
                           "total_orig_len": total_orig_len, "total_comp_len": total_comp_len}, f)
            elapsed = time.time() - t0
            rate = (i - start_idx + 1) / elapsed
            remaining = (n - i - 1) / rate if rate > 0 else 0
            avg_ratio = total_comp_len / max(total_orig_len, 1)
            print(f"\n  [{i+1}/{n}] {rate:.1f} samples/s, ETA {remaining/60:.0f}min, "
                  f"avg compression: {avg_ratio:.0%}")

    elapsed = time.time() - t0
    avg_ratio = total_comp_len / max(total_orig_len, 1)
    print(f"\n  Done: {n} canvases in {elapsed/60:.1f}min")
    print(f"  Compressed {compress_count} texts, avg ratio: {avg_ratio:.0%}")
    with open(progress_file, "w") as f:
        json.dump({"last_completed": n - 1, "compress_count": compress_count,
                   "total_orig_len": total_orig_len, "total_comp_len": total_comp_len,
                   "elapsed_seconds": elapsed}, f, indent=2)
    (out_dir / "done.txt").write_text(str(n))
    return n


def render_hotpotqa_compressed(comp: LoRACompressor):
    """Render all HotpotQA canvases with compressed context paragraphs."""
    print("\n=== HotpotQA Compressed Canvas0415 ===")
    out_dir = Path("/home/cyf/codex/canvas0415_compressed/hotpotqa")
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = Path("/home/cyf/codex/hotpotqa_data/hotpotqa_meta.pkl")
    with open(cache, "rb") as f:
        data = pickle.load(f)
    train = data if isinstance(data, list) else data.get("train", data)

    n = len(train)
    print(f"  Rendering {n} compressed canvases...")

    compress_count = 0
    total_orig_len = 0
    total_comp_len = 0

    progress_file = out_dir / "progress.json"
    start_idx = 0
    if progress_file.exists():
        with open(progress_file) as f:
            prog = json.load(f)
        start_idx = prog.get("last_completed", 0) + 1
        compress_count = prog.get("compress_count", 0)
        total_orig_len = prog.get("total_orig_len", 0)
        total_comp_len = prog.get("total_comp_len", 0)
        print(f"  Resuming from index {start_idx} ({start_idx}/{n} done)")

    t0 = time.time()
    for i in tqdm(range(start_idx, n), initial=start_idx, total=n, desc="HotpotQA Compress+Render"):
        out = out_dir / f"{i:05d}.png"
        s = train[i]
        blocks = []

        # Type + level header
        qtype = s.get("type", "")
        level = s.get("level", "")
        if qtype or level:
            blocks.append(measure_text(f"[{qtype}] Level: {level}", font_size=FONT_HEADER, ref_width=REF_WIDTH))

        # Supporting context paragraphs — compress each
        for para in s.get("paragraphs", [])[:4]:
            title = para["title"]
            text = para["text"][:400].strip()
            if text:
                compressed = comp.compress(text)
                total_orig_len += len(text)
                total_comp_len += len(compressed)
                compress_count += 1
                blocks.append(measure_text(f"[{title}]\n{compressed}", font_size=FONT_BODY, ref_width=REF_WIDTH))

        # Question + answer
        q = s.get("question", "")
        ans = s.get("answer", "")
        blocks.append(measure_text(f"Q: {q}\n>>> A: {ans}", font_size=FONT_QA, ref_width=REF_WIDTH))

        if not blocks:
            blocks.append(measure_text("(empty)", font_size=FONT_BODY, ref_width=REF_WIDTH))

        layout = layout_single_column(blocks, target_width=CANVAS_WIDTH)
        img_out = render_layout(layout)
        buf = io.BytesIO()
        img_out.save(buf, format="PNG", optimize=True)
        out.write_bytes(buf.getvalue())

        if (i + 1) % 500 == 0:
            with open(progress_file, "w") as f:
                json.dump({"last_completed": i, "compress_count": compress_count,
                           "total_orig_len": total_orig_len, "total_comp_len": total_comp_len}, f)
            elapsed = time.time() - t0
            rate = (i - start_idx + 1) / elapsed
            remaining = (n - i - 1) / rate if rate > 0 else 0
            avg_ratio = total_comp_len / max(total_orig_len, 1)
            print(f"\n  [{i+1}/{n}] {rate:.1f} samples/s, ETA {remaining/60:.0f}min, "
                  f"avg compression: {avg_ratio:.0%}")

    elapsed = time.time() - t0
    avg_ratio = total_comp_len / max(total_orig_len, 1)
    print(f"\n  Done: {n} canvases in {elapsed/60:.1f}min")
    print(f"  Compressed {compress_count} texts, avg ratio: {avg_ratio:.0%}")
    with open(progress_file, "w") as f:
        json.dump({"last_completed": n - 1, "compress_count": compress_count,
                   "total_orig_len": total_orig_len, "total_comp_len": total_comp_len,
                   "elapsed_seconds": elapsed}, f, indent=2)
    (out_dir / "done.txt").write_text(str(n))
    return n


def compute_clip_embeddings(canvas_dir: Path, n: int):
    """Compute CLIP image embeddings. Symlink text/query embeddings from original."""
    from transformers import CLIPProcessor, CLIPModel

    img_emb_path = canvas_dir / "clip_img_emb.npy"
    if img_emb_path.exists():
        existing = np.load(img_emb_path)
        if existing.shape[0] == n:
            print(f"  CLIP img embeddings already computed: {existing.shape}")
            return
        print(f"  Existing CLIP has {existing.shape[0]} entries, need {n}. Recomputing...")

    print(f"  Computing CLIP image embeddings for {n} canvases...")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    all_emb = []
    bs = 32
    for i in tqdm(range(0, n, bs), desc="CLIP img"):
        imgs = []
        for j in range(i, min(i + bs, n)):
            imgs.append(Image.open(canvas_dir / f"{j:05d}.png").convert("RGB"))
        inp = proc(images=imgs, return_tensors="pt", padding=True)
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            feat = clip.get_image_features(**inp)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        all_emb.append(feat.cpu().numpy())
    emb = np.concatenate(all_emb)
    np.save(img_emb_path, emb)
    print(f"  Saved: {emb.shape} -> {img_emb_path}")

    del clip, proc
    torch.cuda.empty_cache()

    # Symlink text and query embeddings from original canvas0415
    orig_dir = Path("/home/cyf/codex/canvas0415/scienceqa")
    for name in ["clip_txt_emb.npy", "clip_query_emb.npy"]:
        dst = canvas_dir / name
        if not dst.exists():
            # Follow symlinks to get the real target
            src = orig_dir / name
            real_src = src.resolve()
            os.symlink(real_src, dst)
            print(f"  Symlinked: {name} -> {real_src}")


BENCHMARK_DIRS = {
    "scienceqa": Path("/home/cyf/codex/canvas0415_compressed/scienceqa"),
    "okvqa": Path("/home/cyf/codex/canvas0415_compressed/okvqa"),
    "mmqa": Path("/home/cyf/codex/canvas0415_compressed/mmqa"),
    "hotpotqa": Path("/home/cyf/codex/canvas0415_compressed/hotpotqa"),
}

CLIP_SYMLINK_SOURCES = {
    "scienceqa": Path("/home/cyf/codex/scienceqa_smart_canvases"),
    "okvqa": Path("/home/cyf/codex/okvqa_data/canvases_smart"),
    "mmqa": Path("/home/cyf/codex/mmqa_data/canvases_smart"),
    "hotpotqa": None,  # No originals exist — must compute all
}

RENDER_FUNCS = {
    "scienceqa": render_scienceqa_compressed,
    "okvqa": render_okvqa_compressed,
    "mmqa": render_mmqa_compressed,
    "hotpotqa": render_hotpotqa_compressed,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["scienceqa", "okvqa", "mmqa", "hotpotqa", "all"],
                        default="scienceqa")
    parser.add_argument("--embed", action="store_true", help="Also compute CLIP embeddings")
    args = parser.parse_args()

    benchmarks = list(RENDER_FUNCS.keys()) if args.benchmark == "all" else [args.benchmark]

    comp = LoRACompressor(device="cuda")

    results = {}
    for bm in benchmarks:
        n = RENDER_FUNCS[bm](comp)
        results[bm] = n

    # Free compressor GPU memory before CLIP
    del comp
    torch.cuda.empty_cache()

    if args.embed:
        for bm, n in results.items():
            canvas_dir = BENCHMARK_DIRS[bm]
            compute_clip_embeddings(canvas_dir, n)
            # Symlink txt/query embeddings from originals
            src_dir = CLIP_SYMLINK_SOURCES.get(bm)
            if src_dir:
                for name in ["clip_txt_emb.npy", "clip_query_emb.npy"]:
                    dst = canvas_dir / name
                    if not dst.exists():
                        real_src = (src_dir / name).resolve()
                        if real_src.exists():
                            os.symlink(real_src, dst)
                            print(f"  Symlinked: {name} -> {real_src}")


if __name__ == "__main__":
    main()
