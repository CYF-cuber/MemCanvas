#!/usr/bin/env python3
"""
ScienceQA text compression experiment.
Compress lecture + solution text with Qwen2.5-VL-3B before rendering canvases.
Then CLIP embed + retrieve + evaluate with Qwen2.5-VL-7B.
"""
import io, json, os, pickle, re, string, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

sys.path.insert(0, "/home/cyf/codex")
from smart_canvas_layout import measure_text, measure_image, choose_best_layout, render_layout

COMPRESSOR_MODEL = "/home/cyf/Qwen2.5-VL-3B-Instruct"
VLM_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
DATA_PATH = "/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl"
OUTPUT_DIR = Path("/home/cyf/memcanvas0402/scienceqa")
ALPHA = 0.75
TOP_K = 2

# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------
COMPRESS_PROMPTS = {
    "light": "Extract only the key factual sentences from this educational text. Remove filler and keep only essential facts needed to answer science questions.",
    "heavy": "Compress this educational text into the shortest possible form. Use 'concept: fact' format. Maximum 2 lines.",
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

def compress_text(model, processor, text, level):
    if not text or len(text) < 30:
        return text
    prompt = COMPRESS_PROMPTS[level]
    content = [{"type": "text", "text": f"{prompt}\n\nText:\n{text[:800]}"}]
    messages = [{"role": "user", "content": content}]
    t = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[t], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature=1.0)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# Canvas rendering
# ---------------------------------------------------------------------------
def render_sciqa_canvas(sample, compressed_lecture, compressed_solution, hf_image=None):
    blocks = []
    subj = sample.get("subject", "")
    topic = sample.get("topic", "")
    if subj or topic:
        blocks.append(measure_text(f"[{subj}] {topic}", font_size=12, ref_width=600))

    hint = sample.get("hint", "")
    if hint:
        blocks.append(measure_text(hint, font_size=14, ref_width=600))

    if hf_image is not None:
        blocks.append(measure_image(hf_image.convert("RGB"), max_dim=400))

    if compressed_lecture:
        blocks.append(measure_text(compressed_lecture, font_size=14, ref_width=600))

    # Question + choices + answer
    q = sample.get("question", "")
    choices = sample.get("choices", [])
    answer_idx = sample.get("answer", 0)
    lines = [f"Q: {q}"]
    for ci, ch in enumerate(choices):
        mark = "✓" if ci == answer_idx else " "
        lines.append(f" {mark} {chr(65+ci)}. {ch}")
    blocks.append(measure_text("\n".join(lines), font_size=15, ref_width=600))

    if compressed_solution:
        blocks.append(measure_text(compressed_solution, font_size=13, ref_width=600))

    if not blocks:
        blocks.append(measure_text(f"Q: {q}", font_size=15, ref_width=600))

    layout = choose_best_layout(blocks)
    img = render_layout(layout)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def extract_answer(text, n_choices):
    text = text.strip().upper()
    for i in range(n_choices):
        letter = chr(65 + i)
        if text.startswith(letter) or text.startswith(f"({letter})") or text.startswith(f"ANSWER: {letter}"):
            return i
    for i in range(n_choices):
        if chr(65 + i) in text[:10]:
            return i
    return -1

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True, choices=["light", "heavy"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=500)
    args = parser.parse_args()

    level = args.level
    device = f"cuda:{args.gpu}"
    level_dir = OUTPUT_DIR / level
    canvas_dir = level_dir / "canvases"
    canvas_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = pickle.load(open(DATA_PATH, "rb"))
    train = data["train"]
    test = data["test"]
    print(f"Train: {len(train)}, Test: {len(test)}")

    # --- Phase 1: Compress lecture + solution text ---
    compressed_cache = level_dir / "compressed_texts.pkl"
    if compressed_cache.exists():
        compressed = pickle.load(open(compressed_cache, "rb"))
        print(f"Loaded {len(compressed)} compressed samples from cache")
    else:
        comp_model, comp_proc = load_compressor(device)
        compressed = {}
        total_in, total_out = 0, 0

        for i in tqdm(range(len(train)), desc=f"Compressing ({level})"):
            s = train[i]
            lecture = s.get("lecture", "")
            solution = s.get("solution", "")
            cl = compress_text(comp_model, comp_proc, lecture, level) if lecture else ""
            cs = compress_text(comp_model, comp_proc, solution, level) if solution else ""
            compressed[i] = {"lecture": cl, "solution": cs}
            total_in += len(lecture) + len(solution)
            total_out += len(cl) + len(cs)

            if (i + 1) % 500 == 0:
                pickle.dump(compressed, open(compressed_cache, "wb"))

        pickle.dump(compressed, open(compressed_cache, "wb"))
        ratio = total_out / total_in * 100 if total_in > 0 else 0
        print(f"\n=== Compression Stats ({level}) ===")
        print(f"  Samples: {len(compressed)}")
        print(f"  Compression ratio: {ratio:.1f}%")

        del comp_model, comp_proc
        torch.cuda.empty_cache()

    # --- Phase 2: Build canvases + CLIP embeddings ---
    print("\nBuilding canvases...")
    # Load HF images
    from datasets import load_dataset
    hf_ds = load_dataset("derek-thomas/ScienceQA", split="train")

    n_train = len(train)
    done_marker = canvas_dir / "done.txt"
    if not done_marker.exists():
        for i in tqdm(range(n_train), desc="Canvas"):
            out = canvas_dir / f"{i:05d}.png"
            if out.exists():
                continue
            s = train[i]
            cl = compressed.get(i, {}).get("lecture", s.get("lecture", ""))
            cs = compressed.get(i, {}).get("solution", s.get("solution", ""))
            hf_img = hf_ds[i].get("image") if i < len(hf_ds) and hf_ds[i].get("image") is not None else None
            canvas_bytes = render_sciqa_canvas(s, cl, cs, hf_img)
            out.write_bytes(canvas_bytes)
        done_marker.write_text(str(n_train))
    print(f"  {n_train} canvases ready")

    # CLIP embeddings
    print("Computing CLIP embeddings...")
    from transformers import CLIPProcessor, CLIPModel
    clip = CLIPModel.from_pretrained(CLIP_MODEL_NAME).cuda().eval()
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    img_emb_file = level_dir / "clip_img_emb.npy"
    if not img_emb_file.exists():
        all_emb = []
        bs = 32
        for i in tqdm(range(0, n_train, bs), desc="CLIP img"):
            imgs = []
            for j in range(i, min(i + bs, n_train)):
                imgs.append(Image.open(canvas_dir / f"{j:05d}.png").convert("RGB"))
            inp = clip_proc(images=imgs, return_tensors="pt", padding=True)
            inp = {k: v.cuda() for k, v in inp.items()}
            with torch.no_grad():
                feat = clip.get_image_features(**inp)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            all_emb.append(feat.cpu().numpy())
        img_emb = np.concatenate(all_emb)
        np.save(img_emb_file, img_emb)
    else:
        img_emb = np.load(img_emb_file)
    print(f"  Image emb: {img_emb.shape}")

    # Text embeddings for train
    txt_emb_file = level_dir / "clip_txt_emb.npy"
    if not txt_emb_file.exists():
        all_emb = []
        bs = 64
        for i in tqdm(range(0, n_train, bs), desc="CLIP txt"):
            texts = []
            for j in range(i, min(i + bs, n_train)):
                s = train[j]
                t = f"{s.get('question','')} {s.get('hint','')}"[:77]
                texts.append(t)
            inp = clip_proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inp = {k: v.cuda() for k, v in inp.items()}
            with torch.no_grad():
                feat = clip.get_text_features(**inp)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            all_emb.append(feat.cpu().numpy())
        txt_emb = np.concatenate(all_emb)
        np.save(txt_emb_file, txt_emb)
    else:
        txt_emb = np.load(txt_emb_file)
    print(f"  Text emb: {txt_emb.shape}")

    # Query embeddings for test
    max_test = min(args.max_test, len(test))
    query_emb_file = level_dir / "clip_query_emb.npy"
    if not query_emb_file.exists():
        all_emb = []
        bs = 64
        for i in tqdm(range(0, max_test, bs), desc="CLIP query"):
            texts = [test[j].get("question", "")[:77] for j in range(i, min(i + bs, max_test))]
            inp = clip_proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inp = {k: v.cuda() for k, v in inp.items()}
            with torch.no_grad():
                feat = clip.get_text_features(**inp)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            all_emb.append(feat.cpu().numpy())
        query_emb = np.concatenate(all_emb)
        np.save(query_emb_file, query_emb)
    else:
        query_emb = np.load(query_emb_file)
    print(f"  Query emb: {query_emb.shape}")

    del clip, clip_proc
    torch.cuda.empty_cache()

    # Retrieval
    key_emb = ALPHA * img_emb + (1 - ALPHA) * txt_emb
    key_emb = key_emb / np.linalg.norm(key_emb, axis=1, keepdims=True).clip(1e-8)
    q_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True).clip(1e-8)
    sims = q_norm @ key_emb.T

    retrieval_map = {}
    for i in range(len(query_emb)):
        top_idx = np.argsort(sims[i])[::-1][:TOP_K]
        retrieval_map[i] = [(int(j), float(sims[i][j])) for j in top_idx if sims[i][j] >= 0.1]

    # --- Phase 3: VLM evaluation ---
    # Load test images
    from datasets import load_dataset as ld2
    hf_test_ds = ld2("derek-thomas/ScienceQA", split="test")

    print("\nLoading VLM...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    proc = AutoProcessor.from_pretrained(VLM_MODEL)

    ckpt_file = level_dir / "checkpoint.json"
    results = json.load(open(ckpt_file)) if ckpt_file.exists() else {}
    done = set(results.keys())

    for i in tqdm(range(max_test), desc="Evaluating"):
        key = str(i)
        if key in done:
            continue

        s = test[i]
        question = s.get("question", "")
        choices = s.get("choices", [])
        answer_idx = s.get("answer", 0)
        n_choices = len(choices)
        choice_text = "\n".join(f"  {chr(65+ci)}. {ch}" for ci, ch in enumerate(choices))

        retrieved = retrieval_map.get(i, [])

        # MemCanvas prediction
        content = []
        canvas_imgs = []
        for cidx, sim in retrieved[:TOP_K]:
            img = Image.open(canvas_dir / f"{cidx:05d}.png").convert("RGB")
            canvas_imgs.append(img)
            content.append({"type": "image", "image": img})

        # Also include test image if available
        test_img = None
        if i < len(hf_test_ds) and hf_test_ds[i].get("image") is not None:
            test_img = hf_test_ds[i]["image"].convert("RGB")
            content.append({"type": "image", "image": test_img})
            canvas_imgs.append(test_img)

        user_text = (
            "Study the memory canvases above showing similar science questions and answers.\n\n"
            f"Question: {question}\n{choice_text}\n\n"
            "Answer with just the letter (A, B, C, etc.):"
        )
        content.append({"type": "text", "text": user_text})

        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if canvas_imgs:
            inp = proc(text=[txt], images=canvas_imgs, return_tensors="pt", padding=True)
        else:
            inp = proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}

        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=32, do_sample=False)
        pred_text = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred_idx = extract_answer(pred_text, n_choices)
        correct = int(pred_idx == answer_idx)

        results[key] = {
            "question": question, "pred": pred_text, "pred_idx": pred_idx,
            "answer_idx": answer_idx, "correct": correct,
            "subject": s.get("subject", ""),
        }

        if len(results) % 50 == 0:
            json.dump(results, open(ckpt_file, "w"))

    json.dump(results, open(ckpt_file, "w"))

    # Summary
    n = len(results)
    acc = np.mean([v["correct"] for v in results.values()]) * 100

    # Per-subject
    subjects = {}
    for v in results.values():
        subj = v.get("subject", "unknown")
        subjects.setdefault(subj, []).append(v["correct"])

    summary = {
        "n": n,
        "compression": level,
        "overall_accuracy": acc,
        "per_subject": {s: np.mean(vs) * 100 for s, vs in subjects.items()},
        "config": {"alpha": ALPHA, "top_k": TOP_K, "vlm": "Qwen2.5-VL-7B"},
        "reference": {"baseline": 89.0, "original_memcanvas": 89.4},
    }
    json.dump(summary, open(level_dir / "summary.json", "w"), indent=2)

    print(f"\n{'='*60}")
    print(f"ScienceQA Results — {level} compression ({n} samples)")
    print(f"{'='*60}")
    print(f"  Original Baseline:    89.0%")
    print(f"  Original MemCanvas:   89.4%")
    print(f"  Compressed ({level:5s}):  {acc:.1f}%")
    for s, a in sorted(subjects.items()):
        print(f"    {s}: {np.mean(a)*100:.1f}%")
    print(f"{'='*60}")

    del vlm, proc
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
