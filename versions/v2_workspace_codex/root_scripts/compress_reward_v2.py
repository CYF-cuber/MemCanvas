#!/usr/bin/env python3
"""
Reward function for Text Compressor GRPO v2 — with VLM readability reward.

R = 0.35 * answer_preserved + 0.25 * fact_preserved + 0.2 * conciseness + 0.2 * vlm_readability

VLM readability: render compressed text → canvas → VLM reads back → compare with original.
Uses Qwen2.5-VL-7B-Instruct as the reader (NOT external OCR).

Called by verl's NaiveRewardManager:
    score = compute_score(data_source, solution_str, ground_truth, extra_info)

Ground truth format (JSON):
{
    "answer": "the correct answer text",
    "key_facts": ["fact1", "fact2"],
    "original_length": 500,
    "question": "the question",
    "font_size": 16,         # canvas font size for rendering
    "canvas_width": 830      # canvas width for rendering
}
"""

import io
import json
import os
import re
import sys
import time
from typing import Optional, Dict

# Make smart_canvas_layout importable
sys.path.insert(0, "/home/cyf/codex")

# ============================================================
# VLM singleton — loaded once, reused across calls
# ============================================================
_VLM_MODEL = None
_VLM_PROCESSOR = None
_VLM_ENABLED = os.environ.get("VLM_REWARD_DISABLE", "") != "1"

VLM_MODEL_PATH = os.environ.get(
    "VLM_READABILITY_MODEL",
    "/home/cyf/Qwen2.5-VL-3B-Instruct",  # 3B for training (fits alongside actor); 7B for eval
)
# Device for VLM reward inference. During GRPO with param_offload, GPU is free
# during reward phase. Use "cuda" for speed, "cpu" for safety (no OOM risk).
VLM_DEVICE = os.environ.get("VLM_REWARD_DEVICE", "cuda")

# Canvas rendering parameters (defaults, can be overridden per-sample via ground_truth)
DEFAULT_FONT_SIZE = 16
DEFAULT_CANVAS_WIDTH = 830
DEFAULT_REF_WIDTH = 800


def _load_vlm():
    """Lazy-load VLM model and processor."""
    global _VLM_MODEL, _VLM_PROCESSOR
    if _VLM_MODEL is not None:
        return _VLM_MODEL, _VLM_PROCESSOR

    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"[compress_reward_v2] Loading VLM from {VLM_MODEL_PATH} on {VLM_DEVICE}...")
    t0 = time.time()

    if VLM_DEVICE == "cpu":
        _VLM_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_PATH, dtype=torch.float32,
        ).to("cpu").eval()
    else:
        # Explicitly load to GPU — device_map="auto" fails inside Ray actors
        _VLM_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_PATH, dtype=torch.bfloat16,
        ).to(VLM_DEVICE).eval()

    _VLM_PROCESSOR = AutoProcessor.from_pretrained(VLM_MODEL_PATH)
    actual_device = next(_VLM_MODEL.parameters()).device
    print(f"[compress_reward_v2] VLM loaded in {time.time()-t0:.1f}s on {actual_device}")
    return _VLM_MODEL, _VLM_PROCESSOR


def _render_text_to_canvas(text: str, font_size: int, canvas_width: int, ref_width: int):
    """Render text to a canvas image using smart_canvas_layout."""
    from smart_canvas_layout import measure_text, layout_single_column, render_layout

    block = measure_text(text, font_size=font_size, ref_width=ref_width)
    layout = layout_single_column([block], target_width=canvas_width)
    img = render_layout(layout)
    return img


def _vlm_read_canvas(canvas_img) -> str:
    """Feed canvas image to VLM, ask it to read all text back."""
    import torch

    model, processor = _load_vlm()

    messages = [{"role": "user", "content": [
        {"type": "image", "image": canvas_img},
        {"type": "text", "text": (
            "Read ALL the text from this image exactly as written. "
            "Output only the text content, preserving the original structure. "
            "Do not add any commentary."
        )},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[canvas_img], return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def _vlm_read_canvas_batch(canvas_imgs: list) -> list:
    """Batch VLM read for multiple canvases (more efficient)."""
    import torch

    if not canvas_imgs:
        return []

    model, processor = _load_vlm()
    device = next(model.parameters()).device

    prompt_text = (
        "Read ALL the text from this image exactly as written. "
        "Output only the text content, preserving the original structure. "
        "Do not add any commentary."
    )

    results = []
    # Process in mini-batches of 4 (VLM with images is memory-heavy)
    bs = 4
    for i in range(0, len(canvas_imgs), bs):
        batch_imgs = canvas_imgs[i:i+bs]
        batch_texts = []
        batch_images = []

        for img in batch_imgs:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text},
            ]}]
            t = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_texts.append(t)
            batch_images.append(img)

        # Process each individually (VLM batch with images is tricky)
        for t, img in zip(batch_texts, batch_images):
            inputs = processor(text=[t], images=[img], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            resp = processor.decode(gen_ids, skip_special_tokens=True).strip()
            results.append(resp)

    return results


# ============================================================
# Text utilities
# ============================================================
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def fact_preserved(compressed: str, fact: str) -> float:
    """Check if a fact is preserved in compressed text (fuzzy)."""
    cn = normalize(compressed)
    fn = normalize(fact)
    if fn in cn:
        return 1.0
    fw = set(fn.split())
    cw = set(cn.split())
    if not fw:
        return 1.0
    return len(fw & cw) / len(fw)


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity ratio (0-1)."""
    try:
        from Levenshtein import ratio
        return ratio(s1, s2)
    except ImportError:
        # Fallback: simple word overlap ratio
        w1 = set(normalize(s1).split())
        w2 = set(normalize(s2).split())
        if not w1 and not w2:
            return 1.0
        if not w1 or not w2:
            return 0.0
        return 2 * len(w1 & w2) / (len(w1) + len(w2))


# ============================================================
# VLM readability score
# ============================================================
def vlm_readability_score(
    compressed_text: str,
    font_size: int = DEFAULT_FONT_SIZE,
    canvas_width: int = DEFAULT_CANVAS_WIDTH,
    ref_width: int = DEFAULT_REF_WIDTH,
) -> float:
    """
    Render compressed text to canvas → VLM reads back → compare.

    Returns:
        float in [0, 1]: similarity between VLM-read text and original.
    """
    if not _VLM_ENABLED or not compressed_text.strip():
        return 0.0

    try:
        # 1. Render to canvas
        canvas_img = _render_text_to_canvas(compressed_text, font_size, canvas_width, ref_width)

        # 2. VLM reads back
        vlm_text = _vlm_read_canvas(canvas_img)

        # 3. Compare
        score = levenshtein_ratio(normalize(compressed_text), normalize(vlm_text))
        return score

    except Exception as e:
        print(f"[compress_reward_v2] VLM readability error: {e}")
        return 0.0


# ============================================================
# Main reward function (verl interface)
# ============================================================
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
    **kwargs,
) -> float:
    """
    Compute reward for a compression response.

    Components:
        0.35 * answer_preservation
        0.25 * fact_preservation
        0.20 * conciseness
        0.20 * vlm_readability
    """
    try:
        gt = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    compressed = solution_str.strip()
    compressed = re.sub(r"<think>.*?</think>", "", compressed, flags=re.DOTALL).strip()

    if not compressed:
        return 0.0

    answer = gt.get("answer", "")
    key_facts = gt.get("key_facts", [])
    original_length = gt.get("original_length", 500)
    font_size = gt.get("font_size", DEFAULT_FONT_SIZE)
    canvas_width = gt.get("canvas_width", DEFAULT_CANVAS_WIDTH)
    ref_width = gt.get("ref_width", DEFAULT_REF_WIDTH)

    # --- Component 1: Answer preservation (0.35) ---
    answer_score = fact_preserved(compressed, answer)

    # --- Component 2: Key fact preservation (0.25) ---
    if key_facts:
        fact_scores = [fact_preserved(compressed, f) for f in key_facts]
        fact_score = sum(fact_scores) / len(fact_scores)
    else:
        fact_score = 1.0 if answer_score > 0.5 else 0.0

    # --- Component 3: Conciseness (0.20) ---
    ratio = len(compressed) / max(original_length, 1)
    if ratio < 0.05:
        conciseness = 0.0
    elif ratio < 0.1:
        conciseness = 0.5
    elif ratio < 0.5:
        conciseness = 1.0
    elif ratio < 0.8:
        conciseness = 0.7
    else:
        conciseness = 0.3

    # --- Component 4: VLM readability (0.20) ---
    if _VLM_ENABLED:
        readability = vlm_readability_score(compressed, font_size, canvas_width, ref_width)
    else:
        readability = 0.5  # neutral default when VLM disabled

    total = 0.35 * answer_score + 0.25 * fact_score + 0.20 * conciseness + 0.20 * readability
    return total


# ============================================================
# Batch reward (for efficiency: render all canvases, then batch VLM)
# ============================================================
def compute_scores_batch(
    data_sources: list,
    solution_strs: list,
    ground_truths: list,
    extra_infos: list = None,
) -> list:
    """
    Batch reward computation. Renders all canvases first, then runs VLM
    in batch for readability scores. Much more efficient than per-sample.
    """
    n = len(solution_strs)
    if extra_infos is None:
        extra_infos = [None] * n

    # Parse all ground truths and compressed texts
    parsed = []
    for i in range(n):
        try:
            gt = json.loads(ground_truths[i])
        except (json.JSONDecodeError, TypeError):
            gt = {}
        compressed = solution_strs[i].strip()
        compressed = re.sub(r"<think>.*?</think>", "", compressed, flags=re.DOTALL).strip()
        parsed.append((gt, compressed))

    # Compute text-based scores (fast)
    text_scores = []
    for gt, compressed in parsed:
        if not compressed:
            text_scores.append((0.0, 0.0, 0.0))
            continue

        answer = gt.get("answer", "")
        key_facts = gt.get("key_facts", [])
        original_length = gt.get("original_length", 500)

        answer_score = fact_preserved(compressed, answer)
        if key_facts:
            fs = [fact_preserved(compressed, f) for f in key_facts]
            fact_score = sum(fs) / len(fs)
        else:
            fact_score = 1.0 if answer_score > 0.5 else 0.0

        r = len(compressed) / max(original_length, 1)
        if r < 0.05:
            conciseness = 0.0
        elif r < 0.1:
            conciseness = 0.5
        elif r < 0.5:
            conciseness = 1.0
        elif r < 0.8:
            conciseness = 0.7
        else:
            conciseness = 0.3

        text_scores.append((answer_score, fact_score, conciseness))

    # Render canvases and batch VLM readability
    readability_scores = [0.0] * n
    if _VLM_ENABLED:
        canvas_imgs = []
        canvas_indices = []  # track which samples have canvases

        for i, (gt, compressed) in enumerate(parsed):
            if not compressed:
                continue
            try:
                font_size = gt.get("font_size", DEFAULT_FONT_SIZE)
                canvas_width = gt.get("canvas_width", DEFAULT_CANVAS_WIDTH)
                ref_width = gt.get("ref_width", DEFAULT_REF_WIDTH)
                img = _render_text_to_canvas(compressed, font_size, canvas_width, ref_width)
                canvas_imgs.append(img)
                canvas_indices.append(i)
            except Exception:
                pass

        if canvas_imgs:
            vlm_texts = _vlm_read_canvas_batch(canvas_imgs)
            for idx, vlm_text in zip(canvas_indices, vlm_texts):
                _, compressed = parsed[idx]
                readability_scores[idx] = levenshtein_ratio(
                    normalize(compressed), normalize(vlm_text)
                )
    else:
        readability_scores = [0.5] * n

    # Combine
    results = []
    for i in range(n):
        if not parsed[i][1]:  # empty compressed
            results.append(0.0)
            continue
        a, f, c = text_scores[i]
        r = readability_scores[i]
        total = 0.35 * a + 0.25 * f + 0.20 * c + 0.20 * r
        results.append(total)

    return results


# ============================================================
# Standalone VLM readability evaluation
# ============================================================
def evaluate_vlm_readability(
    texts: list,
    font_size: int = DEFAULT_FONT_SIZE,
    canvas_width: int = DEFAULT_CANVAS_WIDTH,
    ref_width: int = DEFAULT_REF_WIDTH,
) -> dict:
    """
    Evaluate VLM readability for a list of texts.
    Returns dict with per-sample and aggregate scores.
    """
    from tqdm import tqdm

    canvas_imgs = []
    for text in tqdm(texts, desc="Rendering canvases"):
        img = _render_text_to_canvas(text, font_size, canvas_width, ref_width)
        canvas_imgs.append(img)

    vlm_texts = _vlm_read_canvas_batch(canvas_imgs)

    scores = []
    details = []
    for orig, vlm_out in zip(texts, vlm_texts):
        s = levenshtein_ratio(normalize(orig), normalize(vlm_out))
        scores.append(s)
        details.append({
            "original": orig[:200],
            "vlm_read": vlm_out[:200],
            "score": s,
        })

    import numpy as np
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n": len(scores),
        "details": details,
    }


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    print("Compress Reward v2 — Self-test")
    print("=" * 60)

    gt = json.dumps({
        "answer": "Paris",
        "key_facts": ["capital of France", "population 2.1 million"],
        "original_length": 500,
        "question": "What is the capital of France?",
        "font_size": 16,
        "canvas_width": 830,
        "ref_width": 800,
    })

    # Test without VLM first
    import compress_reward_v2 as self_mod
    self_mod._VLM_ENABLED = False

    resp1 = "Paris: capital of France, pop 2.1M, on Seine river"
    s1 = compute_score("memcanvas_compress", resp1, gt)
    print(f"Good compression (no VLM): {s1:.3f}")

    resp2 = "Paris is the capital and most populous city of France. " * 8
    s2 = compute_score("memcanvas_compress", resp2, gt)
    print(f"Verbose (no VLM):          {s2:.3f}")

    resp3 = ""
    s3 = compute_score("memcanvas_compress", resp3, gt)
    print(f"Empty (no VLM):            {s3:.3f}")

    # Test with VLM
    print("\n--- Testing VLM readability ---")
    self_mod._VLM_ENABLED = True

    try:
        # Test canvas rendering
        img = _render_text_to_canvas(resp1, 16, 830, 800)
        print(f"Canvas rendered: {img.size}")

        # Test VLM read
        vlm_text = _vlm_read_canvas(img)
        print(f"VLM read: {vlm_text[:100]}")

        score = levenshtein_ratio(normalize(resp1), normalize(vlm_text))
        print(f"Readability score: {score:.3f}")

        # Full reward with VLM
        s4 = compute_score("memcanvas_compress", resp1, gt)
        print(f"Full reward (with VLM): {s4:.3f}")

    except Exception as e:
        print(f"VLM test skipped: {e}")

    print("\nDone!")
