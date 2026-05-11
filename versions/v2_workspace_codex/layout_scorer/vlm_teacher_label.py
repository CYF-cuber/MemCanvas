#!/usr/bin/env python3
"""
Phase 0b: VLM Teacher scoring — use Qwen2.5-VL-7B to score candidate layouts.

For each candidate image, the VLM rates 5 dimensions (1-5 scale):
  s1: Space utilization
  s2: Text readability
  s3: Visual clarity
  s4: Reading order
  s5: Overall aesthetics
  total: average of s1-s5

Outputs:
- layout_scorer_data/vlm_scores.json

Usage:
    python -m layout_scorer.vlm_teacher_label [--batch_size 4] [--resume]
"""

import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, "/home/cyf/codex")

CANDIDATES_DIR = "/home/cyf/codex/layout_scorer_data/candidates"
META_PATH = "/home/cyf/codex/layout_scorer_data/candidates_meta.json"
OUTPUT_PATH = "/home/cyf/codex/layout_scorer_data/vlm_scores.json"


SCORING_PROMPT = """Rate this canvas layout image on 5 dimensions (1-5 scale each):

1. Space utilization: Is content area large relative to canvas? Little wasted whitespace?
2. Text readability: Are fonts appropriate size? Line width comfortable? No awkward word breaks?
3. Visual clarity: Are images/charts/tables clear and legible?
4. Reading order: Is information arranged in a natural reading flow?
5. Overall aesthetics: Are margins even? Elements aligned? Visually clean?

Output ONLY a JSON object, nothing else:
{"s1": _, "s2": _, "s3": _, "s4": _, "s5": _, "total": _}

where total = average of s1 through s5."""


def load_vlm():
    """Load Qwen2.5-VL-7B for scoring."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch

    model_path = "/home/cyf/Qwen2.5-VL-7B-Instruct"
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

    print(f"Loading VLM from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("VLM loaded.")
    return model, processor


def score_image(model, processor, image_path: str, max_retries: int = 2) -> Optional[Dict]:
    """Score a single candidate image with the VLM."""
    from PIL import Image
    import torch

    img = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": SCORING_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[img],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    for attempt in range(max_retries + 1):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only generated tokens
        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Parse JSON from response
        scores = parse_scores(response)
        if scores is not None:
            return scores

        if attempt < max_retries:
            # Retry with temperature
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                )
            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
            response = processor.decode(generated_ids, skip_special_tokens=True).strip()
            scores = parse_scores(response)
            if scores is not None:
                return scores

    return None


def parse_scores(response: str) -> Optional[Dict]:
    """Parse VLM response into score dict."""
    # Try direct JSON parse
    try:
        # Find JSON in response
        match = re.search(r'\{[^}]+\}', response)
        if match:
            data = json.loads(match.group())
            # Validate
            required = ["s1", "s2", "s3", "s4", "s5"]
            if all(k in data for k in required):
                scores = {}
                for k in required:
                    v = float(data[k])
                    scores[k] = max(1.0, min(5.0, v))  # clamp to [1,5]
                scores["total"] = sum(scores[k] for k in required) / 5.0
                return scores
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try regex extraction
    try:
        pattern = r'"s(\d)":\s*([\d.]+)'
        matches = re.findall(pattern, response)
        if len(matches) >= 5:
            scores = {}
            for idx, val in matches[:5]:
                scores[f"s{idx}"] = max(1.0, min(5.0, float(val)))
            scores["total"] = sum(scores[f"s{i}"] for i in range(1, 6)) / 5.0
            return scores
    except (ValueError, TypeError):
        pass

    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip already scored candidates")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 0b: VLM Teacher Scoring")
    print("=" * 60)

    # Load metadata
    with open(META_PATH) as f:
        meta_list = json.load(f)
    print(f"Total candidates: {len(meta_list)}")

    # Load existing scores
    existing_scores = {}
    if args.resume and os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            existing_scores = json.load(f)
        print(f"Resuming: {len(existing_scores)} already scored")

    # Load VLM
    model, processor = load_vlm()

    scores = dict(existing_scores)
    n_scored = 0
    n_failed = 0
    t0 = time.time()

    for i, meta in enumerate(meta_list):
        key = meta["key"]

        if key in scores:
            continue

        img_path = os.path.join(CANDIDATES_DIR, f"{key}.png")
        if not os.path.exists(img_path):
            print(f"  [WARN] Missing image: {img_path}")
            continue

        result = score_image(model, processor, img_path)

        if result is not None:
            scores[key] = result
            n_scored += 1
        else:
            # Store a default score for failed parses
            scores[key] = {"s1": 3.0, "s2": 3.0, "s3": 3.0, "s4": 3.0, "s5": 3.0, "total": 3.0, "failed": True}
            n_failed += 1

        if (n_scored + n_failed) % 50 == 0:
            elapsed = time.time() - t0
            rate = (n_scored + n_failed) / elapsed if elapsed > 0 else 0
            remaining = len(meta_list) - len(scores)
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{len(scores)}/{len(meta_list)}] Scored: {n_scored}, "
                  f"Failed: {n_failed}, Rate: {rate:.2f}/s, ETA: {eta/60:.0f}min")

            # Save checkpoint
            with open(OUTPUT_PATH, "w") as f:
                json.dump(scores, f, indent=2)

    # Final save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(scores, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done! Scored {n_scored} candidates ({n_failed} failed) in {elapsed/60:.1f}min")
    print(f"Scores saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
