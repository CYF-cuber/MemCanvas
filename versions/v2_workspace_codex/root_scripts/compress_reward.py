#!/usr/bin/env python3
"""
Reward function for Text Compressor GRPO training.

R_compress = 1[answer_preserved] * (1 - 0.3 * length_ratio)

Since VLM-in-the-loop is too expensive for online GRPO, we use a proxy:
  1. answer_preserved (0.7): check if compressed text preserves key answer info
  2. conciseness bonus (0.3): shorter compression gets higher reward

Called by verl's NaiveRewardManager:
    score = compute_score(data_source, solution_str, ground_truth, extra_info)

Training data ground_truth format (JSON):
{
    "answer": "the correct answer text",
    "key_facts": ["fact1", "fact2"],   # key facts that must be preserved
    "original_length": 500,             # char count of original text
    "question": "the question"
}
"""

import json
import re
from typing import Optional, Dict


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def fact_preserved(compressed: str, fact: str) -> float:
    """Check if a fact is preserved in compressed text (fuzzy)."""
    cn = normalize(compressed)
    fn = normalize(fact)

    # Exact substring match
    if fn in cn:
        return 1.0

    # Word overlap
    fw = set(fn.split())
    cw = set(cn.split())
    if not fw:
        return 1.0
    overlap = len(fw & cw) / len(fw)
    return overlap


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
    **kwargs,
) -> float:
    """
    Compute reward for a compression response.

    Args:
        data_source: "memcanvas_compress"
        solution_str: Model-generated compressed text
        ground_truth: JSON with answer, key_facts, original_length, question

    Returns:
        float: Score in [0.0, 1.0]
    """
    try:
        gt = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    compressed = solution_str.strip()
    # Remove thinking tags if present
    compressed = re.sub(r"<think>.*?</think>", "", compressed, flags=re.DOTALL).strip()

    if not compressed:
        return 0.0

    answer = gt.get("answer", "")
    key_facts = gt.get("key_facts", [])
    original_length = gt.get("original_length", 500)

    # --- Component 1: Answer preservation (weight 0.4) ---
    # The compressed text must preserve the answer
    answer_score = fact_preserved(compressed, answer)

    # --- Component 2: Key fact preservation (weight 0.3) ---
    if key_facts:
        fact_scores = [fact_preserved(compressed, f) for f in key_facts]
        fact_score = sum(fact_scores) / len(fact_scores)
    else:
        fact_score = 1.0 if answer_score > 0.5 else 0.0

    # --- Component 3: Conciseness (weight 0.3) ---
    # Reward shorter compressions, but not too short (< 10% might lose info)
    ratio = len(compressed) / max(original_length, 1)
    if ratio < 0.05:
        # Too short, probably garbage
        conciseness = 0.0
    elif ratio < 0.1:
        conciseness = 0.5
    elif ratio < 0.5:
        # Sweet spot: 10-50% of original
        conciseness = 1.0
    elif ratio < 0.8:
        # OK but not great compression
        conciseness = 0.7
    else:
        # Barely compressed
        conciseness = 0.3

    # Combined score
    total = 0.4 * answer_score + 0.3 * fact_score + 0.3 * conciseness
    return total


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Compress Reward — Self-test")
    print("=" * 50)

    gt = json.dumps({
        "answer": "Paris",
        "key_facts": ["capital of France", "population 2.1 million"],
        "original_length": 500,
        "question": "What is the capital of France?"
    })

    # Good compression
    resp1 = "Paris: capital of France, pop 2.1M, on Seine river"
    s1 = compute_score("memcanvas_compress", resp1, gt)
    print(f"Good compression: {s1:.2f} (expect ~0.85+)")

    # Too verbose (barely compressed)
    resp2 = "Paris is the capital and most populous city of France. " * 8
    s2 = compute_score("memcanvas_compress", resp2, gt)
    print(f"Verbose (bad):    {s2:.2f} (expect ~0.45)")

    # Missing answer
    resp3 = "France: European country, population 67M"
    s3 = compute_score("memcanvas_compress", resp3, gt)
    print(f"Missing answer:   {s3:.2f} (expect ~0.30)")

    # Empty
    s4 = compute_score("memcanvas_compress", "", gt)
    print(f"Empty:            {s4:.2f} (expect 0.00)")

    # Perfect but short
    resp5 = "Paris: capital of France, population 2.1 million"
    s5 = compute_score("memcanvas_compress", resp5, gt)
    print(f"Perfect short:    {s5:.2f} (expect ~0.95+)")

    print("\nAll tests done!")
