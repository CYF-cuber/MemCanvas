#!/usr/bin/env python3
"""
layout_reward.py — GRPO reward function for Layout Agent

Reward = format_score(0.1) + quality_score(0.3) + qa_score(0.6)

Called by verl's reward manager:
    score = compute_score(data_source, solution_str, ground_truth, extra_info)
"""

import json
import math
import re
from typing import Any, Dict, Optional

# Import strategy definitions for validation
from .strategies import (
    STRATEGY_NAMES, WIDTH_MAP, IMAGE_SCALE_MAP, FONT_OFFSET, TRUNCATION_MAP,
    LayoutParams,
)


def parse_layout_json(response_str: str) -> Optional[Dict]:
    """Parse agent's JSON response, handling thinking tags and markdown."""
    text = response_str.strip()

    # Remove thinking tags (Qwen3)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try to find JSON in markdown code block
    md_match = re.search(r"```(?:json)?\s*(\{[^`]+\})\s*```", text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find a JSON object
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try full text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return None


def validate_layout_params(d: Dict) -> bool:
    """Check if parsed JSON has valid layout parameter values."""
    strategy = d.get("strategy", "")
    if strategy not in STRATEGY_NAMES:
        return False
    width_class = d.get("width_class", "")
    if width_class not in WIDTH_MAP:
        return False
    # Other fields have defaults, so partial validity is OK
    return True


def compute_quality_from_params(params: LayoutParams, block_info: Dict) -> float:
    """Estimate quality score from params + block metadata without rendering.

    Uses heuristics based on the parameter choices and content properties.
    This is a fast approximation for the full render-based quality score.
    """
    score = 0.0

    # 1. Width appropriateness (0.25)
    tw = params.target_width
    total_chars = block_info.get("total_chars", 500)
    has_image = block_info.get("has_image", False)
    n_blocks = block_info.get("n_blocks", 3)

    # Narrow is good for short content, wide for long
    if total_chars < 300:
        width_fit = {"narrow": 1.0, "medium": 0.8, "wide": 0.5}
    elif total_chars < 800:
        width_fit = {"narrow": 0.7, "medium": 1.0, "wide": 0.8}
    else:
        width_fit = {"narrow": 0.5, "medium": 0.8, "wide": 1.0}
    score += width_fit.get(params.width_class, 0.7) * 0.25

    # 2. Strategy fit (0.25)
    strategy_score = 0.7  # default
    if has_image:
        if params.strategy in ("side_by_side", "text_wrap", "grid"):
            strategy_score = 1.0
        elif params.strategy in ("single_column", "emphasis"):
            strategy_score = 0.8
    else:
        if params.strategy in ("single_column", "two_column", "compact"):
            strategy_score = 1.0
        elif params.strategy in ("emphasis",):
            strategy_score = 0.9
        elif params.strategy in ("side_by_side", "text_wrap", "grid"):
            strategy_score = 0.5  # these need images
    score += strategy_score * 0.25

    # 3. Readability (0.20)
    read_score = 0.8
    if params.font_profile == "compact" and total_chars > 800:
        read_score = 0.6  # too much small text
    elif params.font_profile == "large" and total_chars > 600:
        read_score = 0.7  # large font + lots of text = overflow
    elif params.font_profile == "normal":
        read_score = 1.0
    score += read_score * 0.20

    # 4. Truncation appropriateness (0.15)
    has_knowledge = block_info.get("has_knowledge", False)
    if has_knowledge and params.truncation == "heavy" and total_chars < 400:
        trunc_score = 0.5  # over-truncating short content
    elif not has_knowledge and params.truncation != "none":
        trunc_score = 0.8  # truncating when nothing to truncate
    else:
        trunc_score = 1.0
    score += trunc_score * 0.15

    # 5. Token efficiency estimate (0.15)
    # Rough: narrow→small canvas, wide→large
    est_tokens = {"narrow": 600, "medium": 900, "wide": 1400}
    t = est_tokens.get(params.width_class, 900)
    token_eff = 1.0 if t <= 1280 else max(0, 1.0 - (t - 1280) / 1280)
    score += token_eff * 0.15

    return score * 0.3  # scale to [0, 0.3]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
    **kwargs,
) -> float:
    """
    Compute reward for Layout Agent response.

    Args:
        data_source: "layout_agent"
        solution_str: Agent's JSON layout specification
        ground_truth: JSON with qa_reward_table + block_info
        extra_info: Optional metadata

    Returns:
        float: Score in [0.0, 1.0]
    """
    # Parse ground truth
    try:
        gt = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    qa_table = gt.get("qa_table", {})      # config_hash → qa_accuracy
    block_info = gt.get("block_info", {})   # content metadata

    # --- Component 1: Format score (0.1) ---
    parsed = parse_layout_json(solution_str)
    if parsed is None:
        return 0.0

    if not validate_layout_params(parsed):
        return 0.02  # partial credit for valid JSON but bad values

    format_score = 0.1

    # --- Component 2: Quality score (0.3) ---
    params = LayoutParams.from_dict(parsed)
    quality_score = compute_quality_from_params(params, block_info)

    # --- Component 3: QA score (0.6) ---
    config_hash = params.config_hash()

    if config_hash in qa_table:
        # Exact match in pre-computed table
        qa_accuracy = qa_table[config_hash]
        qa_score = qa_accuracy * 0.6
    else:
        # No exact match — find nearest by strategy + width
        partial_key = f"{params.strategy}_{params.width_class}"
        matching = {k: v for k, v in qa_table.items() if k.startswith(partial_key)}
        if matching:
            qa_accuracy = sum(matching.values()) / len(matching)
            qa_score = qa_accuracy * 0.6 * 0.9  # 10% penalty for inexact match
        else:
            # Fallback: average across all strategies with same width
            width_match = {k: v for k, v in qa_table.items()
                          if f"_{params.width_class}_" in k}
            if width_match:
                qa_accuracy = sum(width_match.values()) / len(width_match)
                qa_score = qa_accuracy * 0.6 * 0.7  # 30% penalty
            else:
                qa_score = 0.3 * 0.6  # assume 30% baseline if no match at all

    total = format_score + quality_score + qa_score
    return min(total, 1.0)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Layout Reward — Self-test")
    print("=" * 60)

    # Mock ground truth with QA table
    gt = json.dumps({
        "qa_table": {
            "single_column_medium_medium_balanced_none_normal": 0.85,
            "compact_narrow_medium_balanced_none_normal": 0.82,
            "side_by_side_medium_medium_balanced_none_normal": 0.90,
            "emphasis_medium_medium_balanced_none_normal": 0.88,
        },
        "block_info": {
            "total_chars": 500,
            "has_image": True,
            "n_blocks": 5,
            "has_knowledge": True,
        }
    })

    # Test 1: Perfect match
    resp1 = '{"strategy": "side_by_side", "width_class": "medium", "image_scale": "medium", "content_priority": "balanced", "truncation": "none", "font_profile": "normal"}'
    s1 = compute_score("layout_agent", resp1, gt)
    print(f"Test 1 (exact match, best QA):  {s1:.3f}")

    # Test 2: Valid but suboptimal
    resp2 = '{"strategy": "compact", "width_class": "narrow"}'
    s2 = compute_score("layout_agent", resp2, gt)
    print(f"Test 2 (compact narrow):        {s2:.3f}")

    # Test 3: Inexact match (partial key lookup)
    resp3 = '{"strategy": "single_column", "width_class": "medium", "image_scale": "large", "truncation": "light"}'
    s3 = compute_score("layout_agent", resp3, gt)
    print(f"Test 3 (inexact match):         {s3:.3f}")

    # Test 4: Unparseable
    resp4 = "I think single column is best"
    s4 = compute_score("layout_agent", resp4, gt)
    print(f"Test 4 (unparseable):           {s4:.3f}")

    # Test 5: Valid JSON but invalid strategy
    resp5 = '{"strategy": "zigzag", "width_class": "medium"}'
    s5 = compute_score("layout_agent", resp5, gt)
    print(f"Test 5 (invalid strategy):      {s5:.3f}")

    # Test 6: Markdown-wrapped JSON
    resp6 = '```json\n{"strategy": "emphasis", "width_class": "medium"}\n```'
    s6 = compute_score("layout_agent", resp6, gt)
    print(f"Test 6 (markdown wrapped):      {s6:.3f}")

    print("\nAll tests passed!")
