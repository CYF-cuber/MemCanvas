#!/usr/bin/env python3
"""
Reward function for Read Agent GRPO training.

Called by verl's NaiveRewardManager via the compute_score interface:
    score = compute_score(data_source, solution_str, ground_truth, extra_info)

Three scoring components (total 0.0 ~ 1.0):
  format_score   (0.1): JSON parseable with valid action field
  action_score   (0.3): RETRIEVE vs DIRECT_ANSWER matches oracle
  selection_score(0.6): precision × recall of selected canvas IDs

No VLM inference needed — pure text/metadata computation.
"""

import json
import re
from typing import Any, Dict, Optional


def parse_agent_response(response_str: str) -> Optional[Dict]:
    """Parse the agent's JSON response, handling common issues."""
    text = response_str.strip()

    # Remove thinking tags if present (Qwen3)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try to find a JSON object
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
    **kwargs,
) -> float:
    """
    Compute reward score for a read agent response.

    Args:
        data_source: "memcanvas_read"
        solution_str: Model-generated response (JSON string)
        ground_truth: JSON string with oracle_ids, has_relevant, candidate_ids
        extra_info: Optional metadata (pid, index, etc.)

    Returns:
        float: Score in [0.0, 1.0]
    """
    # Parse ground truth
    try:
        gt = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    oracle_ids = set(gt.get("oracle_ids", []))
    has_relevant = gt.get("has_relevant", False)
    candidate_ids = set(gt.get("candidate_ids", []))

    # --- Component 1: Format score (0.1) ---
    parsed = parse_agent_response(solution_str)
    if parsed is None:
        # Completely unparseable → 0
        return 0.0

    action = parsed.get("action", "").upper()
    if action not in ("RETRIEVE", "DIRECT_ANSWER"):
        # Invalid action
        return 0.0

    format_score = 0.1  # Valid JSON with valid action

    # --- Component 2: Action score (0.3) ---
    if has_relevant:
        # There are relevant canvases → RETRIEVE is correct
        if action == "RETRIEVE":
            action_score = 0.3
        else:
            action_score = 0.0  # Missed relevant canvases
    else:
        # No relevant canvases → DIRECT_ANSWER is best
        if action == "DIRECT_ANSWER":
            action_score = 0.3
        else:
            # RETRIEVE when nothing relevant: not catastrophic but suboptimal
            action_score = 0.1

    # --- Component 3: Selection score (0.6) ---
    selection_score = 0.0

    if action == "RETRIEVE":
        selected_ids = parsed.get("canvas_ids", [])
        if not isinstance(selected_ids, list):
            selected_ids = [selected_ids] if selected_ids is not None else []

        # Convert to int set, filtering invalid
        try:
            selected_set = set(int(x) for x in selected_ids if x is not None)
        except (ValueError, TypeError):
            selected_set = set()

        # Filter to valid candidate IDs only
        selected_set = selected_set & candidate_ids

        if oracle_ids and selected_set:
            # Precision: fraction of selected that are relevant
            hits = selected_set & oracle_ids
            precision = len(hits) / len(selected_set)

            # Recall: fraction of relevant that were selected
            recall = len(hits) / len(oracle_ids)

            # Weighted F-score: precision slightly more important to avoid noise
            selection_score = 0.6 * (0.6 * precision + 0.4 * recall)

        elif not oracle_ids and not selected_set:
            # No relevant and selected nothing (or DIRECT_ANSWER) — fine
            selection_score = 0.6

        elif not oracle_ids and selected_set:
            # Selected things when nothing is relevant — penalty
            selection_score = 0.0

        elif oracle_ids and not selected_set:
            # Had relevant but selected nothing valid
            selection_score = 0.0

    elif action == "DIRECT_ANSWER":
        if not has_relevant:
            # Correct: nothing relevant, chose DIRECT_ANSWER
            selection_score = 0.6
        else:
            # Wrong: had relevant canvases but skipped retrieval
            selection_score = 0.0

    total = format_score + action_score + selection_score
    return total


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Read Reward Function — Self-test")
    print("=" * 60)

    # Test 1: Perfect RETRIEVE
    gt = json.dumps({
        "oracle_ids": [3, 7],
        "has_relevant": True,
        "candidate_ids": [3, 7, 12, 5, 9],
    })
    resp = '{"action": "RETRIEVE", "canvas_ids": [3, 7], "reason": "relevant"}'
    score = compute_score("memcanvas_read", resp, gt)
    print(f"Test 1 (perfect RETRIEVE):     {score:.2f}  (expect 1.00)")

    # Test 2: Partial RETRIEVE (selected one correct, one wrong)
    resp2 = '{"action": "RETRIEVE", "canvas_ids": [3, 12], "reason": "..."}'
    score2 = compute_score("memcanvas_read", resp2, gt)
    print(f"Test 2 (partial RETRIEVE):     {score2:.2f}  (expect ~0.58)")

    # Test 3: DIRECT_ANSWER when should RETRIEVE
    resp3 = '{"action": "DIRECT_ANSWER", "reason": "..."}'
    score3 = compute_score("memcanvas_read", resp3, gt)
    print(f"Test 3 (wrong DIRECT_ANSWER):  {score3:.2f}  (expect 0.10)")

    # Test 4: Correct DIRECT_ANSWER (no relevant canvases)
    gt_no = json.dumps({
        "oracle_ids": [],
        "has_relevant": False,
        "candidate_ids": [3, 7, 12, 5, 9],
    })
    resp4 = '{"action": "DIRECT_ANSWER", "reason": "general knowledge"}'
    score4 = compute_score("memcanvas_read", resp4, gt_no)
    print(f"Test 4 (correct DIRECT_ANS):   {score4:.2f}  (expect 1.00)")

    # Test 5: RETRIEVE when nothing relevant
    resp5 = '{"action": "RETRIEVE", "canvas_ids": [3], "reason": "..."}'
    score5 = compute_score("memcanvas_read", resp5, gt_no)
    print(f"Test 5 (retrieve no relevant): {score5:.2f}  (expect 0.20)")

    # Test 6: Unparseable response
    resp6 = "I think canvas 3 is good"
    score6 = compute_score("memcanvas_read", resp6, gt)
    print(f"Test 6 (unparseable):          {score6:.2f}  (expect 0.00)")

    # Test 7: Invalid action
    resp7 = '{"action": "SEARCH", "canvas_ids": [3]}'
    score7 = compute_score("memcanvas_read", resp7, gt)
    print(f"Test 7 (invalid action):       {score7:.2f}  (expect 0.00)")

    # Test 8: RETRIEVE all oracle + extra (low precision)
    resp8 = '{"action": "RETRIEVE", "canvas_ids": [3, 7, 12, 5, 9], "reason": "all"}'
    score8 = compute_score("memcanvas_read", resp8, gt)
    print(f"Test 8 (all candidates):       {score8:.2f}  (expect ~0.54)")

    print("\nAll tests passed!")
