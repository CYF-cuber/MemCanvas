"""Evaluation metrics for MemCanvas experiments."""

from __future__ import annotations

import re
from collections import Counter


def normalize_answer(value: object) -> str:
    text = str(value).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def exact_match(prediction: object, ground_truth: object) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: object, ground_truth: object) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens) if pred_tokens else 0.0
    recall = n_common / len(gold_tokens) if gold_tokens else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def vqa_accuracy(prediction: object, answers: list[object]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(answer) for answer in answers))
