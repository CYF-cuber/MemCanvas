#!/usr/bin/env python3
"""
VLM Prompt Improvement Experiment for Memory Canvas on ScienceQA.

Tests 4 prompt variants to measure how prompt design affects
memory-augmented VLM accuracy:

  v0 (current)       — existing prompt, already evaluated (80.71%)
  v1 (guided)        — tell VLM what the canvas contains and how to use it
  v2 (guided+CoT)    — v1 + allow step-by-step reasoning before answering
  v3 (sys+guided+CoT)— v2 + a system prompt

Conditions evaluated:
  - Oracle: all 12,726 memories, PNG quality, top-2 retrieval
  - Baseline: no memory (also tested with each prompt variant)

Usage:
  CUDA_VISIBLE_DEVICES=0 python /home/cyf/codex/prompt_improvement_eval.py
  CUDA_VISIBLE_DEVICES=0 python /home/cyf/codex/prompt_improvement_eval.py --resume /path/to/output_dir
  CUDA_VISIBLE_DEVICES=0 python /home/cyf/codex/prompt_improvement_eval.py --variants v1 v2
"""

import argparse
import copy
import json
import os
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Imports from existing experiment infrastructure
# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/cyf/memory/memory_canvas/experiments")
from scienceqa_qwen25vl_full_experiment import (  # noqa: E402
    CLIPLargeMemoryBuilder,
    ExperimentConfig,
    MemoryEntry,  # needed for pickle
    MemoryIndex,
    Qwen25VLEvaluator,
    ScienceQADataLoader,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MEMORY_INDEX = (
    "/home/cyf/memory/experiments/scienceqa_qwen25vl_full/memory_index_qwen25vl_full.pkl"
)
DEFAULT_OUTPUT_ROOT = "/home/cyf/codex"
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]

# v0 known result — skip re-running unless forced
V0_KNOWN_RESULT = {
    "oracle": {"accuracy": 80.71},
    "baseline": {"accuracy": 78.73},
}


# ---------------------------------------------------------------------------
# Memory index loading (same as memory_forgetting_eval.py)
# ---------------------------------------------------------------------------
def load_memory_index(path: str) -> Tuple[List, np.ndarray, int]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    memories = data["memories"]
    embeddings = data["embeddings"]
    embedding_dim = data.get("embedding_dim", 768)
    return memories, embeddings, embedding_dim


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
def extract_answer_first(response: str) -> str:
    """Extract answer from short response: first A-F character found."""
    response = response.upper().strip()
    for char in response:
        if char in "ABCDEF":
            return char
    return "A"


def extract_answer_last(response: str) -> str:
    """Extract answer from CoT response: last A-F character on its own."""
    response = response.strip()
    lines = response.strip().split("\n")
    # Search from last line backwards for common answer patterns
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        # Pattern 1: "The answer is X", "Answer: X", "answer is X"
        m = re.search(r'answer\s*(?:is|:)\s*\(?([A-Fa-f])\)?', line, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Pattern 2: standalone letter possibly with markdown bold: "A", "**A**", "(A)", "A."
        m = re.match(r'^[\s*\(\[]*([A-Fa-f])[\s*\)\].:]*$', line)
        if m:
            return m.group(1).upper()
    # Fallback: find the last standalone A-F in the entire response
    # Match letters surrounded by word boundaries or common delimiters
    matches = re.findall(r'(?:^|[\s*\(\[])([A-Fa-f])(?:[\s*\)\].,;:!?]|$)', response)
    if matches:
        return matches[-1].upper()
    # Last resort: any A-F character, scanning from end
    for char in reversed(response.upper()):
        if char in "ABCDEF":
            return char
    return "A"


# ---------------------------------------------------------------------------
# Prompt variant definitions
# ---------------------------------------------------------------------------

def build_prompt_v0(
    problem: Dict,
    retrieved_memories: Optional[List[Tuple[MemoryEntry, float]]],
) -> Tuple[Optional[str], str, List[Image.Image]]:
    """
    v0 (current): Original prompt from the experiment.
    Returns (system_prompt, user_prompt, memory_images).
    """
    question = problem.get("question", "")
    choices = problem.get("choices", [])
    hint = problem.get("hint", "")
    choices_text = "\n".join(
        f"{CHOICE_LABELS[i]}. {c}" for i, c in enumerate(choices) if i < len(CHOICE_LABELS)
    )

    memory_images = []
    prompt_parts = []

    if retrieved_memories:
        prompt_parts.append("Below are reference examples shown as images:")
        for i, (mem, sim) in enumerate(retrieved_memories):
            canvas_img = mem.get_canvas_image()
            if canvas_img:
                memory_images.append(canvas_img)
                prompt_parts.append(f"Reference {i+1} ({mem.subject} - {mem.topic})")
        prompt_parts.append("")
        prompt_parts.append("---")

    prompt_parts.append("Answer the following question by selecting the correct option.")
    if hint:
        prompt_parts.append(f"Context: {hint}")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append(f"Options:\n{choices_text}")
    prompt_parts.append("Answer with just the letter (A, B, C, D, etc.):")

    return None, "\n".join(prompt_parts), memory_images


def build_prompt_v1(
    problem: Dict,
    retrieved_memories: Optional[List[Tuple[MemoryEntry, float]]],
) -> Tuple[Optional[str], str, List[Image.Image]]:
    """
    v1 (guided): Tell VLM what the canvas contains and how to use it.
    """
    question = problem.get("question", "")
    choices = problem.get("choices", [])
    hint = problem.get("hint", "")
    choices_text = "\n".join(
        f"{CHOICE_LABELS[i]}. {c}" for i, c in enumerate(choices) if i < len(CHOICE_LABELS)
    )

    memory_images = []
    prompt_parts = []

    if retrieved_memories:
        prompt_parts.append(
            "Below are memory canvas images from similar problems you solved before."
        )
        prompt_parts.append(
            "Each canvas contains: the original question, answer choices "
            "(correct answer marked with \u2713), background knowledge, and solution reasoning."
        )
        prompt_parts.append(
            "Study these canvases and use the knowledge to help answer the new question."
        )
        prompt_parts.append("")
        for i, (mem, sim) in enumerate(retrieved_memories):
            canvas_img = mem.get_canvas_image()
            if canvas_img:
                memory_images.append(canvas_img)
                prompt_parts.append(f"[Canvas {i+1}] ({mem.subject} - {mem.topic})")
        prompt_parts.append("")
        prompt_parts.append("---")

    prompt_parts.append("Answer the following question by selecting the correct option.")
    if hint:
        prompt_parts.append(f"Context: {hint}")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append(f"Options:\n{choices_text}")
    prompt_parts.append("Answer with just the letter (A, B, C, D, etc.):")

    return None, "\n".join(prompt_parts), memory_images


def build_prompt_v2(
    problem: Dict,
    retrieved_memories: Optional[List[Tuple[MemoryEntry, float]]],
) -> Tuple[Optional[str], str, List[Image.Image]]:
    """
    v2 (guided+CoT): v1 + allow step-by-step reasoning.
    """
    question = problem.get("question", "")
    choices = problem.get("choices", [])
    hint = problem.get("hint", "")
    choices_text = "\n".join(
        f"{CHOICE_LABELS[i]}. {c}" for i, c in enumerate(choices) if i < len(CHOICE_LABELS)
    )

    memory_images = []
    prompt_parts = []

    if retrieved_memories:
        prompt_parts.append(
            "Below are memory canvas images from similar problems you solved before."
        )
        prompt_parts.append(
            "Each canvas contains: the original question, answer choices "
            "(correct answer marked with \u2713), background knowledge, and solution reasoning."
        )
        prompt_parts.append(
            "Study these canvases carefully and apply the relevant knowledge to the new question."
        )
        prompt_parts.append("")
        for i, (mem, sim) in enumerate(retrieved_memories):
            canvas_img = mem.get_canvas_image()
            if canvas_img:
                memory_images.append(canvas_img)
                prompt_parts.append(f"[Canvas {i+1}] ({mem.subject} - {mem.topic})")
        prompt_parts.append("")
        prompt_parts.append("---")

    prompt_parts.append("Answer the following question by selecting the correct option.")
    if hint:
        prompt_parts.append(f"Context: {hint}")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append(f"Options:\n{choices_text}")
    prompt_parts.append(
        "Think step by step, then give your final answer as a single letter "
        "(A, B, C, D, etc.) on the last line."
    )

    return None, "\n".join(prompt_parts), memory_images


def build_prompt_v3(
    problem: Dict,
    retrieved_memories: Optional[List[Tuple[MemoryEntry, float]]],
) -> Tuple[Optional[str], str, List[Image.Image]]:
    """
    v3 (sys+guided+CoT): v2 + system prompt.
    """
    system_prompt = (
        "You are a knowledgeable student who learns from past examples. "
        "When given reference canvases from similar problems, carefully study "
        "the solutions and background knowledge shown in them, then apply that "
        "knowledge to solve new problems."
    )
    # User prompt is the same as v2
    _, user_prompt, memory_images = build_prompt_v2(problem, retrieved_memories)
    return system_prompt, user_prompt, memory_images


# Registry of prompt builders
PROMPT_BUILDERS = {
    "v0": build_prompt_v0,
    "v1": build_prompt_v1,
    "v2": build_prompt_v2,
    "v3": build_prompt_v3,
}

# Which variants use CoT (need last-letter extraction and longer generation)
COT_VARIANTS = {"v2", "v3"}


# ---------------------------------------------------------------------------
# Custom VLM prediction
# ---------------------------------------------------------------------------
def predict_with_variant(
    vlm: Qwen25VLEvaluator,
    problem: Dict,
    retrieved_memories: Optional[List[Tuple[MemoryEntry, float]]],
    variant: str,
) -> Tuple[str, str]:
    """
    Run VLM prediction with a specific prompt variant.
    Returns (extracted_answer, raw_response).
    """
    builder = PROMPT_BUILDERS[variant]
    system_prompt, user_prompt, memory_images = builder(problem, retrieved_memories)

    # Build messages
    content = []
    for img in memory_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": user_prompt})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    # Apply chat template
    text = vlm.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if memory_images:
        inputs = vlm.processor(
            text=[text], images=memory_images, return_tensors="pt", padding=True
        )
    else:
        inputs = vlm.processor(text=[text], return_tensors="pt", padding=True)

    inputs = {k: v.to(vlm.model.device) for k, v in inputs.items()}

    max_new_tokens = 512 if variant in COT_VARIANTS else 20

    with torch.no_grad():
        outputs = vlm.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw_response = vlm.processor.decode(generated_ids, skip_special_tokens=True).strip()

    # Extract answer
    if variant in COT_VARIANTS:
        answer = extract_answer_last(raw_response)
    else:
        answer = extract_answer_first(raw_response)

    return answer, raw_response


# ---------------------------------------------------------------------------
# Pre-compute retrieval (reused from memory_forgetting_eval.py)
# ---------------------------------------------------------------------------
def precompute_retrieval_top2(
    test_pids: List[str],
    test_data: Dict,
    text_encoder: CLIPLargeMemoryBuilder,
    mem_index: MemoryIndex,
    top_k: int = 2,
    threshold: float = 0.1,
) -> Dict[str, List[Tuple[MemoryEntry, float]]]:
    """
    Pre-compute top-k retrieval for all test samples.
    Returns mapping pid -> list of (MemoryEntry, similarity).
    """
    retrieved_map: Dict[str, List[Tuple[MemoryEntry, float]]] = {}

    for pid in tqdm(test_pids, desc="Pre-computing retrieval"):
        problem = test_data[pid]
        question_text = (
            problem.get("question", "") + " " + problem.get("hint", "")
        ).strip()
        query_embedding = text_encoder.encode_text_query(question_text)
        retrieved = mem_index.search(query_embedding, top_k=top_k + 2, threshold=threshold)
        # Filter out self
        filtered = [(m, s) for m, s in retrieved if m.pid != pid][:top_k]
        retrieved_map[pid] = filtered

    return retrieved_map


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
class CheckpointManager:
    """Manages checkpoint save/load for evaluation progress."""

    def __init__(self, output_dir: Path):
        self.path = output_dir / "checkpoint.json"
        self.data: Dict[str, Any] = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            print(f"  Checkpoint loaded: {self.path}")

    def get_variant_progress(self, variant: str, condition: str) -> Dict:
        """Get progress for a specific (variant, condition) combination."""
        key = f"{variant}_{condition}"
        return self.data.get(key, {})

    def save_variant_progress(
        self, variant: str, condition: str, progress: Dict
    ) -> None:
        key = f"{variant}_{condition}"
        self.data[key] = progress
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def is_complete(self, variant: str, condition: str) -> bool:
        key = f"{variant}_{condition}"
        entry = self.data.get(key, {})
        return entry.get("complete", False)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate_variant(
    variant: str,
    condition: str,  # "oracle" or "baseline"
    vlm: Qwen25VLEvaluator,
    test_pids: List[str],
    test_data: Dict,
    retrieved_map: Optional[Dict[str, List[Tuple[MemoryEntry, float]]]],
    ckpt: CheckpointManager,
    save_interval: int = 100,
) -> Dict:
    """
    Evaluate a single (variant, condition) combination.
    Supports checkpoint/resume.
    """
    label = f"{variant}_{condition}"

    # Check if already complete
    if ckpt.is_complete(variant, condition):
        progress = ckpt.get_variant_progress(variant, condition)
        print(f"  [{label}] Already complete: {progress['accuracy']:.2f}%")
        return progress

    # Load partial progress
    progress = ckpt.get_variant_progress(variant, condition)
    predictions = progress.get("predictions", [])
    start_idx = len(predictions)
    correct = progress.get("correct", 0)
    total = progress.get("total", 0)

    if start_idx > 0:
        print(f"  [{label}] Resuming from sample {start_idx}/{len(test_pids)}")

    for idx in tqdm(
        range(start_idx, len(test_pids)),
        desc=f"{label}",
        initial=start_idx,
        total=len(test_pids),
    ):
        pid = test_pids[idx]
        problem = test_data[pid]
        answer_idx = problem.get("answer", 0)
        correct_answer = (
            CHOICE_LABELS[answer_idx] if answer_idx < len(CHOICE_LABELS) else "A"
        )

        # Get memories for oracle, None for baseline
        memories_for_query = None
        if condition == "oracle" and retrieved_map is not None:
            memories_for_query = retrieved_map.get(pid, [])
            if not memories_for_query:
                memories_for_query = None

        try:
            pred, raw_response = predict_with_variant(
                vlm, problem, memories_for_query, variant
            )
        except Exception as e:
            print(f"\n  Warning: prediction failed for {pid}: {e}")
            pred = "A"
            raw_response = f"ERROR: {e}"

        is_correct = pred == correct_answer
        correct += int(is_correct)
        total += 1

        pred_entry = {
            "pid": pid,
            "subject": problem.get("subject", ""),
            "predicted": pred,
            "correct": correct_answer,
            "is_correct": is_correct,
        }
        # Only save raw_response for CoT variants (they're longer and more interesting)
        if variant in COT_VARIANTS:
            # Truncate to save space
            pred_entry["raw_response"] = raw_response[:500]

        predictions.append(pred_entry)

        # Periodic save
        if (idx + 1) % save_interval == 0 or idx == len(test_pids) - 1:
            acc = correct / total * 100 if total > 0 else 0
            ckpt.save_variant_progress(variant, condition, {
                "correct": correct,
                "total": total,
                "accuracy": acc,
                "predictions": predictions,
                "complete": False,
            })
            if (idx + 1) % save_interval == 0:
                print(f"\n  [{label}] Progress: {idx+1}/{len(test_pids)}, acc={acc:.2f}%")

    accuracy = correct / total * 100 if total > 0 else 0

    # Mark complete
    result = {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "predictions": predictions,
        "complete": True,
    }
    ckpt.save_variant_progress(variant, condition, result)

    return result


# ---------------------------------------------------------------------------
# Subject-level analysis
# ---------------------------------------------------------------------------
def compute_subject_stats(predictions: List[Dict]) -> Dict[str, Dict]:
    """Compute per-subject accuracy from predictions."""
    stats: Dict[str, Dict] = {}
    for p in predictions:
        subj = p.get("subject", "unknown")
        if subj not in stats:
            stats[subj] = {"correct": 0, "total": 0}
        stats[subj]["total"] += 1
        if p["is_correct"]:
            stats[subj]["correct"] += 1
    for subj in stats:
        s = stats[subj]
        s["accuracy"] = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
    return stats


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(all_results: Dict, output_dir: Path) -> Path:
    """Generate a Chinese markdown report of the experiment."""
    report_path = output_dir / "prompt_report.md"
    lines = []

    lines.append("# VLM textexperiment report")
    lines.append("")
    lines.append(f"*text: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. text")
    lines.append("")
    lines.append("text Memory Canvas text VLM text ScienceQA text。")
    lines.append("text（v0）textVLMtext（text、text、text），")
    lines.append("causesOracleaccuracytext80.71%，compared withno memorybaseline(78.73%)limited improvement。")
    lines.append("")

    lines.append("## 2. textvariant")
    lines.append("")
    lines.append("| variant | text | max_new_tokens | text |")
    lines.append("|------|------|---------------|---------|")
    lines.append("| v0 (current) | text | 20 | textA-Ftext |")
    lines.append("| v1 (guided) | textVLMtext | 20 | textA-Ftext |")
    lines.append("| v2 (guided+CoT) | v1 + text | 512 | textA-Ftext |")
    lines.append("| v3 (sys+guided+CoT) | v2 + system prompt | 512 | textA-Ftext |")
    lines.append("")

    lines.append("## 3. text")
    lines.append("")
    lines.append("- **Oracle**: all12,726memories，PNGquality，top-2text")
    lines.append("- **Baseline**: no memory")
    lines.append("- text: Qwen2.5-VL-7B-Instruct")
    lines.append("- text: CLIP-L/14 (768text)")
    lines.append("")

    lines.append("## 4. text")
    lines.append("")
    lines.append("| variant | Baseline (no memory) | Oracle (with memory) | memory gain |")
    lines.append("|------|------------------|----------------|---------|")

    variants_in_results = sorted(
        set(k.split("_")[0] for k in all_results.keys()),
        key=lambda x: ["v0", "v1", "v2", "v3"].index(x) if x in ["v0", "v1", "v2", "v3"] else 99,
    )

    for variant in variants_in_results:
        baseline_key = f"{variant}_baseline"
        oracle_key = f"{variant}_oracle"
        b_acc = all_results.get(baseline_key, {}).get("accuracy", 0)
        o_acc = all_results.get(oracle_key, {}).get("accuracy", 0)
        improvement = o_acc - b_acc
        lines.append(
            f"| {variant} | {b_acc:.2f}% | {o_acc:.2f}% | {improvement:+.2f}% |"
        )
    lines.append("")

    # Best variant analysis
    oracle_accs = {
        k.split("_")[0]: v["accuracy"]
        for k, v in all_results.items()
        if k.endswith("_oracle") and v.get("accuracy") is not None
    }
    if oracle_accs:
        best_variant = max(oracle_accs, key=oracle_accs.get)
        lines.append(f"**textOraclevariant**: {best_variant} ({oracle_accs[best_variant]:.2f}%)")
        lines.append("")

    # Subject-level breakdown for each variant
    lines.append("## 5. text (Oracletext)")
    lines.append("")

    for variant in variants_in_results:
        oracle_key = f"{variant}_oracle"
        result = all_results.get(oracle_key, {})
        preds = result.get("predictions", [])
        if not preds:
            continue

        subject_stats = compute_subject_stats(preds)
        lines.append(f"### {variant}")
        lines.append("")
        lines.append("| text | text | text | accuracy |")
        lines.append("|------|------|------|--------|")
        for subj in sorted(subject_stats.keys()):
            s = subject_stats[subj]
            lines.append(f"| {subj} | {s['correct']} | {s['total']} | {s['accuracy']:.2f}% |")
        lines.append("")

    # CoT answer extraction analysis
    lines.append("## 6. CoT text")
    lines.append("")
    for variant in ["v2", "v3"]:
        oracle_key = f"{variant}_oracle"
        result = all_results.get(oracle_key, {})
        preds = result.get("predictions", [])
        if not preds:
            continue
        with_response = [p for p in preds if "raw_response" in p]
        if not with_response:
            continue
        avg_len = np.mean([len(p["raw_response"]) for p in with_response])
        lines.append(f"### {variant}")
        lines.append(f"- text: {avg_len:.0f} text (text500)")
        lines.append(f"- text: {len(with_response)}")
        # Show a few examples
        lines.append("")
        lines.append("text（text3text）:")
        lines.append("")
        for p in with_response[:3]:
            lines.append(f"- **PID {p['pid']}**: predicted={p['predicted']}, "
                        f"correct={p['correct']}, is_correct={p['is_correct']}")
            lines.append(f"  ```")
            lines.append(f"  {p['raw_response'][:200]}...")
            lines.append(f"  ```")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*text: NVIDIA GPU, Qwen2.5-VL-7B-Instruct, CLIP-L/14*")

    report_content = "\n".join(lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n  Report saved: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="VLM Prompt Improvement Experiment for ScienceQA"
    )
    parser.add_argument(
        "--memory-index", default=DEFAULT_MEMORY_INDEX,
        help="Path to memory index pickle",
    )
    parser.add_argument(
        "--output-root", default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for output",
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="Resume from existing output directory",
    )
    parser.add_argument(
        "--variants", nargs="+", default=["v1", "v2", "v3"],
        help="Which prompt variants to evaluate (default: v1 v2 v3; v0 uses known result)",
    )
    parser.add_argument(
        "--max-test-samples", type=int, default=0,
        help="Limit test samples (0 = full set)",
    )
    parser.add_argument(
        "--top-k", type=int, default=2,
        help="Number of memories to retrieve per test sample",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="Similarity threshold for retrieval",
    )
    parser.add_argument(
        "--save-interval", type=int, default=100,
        help="Save checkpoint every N samples",
    )

    args = parser.parse_args()

    # Output directory
    if args.resume:
        output_dir = Path(args.resume)
        assert output_dir.exists(), f"Resume dir not found: {output_dir}"
        print(f"Resuming from: {output_dir}")
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_root) / f"prompt_eval_{ts}"
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Determine which variants to actually run
    variants_to_run = list(args.variants)
    # Always include v0 in results (from known data)
    all_variants = ["v0"] + [v for v in variants_to_run if v != "v0"]

    print(f"\nVariants to evaluate: {variants_to_run}")
    print(f"All variants for report: {all_variants}")

    # Load memory index
    print(f"\nLoading memory index: {args.memory_index}")
    memories, embeddings, embedding_dim = load_memory_index(args.memory_index)
    print(f"  Loaded {len(memories)} memories, embedding dim={embedding_dim}")

    # Load test data
    loader = ScienceQADataLoader()
    test_data = loader.get_split("test")
    try:
        test_pids = sorted(test_data.keys(), key=lambda x: int(x))
    except Exception:
        test_pids = sorted(test_data.keys())

    if args.max_test_samples and args.max_test_samples > 0:
        test_pids = test_pids[: args.max_test_samples]
    print(f"  Test samples: {len(test_pids)}")

    # Initialize CLIP text encoder
    exp_config = ExperimentConfig(
        max_memories_to_retrieve=args.top_k,
        similarity_threshold=args.threshold,
    )
    print("\nLoading CLIP text encoder...")
    text_encoder = CLIPLargeMemoryBuilder(exp_config)

    # Build MemoryIndex for retrieval
    mem_index = MemoryIndex(embedding_dim=embedding_dim)
    mem_index.memories = memories
    mem_index.embeddings = embeddings

    # Pre-compute retrieval
    print("\nPre-computing retrieval for all test samples...")
    retrieved_map = precompute_retrieval_top2(
        test_pids, test_data, text_encoder, mem_index,
        top_k=args.top_k, threshold=args.threshold,
    )
    print(f"  Retrieval computed for {len(retrieved_map)} test samples")

    # Free CLIP model to save GPU memory
    del text_encoder
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("  CLIP encoder freed from GPU")

    # Initialize VLM
    print("\nLoading VLM evaluator...")
    vlm = Qwen25VLEvaluator(exp_config)

    # Initialize checkpoint manager
    ckpt = CheckpointManager(output_dir)

    # Collect all results
    all_results: Dict[str, Dict] = {}

    # Add v0 known results
    if "v0" in all_variants:
        all_results["v0_baseline"] = {
            "correct": 0, "total": 0,
            "accuracy": V0_KNOWN_RESULT["baseline"]["accuracy"],
            "predictions": [], "complete": True,
            "source": "known_result",
        }
        all_results["v0_oracle"] = {
            "correct": 0, "total": 0,
            "accuracy": V0_KNOWN_RESULT["oracle"]["accuracy"],
            "predictions": [], "complete": True,
            "source": "known_result",
        }
        print(f"\n  v0 results from prior experiment: "
              f"baseline={V0_KNOWN_RESULT['baseline']['accuracy']:.2f}%, "
              f"oracle={V0_KNOWN_RESULT['oracle']['accuracy']:.2f}%")

    # Run evaluations for each variant
    for variant in variants_to_run:
        if variant == "v0":
            # v0 already added from known results, skip unless forced
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating variant: {variant}")
        print(f"{'='*60}")

        # Baseline (no memory)
        print(f"\n  --- {variant} baseline (no memory) ---")
        baseline_result = evaluate_variant(
            variant=variant,
            condition="baseline",
            vlm=vlm,
            test_pids=test_pids,
            test_data=test_data,
            retrieved_map=None,
            ckpt=ckpt,
            save_interval=args.save_interval,
        )
        all_results[f"{variant}_baseline"] = baseline_result
        print(f"  [{variant}_baseline] Accuracy: {baseline_result['accuracy']:.2f}%")

        # Oracle (with memory)
        print(f"\n  --- {variant} oracle (with memory) ---")
        oracle_result = evaluate_variant(
            variant=variant,
            condition="oracle",
            vlm=vlm,
            test_pids=test_pids,
            test_data=test_data,
            retrieved_map=retrieved_map,
            ckpt=ckpt,
            save_interval=args.save_interval,
        )
        all_results[f"{variant}_oracle"] = oracle_result
        print(f"  [{variant}_oracle] Accuracy: {oracle_result['accuracy']:.2f}%")

    # Save final results (without large predictions for the summary file)
    results_summary = {}
    for key, val in all_results.items():
        results_summary[key] = {
            "correct": val.get("correct", 0),
            "total": val.get("total", 0),
            "accuracy": val.get("accuracy", 0),
            "source": val.get("source", "evaluated"),
        }

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {results_path}")

    # Generate report
    generate_report(all_results, output_dir)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Variant':<25} {'Baseline':>12} {'Oracle':>12} {'Improvement':>12}")
    print("-" * 65)
    for variant in all_variants:
        b_key = f"{variant}_baseline"
        o_key = f"{variant}_oracle"
        b_acc = all_results.get(b_key, {}).get("accuracy", 0)
        o_acc = all_results.get(o_key, {}).get("accuracy", 0)
        imp = o_acc - b_acc
        print(f"{variant:<25} {b_acc:>11.2f}% {o_acc:>11.2f}% {imp:>+11.2f}%")
    print(f"\nResults directory: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
