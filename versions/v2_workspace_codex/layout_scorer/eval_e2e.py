#!/usr/bin/env python3
"""
Phase 3b: End-to-end ScienceQA evaluation with SmartCanvas + Scorer memories.

Loads the SmartCanvas memory index, then runs the standard evaluation pipeline:
    1. CLIP text query → retrieve top-k memories
    2. VLM (Qwen2.5-VL-7B) predicts answer with retrieved canvas images
    3. Compare with ground truth

Also runs No-Memory baseline for fair comparison.

Usage:
    python -m layout_scorer.eval_e2e [--max_test None] [--top_k 2]
"""

import io
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/codex")
sys.path.insert(0, "/home/cyf/memory")

from layout_scorer.build_memory import MemoryEntry, MemoryIndex

# Paths
SCIENCEQA_DIR = "/home/cyf/memory/ScienceQA"
PROBLEMS_PATH = f"{SCIENCEQA_DIR}/data/scienceqa/problems.json"
IMAGES_DIR = f"{SCIENCEQA_DIR}/data/scienceqa/images"

SMART_INDEX_PATH = "/home/cyf/memory/experiments/scienceqa_smart_scorer/memory_index_smart.pkl"
DYNAMIC_INDEX_PATH = "/home/cyf/memory/experiments/scienceqa_qwen25vl_full/memory_index_checkpoint_12500.pkl"
OUTPUT_DIR = "/home/cyf/memory/experiments/scienceqa_smart_scorer"


class Qwen25VLEvaluator:
    """Qwen2.5-VL evaluator — same interface as existing pipeline."""

    def __init__(self, model_path: str = "/home/cyf/Qwen2.5-VL-7B-Instruct"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._init_model()

    def _init_model(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"Loading VLM: {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print("  VLM loaded")

    def predict(self, problem: Dict, retrieved_memories=None,
                problem_image: Optional[Image.Image] = None) -> str:
        """Predict answer using VLM with optional retrieved memories."""
        question = problem.get('question', '')
        choices = problem.get('choices', [])
        hint = problem.get('hint', '')

        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        choices_text = '\n'.join([
            f"{choice_labels[i]}. {c}"
            for i, c in enumerate(choices) if i < len(choice_labels)
        ])

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

        prompt = '\n'.join(prompt_parts)

        content = []
        for img in memory_images:
            content.append({"type": "image", "image": img})
        if problem_image is not None:
            content.append({"type": "image", "image": problem_image})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        all_images = memory_images + ([problem_image] if problem_image else [])
        if all_images:
            inputs = self.processor(text=[text], images=all_images, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)

        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        return self._extract_answer(response)

    def _extract_answer(self, response: str) -> str:
        response = response.upper().strip()
        if response and response[0] in 'ABCDEF':
            return response[0]
        for char in response:
            if char in 'ABCDEF':
                return char
        return 'A'


def run_evaluation(
    memory_index: MemoryIndex,
    evaluator: Qwen25VLEvaluator,
    clip_model,
    clip_processor,
    problems: Dict,
    test_pids: List[str],
    top_k: int = 2,
    threshold: float = 0.1,
    device: str = "cuda",
    label: str = "SmartCanvas+Scorer",
) -> Dict:
    """Run evaluation with a memory index."""
    print(f"\nEvaluating: {label}")
    print(f"  Test samples: {len(test_pids)}, Top-k: {top_k}")

    results = {'correct': 0, 'total': 0, 'predictions': []}
    choice_labels = ['A', 'B', 'C', 'D', 'E', 'F']

    for idx in tqdm(range(len(test_pids)), desc=f"Eval {label}"):
        pid = test_pids[idx]
        problem = problems[pid]
        answer_idx = problem.get('answer', 0)
        correct_answer = choice_labels[answer_idx] if answer_idx < len(choice_labels) else 'A'

        # Encode query
        query_text = problem.get('question', '') + ' ' + problem.get('hint', '')
        inputs = clip_processor(text=query_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            query_feat = clip_model.get_text_features(**inputs)
        query_feat = query_feat / query_feat.norm(dim=-1, keepdim=True)
        query_embedding = query_feat.cpu().numpy().squeeze()

        # Retrieve
        retrieved = memory_index.search(query_embedding, top_k=top_k, threshold=threshold)
        retrieved = [(m, s) for m, s in retrieved if m.pid != pid]

        # Predict
        pred = evaluator.predict(problem, retrieved_memories=retrieved)

        is_correct = (pred == correct_answer)
        results['correct'] += int(is_correct)
        results['total'] += 1
        results['predictions'].append({
            'pid': pid,
            'predicted': pred,
            'correct': correct_answer,
            'is_correct': is_correct,
            'retrieved_count': len(retrieved),
        })

        if (idx + 1) % 100 == 0:
            acc = results['correct'] / results['total'] * 100
            print(f"  [{idx+1}/{len(test_pids)}] {label}: {acc:.2f}%")

    return results


def run_baseline(
    evaluator: Qwen25VLEvaluator,
    problems: Dict,
    test_pids: List[str],
) -> Dict:
    """Run no-memory baseline."""
    print(f"\nEvaluating: No-Memory Baseline")
    print(f"  Test samples: {len(test_pids)}")

    results = {'correct': 0, 'total': 0, 'predictions': []}
    choice_labels = ['A', 'B', 'C', 'D', 'E', 'F']

    for idx in tqdm(range(len(test_pids)), desc="Eval Baseline"):
        pid = test_pids[idx]
        problem = problems[pid]
        answer_idx = problem.get('answer', 0)
        correct_answer = choice_labels[answer_idx] if answer_idx < len(choice_labels) else 'A'

        pred = evaluator.predict(problem, retrieved_memories=None)

        is_correct = (pred == correct_answer)
        results['correct'] += int(is_correct)
        results['total'] += 1
        results['predictions'].append({
            'pid': pid,
            'predicted': pred,
            'correct': correct_answer,
            'is_correct': is_correct,
        })

        if (idx + 1) % 100 == 0:
            acc = results['correct'] / results['total'] * 100
            print(f"  [{idx+1}/{len(test_pids)}] Baseline: {acc:.2f}%")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_dynamic", action="store_true",
                        help="Skip DynamicCanvas comparison (use saved results)")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3b: End-to-End ScienceQA Evaluation")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading ScienceQA data...")
    with open(PROBLEMS_PATH) as f:
        problems = json.load(f)

    test_pids = [pid for pid, p in problems.items() if p.get('split') == 'test']
    if args.max_test:
        test_pids = test_pids[:args.max_test]
    print(f"Test samples: {len(test_pids)}")

    # Load CLIP model for query encoding
    from transformers import CLIPModel, CLIPProcessor
    print("Loading CLIP model...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(args.device)
    clip_model.eval()

    # Load SmartCanvas memory index
    print(f"Loading SmartCanvas index: {SMART_INDEX_PATH}")
    smart_index = MemoryIndex()
    smart_index.load(SMART_INDEX_PATH)

    # Load VLM evaluator
    evaluator = Qwen25VLEvaluator()

    all_results = {}

    # 1. No-Memory Baseline
    if not args.skip_baseline:
        baseline_results = run_baseline(evaluator, problems, test_pids)
        baseline_acc = baseline_results['correct'] / baseline_results['total'] * 100
        all_results['baseline'] = {
            'correct': baseline_results['correct'],
            'total': baseline_results['total'],
            'accuracy': baseline_acc,
        }
        print(f"\n  Baseline: {baseline_acc:.2f}%")
    else:
        # Use known result
        all_results['baseline'] = {'correct': 3339, 'total': 4241, 'accuracy': 78.73}

    # 2. SmartCanvas + Scorer
    smart_results = run_evaluation(
        smart_index, evaluator, clip_model, clip_processor,
        problems, test_pids, top_k=args.top_k, threshold=args.threshold,
        device=args.device, label="SmartCanvas+Scorer",
    )
    smart_acc = smart_results['correct'] / smart_results['total'] * 100
    all_results['smart_scorer'] = {
        'correct': smart_results['correct'],
        'total': smart_results['total'],
        'accuracy': smart_acc,
    }

    # 3. DynamicCanvas comparison (from saved results or re-run)
    if not args.skip_dynamic and os.path.exists(DYNAMIC_INDEX_PATH):
        print(f"\nLoading DynamicCanvas index: {DYNAMIC_INDEX_PATH}")
        dynamic_index = MemoryIndex()
        dynamic_index.load(DYNAMIC_INDEX_PATH)

        dynamic_results = run_evaluation(
            dynamic_index, evaluator, clip_model, clip_processor,
            problems, test_pids, top_k=args.top_k, threshold=args.threshold,
            device=args.device, label="DynamicCanvas",
        )
        dynamic_acc = dynamic_results['correct'] / dynamic_results['total'] * 100
        all_results['dynamic_canvas'] = {
            'correct': dynamic_results['correct'],
            'total': dynamic_results['total'],
            'accuracy': dynamic_acc,
        }
    else:
        # Use known result
        all_results['dynamic_canvas'] = {'correct': 3423, 'total': 4241, 'accuracy': 80.71}

    # Report
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\n{'Method':<35} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 70)
    for name, r in all_results.items():
        print(f"{name:<35} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>9.2f}%")
    print("-" * 70)

    baseline_acc = all_results['baseline']['accuracy']
    smart_acc = all_results['smart_scorer']['accuracy']
    dynamic_acc = all_results['dynamic_canvas']['accuracy']
    print(f"\nSmartCanvas+Scorer vs Baseline: {smart_acc - baseline_acc:+.2f}%")
    print(f"SmartCanvas+Scorer vs DynamicCanvas: {smart_acc - dynamic_acc:+.2f}%")

    # Save results
    results_path = f"{OUTPUT_DIR}/results_smart_scorer.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    # Save detailed predictions
    predictions_path = f"{OUTPUT_DIR}/predictions_smart_scorer.json"
    with open(predictions_path, 'w') as f:
        json.dump(smart_results['predictions'], f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
