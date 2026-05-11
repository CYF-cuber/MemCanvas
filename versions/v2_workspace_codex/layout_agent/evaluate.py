#!/usr/bin/env python3
"""
evaluate.py — End-to-end evaluation of Layout Agent

Pipeline:
  1. Load trained Layout Agent (Qwen2.5-VL-3B + LoRA)
  2. For each test sample: agent decides layout → render canvas → Reader VLM answers QA
  3. Compare: baseline (heuristic) vs agent-selected layout vs DynamicCanvas

Usage:
    python -m layout_agent.evaluate [--agent_path PATH] [--device cuda:0]
"""

import argparse
import json
import math
import os
import pickle
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from layout_agent.strategies import (
    STRATEGY_NAMES, LayoutParams, ContentBlock,
    adapt_scienceqa_knowledge_only, adapt_scienceqa,
    generate_layout, render_layout, compute_quality_score,
    generate_all_candidates, build_agent_prompt,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCIQA_CACHE = "/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl"
READER_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
AGENT_MODEL = "/home/cyf/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "/home/cyf/codex/layout_agent/eval_results"


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_reader_vlm(model_path: str, device: str = "cuda:0"):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def load_layout_agent(model_path: str, lora_path: Optional[str] = None,
                      device: str = "cuda:1"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    if lora_path and os.path.exists(lora_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"Loaded LoRA from {lora_path}")
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Agent inference
# ---------------------------------------------------------------------------

def agent_decide_layout(model, tokenizer, blocks: List[ContentBlock],
                        device: str = "cuda:1") -> LayoutParams:
    """Ask the layout agent to decide the best layout."""
    prompt = build_agent_prompt(blocks)
    system = (
        "You are a canvas layout optimization agent. "
        "Given content blocks, output the optimal layout as JSON. "
        "Output ONLY a JSON object."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128,
                                do_sample=False, temperature=None, top_p=None)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()

    # Parse JSON
    import re
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    match = re.search(r"\{[^{}]*\}", response)
    if match:
        try:
            d = json.loads(match.group())
            params = LayoutParams.from_dict(d)
            if params.is_valid():
                return params
        except json.JSONDecodeError:
            pass

    # Fallback to default
    return LayoutParams()


# ---------------------------------------------------------------------------
# QA evaluation
# ---------------------------------------------------------------------------

def reader_qa(model, processor, canvas: Image.Image, question: str,
              choices: List[str], device: str = "cuda:0") -> str:
    """Ask the reader VLM a question given a canvas image."""
    q_text = (
        "The image shows reference material. "
        "Use it to answer the following question.\n\n"
        f"Q: {question}\n"
    )
    for i, c in enumerate(choices):
        q_text += f"  {chr(65+i)}. {c}\n"
    q_text += "\nAnswer with ONLY the letter."

    messages = [{"role": "user", "content": [
        {"type": "image", "image": canvas},
        {"type": "text", "text": q_text},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    inputs = processor(text=[text], images=[canvas], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=16,
                                do_sample=False, temperature=None, top_p=None)
    response = processor.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True).strip()
    for ch in response.upper():
        if ch in "ABCDEFGH":
            return ch
    return ""


# ---------------------------------------------------------------------------
# Heuristic baseline
# ---------------------------------------------------------------------------

def heuristic_best_layout(blocks: List[ContentBlock]) -> LayoutParams:
    """Select best layout using the original heuristic scorer."""
    candidates = generate_all_candidates(blocks)
    best_params, best_layout = max(candidates,
                                    key=lambda x: compute_quality_score(x[1]))
    return best_params


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(agent_path: str = AGENT_MODEL,
                   lora_path: Optional[str] = None,
                   reader_path: str = READER_MODEL,
                   num_test: int = 500,
                   device_reader: str = "cuda:0",
                   device_agent: str = "cuda:1"):
    """Run end-to-end evaluation comparing agent vs heuristic vs no-memory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading ScienceQA data...")
    with open(SCIQA_CACHE, "rb") as f:
        cache = pickle.load(f)
    train_data = cache["train"] if isinstance(cache, dict) else cache[0]
    test_data = cache.get("test", train_data) if isinstance(cache, dict) else cache[1]

    # Load HF dataset for images
    hf_dataset = None
    try:
        from datasets import load_dataset
        hf_dataset = load_dataset("derek-thomas/ScienceQA", split="test")
        print(f"HF test dataset: {len(hf_dataset)} samples")
    except Exception as e:
        print(f"HF dataset not available: {e}")

    # Load models
    print("Loading Reader VLM...")
    reader_model, reader_processor = load_reader_vlm(reader_path, device_reader)

    print("Loading Layout Agent...")
    agent_model, agent_tokenizer = load_layout_agent(agent_path, lora_path, device_agent)

    # Evaluate
    results = {"agent": [], "heuristic": [], "no_memory": []}
    test_indices = list(range(min(num_test, len(test_data))))

    t0 = time.time()
    for ti, idx in enumerate(test_indices):
        problem = test_data[idx]
        question = problem.get("question", "")
        choices = problem.get("choices", [])
        answer_idx = problem.get("answer", 0)
        answer_letter = chr(65 + answer_idx) if answer_idx < len(choices) else "A"

        # Get image
        img = None
        if hf_dataset is not None:
            try:
                if hf_dataset[idx].get("image") is not None:
                    img = hf_dataset[idx]["image"].convert("RGB")
            except Exception:
                pass

        # Knowledge-only blocks for canvas
        blocks = adapt_scienceqa_knowledge_only(problem, img)

        if blocks:
            # --- Agent layout ---
            agent_params = agent_decide_layout(agent_model, agent_tokenizer,
                                                blocks, device_agent)
            agent_layout = generate_layout(blocks, agent_params)
            agent_canvas = render_layout(agent_layout)
            agent_pred = reader_qa(reader_model, reader_processor, agent_canvas,
                                    question, choices, device_reader)
            agent_correct = agent_pred == answer_letter

            # --- Heuristic layout ---
            heur_params = heuristic_best_layout(blocks)
            heur_layout = generate_layout(blocks, heur_params)
            heur_canvas = render_layout(heur_layout)
            heur_pred = reader_qa(reader_model, reader_processor, heur_canvas,
                                   question, choices, device_reader)
            heur_correct = heur_pred == answer_letter
        else:
            agent_correct = False
            heur_correct = False
            agent_params = LayoutParams()
            heur_params = LayoutParams()

        # --- No memory baseline ---
        no_mem_pred = reader_qa(reader_model, reader_processor,
                                 Image.new("RGB", (100, 100), (255, 255, 255)),
                                 question, choices, device_reader)
        no_mem_correct = no_mem_pred == answer_letter

        results["agent"].append(agent_correct)
        results["heuristic"].append(heur_correct)
        results["no_memory"].append(no_mem_correct)

        if (ti + 1) % 20 == 0:
            a_acc = sum(results["agent"]) / len(results["agent"]) * 100
            h_acc = sum(results["heuristic"]) / len(results["heuristic"]) * 100
            n_acc = sum(results["no_memory"]) / len(results["no_memory"]) * 100
            elapsed = time.time() - t0
            print(f"  [{ti+1}/{len(test_indices)}] "
                  f"agent={a_acc:.1f}% heuristic={h_acc:.1f}% no_mem={n_acc:.1f}% "
                  f"({elapsed:.0f}s)")

    # Final results
    elapsed = time.time() - t0
    final = {}
    for method in ["agent", "heuristic", "no_memory"]:
        correct = sum(results[method])
        total = len(results[method])
        final[method] = {
            "accuracy": correct / total * 100,
            "correct": correct,
            "total": total,
        }
        print(f"\n{method:12s}: {correct}/{total} = {correct/total*100:.2f}%")

    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/len(test_indices):.1f}s/sample)")

    # Save
    output_path = os.path.join(OUTPUT_DIR, "e2e_results.json")
    with open(output_path, "w") as f:
        json.dump({"results": final, "per_sample": results}, f, indent=1)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_path", default=AGENT_MODEL)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--reader_path", default=READER_MODEL)
    parser.add_argument("--num_test", type=int, default=500)
    parser.add_argument("--device_reader", default="cuda:0")
    parser.add_argument("--device_agent", default="cuda:1")
    args = parser.parse_args()

    run_evaluation(
        agent_path=args.agent_path,
        lora_path=args.lora_path,
        reader_path=args.reader_path,
        num_test=args.num_test,
        device_reader=args.device_reader,
        device_agent=args.device_agent,
    )


if __name__ == "__main__":
    main()
