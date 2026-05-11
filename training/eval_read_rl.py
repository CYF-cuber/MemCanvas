#!/usr/bin/env python3
"""
Evaluate RL-trained Read Agent on the full MemCanvas pipeline.

Uses the oracle canvas library (1 canvas/sample, not agent-built fragmented
canvases) and compares:
  - RL read agent
  - SFT read agent
  - Embedding Top-K only (no agent)
  - No-memory baseline

Usage:
  # Evaluate RL agent (after merging RL LoRA)
  python eval_read_rl.py --rl-model /home/cyf/codex/agent_rl_merged

  # Compare all methods
  python eval_read_rl.py --rl-model /home/cyf/codex/agent_rl_merged --compare-all

  # Quick test with fewer samples
  python eval_read_rl.py --rl-model /home/cyf/codex/agent_rl_merged --max-test 100
"""

import argparse
import io
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/memory")
sys.path.insert(0, "/home/cyf/codex")

from canvas_agent import AgentConfig, CanvasManagerAgent, CanvasMeta
from agent_experiment import (
    VLM_MODEL_PATH,
    extract_answer,
    load_scienceqa,
    render_sample_to_canvas,
    sample_to_knowledge_text,
    vlm_predict,
)
from prepare_agent_sft_data import SimLibrary, answer_info_in_canvas

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RL_DATA_DIR = Path("/home/cyf/codex/agent_rl_data")
OUTPUT_DIR = Path("/home/cyf/codex/agent_rl_eval_output")
AGENT_MODEL_PATH = "/home/cyf/Qwen3-4B"


# ---------------------------------------------------------------------------
# Build baseline canvas library (1 canvas per training sample)
# ---------------------------------------------------------------------------
def build_baseline_canvases(
    train_data: List[Dict], max_train: int = 2000
) -> Tuple[List[Dict], np.ndarray]:
    """Build baseline canvases: one per training sample, with CLIP embeddings."""
    from memory_canvas.dynamic_canvas import DynamicCanvas, DynamicCanvasConfig
    from transformers import CLIPModel, CLIPProcessor

    cache_file = OUTPUT_DIR / "baseline_canvases.pkl"
    if cache_file.exists():
        print(f"Loading cached baseline canvases from {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        return cached["canvas_data"], cached["canvas_embeddings"]

    train_data = train_data[:max_train]
    print(f"\nBuilding baseline canvases ({len(train_data)} samples)...")

    # Render each sample into its own canvas
    canvas_data = []
    for i, sample in enumerate(tqdm(train_data, desc="Rendering canvases")):
        config = DynamicCanvasConfig(patch_size=640)
        canvas = DynamicCanvas(config)
        render_sample_to_canvas(sample, canvas)

        patches = canvas.get_patches()
        if not patches:
            continue

        # Combine patches into single image
        patch_imgs = [p.image for p in patches]
        total_h = sum(img.size[1] for img in patch_imgs)
        combined = Image.new("RGB", (640, total_h), (255, 255, 255))
        y = 0
        for img in patch_imgs:
            combined.paste(img, (0, y))
            y += img.size[1]

        knowledge = sample_to_knowledge_text(sample)
        buf = io.BytesIO()
        combined.save(buf, format="PNG")

        canvas_data.append({
            "canvas_id": i,
            "topic": f"{sample.get('subject', '')} / {sample.get('topic', '')}",
            "num_items": 1,
            "num_patches": len(patches),
            "text_summary": knowledge[:200],
            "items": [knowledge],
            "image_bytes": buf.getvalue(),
        })

    # Compute CLIP embeddings
    print(f"Computing CLIP embeddings for {len(canvas_data)} canvases...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device)

    canvas_embeddings = []
    for cd in tqdm(canvas_data, desc="CLIP encoding"):
        img = Image.open(io.BytesIO(cd["image_bytes"])).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        canvas_embeddings.append(emb.cpu().numpy().flatten())

    canvas_embeddings = np.array(canvas_embeddings)

    del clip_model, clip_processor
    torch.cuda.empty_cache()

    # Cache
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump({
            "canvas_data": canvas_data,
            "canvas_embeddings": canvas_embeddings,
        }, f)

    print(f"  {len(canvas_data)} canvases, embeddings: {canvas_embeddings.shape}")
    return canvas_data, canvas_embeddings


# ---------------------------------------------------------------------------
# Encode test queries with CLIP
# ---------------------------------------------------------------------------
def encode_test_queries(
    test_data: List[Dict],
) -> np.ndarray:
    """Encode test queries using CLIP text encoder."""
    from transformers import CLIPModel, CLIPProcessor

    cache_file = OUTPUT_DIR / "test_query_embeddings.pkl"
    if cache_file.exists():
        print(f"Loading cached query embeddings from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"Encoding {len(test_data)} test queries with CLIP...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device)

    query_embeddings = []
    for sample in tqdm(test_data, desc="Encoding queries"):
        query = sample["question"]
        if sample.get("hint"):
            query += " " + sample["hint"]
        inputs = clip_processor(
            text=query, return_tensors="pt", truncation=True, max_length=77
        ).to(device)
        with torch.no_grad():
            emb = clip_model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        query_embeddings.append(emb.cpu().numpy().flatten())

    query_embeddings = np.array(query_embeddings)

    del clip_model, clip_processor
    torch.cuda.empty_cache()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(query_embeddings, f)

    return query_embeddings


# ---------------------------------------------------------------------------
# Evaluate a single method
# ---------------------------------------------------------------------------
def evaluate_method(
    method_name: str,
    test_data: List[Dict],
    canvas_data: List[Dict],
    canvas_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    vlm_model,
    vlm_processor,
    agent: Optional[CanvasManagerAgent] = None,
    top_k: int = 5,
    agent_select_k: int = 2,
) -> Dict:
    """Evaluate a retrieval method on test data."""
    print(f"\n--- Evaluating: {method_name} ---")

    # Build canvas metadata for agent
    canvas_metas = []
    for cd in canvas_data:
        canvas_metas.append(
            CanvasMeta(
                canvas_id=cd["canvas_id"],
                topic=cd["topic"],
                num_items=cd["num_items"],
                num_patches=cd["num_patches"],
                max_patches=4,
                text_summary=cd["text_summary"],
            )
        )

    correct = 0
    results = []

    for i, sample in enumerate(tqdm(test_data, desc=method_name)):
        query_emb = query_embeddings[i]

        if method_name == "no_memory":
            # No retrieval, direct VLM answer
            pred = vlm_predict(vlm_model, vlm_processor, sample, [])
        else:
            # CLIP pre-filter
            similarities = canvas_embeddings @ query_emb
            top_indices = np.argsort(similarities)[::-1][:top_k]

            if method_name == "embedding_topk":
                # Use top-1 embedding match directly
                selected_ids = [canvas_data[top_indices[0]]["canvas_id"]]
            elif agent is not None:
                # Agent selection from candidates
                candidates = [
                    (canvas_metas[idx], float(similarities[idx]))
                    for idx in top_indices
                    if similarities[idx] > 0.1
                ]
                if candidates:
                    read_action = agent.read_decision(
                        sample["question"], candidates
                    )
                    if (
                        read_action.action == "RETRIEVE"
                        and read_action.canvas_ids
                    ):
                        selected_ids = read_action.canvas_ids[:agent_select_k]
                    else:
                        selected_ids = []  # DIRECT_ANSWER
                else:
                    selected_ids = []
            else:
                selected_ids = [canvas_data[top_indices[0]]["canvas_id"]]

            # Load canvas images
            memory_images = []
            for sid in selected_ids:
                for cd in canvas_data:
                    if cd["canvas_id"] == sid:
                        img = Image.open(io.BytesIO(cd["image_bytes"])).convert("RGB")
                        memory_images.append(img)
                        break

            pred = vlm_predict(vlm_model, vlm_processor, sample, memory_images)

        is_correct = pred == sample["answer"]
        if is_correct:
            correct += 1

        results.append({
            "pid": sample.get("pid", str(i)),
            "prediction": pred,
            "correct": is_correct,
        })

        if (i + 1) % 100 == 0:
            acc = correct / (i + 1) * 100
            print(f"  [{i+1}/{len(test_data)}] {method_name}: {acc:.1f}%")

    accuracy = correct / len(test_data) * 100
    print(f"  Final: {correct}/{len(test_data)} = {accuracy:.2f}%")

    return {
        "method": method_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_data),
        "predictions": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate RL Read Agent")
    parser.add_argument(
        "--rl-model",
        type=str,
        required=True,
        help="Path to RL-trained model (merged RL LoRA, or base+LoRA path)",
    )
    parser.add_argument(
        "--rl-lora-path",
        type=str,
        default=None,
        help="If set, load this LoRA on top of --rl-model (for un-merged checkpoints)",
    )
    parser.add_argument(
        "--sft-model",
        type=str,
        default="/home/cyf/codex/agent_sft_merged",
        help="Path to SFT-merged model for comparison",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=2000,
        help="Max training samples for baseline canvases",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=0,
        help="Max test samples (0 = all)",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare RL agent, SFT agent, embedding Top-K, and no-memory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: agent_rl_eval_output)",
    )
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading ScienceQA data...")
    train_data, test_data = load_scienceqa()
    if args.max_test > 0:
        test_data = test_data[: args.max_test]
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    # Build baseline canvases + CLIP embeddings
    canvas_data, canvas_embeddings = build_baseline_canvases(
        train_data, max_train=args.max_train
    )
    query_embeddings = encode_test_queries(test_data)

    # Load VLM
    print("Loading VLM...")
    from transformers import AutoModelForVision2Seq, AutoProcessor

    vlm_model = AutoModelForVision2Seq.from_pretrained(
        VLM_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    vlm_processor = AutoProcessor.from_pretrained(
        VLM_MODEL_PATH, trust_remote_code=True
    )
    vlm_model.eval()

    all_results = {}

    # --- Evaluate RL Read Agent ---
    print("\n" + "=" * 60)
    print("Evaluating RL Read Agent")
    print("=" * 60)

    rl_agent = CanvasManagerAgent(
        AgentConfig(
            model_path=args.rl_model,
            lora_path=args.rl_lora_path,
            device="cuda",
            do_sample=False,  # greedy for evaluation
        )
    )
    rl_agent.load()

    rl_result = evaluate_method(
        "rl_agent",
        test_data,
        canvas_data,
        canvas_embeddings,
        query_embeddings,
        vlm_model,
        vlm_processor,
        agent=rl_agent,
    )
    all_results["rl_agent"] = rl_result
    rl_agent.unload()

    if args.compare_all:
        # --- SFT Read Agent ---
        if os.path.exists(args.sft_model):
            print("\n" + "=" * 60)
            print("Evaluating SFT Read Agent")
            print("=" * 60)

            sft_agent = CanvasManagerAgent(
                AgentConfig(
                    model_path=args.sft_model,
                    device="cuda",
                    do_sample=False,
                )
            )
            sft_agent.load()

            sft_result = evaluate_method(
                "sft_agent",
                test_data,
                canvas_data,
                canvas_embeddings,
                query_embeddings,
                vlm_model,
                vlm_processor,
                agent=sft_agent,
            )
            all_results["sft_agent"] = sft_result
            sft_agent.unload()

        # --- Embedding Top-K only ---
        print("\n" + "=" * 60)
        print("Evaluating Embedding Top-K")
        print("=" * 60)
        topk_result = evaluate_method(
            "embedding_topk",
            test_data,
            canvas_data,
            canvas_embeddings,
            query_embeddings,
            vlm_model,
            vlm_processor,
        )
        all_results["embedding_topk"] = topk_result

        # --- No Memory ---
        print("\n" + "=" * 60)
        print("Evaluating No-Memory Baseline")
        print("=" * 60)
        nomem_result = evaluate_method(
            "no_memory",
            test_data,
            canvas_data,
            canvas_embeddings,
            query_embeddings,
            vlm_model,
            vlm_processor,
        )
        all_results["no_memory"] = nomem_result

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in all_results.items():
        print(f"  {name:20s}: {result['accuracy']:.2f}%  ({result['correct']}/{result['total']})")

    # Save results
    summary = {
        name: {k: v for k, v in result.items() if k != "predictions"}
        for name, result in all_results.items()
    }
    with open(output_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "eval_full_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
