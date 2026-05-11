#!/usr/bin/env python3
"""
Agent MemCanvas Experiment — ScienceQA

Compares:
  (A) MemCanvas baseline: one canvas per training sample, embedding Top-K retrieval
  (C) MemCanvas + Prompt Agent: Qwen3-4B agent decides canvas organization & retrieval

Pipeline:
  Phase 1 (prep): Load data, agent organizes knowledge into canvases
  Phase 2 (embed): Compute CLIP embeddings for canvases + test queries
  Phase 3 (eval): Agent retrieval + VLM reading → compare with baseline

Usage:
  # Full pipeline
  CUDA_VISIBLE_DEVICES=0 python -u agent_experiment.py --phase all

  # Phase by phase
  CUDA_VISIBLE_DEVICES=0 python -u agent_experiment.py --phase prep --max-train 2000
  CUDA_VISIBLE_DEVICES=0 python -u agent_experiment.py --phase embed
  CUDA_VISIBLE_DEVICES=0 python -u agent_experiment.py --phase eval --max-test 500
"""

import argparse
import io
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/cyf/memory")
sys.path.insert(0, "/home/cyf/codex")

from memory_canvas.dynamic_canvas import DynamicCanvas, DynamicCanvasConfig
from canvas_agent import CanvasManagerAgent, AgentConfig, CanvasMeta
from canvas_executor import CanvasExecutor, ExecutorConfig, WriteAction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCIQA_DATA_DIR = Path("/home/cyf/memory/ScienceQA/data/scienceqa")
OUTPUT_DIR = Path("/home/cyf/codex/agent_experiment_output")
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
VLM_MODEL_PATH = "/home/cyf/Qwen2.5-VL-7B-Instruct"
AGENT_MODEL_PATH = "/home/cyf/Qwen3-4B"

TOP_K = 5  # pre-filter candidates for agent
AGENT_SELECT_K = 2  # agent selects from candidates
AGENT_LORA_PATH = None  # set by --lora-path CLI arg


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_scienceqa() -> Tuple[List[Dict], List[Dict]]:
    """Load ScienceQA train and test splits."""
    cache_file = OUTPUT_DIR / "sciqa_cached.pkl"
    if cache_file.exists():
        print(f"Loading cached ScienceQA from {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        print(f"  Train: {len(data['train'])}, Test: {len(data['test'])}")
        return data["train"], data["test"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Try loading from local ScienceQA data
    problems_file = SCIQA_DATA_DIR / "problems.json"
    if problems_file.exists():
        print(f"Loading from {problems_file}...")
        with open(problems_file) as f:
            problems = json.load(f)

        captions_file = SCIQA_DATA_DIR / "captions.json"
        captions = {}
        if captions_file.exists():
            with open(captions_file) as f:
                captions = json.load(f)

        train_data, test_data = [], []
        img_base = SCIQA_DATA_DIR
        for pid, prob in problems.items():
            # Get image path
            img_path = None
            if prob.get("image"):
                img_path = str(img_base / prob["split"] / pid / "image.png")
                if not os.path.exists(img_path):
                    img_path = None

            entry = {
                "pid": pid,
                "question": prob["question"],
                "choices": prob["choices"],
                "answer": prob["answer"],
                "subject": prob.get("subject", ""),
                "topic": prob.get("topic", ""),
                "hint": prob.get("hint", ""),
                "lecture": prob.get("lecture", ""),
                "solution": prob.get("solution", ""),
                "image_path": img_path,
                "caption": captions.get(pid, ""),
            }

            if prob["split"] == "train":
                train_data.append(entry)
            elif prob["split"] == "test":
                test_data.append(entry)
    else:
        # Fallback: HuggingFace
        print("Loading ScienceQA from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("derek-thomas/ScienceQA")
        train_data = _convert_hf_split(ds["train"])
        test_data = _convert_hf_split(ds["test"])

    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    with open(cache_file, "wb") as f:
        pickle.dump({"train": train_data, "test": test_data}, f)
    return train_data, test_data


def _convert_hf_split(split) -> List[Dict]:
    """Convert HuggingFace dataset split to our format."""
    data = []
    for i, sample in enumerate(split):
        img_path = None
        if sample.get("image") is not None:
            img_dir = OUTPUT_DIR / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = str(img_dir / f"{i:05d}.png")
            if not os.path.exists(img_path):
                sample["image"].save(img_path)

        data.append({
            "pid": str(i),
            "question": sample["question"],
            "choices": sample["choices"],
            "answer": sample["answer"],
            "subject": sample.get("subject", ""),
            "topic": sample.get("topic", ""),
            "hint": sample.get("hint", ""),
            "lecture": sample.get("lecture", ""),
            "solution": sample.get("solution", ""),
            "image_path": img_path,
            "caption": "",
        })
    return data


# ---------------------------------------------------------------------------
# Knowledge text for agent (what the agent sees about each training sample)
# ---------------------------------------------------------------------------
def sample_to_knowledge_text(sample: Dict) -> str:
    """Convert a training sample to knowledge text for the agent."""
    parts = []
    parts.append(f"Subject: {sample['subject']} / {sample['topic']}")
    if sample["hint"]:
        parts.append(f"Context: {sample['hint'][:200]}")
    parts.append(f"Q: {sample['question']}")
    correct_idx = sample["answer"]
    choices = sample["choices"]
    correct_answer = choices[correct_idx] if correct_idx < len(choices) else ""
    parts.append(f"A: {correct_answer}")
    if sample["lecture"]:
        parts.append(f"Knowledge: {sample['lecture'][:200]}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Canvas rendering (same as existing experiment)
# ---------------------------------------------------------------------------
def render_sample_to_canvas(sample: Dict, canvas: DynamicCanvas) -> None:
    """Render a ScienceQA sample onto a canvas (in-place)."""
    # Header
    subj = sample.get("subject", "")
    topic = sample.get("topic", "")
    if subj and topic:
        canvas.add_text(f"[{subj} - {topic}]", font_size=14)

    # Context/hint
    hint = sample.get("hint", "")
    if hint:
        canvas.add_text(f"Context: {hint[:500]}", font_size=16)

    # Image
    img_path = sample.get("image_path")
    if img_path and os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            canvas.add_image(img, max_height=280)
        except Exception:
            pass

    # Caption
    caption = sample.get("caption", "")
    if caption:
        canvas.add_text(f"[Image] {caption[:200]}", font_size=14)

    canvas.add_separator()

    # Question
    canvas.add_text(f"Q: {sample['question']}", font_size=18)

    # Choices
    correct_idx = sample["answer"]
    for i, choice in enumerate(sample["choices"]):
        letter = chr(65 + i)
        marker = "✓" if i == correct_idx else "○"
        canvas.add_text(f"  {marker} {letter}. {choice}", font_size=16)

    # Lecture (background knowledge)
    lecture = sample.get("lecture", "")
    if lecture:
        canvas.add_text(f"Background: {lecture[:400]}", font_size=14)

    # Solution
    solution = sample.get("solution", "")
    if solution:
        canvas.add_text(f"Solution: {solution[:300]}", font_size=14)


# ---------------------------------------------------------------------------
# Phase 1: Agent organizes training data into canvases
# ---------------------------------------------------------------------------
def phase_prep(train_data: List[Dict], max_train: int = 2000):
    """Agent organizes training samples into canvases."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = OUTPUT_DIR / "agent_canvases.pkl"

    if cache_file.exists():
        print(f"Loading cached agent canvases from {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        print(f"  {cached['num_canvases']} canvases, {cached['total_items']} items")
        return cached

    train_data = train_data[:max_train]
    print(f"\n{'='*60}")
    print(f"Phase 1: Agent canvas organization ({len(train_data)} samples)")
    print(f"{'='*60}")

    # Load sentence-transformer for top-K canvas filtering
    # (must match SFT training: only show top-15 most similar canvases)
    MAX_DISPLAY_CANVASES = 15
    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformers for canvas filtering...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    topic_embeddings = {}  # canvas_id → embedding

    # Initialize agent and executor
    agent = CanvasManagerAgent(AgentConfig(
        model_path=AGENT_MODEL_PATH,
        lora_path=AGENT_LORA_PATH,
        do_sample=False,  # greedy for reproducibility
    ))
    agent.load()

    executor = CanvasExecutor(ExecutorConfig(max_patches_per_canvas=4))

    # Process each training sample
    decisions_log = []
    t0 = time.time()

    for i, sample in enumerate(tqdm(train_data, desc="Agent organizing")):
        knowledge_text = sample_to_knowledge_text(sample)
        all_metas = executor.canvas_metas

        # Filter to top-K most similar canvases (matching SFT training distribution)
        if len(all_metas) <= MAX_DISPLAY_CANVASES:
            display_metas = all_metas
        else:
            knowledge_emb = embedder.encode(knowledge_text, normalize_embeddings=True)
            canvas_ids = list(topic_embeddings.keys())
            emb_matrix = np.array([topic_embeddings[cid] for cid in canvas_ids])
            sims = emb_matrix @ knowledge_emb
            top_idx = np.argsort(sims)[::-1][:MAX_DISPLAY_CANVASES]
            top_cids = set(canvas_ids[j] for j in top_idx)
            display_metas = [m for m in all_metas if m.canvas_id in top_cids]

        # Agent decides (with filtered canvas view)
        action = agent.write_decision(
            canvas_library=display_metas,
            new_knowledge=knowledge_text,
        )

        # Execute the action
        img = None
        img_path = sample.get("image_path")
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                pass

        # For CREATE/APPEND, render the full canvas content
        if action.action in ("CREATE", "APPEND"):
            if action.action == "CREATE":
                # Create new canvas via executor
                topic = action.params.get("topic", f"{sample['subject']} - {sample['topic']}")
                cid = executor._create(topic, "", img)
                # Now render the full sample content onto this canvas
                render_sample_to_canvas(sample, executor.library[cid].canvas)
                executor.library[cid].meta.num_patches = len(
                    executor.library[cid].canvas.get_patches()
                )
                executor.library[cid].meta.num_items = 1
                executor.library[cid].items = [knowledge_text]
                executor.library[cid].meta.text_summary = knowledge_text[:200]
                # Store embedding for new canvas
                topic_embeddings[cid] = embedder.encode(
                    topic + " " + knowledge_text[:200], normalize_embeddings=True
                )
            else:
                cid = action.params.get("canvas_id", 0)
                if cid in executor.library:
                    entry = executor.library[cid]
                    if not entry.meta.is_full:
                        entry.canvas.add_separator(style="light")
                        render_sample_to_canvas(sample, entry.canvas)
                        entry.meta.num_items += 1
                        entry.meta.num_patches = len(entry.canvas.get_patches())
                        entry.items.append(knowledge_text)
                        entry.meta.text_summary = " | ".join(entry.items[-3:])[:200]
                        # Update embedding
                        topic_embeddings[cid] = embedder.encode(
                            entry.meta.topic + " " + entry.meta.text_summary,
                            normalize_embeddings=True,
                        )
                    else:
                        # Canvas full, create new
                        topic = entry.meta.topic + " (cont.)"
                        new_cid = executor._create(topic, "", img)
                        render_sample_to_canvas(sample, executor.library[new_cid].canvas)
                        executor.library[new_cid].meta.num_patches = len(
                            executor.library[new_cid].canvas.get_patches()
                        )
                        topic_embeddings[new_cid] = embedder.encode(
                            topic + " " + knowledge_text[:200], normalize_embeddings=True
                        )
                        cid = new_cid
                else:
                    # Canvas not found, create
                    topic = f"{sample['subject']} - {sample['topic']}"
                    cid = executor._create(topic, "", img)
                    render_sample_to_canvas(sample, executor.library[cid].canvas)
                    executor.library[cid].meta.num_patches = len(
                        executor.library[cid].canvas.get_patches()
                    )
                    topic_embeddings[cid] = embedder.encode(
                        topic + " " + knowledge_text[:200], normalize_embeddings=True
                    )

        decisions_log.append({
            "sample_idx": i,
            "pid": sample["pid"],
            "action": action.action,
            "params": action.params,
            "reason": action.reason,
        })

        # Progress
        if (i + 1) % 100 == 0:
            stats = executor.get_statistics()
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(train_data)}] "
                  f"Canvases: {stats['num_canvases']}, "
                  f"Items: {stats['total_items']}, "
                  f"Patches: {stats['total_patches']}, "
                  f"Time: {elapsed:.0f}s")

    agent.unload()
    del embedder  # free sentence-transformer

    # Collect all canvas images
    stats = executor.get_statistics()
    print(f"\nFinal: {stats['num_canvases']} canvases, "
          f"{stats['total_items']} items, "
          f"{stats['total_patches']} patches")

    # Save canvas images and metadata
    canvas_data = []
    for cid, entry in executor.library.items():
        img = executor.get_canvas_image(cid)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        canvas_data.append({
            "canvas_id": cid,
            "topic": entry.meta.topic,
            "num_items": entry.meta.num_items,
            "num_patches": entry.meta.num_patches,
            "text_summary": entry.meta.text_summary,
            "items": entry.items,
            "image_bytes": buf.getvalue(),
        })

    # Action distribution
    action_counts = Counter(d["action"] for d in decisions_log)
    print(f"\nAction distribution: {dict(action_counts)}")

    cached = {
        "canvas_data": canvas_data,
        "decisions_log": decisions_log,
        "action_counts": dict(action_counts),
        "num_canvases": len(canvas_data),
        "total_items": stats["total_items"],
        "total_patches": stats["total_patches"],
    }

    with open(cache_file, "wb") as f:
        pickle.dump(cached, f)

    # Also save decisions log as JSON for analysis
    with open(OUTPUT_DIR / "decisions_log.json", "w") as f:
        json.dump(decisions_log, f, indent=2)

    print(f"Saved to {cache_file}")
    return cached


# ---------------------------------------------------------------------------
# Phase 2: Compute CLIP embeddings
# ---------------------------------------------------------------------------
def phase_embed(cached_prep: Dict):
    """Compute CLIP embeddings for agent-organized canvases."""
    embed_file = OUTPUT_DIR / "agent_embeddings.pkl"
    if embed_file.exists():
        print(f"Loading cached embeddings from {embed_file}")
        with open(embed_file, "rb") as f:
            return pickle.load(f)

    print(f"\n{'='*60}")
    print(f"Phase 2: Computing CLIP embeddings")
    print(f"{'='*60}")

    from transformers import CLIPModel, CLIPProcessor

    print(f"Loading CLIP model: {CLIP_MODEL_NAME}")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device)

    # Encode canvas images
    canvas_data = cached_prep["canvas_data"]
    print(f"Encoding {len(canvas_data)} canvas images...")
    canvas_embeddings = []

    for cd in tqdm(canvas_data, desc="Encoding canvases"):
        img = Image.open(io.BytesIO(cd["image_bytes"])).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        canvas_embeddings.append(emb.cpu().numpy().flatten())

    canvas_embeddings = np.array(canvas_embeddings)
    print(f"  Canvas embeddings shape: {canvas_embeddings.shape}")

    # Clean up CLIP
    del clip_model, clip_processor
    torch.cuda.empty_cache()

    result = {
        "canvas_embeddings": canvas_embeddings,
    }

    with open(embed_file, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved to {embed_file}")
    return result


# ---------------------------------------------------------------------------
# Phase 3: Evaluation
# ---------------------------------------------------------------------------
def phase_eval(
    test_data: List[Dict],
    cached_prep: Dict,
    cached_embed: Dict,
    max_test: int = 500,
    skip_baseline: bool = False,
):
    """Evaluate Agent MemCanvas vs baseline."""
    test_data = test_data[:max_test]
    print(f"\n{'='*60}")
    print(f"Phase 3: Evaluation ({len(test_data)} test samples)")
    print(f"{'='*60}")

    canvas_data = cached_prep["canvas_data"]
    canvas_embeddings = cached_embed["canvas_embeddings"]

    # Load CLIP for query encoding
    from transformers import CLIPModel, CLIPProcessor
    print("Loading CLIP for query encoding...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device)

    # Encode all test queries
    print("Encoding test queries...")
    query_embeddings = []
    for sample in tqdm(test_data, desc="Encoding queries"):
        query = sample["question"]
        if sample["hint"]:
            query += " " + sample["hint"]
        inputs = clip_processor(text=query, return_tensors="pt", truncation=True, max_length=77).to(device)
        with torch.no_grad():
            emb = clip_model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        query_embeddings.append(emb.cpu().numpy().flatten())
    query_embeddings = np.array(query_embeddings)

    del clip_model, clip_processor
    torch.cuda.empty_cache()

    # Load VLM
    print("Loading VLM...")
    from transformers import AutoModelForVision2Seq, AutoProcessor
    vlm_model = AutoModelForVision2Seq.from_pretrained(
        VLM_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_PATH, trust_remote_code=True)
    vlm_model.eval()

    # Load Agent for read decisions
    agent = CanvasManagerAgent(AgentConfig(
        model_path=AGENT_MODEL_PATH,
        lora_path=AGENT_LORA_PATH,
        device="cuda",  # agent on GPU alongside VLM
        do_sample=False,
    ))
    agent.load()

    # Build canvas metadata for agent
    canvas_metas = []
    for cd in canvas_data:
        canvas_metas.append(CanvasMeta(
            canvas_id=cd["canvas_id"],
            topic=cd["topic"],
            num_items=cd["num_items"],
            num_patches=cd["num_patches"],
            max_patches=4,
            text_summary=cd["text_summary"],
        ))

    # Evaluate
    results = {"agent": [], "baseline": []}
    agent_correct = 0
    baseline_correct = 0

    for i, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        query_emb = query_embeddings[i]

        # --- Agent MemCanvas ---
        # Step 1: Embedding pre-filter (Top-K candidates)
        similarities = canvas_embeddings @ query_emb
        top_indices = np.argsort(similarities)[::-1][:TOP_K]

        candidates = [
            (canvas_metas[idx], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > 0.1
        ]

        # Step 2: Agent selects from candidates
        if candidates:
            read_action = agent.read_decision(sample["question"], candidates)

            if read_action.action == "RETRIEVE" and read_action.canvas_ids:
                selected_ids = read_action.canvas_ids[:AGENT_SELECT_K]
            else:
                # Fallback: use top-1
                selected_ids = [candidates[0][0].canvas_id]
        else:
            selected_ids = []

        # Step 3: Get canvas images
        memory_images = []
        for sid in selected_ids:
            for cd in canvas_data:
                if cd["canvas_id"] == sid:
                    img = Image.open(io.BytesIO(cd["image_bytes"])).convert("RGB")
                    memory_images.append(img)
                    break

        # Step 4: VLM answer with agent-selected canvases
        agent_pred = vlm_predict(
            vlm_model, vlm_processor, sample, memory_images
        )
        agent_is_correct = (agent_pred == sample["answer"])
        if agent_is_correct:
            agent_correct += 1

        results["agent"].append({
            "pid": sample["pid"],
            "prediction": agent_pred,
            "correct": agent_is_correct,
            "selected_canvases": selected_ids,
            "agent_action": read_action.action if candidates else "NO_CANDIDATES",
        })

        # --- Baseline (no memory) ---
        if not skip_baseline:
            base_pred = vlm_predict(vlm_model, vlm_processor, sample, [])
            base_is_correct = (base_pred == sample["answer"])
            if base_is_correct:
                baseline_correct += 1
            results["baseline"].append({
                "pid": sample["pid"],
                "prediction": base_pred,
                "correct": base_is_correct,
            })

        # Progress
        if (i + 1) % 50 == 0:
            n = i + 1
            agent_acc = agent_correct / n * 100
            if not skip_baseline:
                base_acc = baseline_correct / n * 100
                print(f"  [{n}/{len(test_data)}] Agent: {agent_acc:.1f}% | Baseline: {base_acc:.1f}%")
            else:
                print(f"  [{n}/{len(test_data)}] Agent: {agent_acc:.1f}%")

    agent.unload()

    # Final results
    n = len(test_data)
    agent_acc = agent_correct / n * 100
    print(f"\n{'='*60}")
    print(f"RESULTS ({n} test samples)")
    print(f"{'='*60}")
    print(f"  Agent MemCanvas:  {agent_correct}/{n} = {agent_acc:.2f}%")
    if not skip_baseline:
        base_acc = baseline_correct / n * 100
        print(f"  Baseline:         {baseline_correct}/{n} = {base_acc:.2f}%")
        print(f"  Improvement:      {agent_acc - base_acc:+.2f}%")

    # Save results
    with open(OUTPUT_DIR / "eval_results.json", "w") as f:
        json.dump({
            "agent_accuracy": agent_acc,
            "baseline_accuracy": baseline_correct / n * 100 if not skip_baseline else None,
            "agent_correct": agent_correct,
            "baseline_correct": baseline_correct if not skip_baseline else None,
            "total": n,
            "prep_stats": {
                "num_canvases": cached_prep["num_canvases"],
                "total_items": cached_prep["total_items"],
                "action_counts": cached_prep["action_counts"],
            },
        }, f, indent=2)

    with open(OUTPUT_DIR / "eval_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {OUTPUT_DIR}")
    return results


# ---------------------------------------------------------------------------
# VLM prediction
# ---------------------------------------------------------------------------
def vlm_predict(
    model, processor, sample: Dict, memory_images: List[Image.Image],
) -> int:
    """Use VLM to answer a ScienceQA question, optionally with memory canvases."""
    # Build prompt
    choices_text = "\n".join(
        f"  {chr(65+i)}. {c}" for i, c in enumerate(sample["choices"])
    )

    if memory_images:
        user_text = (
            "Below are memory canvas images from similar problems you solved before. "
            "Study them carefully and apply the relevant knowledge.\n\n"
        )
        # Image placeholders will be added via messages
    else:
        user_text = ""

    question_text = ""
    if sample["hint"]:
        question_text += f"Context: {sample['hint']}\n"
    question_text += f"Question: {sample['question']}\n"
    question_text += f"Options:\n{choices_text}\n\n"
    question_text += "Answer with just the letter (A, B, C, D, etc.)."

    # Build messages
    content = []

    # Add memory canvas images
    for img in memory_images:
        content.append({"type": "image", "image": img})

    if memory_images:
        content.append({"type": "text", "text": user_text + "---\n\n" + question_text})
    else:
        content.append({"type": "text", "text": question_text})

    # Add test image if available
    img_path = sample.get("image_path")
    if img_path and os.path.exists(img_path):
        try:
            test_img = Image.open(img_path).convert("RGB")
            content.insert(len(memory_images), {"type": "image", "image": test_img})
        except Exception:
            pass

    messages = [{"role": "user", "content": content}]

    # Process and generate
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    all_images = [c["image"] for c in content if c.get("type") == "image"]
    proc_kwargs = dict(text=[text], return_tensors="pt", padding=True)
    if all_images:
        proc_kwargs["images"] = all_images
    inputs = processor(**proc_kwargs).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True).strip()

    # Extract answer letter
    return extract_answer(response, len(sample["choices"]))


def extract_answer(response: str, num_choices: int) -> int:
    """Extract answer index from VLM response."""
    import re
    response = response.strip().upper()

    # Look for letter patterns
    valid_letters = [chr(65 + i) for i in range(num_choices)]

    # Try common patterns
    for pattern in [
        r"(?:answer|option)\s*(?:is\s*)?([A-F])",
        r"\b([A-F])\b",
        r"([A-F])\.",
        r"([A-F])\s*$",
    ]:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            letter = matches[-1].upper()
            if letter in valid_letters:
                return ord(letter) - 65

    # Fallback: first valid letter found
    for ch in response:
        if ch.upper() in valid_letters:
            return ord(ch.upper()) - 65

    return 0  # default to A


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prep", "embed", "eval", "all"], default="all")
    parser.add_argument("--max-train", type=int, default=2000,
                        help="Max training samples for agent organization")
    parser.add_argument("--max-test", type=int, default=500,
                        help="Max test samples for evaluation")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to LoRA adapter for SFT/RL agent (e.g., agent_sft_output/final)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory (default: agent_experiment_output)")
    args = parser.parse_args()

    # Override output directory if specified
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    # Override agent model path with LoRA if specified
    global AGENT_LORA_PATH
    AGENT_LORA_PATH = args.lora_path

    train_data, test_data = load_scienceqa()

    if args.phase in ("prep", "all"):
        cached_prep = phase_prep(train_data, max_train=args.max_train)
    else:
        cache_file = OUTPUT_DIR / "agent_canvases.pkl"
        with open(cache_file, "rb") as f:
            cached_prep = pickle.load(f)

    if args.phase in ("embed", "all"):
        cached_embed = phase_embed(cached_prep)
    else:
        embed_file = OUTPUT_DIR / "agent_embeddings.pkl"
        with open(embed_file, "rb") as f:
            cached_embed = pickle.load(f)

    if args.phase in ("eval", "all"):
        phase_eval(test_data, cached_prep, cached_embed,
                   max_test=args.max_test, skip_baseline=args.skip_baseline)


if __name__ == "__main__":
    main()
