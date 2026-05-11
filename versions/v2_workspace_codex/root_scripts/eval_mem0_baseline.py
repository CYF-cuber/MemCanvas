#!/usr/bin/env python3
"""
Mem0 baseline evaluation on 4 multimodal benchmarks.

Feeds ONLY the text modality from each benchmark's training data into Mem0's
memory system, then retrieves text memories at test time. Compares against
MemCanvas's visual canvas memory to demonstrate the necessity of multimodal
visual memory.

Prerequisites:
    pip install mem0ai
    # Start vLLM server for Mem0's LLM backend:
    CUDA_VISIBLE_DEVICES=0 vllm serve /home/cyf/Qwen2.5-7B-Instruct \
        --port 8100 --gpu-memory-utilization 0.45 --max-model-len 4096

Usage:
    # Phase 1: Populate Mem0 memory for each benchmark
    python eval_mem0_baseline.py --phase populate --benchmark scienceqa
    python eval_mem0_baseline.py --phase populate --benchmark okvqa
    python eval_mem0_baseline.py --phase populate --benchmark mmqa
    python eval_mem0_baseline.py --phase populate --benchmark hotpotqa

    # Phase 2: Evaluate (needs GPU for VLM)
    CUDA_VISIBLE_DEVICES=1 python eval_mem0_baseline.py --phase eval --benchmark scienceqa
    CUDA_VISIBLE_DEVICES=1 python eval_mem0_baseline.py --phase eval --benchmark okvqa
    CUDA_VISIBLE_DEVICES=1 python eval_mem0_baseline.py --phase eval --benchmark mmqa
    CUDA_VISIBLE_DEVICES=1 python eval_mem0_baseline.py --phase eval --benchmark hotpotqa

    # All benchmarks sequentially
    python eval_mem0_baseline.py --phase populate --all
    CUDA_VISIBLE_DEVICES=1 python eval_mem0_baseline.py --phase eval --all
"""

import argparse, json, os, pickle, re, string, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ============================================================
# Config
# ============================================================
VLLM_BASE_URL = "http://localhost:8100/v1"
VLLM_MODEL = "qwen2.5-vl-7b"  # model name as served by our API server
VLM_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
OUT_ROOT = Path("/home/cyf/codex/mem0_baseline_eval")
TOP_K_RETRIEVE = 5  # Mem0 search top-k
MEM0_DB_DIR = OUT_ROOT / "mem0_dbs"

# ============================================================
# Metrics (same as MemCanvas eval)
# ============================================================
def normalize_answer(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())

def compute_em(p, g):
    return float(normalize_answer(p) == normalize_answer(g))

def compute_f1(p, g):
    pt, gt = normalize_answer(p).split(), normalize_answer(g).split()
    c = Counter(pt) & Counter(gt); n = sum(c.values())
    if n == 0: return 0.0
    pr = n / len(pt) if pt else 0; rc = n / len(gt) if gt else 0
    return 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0


# ============================================================
# Mem0 initialization
# ============================================================
def init_mem0(benchmark_name):
    """Initialize Mem0 with vLLM backend + local HuggingFace embeddings."""
    from mem0 import Memory

    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": VLLM_MODEL,
                "openai_base_url": VLLM_BASE_URL,
                "api_key": "dummy",  # vLLM doesn't need real key
                "temperature": 0.1,
                "max_tokens": 1000,
            }
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            }
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": f"mem0_{benchmark_name}",
                "path": str(MEM0_DB_DIR / benchmark_name),
            }
        },
        "version": "v1.1",
    }
    m = Memory.from_config(config)
    return m


# ============================================================
# Data loaders — extract TEXT ONLY from training data
# ============================================================
def load_scienceqa_train():
    """Load ScienceQA training data, extract text content."""
    with open("/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl", "rb") as f:
        cache = pickle.load(f)
    train = cache["train"] if isinstance(cache, dict) else cache[0]

    entries = []
    for item in train:
        parts = []
        parts.append(f"Question: {item['question']}")
        choices = item.get("choices", [])
        if choices:
            choice_str = ", ".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))
            parts.append(f"Choices: {choice_str}")
        if item.get("hint"):
            parts.append(f"Hint: {item['hint']}")
        if item.get("lecture"):
            parts.append(f"Lecture: {item['lecture']}")
        if item.get("solution"):
            parts.append(f"Solution: {item['solution']}")
        answer_idx = item["answer"]
        if choices and answer_idx < len(choices):
            parts.append(f"Answer: {chr(65 + answer_idx)}. {choices[answer_idx]}")

        text = "\n".join(parts)
        entries.append({
            "id": str(item.get("pid", len(entries))),
            "text": text,
            "subject": item.get("subject", ""),
        })
    return entries


def load_okvqa_train():
    """Load OK-VQA training data, extract text content (question + caption)."""
    with open("/home/cyf/codex/okvqa_data/okvqa_cached.pkl", "rb") as f:
        data = pickle.load(f)
    train = data["train"]

    entries = []
    for item in train:
        parts = []
        parts.append(f"Question: {item['question']}")
        if item.get("caption"):
            parts.append(f"Image description: {item['caption']}")
        answers = item.get("answers", [])
        if answers:
            # Most common answer
            ans_counts = Counter(answers)
            top_ans = ans_counts.most_common(1)[0][0]
            parts.append(f"Answer: {top_ans}")

        text = "\n".join(parts)
        entries.append({
            "id": str(item.get("question_id", len(entries))),
            "text": text,
        })
    return entries


def load_mmqa_train():
    """Load MMQA training data, extract text content (question + table + passage text)."""
    with open("/home/cyf/codex/mmqa_data/mmqa_parsed.pkl", "rb") as f:
        parsed = pickle.load(f)
    train_data = parsed.get("train", {})
    if isinstance(train_data, dict):
        items = list(train_data.values())
    else:
        items = list(train_data)
    tables = parsed.get("tables", {})
    texts = parsed.get("texts", {})

    def table_to_text(table_doc, max_rows=10):
        header = table_doc.get("header", [])
        rows = table_doc.get("rows", table_doc.get("data", []))
        if not header and rows:
            header = [f"Col{i}" for i in range(len(rows[0]))]
        lines = [" | ".join(str(h) for h in header)]
        for row in rows[:max_rows]:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)

    entries = []
    for item in items:
        parts = []
        parts.append(f"Question: {item['question']}")

        # Add supporting context (text only)
        for ctx in item.get("supporting_context", []):
            doc_id = ctx["doc_id"]
            doc_part = ctx["doc_part"]
            if doc_part == "text" and doc_id in texts:
                td = texts[doc_id]
                parts.append(f"Passage [{td.get('title', '')}]: {td.get('text', '')[:500]}")
            elif doc_part == "table" and doc_id in tables:
                td = tables[doc_id]
                tbl = td.get("table", td)
                parts.append(f"Table [{td.get('title', '')}]:\n{table_to_text(tbl)}")
            # Skip images — text-only baseline

        answers = [str(a["answer"]) for a in item.get("answers", [])]
        if answers:
            parts.append(f"Answer: {answers[0]}")

        text = "\n".join(parts)
        entries.append({
            "id": item.get("qid", str(len(entries))),
            "text": text,
        })
    return entries


def load_hotpotqa_train():
    """Load HotpotQA training data, extract text content (question + paragraphs)."""
    with open("/home/cyf/codex/hotpotqa_data/hotpotqa_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    train = meta["train"]

    entries = []
    for item in train:
        parts = []
        parts.append(f"Question: {item['question']}")
        for para in item.get("paragraphs", []):
            parts.append(f"[{para['title']}]: {para['text'][:500]}")
        parts.append(f"Answer: {item['answer']}")

        text = "\n".join(parts)
        entries.append({
            "id": item.get("id", str(len(entries))),
            "text": text,
            "type": item.get("type", ""),
        })
    return entries


# ============================================================
# Phase 1: Populate Mem0 memory
# ============================================================
def populate_benchmark(benchmark):
    """Store training data as Mem0 memories."""
    print(f"\n{'='*60}")
    print(f"  Populating Mem0 for {benchmark}")
    print(f"{'='*60}")

    loader = {
        "scienceqa": load_scienceqa_train,
        "okvqa": load_okvqa_train,
        "mmqa": load_mmqa_train,
        "hotpotqa": load_hotpotqa_train,
    }[benchmark]

    entries = loader()
    print(f"  Loaded {len(entries)} training entries")

    m = init_mem0(benchmark)

    # Checkpoint for resuming
    ckpt_path = OUT_ROOT / f"{benchmark}_populate_checkpoint.json"
    done_ids = set()
    if ckpt_path.exists():
        done_ids = set(json.load(open(ckpt_path)))
        print(f"  Resuming from checkpoint: {len(done_ids)} already done")

    start = time.time()
    errors = 0
    for i, entry in enumerate(tqdm(entries, desc=f"Populating {benchmark}")):
        if entry["id"] in done_ids:
            continue
        try:
            m.add(
                entry["text"],
                user_id=benchmark,
                metadata={"entry_id": entry["id"]},
                infer=False,  # Skip LLM fact extraction — store raw text directly
            )
            done_ids.add(entry["id"])
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  Error on entry {entry['id']}: {e}")
            if errors == 5:
                print("  (suppressing further error messages)")

        # Checkpoint every 500 entries
        if (i + 1) % 500 == 0:
            json.dump(list(done_ids), open(ckpt_path, "w"))
            elapsed = time.time() - start
            rate = len(done_ids) / elapsed if elapsed > 0 else 0
            remaining = (len(entries) - len(done_ids)) / rate if rate > 0 else 0
            print(f"\n  Progress: {len(done_ids)}/{len(entries)} "
                  f"({rate:.1f} entries/s, ~{remaining/60:.0f}min remaining)")

    json.dump(list(done_ids), open(ckpt_path, "w"))
    elapsed = time.time() - start
    print(f"\n  Done: {len(done_ids)}/{len(entries)} entries in {elapsed/60:.1f}min "
          f"({errors} errors)")


# ============================================================
# Phase 2: Evaluate with Mem0 retrieval + VLM
# ============================================================
def load_vlm():
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading VLM...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)
    return model, proc


def mem0_search(m, query, benchmark, top_k=TOP_K_RETRIEVE):
    """Search Mem0 and return list of memory text strings."""
    try:
        results = m.search(query, user_id=benchmark, limit=top_k)
        # Mem0 returns {"results": [...]} where each has "memory" key
        if isinstance(results, dict):
            memories = results.get("results", [])
        else:
            memories = results
        return [r.get("memory", "") for r in memories if r.get("memory")]
    except Exception as e:
        print(f"  Search error: {e}")
        return []


def eval_scienceqa(m):
    import torch
    from PIL import Image
    print(f"\n{'='*60}\n  Eval ScienceQA with Mem0 baseline\n{'='*60}")

    from datasets import load_dataset
    test_ds = load_dataset("derek-thomas/ScienceQA", split="test")
    print(f"  Test set: {len(test_ds)} samples")

    vlm, proc = load_vlm()
    OUT = OUT_ROOT / "scienceqa"
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    for i in tqdm(range(len(test_ds)), desc="ScienceQA"):
        if str(i) in results:
            continue
        item = test_ds[i]
        q = item["question"]
        choices = item["choices"]
        gt = chr(65 + item["answer"])
        hint = item.get("hint", "") or ""
        choice_txt = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))

        # Retrieve Mem0 memories
        search_query = f"{q} {hint}".strip()
        memories = mem0_search(m, search_query, "scienceqa")

        # Build prompt with text memories + test image
        content = []
        if memories:
            mem_text = "\n\n---\n\n".join(memories[:3])
            content.append({"type": "text", "text":
                f"Reference memories from similar questions:\n{mem_text}\n\n---\n"})

        if item.get("image") is not None:
            content.append({"type": "image", "image": item["image"].convert("RGB")})

        prompt = (f"{hint}\n\nQuestion: {q}\n{choice_txt}\n"
                  f"Think step by step, then answer with just the letter:")
        content.append({"type": "text", "text": prompt})

        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs = [item["image"].convert("RGB")] if item.get("image") is not None else []
        if imgs:
            inp = proc(text=[txt], images=imgs, return_tensors="pt", padding=True)
        else:
            inp = proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}

        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=512, do_sample=False)
        raw = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        pred = ""
        for c in raw.upper():
            if c in "ABCDEF": pred = c; break

        results[str(i)] = {"gt": gt, "pred": pred, "correct": float(pred == gt),
                           "subject": item.get("subject", ""),
                           "n_memories": len(memories)}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    acc = np.mean([v["correct"] for v in results.values()]) * 100
    summary = {"n": len(results), "accuracy": acc,
               "config": {"top_k": TOP_K_RETRIEVE, "method": "Mem0 (text-only)"}}
    for subj in ["natural science", "social science", "language science"]:
        vals = [v["correct"] for v in results.values() if v.get("subject", "") == subj]
        if vals: summary[subj] = np.mean(vals) * 100
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  ScienceQA Mem0: {acc:.2f}%")
    del vlm, proc; torch.cuda.empty_cache()
    return summary


def eval_okvqa(m):
    import torch
    from PIL import Image
    print(f"\n{'='*60}\n  Eval OK-VQA with Mem0 baseline\n{'='*60}")

    with open("/home/cyf/codex/okvqa_data/okvqa_cached.pkl", "rb") as f:
        data = pickle.load(f)
    test = data["test"]
    print(f"  Test set: {len(test)} samples")

    vlm, proc = load_vlm()
    OUT = OUT_ROOT / "okvqa"
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    for i in tqdm(range(len(test)), desc="OK-VQA"):
        if str(i) in results:
            continue
        s = test[i]
        q = s["question"]
        answers = s.get("answers", [])

        memories = mem0_search(m, q, "okvqa")

        content = []
        if memories:
            mem_text = "\n\n---\n\n".join(memories[:3])
            content.append({"type": "text", "text":
                f"Reference memories from similar questions:\n{mem_text}\n\n---\n"})

        img_path = s.get("image_path", "")
        test_img = None
        if img_path and os.path.exists(img_path):
            test_img = Image.open(img_path).convert("RGB")
            content.append({"type": "image", "image": test_img})

        content.append({"type": "text", "text":
            f"Answer the question about the image.\nQuestion: {q}\nAnswer concisely:"})

        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs = [test_img] if test_img else []
        if imgs:
            inp = proc(text=[txt], images=imgs, return_tensors="pt", padding=True)
        else:
            inp = proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}

        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=32, do_sample=False)
        pred = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        correct = float(any(normalize_answer(pred) == normalize_answer(a) for a in answers))
        results[str(i)] = {"gt": answers, "pred": pred, "correct": correct,
                           "n_memories": len(memories)}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    acc = np.mean([v["correct"] for v in results.values()]) * 100
    summary = {"n": len(results), "accuracy": acc,
               "config": {"top_k": TOP_K_RETRIEVE, "method": "Mem0 (text-only)"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  OK-VQA Mem0: {acc:.2f}%")
    del vlm, proc; torch.cuda.empty_cache()
    return summary


def eval_mmqa(m):
    import torch
    from PIL import Image
    print(f"\n{'='*60}\n  Eval MMQA with Mem0 baseline\n{'='*60}")

    with open("/home/cyf/codex/mmqa_data/mmqa_parsed.pkl", "rb") as f:
        parsed = pickle.load(f)
    dev_data = list(parsed["dev"].values()) if isinstance(parsed["dev"], dict) else parsed["dev"]
    tables = parsed.get("tables", {})
    texts = parsed.get("texts", {})
    images_meta = parsed.get("images", {})
    print(f"  Test set: {len(dev_data)} samples")

    def table_to_md(table_doc, max_rows=10):
        header = table_doc.get("header", [])
        rows = table_doc.get("rows", table_doc.get("data", []))
        if not header and rows:
            header = [f"Col{i}" for i in range(len(rows[0]))]
        lines = ["| " + " | ".join(str(h) for h in header) + " |",
                 "| " + " | ".join("---" for _ in header) + " |"]
        for row in rows[:max_rows]:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)

    IMG_DIR = Path("/home/cyf/codex/mmqa_data/final_dataset_images")
    def load_img(doc_id):
        for ext in [".jpg", ".png", ".jpeg"]:
            p = IMG_DIR / f"{doc_id}{ext}"
            if p.exists():
                return Image.open(p).convert("RGB")
        return None

    vlm, proc = load_vlm()
    OUT = OUT_ROOT / "mmqa"
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    for i in tqdm(range(len(dev_data)), desc="MMQA"):
        if str(i) in results:
            continue
        sample = dev_data[i]
        q = sample["question"]
        gold_answers = [str(a["answer"]) for a in sample.get("answers", [])]

        memories = mem0_search(m, q, "mmqa")

        # Build context from supporting docs (text parts only)
        text_parts = []
        ctx_images = []
        for ctx in sample.get("supporting_context", []):
            doc_id = ctx["doc_id"]
            doc_part = ctx["doc_part"]
            if doc_part == "text" and doc_id in texts:
                td = texts[doc_id]
                text_parts.append(f"[Text: {td.get('title','')}]\n{td.get('text','')[:500]}")
            elif doc_part == "table" and doc_id in tables:
                td = tables[doc_id]
                md = table_to_md(td.get("table", td))
                text_parts.append(f"[Table: {td.get('title','')}]\n{md}")
            elif doc_part == "image":
                img = load_img(doc_id)
                if img:
                    ctx_images.append(img)
        ctx_text = "\n\n".join(text_parts)

        # Build VLM prompt
        content = []
        if memories:
            mem_text = "\n\n---\n\n".join(memories[:3])
            content.append({"type": "text", "text":
                f"Reference memories:\n{mem_text}\n\n---\n"})

        for img in ctx_images:
            content.append({"type": "image", "image": img})

        prompt_parts = []
        if ctx_text:
            prompt_parts.append(f"Context:\n{ctx_text}\n")
        prompt_parts.append(f"Question: {q}\nAnswer concisely:")
        content.append({"type": "text", "text": "\n".join(prompt_parts)})

        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if ctx_images:
            inp = proc(text=[txt], images=ctx_images, return_tensors="pt", padding=True)
        else:
            inp = proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}

        try:
            with torch.no_grad():
                out = vlm.generate(**inp, max_new_tokens=64, do_sample=False)
            raw = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            raw = ""
            print(f"\n  Error {i}: {e}")

        best_em = max(compute_em(raw, ga) for ga in gold_answers) if gold_answers else 0.0
        best_f1 = max(compute_f1(raw, ga) for ga in gold_answers) if gold_answers else 0.0
        results[str(i)] = {"gt": gold_answers, "pred": raw, "em": best_em, "f1": best_f1,
                           "n_memories": len(memories)}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    em = np.mean([v["em"] for v in results.values()]) * 100
    f1 = np.mean([v["f1"] for v in results.values()]) * 100
    summary = {"n": len(results), "em": em, "f1": f1,
               "config": {"top_k": TOP_K_RETRIEVE, "method": "Mem0 (text-only)"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  MMQA Mem0: EM={em:.2f}, F1={f1:.2f}")
    del vlm, proc; torch.cuda.empty_cache()
    return summary


def eval_hotpotqa(m):
    import torch
    print(f"\n{'='*60}\n  Eval HotpotQA with Mem0 baseline\n{'='*60}")

    with open("/home/cyf/codex/hotpotqa_data/hotpotqa_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dev_data = meta["dev"]
    print(f"  Test set: {len(dev_data)} samples")

    vlm, proc = load_vlm()
    OUT = OUT_ROOT / "hotpotqa"
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    for i in tqdm(range(len(dev_data)), desc="HotpotQA"):
        if str(i) in results:
            continue
        sample = dev_data[i]
        q = sample["question"]
        gt = sample["answer"]

        memories = mem0_search(m, q, "hotpotqa")

        # Format context from test-time paragraphs
        ctx_parts = []
        for para in sample.get("paragraphs", []):
            ctx_parts.append(f"[{para['title']}]\n{para['text'][:500]}")
        ctx_text = "\n\n".join(ctx_parts)

        # Build text-only prompt (HotpotQA has no images)
        content = []
        if memories:
            mem_text = "\n\n---\n\n".join(memories[:3])
            content.append({"type": "text", "text":
                f"Reference memories from similar questions:\n{mem_text}\n\n---\n"})

        prompt = (f"Answer the question using the context below.\n\n"
                  f"Context:\n{ctx_text}\n\n"
                  f"Question: {q}\nAnswer concisely:")
        content.append({"type": "text", "text": prompt})

        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}

        try:
            with torch.no_grad():
                out = vlm.generate(**inp, max_new_tokens=64, do_sample=False)
            raw = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            raw = ""
            print(f"\n  Error {i}: {e}")

        em_val = compute_em(raw, gt)
        f1_val = compute_f1(raw, gt)
        results[str(i)] = {"gt": gt, "pred": raw, "em": em_val, "f1": f1_val,
                           "n_memories": len(memories)}
        if len(results) % 200 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    em = np.mean([v["em"] for v in results.values()]) * 100
    f1 = np.mean([v["f1"] for v in results.values()]) * 100
    summary = {"n": len(results), "em": em, "f1": f1,
               "config": {"top_k": TOP_K_RETRIEVE, "method": "Mem0 (text-only)"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  HotpotQA Mem0: EM={em:.2f}, F1={f1:.2f}")
    del vlm, proc; torch.cuda.empty_cache()
    return summary


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mem0 baseline evaluation")
    parser.add_argument("--phase", choices=["populate", "eval"], required=True)
    parser.add_argument("--benchmark", choices=["scienceqa", "okvqa", "mmqa", "hotpotqa"])
    parser.add_argument("--all", action="store_true", help="Run all 4 benchmarks")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MEM0_DB_DIR.mkdir(parents=True, exist_ok=True)

    benchmarks = ["scienceqa", "okvqa", "mmqa", "hotpotqa"] if args.all else [args.benchmark]
    if not args.all and not args.benchmark:
        parser.error("Specify --benchmark or --all")

    if args.phase == "populate":
        for bm in benchmarks:
            populate_benchmark(bm)

    elif args.phase == "eval":
        all_results = {}
        for bm in benchmarks:
            m = init_mem0(bm)
            fn = {"scienceqa": eval_scienceqa, "okvqa": eval_okvqa,
                  "mmqa": eval_mmqa, "hotpotqa": eval_hotpotqa}[bm]
            s = fn(m)
            all_results[bm] = s

        if len(all_results) > 1:
            json.dump(all_results, open(OUT_ROOT / "all_results.json", "w"), indent=2)
            print(f"\n{'='*60}\n  ALL RESULTS (Mem0 baseline)\n{'='*60}")
            for bm, s in all_results.items():
                if "accuracy" in s:
                    print(f"  {bm}: {s['accuracy']:.2f}%")
                else:
                    print(f"  {bm}: EM={s['em']:.2f}, F1={s['f1']:.2f}")
