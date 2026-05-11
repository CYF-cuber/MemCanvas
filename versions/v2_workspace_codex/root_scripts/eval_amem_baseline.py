#!/usr/bin/env python3
"""
A-Mem (Agentic Memory, NeurIPS 2025) baseline evaluation on 4 multimodal benchmarks.

Feeds ONLY the text modality from each benchmark's training data into A-Mem's
Zettelkasten-style memory system, then retrieves text memories at test time.

Prerequisites:
    pip install rank-bm25 nltk sentence-transformers litellm
    python -c "import nltk; nltk.download('punkt_tab')"

    # Start vLLM server for A-Mem's LLM backend (OpenAI-compatible):
    CUDA_VISIBLE_DEVICES=0 vllm serve /home/cyf/Qwen2.5-7B-Instruct \
        --port 8100 --gpu-memory-utilization 0.45 --max-model-len 4096

Usage:
    # Phase 1: Populate A-Mem memory
    python eval_amem_baseline.py --phase populate --benchmark scienceqa
    python eval_amem_baseline.py --phase populate --benchmark hotpotqa

    # Phase 2: Evaluate (needs GPU for VLM)
    CUDA_VISIBLE_DEVICES=1 python eval_amem_baseline.py --phase eval --benchmark scienceqa

    # All benchmarks
    python eval_amem_baseline.py --phase populate --all
    CUDA_VISIBLE_DEVICES=1 python eval_amem_baseline.py --phase eval --all
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
OUT_ROOT = Path("/home/cyf/codex/amem_baseline_eval")
TOP_K_RETRIEVE = 5
AMEM_DIR = Path("/home/cyf/codex/A-mem")

# Add A-Mem to path
sys.path.insert(0, str(AMEM_DIR))

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
# A-Mem initialization
# ============================================================
def init_amem():
    """Initialize A-Mem with vLLM-backed OpenAI-compatible server."""
    # Set env vars so A-Mem's OpenAI controller uses vLLM
    os.environ["OPENAI_API_KEY"] = "dummy"
    os.environ["OPENAI_BASE_URL"] = VLLM_BASE_URL

    from memory_layer import AgenticMemorySystem
    mem = AgenticMemorySystem(
        model_name='all-MiniLM-L6-v2',
        llm_backend='openai',
        llm_model=VLLM_MODEL,
        api_key='dummy',
        evo_threshold=200,  # Consolidate every 200 notes (less frequent = faster)
    )
    return mem


def save_amem(mem, benchmark):
    """Persist A-Mem state to disk."""
    save_path = OUT_ROOT / f"{benchmark}_amem_state.pkl"
    state = {
        "memories": mem.memories,
        "evo_cnt": mem.evo_cnt,
    }
    with open(save_path, "wb") as f:
        pickle.dump(state, f)
    # Also save retriever documents for rebuilding
    retriever_path = OUT_ROOT / f"{benchmark}_amem_retriever.pkl"
    with open(retriever_path, "wb") as f:
        pickle.dump({
            "documents": mem.retriever.documents if hasattr(mem.retriever, 'documents') else [],
            "bm25_corpus": mem.retriever.bm25_corpus if hasattr(mem.retriever, 'bm25_corpus') else [],
        }, f)
    print(f"  Saved A-Mem state: {len(mem.memories)} memories → {save_path}")


def _flatten(x):
    """Recursively flatten nested lists/values to a flat list of strings."""
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        out = []
        for item in x:
            out.extend(_flatten(item))
        return out
    return [str(x)] if x else []


class TfidfAmemWrapper:
    """Wraps A-Mem memories with a TF-IDF retriever for robust search."""
    def __init__(self, memories):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.memory_ids = list(memories.keys())
        self.memories = memories
        docs = []
        for mid in self.memory_ids:
            m = memories[mid]
            parts = [m.content or ""]
            parts.extend(_flatten(m.context))
            parts.extend(_flatten(m.keywords))
            parts.extend(_flatten(m.tags))
            docs.append(" ".join(parts))
        self.vectorizer = TfidfVectorizer(max_features=50000, stop_words='english')
        self.tfidf = self.vectorizer.fit_transform(docs)
        print(f"  Built TF-IDF retriever: {len(docs)} documents, {self.tfidf.shape[1]} features")

    def search(self, query, k=5):
        q_vec = self.vectorizer.transform([query])
        scores = (self.tfidf @ q_vec.T).toarray().flatten()
        top_idx = scores.argsort()[::-1][:k]
        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                results.append(self.memories[self.memory_ids[idx]].content)
        return results


def load_amem(benchmark):
    """Load persisted A-Mem state and build TF-IDF retriever."""
    save_path = OUT_ROOT / f"{benchmark}_amem_state.pkl"
    if not save_path.exists():
        raise FileNotFoundError(f"No A-Mem state for {benchmark}: {save_path}")
    with open(save_path, "rb") as f:
        state = pickle.load(f)
    wrapper = TfidfAmemWrapper(state["memories"])
    return wrapper


# ============================================================
# Data loaders — extract TEXT ONLY (same as Mem0 baseline)
# ============================================================
def load_scienceqa_train():
    with open("/home/cyf/codex/agent_experiment_output/sciqa_cached.pkl", "rb") as f:
        cache = pickle.load(f)
    train = cache["train"] if isinstance(cache, dict) else cache[0]
    entries = []
    for item in train:
        parts = [f"Question: {item['question']}"]
        choices = item.get("choices", [])
        if choices:
            parts.append(f"Choices: {', '.join(f'{chr(65+j)}. {c}' for j, c in enumerate(choices))}")
        if item.get("hint"): parts.append(f"Hint: {item['hint']}")
        if item.get("lecture"): parts.append(f"Lecture: {item['lecture']}")
        if item.get("solution"): parts.append(f"Solution: {item['solution']}")
        answer_idx = item["answer"]
        if choices and answer_idx < len(choices):
            parts.append(f"Answer: {chr(65 + answer_idx)}. {choices[answer_idx]}")
        entries.append({"id": str(item.get("pid", len(entries))), "text": "\n".join(parts)})
    return entries


def load_okvqa_train():
    with open("/home/cyf/codex/okvqa_data/okvqa_cached.pkl", "rb") as f:
        data = pickle.load(f)
    entries = []
    for item in data["train"]:
        parts = [f"Question: {item['question']}"]
        if item.get("caption"): parts.append(f"Image description: {item['caption']}")
        answers = item.get("answers", [])
        if answers:
            top_ans = Counter(answers).most_common(1)[0][0]
            parts.append(f"Answer: {top_ans}")
        entries.append({"id": str(item.get("question_id", len(entries))), "text": "\n".join(parts)})
    return entries


def load_mmqa_train():
    with open("/home/cyf/codex/mmqa_data/mmqa_parsed.pkl", "rb") as f:
        parsed = pickle.load(f)
    train_data = parsed.get("train", {})
    items = list(train_data.values()) if isinstance(train_data, dict) else list(train_data)
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
        parts = [f"Question: {item['question']}"]
        for ctx in item.get("supporting_context", []):
            doc_id, doc_part = ctx["doc_id"], ctx["doc_part"]
            if doc_part == "text" and doc_id in texts:
                td = texts[doc_id]
                parts.append(f"Passage [{td.get('title', '')}]: {td.get('text', '')[:500]}")
            elif doc_part == "table" and doc_id in tables:
                td = tables[doc_id]
                parts.append(f"Table [{td.get('title', '')}]:\n{table_to_text(td.get('table', td))}")
        answers = [str(a["answer"]) for a in item.get("answers", [])]
        if answers: parts.append(f"Answer: {answers[0]}")
        entries.append({"id": item.get("qid", str(len(entries))), "text": "\n".join(parts)})
    return entries


def load_hotpotqa_train():
    with open("/home/cyf/codex/hotpotqa_data/hotpotqa_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    entries = []
    for item in meta["train"]:
        parts = [f"Question: {item['question']}"]
        for para in item.get("paragraphs", []):
            parts.append(f"[{para['title']}]: {para['text'][:500]}")
        parts.append(f"Answer: {item['answer']}")
        entries.append({"id": item.get("id", str(len(entries))), "text": "\n".join(parts)})
    return entries


# ============================================================
# Phase 1: Populate A-Mem memory
# ============================================================
def populate_benchmark(benchmark):
    print(f"\n{'='*60}")
    print(f"  Populating A-Mem for {benchmark}")
    print(f"{'='*60}")

    loader = {
        "scienceqa": load_scienceqa_train, "okvqa": load_okvqa_train,
        "mmqa": load_mmqa_train, "hotpotqa": load_hotpotqa_train,
    }[benchmark]

    entries = loader()
    print(f"  Loaded {len(entries)} training entries")

    # Check for existing state
    state_path = OUT_ROOT / f"{benchmark}_amem_state.pkl"
    ckpt_path = OUT_ROOT / f"{benchmark}_populate_checkpoint.json"
    done_ids = set()
    if ckpt_path.exists():
        done_ids = set(json.load(open(ckpt_path)))
        print(f"  Resuming from checkpoint: {len(done_ids)} already done")

    if state_path.exists() and done_ids:
        mem = load_amem(benchmark)
    else:
        mem = init_amem()

    # Suppress A-Mem's verbose prints during population
    import io
    class SuppressPrint:
        def __init__(self): self.real_stdout = sys.stdout
        def write(self, x):
            if not x.startswith("analysis") and not x.startswith("prompt_memory"):
                self.real_stdout.write(x)
        def flush(self): self.real_stdout.flush()

    old_stdout = sys.stdout
    sys.stdout = SuppressPrint()

    start = time.time()
    errors = 0
    for i, entry in enumerate(tqdm(entries, desc=f"Populating {benchmark}", file=old_stdout)):
        if entry["id"] in done_ids:
            continue
        try:
            mem.add_note(content=entry["text"][:2000])  # Truncate very long entries
            done_ids.add(entry["id"])
        except Exception as e:
            errors += 1
            if errors <= 5:
                old_stdout.write(f"\n  Error on entry {entry['id']}: {e}\n")
            if errors == 5:
                old_stdout.write("  (suppressing further error messages)\n")

        # Checkpoint every 500 entries
        if (i + 1) % 500 == 0:
            sys.stdout = old_stdout
            json.dump(list(done_ids), open(ckpt_path, "w"))
            save_amem(mem, benchmark)
            elapsed = time.time() - start
            rate = len(done_ids) / elapsed if elapsed > 0 else 0
            remaining = (len(entries) - len(done_ids)) / rate if rate > 0 else 0
            print(f"\n  Progress: {len(done_ids)}/{len(entries)} "
                  f"({rate:.1f} entries/s, ~{remaining/60:.0f}min remaining)")
            sys.stdout = SuppressPrint()

    sys.stdout = old_stdout
    json.dump(list(done_ids), open(ckpt_path, "w"))
    save_amem(mem, benchmark)
    elapsed = time.time() - start
    print(f"\n  Done: {len(done_ids)}/{len(entries)} entries in {elapsed/60:.1f}min "
          f"({errors} errors)")


# ============================================================
# Phase 2: Evaluate with A-Mem retrieval + VLM
# ============================================================
def load_vlm():
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading VLM...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)
    return model, proc


def amem_search(mem, query, k=TOP_K_RETRIEVE):
    """Search A-Mem wrapper and return list of memory text strings."""
    try:
        return mem.search(query, k=k)
    except Exception as e:
        print(f"  Search error: {e}")
        return []


def eval_scienceqa(mem):
    import torch
    from PIL import Image
    print(f"\n{'='*60}\n  Eval ScienceQA with A-Mem baseline\n{'='*60}")

    from datasets import load_dataset
    test_ds = load_dataset("derek-thomas/ScienceQA", split="test")
    print(f"  Test set: {len(test_ds)} samples")

    vlm, proc = load_vlm()
    OUT = OUT_ROOT / "scienceqa"
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    for i in tqdm(range(len(test_ds)), desc="ScienceQA"):
        if str(i) in results: continue
        item = test_ds[i]
        q = item["question"]
        choices = item["choices"]
        gt = chr(65 + item["answer"])
        hint = item.get("hint", "") or ""
        choice_txt = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))

        search_query = f"{q} {hint}".strip()
        memories = amem_search(mem, search_query)

        content = []
        if memories:
            mem_text = "\n\n---\n\n".join(memories[:3])
            content.append({"type": "text", "text":
                f"Reference memories from similar questions:\n{mem_text}\n\n---\n"})
        if item.get("image") is not None:
            content.append({"type": "image", "image": item["image"].convert("RGB")})
        content.append({"type": "text", "text":
            f"{hint}\n\nQuestion: {q}\n{choice_txt}\nThink step by step, then answer with just the letter:"})

        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs = [item["image"].convert("RGB")] if item.get("image") is not None else []
        inp = proc(text=[txt], images=imgs, return_tensors="pt", padding=True) if imgs else proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}

        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=512, do_sample=False)
        raw = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        pred = ""
        for c in raw.upper():
            if c in "ABCDEF": pred = c; break

        results[str(i)] = {"gt": gt, "pred": pred, "correct": float(pred == gt),
                           "subject": item.get("subject", ""), "n_memories": len(memories)}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    acc = np.mean([v["correct"] for v in results.values()]) * 100
    summary = {"n": len(results), "accuracy": acc,
               "config": {"top_k": TOP_K_RETRIEVE, "method": "A-Mem (text-only)"}}
    for subj in ["natural science", "social science", "language science"]:
        vals = [v["correct"] for v in results.values() if v.get("subject", "") == subj]
        if vals: summary[subj] = np.mean(vals) * 100
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  ScienceQA A-Mem: {acc:.2f}%")
    del vlm, proc; torch.cuda.empty_cache()
    return summary


def eval_okvqa(mem):
    import torch
    from PIL import Image
    print(f"\n{'='*60}\n  Eval OK-VQA with A-Mem baseline\n{'='*60}")

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
        if str(i) in results: continue
        s = test[i]
        q = s["question"]
        answers = s.get("answers", [])

        memories = amem_search(mem, q)

        content = []
        if memories:
            mem_text = "\n\n---\n\n".join(memories[:3])
            content.append({"type": "text", "text":
                f"Reference memories:\n{mem_text}\n\n---\n"})
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
        inp = proc(text=[txt], images=imgs, return_tensors="pt", padding=True) if imgs else proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}

        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=32, do_sample=False)
        pred = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        correct = float(any(normalize_answer(pred) == normalize_answer(a) for a in answers))
        results[str(i)] = {"gt": answers, "pred": pred, "correct": correct, "n_memories": len(memories)}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    acc = np.mean([v["correct"] for v in results.values()]) * 100
    summary = {"n": len(results), "accuracy": acc,
               "config": {"top_k": TOP_K_RETRIEVE, "method": "A-Mem (text-only)"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  OK-VQA A-Mem: {acc:.2f}%")
    del vlm, proc; torch.cuda.empty_cache()
    return summary


def eval_mmqa(mem):
    import torch
    from PIL import Image
    print(f"\n{'='*60}\n  Eval MMQA with A-Mem baseline\n{'='*60}")

    with open("/home/cyf/codex/mmqa_data/mmqa_parsed.pkl", "rb") as f:
        parsed = pickle.load(f)
    dev_data = list(parsed["dev"].values()) if isinstance(parsed["dev"], dict) else parsed["dev"]
    tables = parsed.get("tables", {})
    texts = parsed.get("texts", {})
    print(f"  Test set: {len(dev_data)} samples")

    def table_to_md(table_doc, max_rows=10):
        header = table_doc.get("header", [])
        rows = table_doc.get("rows", table_doc.get("data", []))
        if not header and rows: header = [f"Col{i}" for i in range(len(rows[0]))]
        lines = ["| " + " | ".join(str(h) for h in header) + " |",
                 "| " + " | ".join("---" for _ in header) + " |"]
        for row in rows[:max_rows]:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)

    IMG_DIR = Path("/home/cyf/codex/mmqa_data/final_dataset_images")
    def load_img(doc_id):
        for ext in [".jpg", ".png", ".jpeg"]:
            p = IMG_DIR / f"{doc_id}{ext}"
            if p.exists(): return Image.open(p).convert("RGB")
        return None

    vlm, proc = load_vlm()
    OUT = OUT_ROOT / "mmqa"
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    for i in tqdm(range(len(dev_data)), desc="MMQA"):
        if str(i) in results: continue
        sample = dev_data[i]
        q = sample["question"]
        gold_answers = [str(a["answer"]) for a in sample.get("answers", [])]

        memories = amem_search(mem, q)

        text_parts, ctx_images = [], []
        for ctx in sample.get("supporting_context", []):
            doc_id, doc_part = ctx["doc_id"], ctx["doc_part"]
            if doc_part == "text" and doc_id in texts:
                td = texts[doc_id]
                text_parts.append(f"[Text: {td.get('title','')}]\n{td.get('text','')[:500]}")
            elif doc_part == "table" and doc_id in tables:
                td = tables[doc_id]
                text_parts.append(f"[Table: {td.get('title','')}]\n{table_to_md(td.get('table', td))}")
            elif doc_part == "image":
                img = load_img(doc_id)
                if img: ctx_images.append(img)
        ctx_text = "\n\n".join(text_parts)

        content = []
        if memories:
            mem_text = "\n\n---\n\n".join(memories[:3])
            content.append({"type": "text", "text": f"Reference memories:\n{mem_text}\n\n---\n"})
        for img in ctx_images:
            content.append({"type": "image", "image": img})
        prompt_parts = []
        if ctx_text: prompt_parts.append(f"Context:\n{ctx_text}\n")
        prompt_parts.append(f"Question: {q}\nAnswer concisely:")
        content.append({"type": "text", "text": "\n".join(prompt_parts)})

        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[txt], images=ctx_images, return_tensors="pt", padding=True) if ctx_images else proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}

        try:
            with torch.no_grad():
                out = vlm.generate(**inp, max_new_tokens=64, do_sample=False)
            raw = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            raw = ""; print(f"\n  Error {i}: {e}")

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
               "config": {"top_k": TOP_K_RETRIEVE, "method": "A-Mem (text-only)"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  MMQA A-Mem: EM={em:.2f}, F1={f1:.2f}")
    del vlm, proc; torch.cuda.empty_cache()
    return summary


def eval_hotpotqa(mem):
    import torch
    print(f"\n{'='*60}\n  Eval HotpotQA with A-Mem baseline\n{'='*60}")

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
        if str(i) in results: continue
        sample = dev_data[i]
        q = sample["question"]
        gt = sample["answer"]

        memories = amem_search(mem, q)

        ctx_parts = [f"[{p['title']}]\n{p['text'][:500]}" for p in sample.get("paragraphs", [])]
        ctx_text = "\n\n".join(ctx_parts)

        content = []
        if memories:
            mem_text = "\n\n---\n\n".join(memories[:3])
            content.append({"type": "text", "text":
                f"Reference memories from similar questions:\n{mem_text}\n\n---\n"})
        content.append({"type": "text", "text":
            f"Answer the question using the context below.\n\nContext:\n{ctx_text}\n\n"
            f"Question: {q}\nAnswer concisely:"})

        msgs = [{"role": "user", "content": content}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[txt], return_tensors="pt", padding=True)
        inp = {k: v.to(vlm.device) for k, v in inp.items()}

        try:
            with torch.no_grad():
                out = vlm.generate(**inp, max_new_tokens=64, do_sample=False)
            raw = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            raw = ""; print(f"\n  Error {i}: {e}")

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
               "config": {"top_k": TOP_K_RETRIEVE, "method": "A-Mem (text-only)"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  HotpotQA A-Mem: EM={em:.2f}, F1={f1:.2f}")
    del vlm, proc; torch.cuda.empty_cache()
    return summary


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A-Mem baseline evaluation")
    parser.add_argument("--phase", choices=["populate", "eval"], required=True)
    parser.add_argument("--benchmark", choices=["scienceqa", "okvqa", "mmqa", "hotpotqa"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    benchmarks = ["scienceqa", "okvqa", "mmqa", "hotpotqa"] if args.all else [args.benchmark]
    if not args.all and not args.benchmark:
        parser.error("Specify --benchmark or --all")

    if args.phase == "populate":
        for bm in benchmarks:
            populate_benchmark(bm)

    elif args.phase == "eval":
        all_results = {}
        for bm in benchmarks:
            mem = load_amem(bm)
            fn = {"scienceqa": eval_scienceqa, "okvqa": eval_okvqa,
                  "mmqa": eval_mmqa, "hotpotqa": eval_hotpotqa}[bm]
            s = fn(mem)
            all_results[bm] = s

        if len(all_results) > 1:
            json.dump(all_results, open(OUT_ROOT / "all_results.json", "w"), indent=2)
            print(f"\n{'='*60}\n  ALL RESULTS (A-Mem baseline)\n{'='*60}")
            for bm, s in all_results.items():
                if "accuracy" in s:
                    print(f"  {bm}: {s['accuracy']:.2f}%")
                else:
                    print(f"  {bm}: EM={s['em']:.2f}, F1={s['f1']:.2f}")
