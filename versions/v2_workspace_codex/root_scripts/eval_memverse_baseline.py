#!/usr/bin/env python3
"""
MemVerse baseline evaluation on OK-VQA and MMQA.

Uses OpenAI API (GPT-4o-mini + text-embedding-3-small) for memory construction
via MemVerse's LightRAG knowledge graph + tripartite memory pipeline.
Uses Qwen2.5-VL-7B for final answer generation (same backbone as other baselines).

Prerequisites:
    export OPENAI_API_KEY="sk-..."
    # MemVerse repo at /home/cyf/memory/MemVerse/

Usage:
    # Phase 1: Populate MemVerse memory
    python eval_memverse_baseline.py --phase populate --benchmark okvqa
    python eval_memverse_baseline.py --phase populate --benchmark mmqa
    python eval_memverse_baseline.py --phase populate --all

    # Phase 2: Evaluate (needs GPU for VLM)
    CUDA_VISIBLE_DEVICES=1 python eval_memverse_baseline.py --phase eval --benchmark okvqa
    CUDA_VISIBLE_DEVICES=1 python eval_memverse_baseline.py --phase eval --all

    # Test with small batch
    python eval_memverse_baseline.py --phase populate --benchmark okvqa --max-entries 5
"""

import argparse, asyncio, json, os, pickle, re, string, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ============================================================
# Config
# ============================================================
MEMVERSE_ROOT = Path("/home/cyf/memory/MemVerse")
VLM_MODEL = "/home/cyf/Qwen2.5-VL-7B-Instruct"
OUT_ROOT = Path("/home/cyf/codex/memverse_baseline_eval")
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Memory agent prompt files (from MemVerse repo)
PROMPT_FILES = {
    "core": MEMVERSE_ROOT / "MemoryKB" / "Long_Term_Memory" / "system" / "core_memory_agent.txt",
    "episodic": MEMVERSE_ROOT / "MemoryKB" / "Long_Term_Memory" / "system" / "episodic_memory_agent.txt",
    "semantic": MEMVERSE_ROOT / "MemoryKB" / "Long_Term_Memory" / "system" / "semantic_memory_agent.txt",
}

# ============================================================
# Metrics (same as MemCanvas/Mem0 eval)
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
# OpenAI client setup
# ============================================================
def get_openai_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")
    return OpenAI(api_key=api_key)


def get_embedding(client, text):
    """Get text-embedding-3-small embedding."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def classify_memory(client, input_text, prompt_type):
    """Classify text into one memory type using GPT-4o-mini."""
    prompt = PROMPT_FILES[prompt_type].read_text(encoding="utf-8").strip()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text}
        ],
        temperature=0,
        max_tokens=1000,
    )
    return resp.choices[0].message.content


# ============================================================
# LightRAG setup
# ============================================================
def setup_lightrag_imports():
    """Add MemVerse paths to sys.path for LightRAG imports."""
    lightrag_path = str(MEMVERSE_ROOT / "MemoryKB" / "Long_Term_Memory" / "Graph_Construction")
    memverse_path = str(MEMVERSE_ROOT)
    for p in [lightrag_path, memverse_path]:
        if p not in sys.path:
            sys.path.insert(0, p)


async def init_rag_instance(working_dir):
    """Initialize a single LightRAG instance."""
    setup_lightrag_imports()
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_embed, gpt_4o_mini_complete
    from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status

    os.makedirs(working_dir, exist_ok=True)
    initialize_share_data(workers=1)

    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def init_three_rags(benchmark):
    """Initialize core/episodic/semantic RAG instances for a benchmark."""
    base = OUT_ROOT / benchmark / "MMKG"
    rags = {}
    for mem_type in ["core", "episodic", "semantic"]:
        rag = await init_rag_instance(str(base / mem_type))
        rags[mem_type] = rag
    return rags


# ============================================================
# Data loaders (reused from eval_mem0_baseline.py)
# ============================================================
def load_okvqa_train():
    with open("/home/cyf/codex/okvqa_data/okvqa_cached.pkl", "rb") as f:
        data = pickle.load(f)
    train = data["train"]
    entries = []
    for item in train:
        q = item["question"]
        caption = item.get("caption", "")
        answers = item.get("answers", [])
        top_ans = Counter(answers).most_common(1)[0][0] if answers else ""

        query_text = f"Question: {q}\nAnswer: {top_ans}"
        entries.append({
            "id": str(item.get("question_id", len(entries))),
            "query": query_text,
            "imagecaption": f"Image description: {caption}" if caption else None,
            "videocaption": None,
            "audiocaption": None,
        })
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
            doc_id = ctx["doc_id"]
            doc_part = ctx["doc_part"]
            if doc_part == "text" and doc_id in texts:
                td = texts[doc_id]
                parts.append(f"Passage [{td.get('title', '')}]: {td.get('text', '')[:500]}")
            elif doc_part == "table" and doc_id in tables:
                td = tables[doc_id]
                tbl = td.get("table", td)
                parts.append(f"Table [{td.get('title', '')}]:\n{table_to_text(tbl)}")

        answers = [str(a["answer"]) for a in item.get("answers", [])]
        if answers:
            parts.append(f"Answer: {answers[0]}")

        entries.append({
            "id": item.get("qid", str(len(entries))),
            "query": "\n".join(parts),
            "imagecaption": None,
            "videocaption": None,
            "audiocaption": None,
        })
    return entries


# ============================================================
# Phase 1: Populate MemVerse memory
# ============================================================
async def _populate_async(benchmark, max_entries=None):
    """Async inner function for populate — single event loop for all LightRAG ops."""
    print(f"\n{'='*60}")
    print(f"  Populating MemVerse for {benchmark}")
    print(f"{'='*60}")

    loader = {"okvqa": load_okvqa_train, "mmqa": load_mmqa_train}[benchmark]
    entries = loader()
    if max_entries:
        entries = entries[:max_entries]
    print(f"  Loaded {len(entries)} training entries")

    client = get_openai_client()

    # Checkpoint
    bench_dir = OUT_ROOT / benchmark
    bench_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = bench_dir / "populate_checkpoint.json"
    done_ids = set()
    if ckpt_path.exists():
        done_ids = set(json.load(open(ckpt_path)))
        print(f"  Resuming from checkpoint: {len(done_ids)} already done")

    # Initialize LightRAG instances (in same event loop)
    print("  Initializing LightRAG...")
    rags = await init_three_rags(benchmark)

    # Memory chunk output files
    chunk_dir = bench_dir / "memory_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    errors = 0
    api_calls = 0

    for i, entry in enumerate(tqdm(entries, desc=f"Populating {benchmark}")):
        if entry["id"] in done_ids:
            continue

        try:
            # Format input text (same as MemVerse's build_memory.py)
            input_parts = [f"Query: {entry['query']}"]
            if entry.get("imagecaption"):
                input_parts.append(f"Image: {entry['imagecaption']}")
            input_text = "\n".join(input_parts)

            # Step 1: Classify into 3 memory types via GPT-4o-mini (parallel)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {mt: pool.submit(classify_memory, client, input_text, mt)
                           for mt in ["core", "episodic", "semantic"]}
                classified = {mt: fut.result() for mt, fut in futures.items()}
            api_calls += 3

            # Save classifications to JSONL
            for mem_type, output_text in classified.items():
                mem_entry = {
                    "id": entry["id"],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "input_text": input_text,
                    "output_text": output_text,
                }
                with open(chunk_dir / f"{mem_type}_memory.jsonl", "a") as f:
                    f.write(json.dumps(mem_entry, ensure_ascii=False) + "\n")

            # Step 2: Insert into LightRAG KG (parallel across 3 types)
            insert_tasks = []
            for mem_type, output_text in classified.items():
                if output_text.strip():
                    insert_tasks.append(rags[mem_type].ainsert(output_text))
            if insert_tasks:
                await asyncio.gather(*insert_tasks)
                api_calls += len(insert_tasks)

            done_ids.add(entry["id"])

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"\n  Error on entry {entry['id']}: {e}")
            if errors == 10:
                print("  (suppressing further error messages)")

        # Checkpoint every 50 entries
        if (len(done_ids) % 50 == 0) and len(done_ids) > 0:
            json.dump(list(done_ids), open(ckpt_path, "w"))
            elapsed = time.time() - start
            rate = len(done_ids) / elapsed if elapsed > 0 else 0
            remaining = (len(entries) - len(done_ids)) / rate if rate > 0 else 0
            print(f"\n  Progress: {len(done_ids)}/{len(entries)} "
                  f"({rate:.2f} entries/s, ~{remaining/60:.0f}min remaining, "
                  f"{api_calls} API calls, {errors} errors)")

    json.dump(list(done_ids), open(ckpt_path, "w"))
    elapsed = time.time() - start
    print(f"\n  Done: {len(done_ids)}/{len(entries)} entries in {elapsed/60:.1f}min "
          f"({errors} errors, {api_calls} API calls)")


def populate_benchmark(benchmark, max_entries=None):
    """Build MemVerse tripartite KG memory for a benchmark."""
    asyncio.run(_populate_async(benchmark, max_entries))


# ============================================================
# Phase 2: Evaluate with MemVerse retrieval + VLM
# ============================================================
def load_vlm():
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading VLM...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)
    return model, proc


def _clean_rag_context(raw, max_chunks=2, max_chars=200):
    """Extract clean text from LightRAG only_need_context JSON output.
    Returns short, concise text to avoid making VLM verbose."""
    import json as _json
    try:
        start = raw.find('[')
        end = raw.rfind(']') + 1
        if start >= 0 and end > start:
            chunks = _json.loads(raw[start:end])
            texts = []
            for c in chunks[:max_chunks]:
                content = c.get("content", "")
                # Strip memory type prefixes
                for prefix in ["Core Memory:\n", "Episodic Memory:\n", "Semantic Memory:\n"]:
                    content = content.replace(prefix, "")
                # Strip "Line N: " prefixes, join into single line
                lines = []
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("Line ") and ": " in line:
                        line = line.split(": ", 1)[1]
                    if line:
                        lines.append(line)
                if lines:
                    texts.append(" ".join(lines))
            return "; ".join(texts)[:max_chars]
    except Exception:
        pass
    return ""


async def query_rag(rags, question, mode="naive"):
    """Query all 3 RAG instances and combine results.
    Uses naive mode + only_need_context to skip LLM calls (much faster)."""
    from lightrag import QueryParam
    results = []
    for mem_type in ["core", "episodic", "semantic"]:
        try:
            result = await rags[mem_type].aquery(
                question, param=QueryParam(mode=mode, only_need_context=True))
            if result and result.strip():
                cleaned = _clean_rag_context(result)
                if cleaned:
                    results.append(cleaned)
        except Exception as e:
            pass  # Skip failed retrievals
    return "\n".join(results) if results else ""


def query_rag_sync(rags, question, mode="naive"):
    """Sync wrapper for query_rag using current event loop."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(query_rag(rags, question, mode))


def eval_okvqa(rags):
    import torch
    from PIL import Image
    print(f"\n{'='*60}\n  Eval OK-VQA with MemVerse baseline\n{'='*60}")

    with open("/home/cyf/codex/okvqa_data/okvqa_cached.pkl", "rb") as f:
        data = pickle.load(f)
    test = data["test"]
    print(f"  Test set: {len(test)} samples")

    vlm, proc = load_vlm()
    OUT = OUT_ROOT / "okvqa"
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "eval_checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    for i in tqdm(range(len(test)), desc="OK-VQA"):
        if str(i) in results:
            continue
        s = test[i]
        q = s["question"]
        answers = s.get("answers", [])

        # Retrieve from MemVerse KG
        retrieved = query_rag_sync(rags, q)

        content = []
        if retrieved:
            content.append({"type": "text", "text":
                f"Reference memories from similar questions:\n{retrieved}\n\n---\n"})

        img_path = s.get("image_path", "")
        test_img = None
        if img_path and os.path.exists(img_path):
            test_img = Image.open(img_path).convert("RGB")
            content.append({"type": "image", "image": test_img})

        content.append({"type": "text", "text":
            f"Question: {q}\nGive a short answer (1-3 words):"})

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
        results[str(i)] = {"gt": answers, "pred": pred, "correct": correct}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    acc = np.mean([v["correct"] for v in results.values()]) * 100
    summary = {"n": len(results), "accuracy": acc,
               "config": {"method": "MemVerse (GPT-4o-mini KG + Qwen2.5-VL-7B)"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  OK-VQA MemVerse: {acc:.2f}%")
    del vlm, proc; torch.cuda.empty_cache()
    return summary


def eval_mmqa(rags):
    import torch
    from PIL import Image
    print(f"\n{'='*60}\n  Eval MMQA with MemVerse baseline\n{'='*60}")

    with open("/home/cyf/codex/mmqa_data/mmqa_parsed.pkl", "rb") as f:
        parsed = pickle.load(f)
    dev_data = list(parsed["dev"].values()) if isinstance(parsed["dev"], dict) else parsed["dev"]
    tables = parsed.get("tables", {})
    texts = parsed.get("texts", {})
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
    ckpt = OUT / "eval_checkpoint.json"
    results = json.load(open(ckpt)) if ckpt.exists() else {}

    for i in tqdm(range(len(dev_data)), desc="MMQA"):
        if str(i) in results:
            continue
        sample = dev_data[i]
        q = sample["question"]
        gold_answers = [str(a["answer"]) for a in sample.get("answers", [])]

        # Retrieve from MemVerse KG
        retrieved = query_rag_sync(rags, q)

        # Build context from supporting docs
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

        content = []
        if retrieved:
            content.append({"type": "text", "text":
                f"Reference memories from similar questions:\n{retrieved}\n\n---\n"})

        for img in ctx_images:
            content.append({"type": "image", "image": img})

        prompt_parts = []
        if ctx_text:
            prompt_parts.append(f"Context:\n{ctx_text}\n")
        prompt_parts.append(f"Question: {q}\nGive a short answer (1-3 words):")
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

        best_em = max(compute_em(raw, ga) for ga in gold_answers) if gold_answers else 0.0
        best_f1 = max(compute_f1(raw, ga) for ga in gold_answers) if gold_answers else 0.0
        results[str(i)] = {"gt": gold_answers, "pred": raw, "em": best_em, "f1": best_f1}
        if len(results) % 100 == 0:
            json.dump(results, open(ckpt, "w"))

    json.dump(results, open(ckpt, "w"))
    em = np.mean([v["em"] for v in results.values()]) * 100
    f1 = np.mean([v["f1"] for v in results.values()]) * 100
    summary = {"n": len(results), "em": em, "f1": f1,
               "config": {"method": "MemVerse (GPT-4o-mini KG + Qwen2.5-VL-7B)"}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n  MMQA MemVerse: EM={em:.2f}, F1={f1:.2f}")
    del vlm, proc; torch.cuda.empty_cache()
    return summary


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemVerse baseline evaluation")
    parser.add_argument("--phase", choices=["populate", "eval"], required=True)
    parser.add_argument("--benchmark", choices=["okvqa", "mmqa"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-entries", type=int, default=None,
                        help="Limit entries for testing (e.g., 5)")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    benchmarks = ["okvqa", "mmqa"] if args.all else [args.benchmark]
    if not args.all and not args.benchmark:
        parser.error("Specify --benchmark or --all")

    if args.phase == "populate":
        for bm in benchmarks:
            populate_benchmark(bm, max_entries=args.max_entries)

    elif args.phase == "eval":
        setup_lightrag_imports()
        # Create a persistent event loop for LightRAG async ops
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        all_results = {}
        for bm in benchmarks:
            rags = loop.run_until_complete(init_three_rags(bm))
            fn = {"okvqa": eval_okvqa, "mmqa": eval_mmqa}[bm]
            s = fn(rags)
            all_results[bm] = s

        if len(all_results) > 1:
            json.dump(all_results, open(OUT_ROOT / "all_results.json", "w"), indent=2)
            print(f"\n{'='*60}\n  ALL RESULTS (MemVerse baseline)\n{'='*60}")
            for bm, s in all_results.items():
                if "accuracy" in s:
                    print(f"  {bm}: {s['accuracy']:.2f}%")
                else:
                    print(f"  {bm}: EM={s['em']:.2f}, F1={s['f1']:.2f}")
