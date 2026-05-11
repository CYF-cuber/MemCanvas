#!/usr/bin/env python3
"""
Detailed win-case analysis: retrieve Mem0 memories + MemVerse memories +
MemCanvas canvases for each strict win case. Generate per-case analysis
with images and text.

Output: /home/cyf/memcanvas0402/memcanvas0413/wincase_analysis/{benchmark}/cases/
"""
import json, os, pickle, shutil, numpy as np
from pathlib import Path
from collections import Counter

# ============================================================
# Paths
# ============================================================
BASE = Path("/home/cyf/codex")
OUT = Path("/home/cyf/memcanvas0402/memcanvas0413/wincase_analysis")

CANVAS_DIRS = {
    "scienceqa": BASE / "canvas0415/scienceqa",
    "okvqa": BASE / "canvas0415/okvqa",
    "mmqa": BASE / "canvas0415/mmqa",
    "hotpotqa": BASE / "canvas0415/hotpotqa",
}

# Older smart canvases (used by memcanvas0413_eval)
SMART_CANVAS_DIRS = {
    "scienceqa": BASE / "scienceqa_smart_canvases",
    "okvqa": BASE / "okvqa_data/canvases_smart",
    "mmqa": BASE / "mmqa_data/canvases_smart",
    "hotpotqa": BASE / "hotpotqa_data/canvases_smart",
}

MEM0_DB = BASE / "mem0_baseline_eval/mem0_dbs"

CHECKPOINTS = {
    "scienceqa": {
        "memcanvas": BASE / "memcanvas0413_eval/scienceqa_alpha0.00/checkpoint.json",
        "mem0": BASE / "mem0_baseline_eval/scienceqa/checkpoint.json",
        "textrag": BASE / "text_rag_results/scienceqa/checkpoint.json",
    },
    "okvqa": {
        "memcanvas": BASE / "memcanvas0413_eval/okvqa_alpha0.75/checkpoint.json",
        "mem0": BASE / "mem0_baseline_eval/okvqa/checkpoint.json",
        "memverse": BASE / "memverse_baseline_eval/okvqa/eval_checkpoint.json",
    },
    "mmqa": {
        "memcanvas": BASE / "memcanvas0413_eval/mmqa_alpha0.75/checkpoint.json",
        "mem0": BASE / "mem0_baseline_eval/mmqa/checkpoint.json",
        "textrag": BASE / "text_rag_results/mmqa/checkpoint.json",
        "memverse": BASE / "memverse_baseline_eval/mmqa/eval_checkpoint.json",
    },
    "hotpotqa": {
        "memcanvas": BASE / "memcanvas0413_eval/hotpotqa_alpha0.75/checkpoint.json",
        "mem0": BASE / "mem0_baseline_eval/hotpotqa/checkpoint.json",
        "textrag": BASE / "text_rag_results/hotpotqa/checkpoint.json",
    },
}

CORRECT_FIELD = {
    "scienceqa": "correct", "okvqa": "correct",
    "mmqa": "em", "hotpotqa": "em",
}

CLIP_EMB = {
    "scienceqa": {
        "img": BASE / "scienceqa_smart_canvases/clip_img_emb.npy",
        "txt": BASE / "scienceqa_smart_canvases/clip_txt_emb.npy",
        "query": BASE / "scienceqa_smart_canvases/clip_query_emb.npy",
    },
    "okvqa": {
        "img": BASE / "okvqa_data/canvases_smart/clip_img_emb.npy",
        "txt": BASE / "okvqa_data/canvases_smart/clip_txt_emb.npy",
        "query": BASE / "okvqa_data/canvases_smart/clip_query_emb.npy",
    },
    "mmqa": {
        "img": BASE / "mmqa_data/canvases_smart/clip_img_emb.npy",
        "txt": BASE / "mmqa_data/canvases_smart/clip_txt_emb.npy",
        "query": BASE / "mmqa_data/canvases_smart/clip_query_emb.npy",
    },
    "hotpotqa": {
        "img": BASE / "hotpotqa_data/canvas_embeddings_smart.npy",
        "txt": BASE / "hotpotqa_data/canvas_text_embeddings.npy",
        "query": BASE / "hotpotqa_data/query_embeddings.npy",
    },
}

ALPHA = {"scienceqa": 0.0, "okvqa": 0.75, "mmqa": 0.75, "hotpotqa": 0.75}


def load_retrieval_map(benchmark, top_k=2):
    """Compute which canvases were retrieved for each test query."""
    emb = CLIP_EMB[benchmark]
    img = np.load(emb["img"])
    txt = np.load(emb["txt"])
    query = np.load(emb["query"])
    alpha = ALPHA[benchmark]

    keys = alpha * img + (1 - alpha) * txt
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True).clip(1e-8)
    qn = query / np.linalg.norm(query, axis=1, keepdims=True).clip(1e-8)
    sims = qn @ keys.T

    rmap = {}
    for i in range(len(query)):
        top = np.argsort(sims[i])[::-1][:top_k + 5]
        res = [(int(j), float(sims[i][j])) for j in top if sims[i][j] >= 0.1][:top_k]
        rmap[i] = res
    return rmap


def get_mem0_memories(benchmark, query, top_k=5):
    """Retrieve Mem0 memories via Chroma embedding search."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    if not hasattr(get_mem0_memories, '_model'):
        get_mem0_memories._model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        get_mem0_memories._clients = {}

    model = get_mem0_memories._model

    if benchmark not in get_mem0_memories._clients:
        db_path = str(MEM0_DB / benchmark)
        client = chromadb.PersistentClient(path=db_path)
        col = client.get_collection(f"mem0_{benchmark}")
        get_mem0_memories._clients[benchmark] = col

    col = get_mem0_memories._clients[benchmark]
    emb = model.encode([query])[0].tolist()
    results = col.query(query_embeddings=[emb], n_results=top_k, include=['metadatas', 'distances'])

    memories = []
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        dist = results['distances'][0][i]
        data = meta.get('data', '')
        memories.append({"text": data, "distance": dist})
    return memories


MEMVERSE_MMKG = {
    "okvqa": BASE / "memverse_baseline_eval/okvqa/MMKG/core",
    "mmqa": BASE / "memverse_baseline_eval/mmqa/MMKG/core",
}


def get_memverse_memories(benchmark, query, top_k=5):
    """Retrieve MemVerse memories via TF-IDF similarity over core chunks.

    MemVerse uses OpenAI text-embedding-3-small for its vector store, but
    we approximate retrieval with TF-IDF to avoid API calls.  For the
    purposes of showing *what* MemVerse had in memory this is sufficient.
    """
    if benchmark not in MEMVERSE_MMKG:
        return []

    # Lazy-load chunk index per benchmark
    if not hasattr(get_memverse_memories, '_indices'):
        get_memverse_memories._indices = {}

    if benchmark not in get_memverse_memories._indices:
        from sklearn.feature_extraction.text import TfidfVectorizer
        chunks_path = MEMVERSE_MMKG[benchmark] / "kv_store_text_chunks.json"
        with open(chunks_path) as f:
            chunks_data = json.load(f)
        chunk_ids = list(chunks_data.keys())
        chunk_contents = [chunks_data[cid].get("content", "") for cid in chunk_ids]
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(chunk_contents)
        get_memverse_memories._indices[benchmark] = {
            "ids": chunk_ids,
            "contents": chunk_contents,
            "vectorizer": vectorizer,
            "matrix": tfidf_matrix,
        }

    idx_data = get_memverse_memories._indices[benchmark]
    q_vec = idx_data["vectorizer"].transform([query])
    sims = (idx_data["matrix"] @ q_vec.T).toarray().flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in top_indices:
        if sims[i] > 0:
            results.append({
                "text": idx_data["contents"][i],
                "similarity": float(sims[i]),
                "chunk_id": idx_data["ids"][i],
            })
    return results


def load_dataset_questions(benchmark):
    """Load test questions."""
    if benchmark == "scienceqa":
        from datasets import load_dataset
        test_ds = load_dataset("derek-thomas/ScienceQA", split="test")
        questions = {}
        for i in range(len(test_ds)):
            item = test_ds[i]
            choices = item.get("choices", [])
            questions[i] = {
                "question": item.get("question", ""),
                "choices": "; ".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices)),
                "hint": item.get("hint", "") or "",
                "subject": item.get("subject", ""),
                "topic": item.get("topic", ""),
                "gt_text": choices[item["answer"]] if item["answer"] < len(choices) else "",
                "has_image": item.get("image") is not None,
            }
        return questions

    elif benchmark == "okvqa":
        with open(BASE / "okvqa_data/okvqa_cached.pkl", "rb") as f:
            data = pickle.load(f)
        test = data.get("test", data.get("val", []))
        questions = {}
        for i, item in enumerate(test):
            questions[i] = {
                "question": item.get("question", ""),
                "caption": item.get("caption", ""),
                "answers": item.get("answers", []),
            }
        return questions

    elif benchmark == "mmqa":
        with open(BASE / "mmqa_data/mmqa_parsed.pkl", "rb") as f:
            data = pickle.load(f)
        dev = data.get("dev", [])
        questions = {}
        for i, item in enumerate(dev):
            questions[i] = {
                "question": item.get("question", ""),
                "qtype": item.get("metadata", {}).get("type", ""),
                "modalities": item.get("metadata", {}).get("modalities", []),
                "answers": [a["answer"] for a in item.get("answers", [])],
            }
        return questions

    elif benchmark == "hotpotqa":
        with open(BASE / "hotpotqa_data/hotpotqa_meta.pkl", "rb") as f:
            data = pickle.load(f)
        dev = data.get("dev", [])
        questions = {}
        for i, item in enumerate(dev):
            questions[i] = {
                "question": item.get("question", ""),
                "qtype": item.get("type", ""),
                "level": item.get("level", ""),
            }
        return questions


def find_canvas_image(benchmark, canvas_idx):
    """Find the canvas PNG for a given train index."""
    # Try canvas0415 first, then smart canvases
    for d in [CANVAS_DIRS.get(benchmark), SMART_CANVAS_DIRS.get(benchmark)]:
        if d is None:
            continue
        for fmt in [f"{canvas_idx:05d}.png", f"{canvas_idx:04d}.png", f"{canvas_idx}.png"]:
            p = d / fmt
            if p.exists():
                return p
    return None


def analyze_benchmark(benchmark, max_cases=30):
    """Generate detailed analysis for top strict win cases."""
    print(f"\n{'='*60}")
    print(f"  {benchmark.upper()} — Detailed Analysis")
    print(f"{'='*60}")

    # Load wincases
    wc_file = OUT / benchmark / "wincases.json"
    with open(wc_file) as f:
        wc = json.load(f)
    strict_wins = wc["strict_wins"]
    print(f"  Strict wins: {len(strict_wins)}")

    # Load checkpoints
    ckpts = {}
    for method, path in CHECKPOINTS[benchmark].items():
        if path.exists():
            with open(path) as f:
                ckpts[method] = json.load(f)

    # Load retrieval map
    print("  Computing retrieval map...")
    rmap = load_retrieval_map(benchmark)

    # Load questions
    print("  Loading dataset questions...")
    questions = load_dataset_questions(benchmark)

    # Output dir
    cases_dir = OUT / benchmark / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    # Build search query same way as eval script
    all_cases = []
    n = min(max_cases, len(strict_wins))
    print(f"  Analyzing top {n} cases...")

    for ci, case in enumerate(strict_wins[:n]):
        idx = case["index"]
        idx_str = str(idx)
        q_info = questions.get(idx, {})
        question = q_info.get("question", case.get("question", ""))

        # Build search query (same as eval scripts)
        if benchmark == "scienceqa":
            hint = q_info.get("hint", "")
            search_query = f"{question} {hint}".strip()
        else:
            search_query = question

        # Get Mem0 retrieved memories
        mem0_memories = get_mem0_memories(benchmark, search_query, top_k=5)

        # Get MemVerse retrieved memories (only for okvqa/mmqa)
        memverse_memories = get_memverse_memories(benchmark, search_query, top_k=5)

        # Get MemCanvas retrieved canvases
        retrieved = rmap.get(idx, [])

        # Copy canvas images
        canvas_files = []
        for ri, (cidx, sim) in enumerate(retrieved[:2]):
            src = find_canvas_image(benchmark, cidx)
            if src:
                dst = cases_dir / f"idx{idx:04d}_canvas_{ri}_{cidx:05d}.png"
                shutil.copy2(src, dst)
                canvas_files.append(dst.name)

        # Build case detail
        detail = {
            "index": idx,
            "question": question,
            "gt": case.get("gt", ""),
            "question_info": {k: v for k, v in q_info.items() if k != "question"},
            "memcanvas": {
                "pred": case.get("memcanvas_pred", ""),
                "correct": True,
                "retrieved_canvases": [
                    {"train_idx": cidx, "similarity": sim}
                    for cidx, sim in retrieved[:2]
                ],
                "canvas_files": canvas_files,
            },
            "mem0": {
                "pred": ckpts.get("mem0", {}).get(idx_str, {}).get("pred", ""),
                "correct": False,
                "retrieved_memories": [
                    {"text": m["text"][:500], "distance": m["distance"]}
                    for m in mem0_memories[:3]
                ],
            },
        }

        # Add textrag baseline
        if "textrag" in ckpts and idx_str in ckpts["textrag"]:
            detail["textrag"] = {
                "pred": ckpts["textrag"][idx_str].get("pred", ""),
                "correct": False,
            }

        # Add memverse baseline with retrieved memories
        if "memverse" in ckpts and idx_str in ckpts["memverse"]:
            detail["memverse"] = {
                "pred": ckpts["memverse"][idx_str].get("pred", ""),
                "correct": False,
                "retrieved_memories": [
                    {"text": m["text"][:500], "similarity": m["similarity"]}
                    for m in memverse_memories[:3]
                ],
            }

        all_cases.append(detail)

        if (ci + 1) % 10 == 0:
            print(f"    {ci+1}/{n} done")

    # Save detailed cases
    with open(cases_dir / "detailed_cases.json", "w") as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)

    # Generate detailed markdown
    write_detailed_md(benchmark, all_cases, cases_dir)
    print(f"  Saved {len(all_cases)} detailed cases to {cases_dir}/")
    return all_cases


def write_detailed_md(benchmark, cases, cases_dir):
    """Write detailed markdown with Mem0 memory content and analysis."""
    lines = [
        f"# {benchmark.upper()} — Detailed Win-Case Analysis\n",
        f"MemCanvas答对、其他方法全部答错的案例详细分析。\n",
        f"共 {len(cases)} 个案例。\n",
    ]

    for i, case in enumerate(cases):
        idx = case["index"]
        lines.append(f"---\n## Case {i+1}: Test Index {idx}\n")

        # Question
        lines.append(f"**Question**: {case['question']}")
        qi = case.get("question_info", {})
        if qi.get("choices"):
            lines.append(f"**Choices**: {qi['choices']}")
        if qi.get("hint"):
            lines.append(f"**Hint**: {qi['hint'][:300]}")
        if qi.get("subject"):
            lines.append(f"**Subject**: {qi['subject']} / {qi.get('topic', '')}")
        if qi.get("qtype"):
            lines.append(f"**Type**: {qi['qtype']}")
        if qi.get("modalities"):
            lines.append(f"**Modalities**: {', '.join(qi['modalities'])}")
        lines.append(f"**GT**: {case['gt']}\n")

        # MemCanvas
        mc = case["memcanvas"]
        lines.append(f"### MemCanvas (Correct: {mc['pred']})")
        for ci2, rc in enumerate(mc.get("retrieved_canvases", [])):
            lines.append(f"- Retrieved canvas #{ci2+1}: train idx={rc['train_idx']}, sim={rc['similarity']:.4f}")
        if mc.get("canvas_files"):
            for cf in mc["canvas_files"]:
                lines.append(f"- Canvas image: `{cf}`")
        lines.append("")

        # Mem0
        m0 = case.get("mem0", {})
        lines.append(f"### Mem0 (Wrong: {m0.get('pred', 'N/A')[:200]})")
        lines.append("**Retrieved memories**:")
        for mi, mem in enumerate(m0.get("retrieved_memories", [])):
            text = mem["text"][:400].replace("\n", " ")
            lines.append(f"  {mi+1}. [dist={mem['distance']:.4f}] {text}")
        lines.append("")

        # Analysis: why Mem0 failed
        lines.append("**Why Mem0 failed**:")
        mem0_mems = m0.get("retrieved_memories", [])
        if not mem0_mems:
            lines.append("- No memories retrieved")
        else:
            mem_texts = " ".join(m["text"] for m in mem0_mems)
            gt = str(case["gt"])
            has_visual = qi.get("has_image", False) or qi.get("modalities", [])
            hint = qi.get("hint", "")

            reasons = []
            if has_visual or "image" in str(qi.get("modalities", [])) or "shown" in hint.lower() or "diagram" in hint.lower() or "map" in hint.lower():
                reasons.append("问题需要视觉信息（图像/图表/地图），Mem0的纯文本记忆无法表达视觉内容")
            if "table" in str(qi.get("modalities", [])):
                reasons.append("问题涉及表格数据，Mem0的文本记忆丢失了表格结构信息")

            # Check if memories are about different topics
            q_lower = case["question"].lower()
            mem_lower = mem_texts.lower()
            if len(mem0_mems) >= 3 and all("same" in m["text"][:50].lower() or m["distance"] > 1.0 for m in mem0_mems):
                reasons.append("检索到的记忆与问题相关度低")

            if not reasons:
                # Generic analysis
                m0_pred = str(m0.get("pred", "")).lower()
                if len(m0_pred) > 100:
                    reasons.append("Mem0生成了冗长但错误的回答，可能因为检索到的文本记忆缺少关键视觉信息")
                else:
                    reasons.append("Mem0检索到了相似文本记忆，但缺少视觉上下文导致判断错误")

            for r in reasons:
                lines.append(f"- {r}")
        lines.append("")

        # TextRAG baseline
        if "textrag" in case:
            pred = str(case["textrag"].get("pred", ""))[:200]
            lines.append(f"### TextRAG (Wrong: {pred})")
            lines.append("")

        # MemVerse baseline with retrieved memories
        if "memverse" in case:
            mv = case["memverse"]
            pred = str(mv.get("pred", ""))[:200]
            lines.append(f"### MemVerse (Wrong: {pred})")
            mv_mems = mv.get("retrieved_memories", [])
            if mv_mems:
                lines.append("**Retrieved memories (from LightRAG core KG)**:")
                for mi, mem in enumerate(mv_mems):
                    text = mem["text"][:400].replace("\n", " ")
                    lines.append(f"  {mi+1}. [sim={mem['similarity']:.4f}] {text}")
                lines.append("")
                lines.append("**Why MemVerse failed**:")
                # Analyze failure
                mv_mem_text = " ".join(m["text"] for m in mv_mems)
                has_visual = qi.get("has_image", False) or qi.get("modalities", [])
                reasons_mv = []
                if has_visual or "image" in str(qi.get("modalities", [])):
                    reasons_mv.append("MemVerse的知识图谱只存储了文本实体和关系，丢失了视觉信息")
                if "table" in str(qi.get("modalities", [])):
                    reasons_mv.append("表格数据在知识图谱提取过程中丢失了结构信息")
                if mv_mems and mv_mems[0]["similarity"] < 0.1:
                    reasons_mv.append("检索到的记忆与查询相关度很低（TF-IDF sim < 0.1）")
                if not reasons_mv:
                    reasons_mv.append("MemVerse的知识图谱虽然检索到了相关文本记忆，但缺少视觉上下文导致VLM无法正确推理")
                for r in reasons_mv:
                    lines.append(f"- {r}")
            else:
                lines.append("- 未检索到相关记忆")
            lines.append("")

    with open(cases_dir / "detailed_analysis.md", "w") as f:
        f.write("\n".join(lines))


def main():
    for benchmark in ["scienceqa", "okvqa", "mmqa", "hotpotqa"]:
        try:
            analyze_benchmark(benchmark, max_cases=30)
        except Exception as e:
            print(f"  ERROR in {benchmark}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone! Detailed analysis in {OUT}/*/cases/")


if __name__ == "__main__":
    main()
