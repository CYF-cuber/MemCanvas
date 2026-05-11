#!/usr/bin/env python3
"""
Streaming memory simulation — no GPU needed.
Simulates memories arriving in 10 sequential batches, measuring memory bank size
and retrieval hit rate under different forgetting strategies.

Usage:
  python run_streaming_simulation.py
"""
import json, math, os
from collections import Counter
from pathlib import Path
import numpy as np

CANVAS_DIR = Path("/home/cyf/codex/scienceqa_smart_canvases")
OUTPUT_DIR = Path("/home/cyf/memcanvas0402/forgetting_experiments")
DEFAULT_ALPHA = 0.00
DEFAULT_TOP_K = 2
N_BATCHES = 10
FREQ_ADAPTIVE_T = 1000
FREQ_ADAPTIVE_S = 1
FIFO_CAPACITY = 3374


def build_retrieval_map(img_emb, txt_emb, query_emb, alpha=DEFAULT_ALPHA, top_k=DEFAULT_TOP_K):
    keys = alpha * img_emb + (1 - alpha) * txt_emb
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True).clip(1e-8)
    qn = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True).clip(1e-8)
    sims = qn @ keys.T
    rmap = {}
    for i in range(len(query_emb)):
        top = np.argsort(sims[i])[::-1][:top_k + 5]
        res = [(int(j), float(sims[i][j])) for j in top if sims[i][j] >= 0.1][:top_k]
        rmap[i] = res
    return rmap


def compute_hit_rate(rmap, n_test, quality, available_set, top_k=DEFAULT_TOP_K):
    """Compute retrieval hit rate: fraction of top-k retrieved memories that are alive."""
    hits = 0
    total = 0
    for qi in range(n_test):
        for mem_idx, sim in rmap.get(qi, [])[:top_k]:
            if mem_idx in available_set and quality[mem_idx] < 4:
                hits += 1
            total += 1
    return hits / total * 100 if total else 0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CLIP embeddings...")
    img_emb = np.load(CANVAS_DIR / "clip_img_emb.npy")
    txt_emb = np.load(CANVAS_DIR / "clip_txt_emb.npy")
    query_emb = np.load(CANVAS_DIR / "clip_query_emb.npy")
    n_memories = len(img_emb)
    n_test = len(query_emb)
    rmap = build_retrieval_map(img_emb, txt_emb, query_emb)

    print(f"  {n_memories} memories, {n_test} test queries, {N_BATCHES} batches")

    # Split memories into N_BATCHES sequential batches
    batch_size = n_memories // N_BATCHES
    batches = []
    for b in range(N_BATCHES):
        start = b * batch_size
        end = start + batch_size if b < N_BATCHES - 1 else n_memories
        batches.append(list(range(start, end)))
    print(f"  Batch sizes: {[len(b) for b in batches]}")

    results = {}

    # ============================================================
    # Strategy 1: No Forgetting (linear growth)
    # ============================================================
    print("\n--- No Forgetting ---")
    no_forget = {"memory_sizes": [], "hit_rates": [], "batch_labels": []}
    available = set()
    quality = [0] * n_memories  # all alive
    for b in range(N_BATCHES):
        available.update(batches[b])
        # All memories in available are alive (quality=0)
        hr = compute_hit_rate(rmap, n_test, quality, available)
        no_forget["memory_sizes"].append(len(available))
        no_forget["hit_rates"].append(hr)
        no_forget["batch_labels"].append(f"B{b}")
        print(f"  Batch {b}: size={len(available)}, hit_rate={hr:.1f}%")
    results["no_forgetting"] = no_forget

    # ============================================================
    # Strategy 2: FIFO (fixed capacity)
    # ============================================================
    print("\n--- FIFO ---")
    fifo_res = {"memory_sizes": [], "hit_rates": [], "batch_labels": []}
    available = set()
    quality = [4] * n_memories  # all dead initially
    arrival_order = []  # track insertion order
    for b in range(N_BATCHES):
        # Add new batch
        for idx in batches[b]:
            available.add(idx)
            quality[idx] = 0
            arrival_order.append(idx)
        # Evict oldest if over capacity
        while len(available) > FIFO_CAPACITY:
            oldest = arrival_order.pop(0)
            if oldest in available:
                available.discard(oldest)
                quality[oldest] = 4
        hr = compute_hit_rate(rmap, n_test, quality, available)
        fifo_res["memory_sizes"].append(len(available))
        fifo_res["hit_rates"].append(hr)
        fifo_res["batch_labels"].append(f"B{b}")
        print(f"  Batch {b}: size={len(available)}, hit_rate={hr:.1f}%")
    results["fifo"] = fifo_res

    # ============================================================
    # Strategy 3: Ebbinghaus Time-Decay
    # ============================================================
    print("\n--- Ebbinghaus ---")
    ebb_res = {"memory_sizes": [], "hit_rates": [], "batch_labels": []}
    quality = [4] * n_memories
    strength = [1.0] * n_memories
    last_access = [0] * n_memories
    available = set()
    # Simulate queries after each batch arrival, applying decay
    queries_so_far = 0
    for b in range(N_BATCHES):
        # Add new batch
        for idx in batches[b]:
            available.add(idx)
            quality[idx] = 0
            strength[idx] = 1.0
            last_access[idx] = queries_so_far

        # Simulate a round of test queries accessing memories
        for qi in range(n_test):
            for mem_idx, sim in rmap.get(qi, []):
                if mem_idx in available and quality[mem_idx] < 4:
                    strength[mem_idx] += 1.0
                    last_access[mem_idx] = queries_so_far + qi
        queries_so_far += n_test

        # Apply Ebbinghaus decay: delete memories with low retention
        retention = {}
        for mi in available:
            elapsed = queries_so_far - last_access[mi]
            R = math.exp(-elapsed / max(strength[mi], 1e-8))
            retention[mi] = R

        # Sort by retention, keep top FIFO_CAPACITY
        if len(available) > FIFO_CAPACITY:
            sorted_mems = sorted(available, key=lambda i: retention.get(i, 0), reverse=True)
            keep_set = set(sorted_mems[:FIFO_CAPACITY])
            for mi in list(available):
                if mi not in keep_set:
                    available.discard(mi)
                    quality[mi] = 4

        hr = compute_hit_rate(rmap, n_test, quality, available)
        ebb_res["memory_sizes"].append(len(available))
        ebb_res["hit_rates"].append(hr)
        ebb_res["batch_labels"].append(f"B{b}")
        print(f"  Batch {b}: size={len(available)}, hit_rate={hr:.1f}%")
    results["ebbinghaus"] = ebb_res

    # ============================================================
    # Strategy 4: Freq-Adaptive (Ours)
    # ============================================================
    print("\n--- Freq-Adaptive (Ours) ---")
    freq_res = {"memory_sizes": [], "hit_rates": [], "batch_labels": []}
    quality = [4] * n_memories
    retrieval_count = [0] * n_memories
    available = set()
    queries_so_far = 0
    T = FREQ_ADAPTIVE_T
    S = FREQ_ADAPTIVE_S
    for b in range(N_BATCHES):
        # Add new batch
        for idx in batches[b]:
            available.add(idx)
            quality[idx] = 0

        # Simulate test queries
        for qi in range(n_test):
            global_qi = queries_so_far + qi
            for mem_idx, sim in rmap.get(qi, []):
                if mem_idx in available and quality[mem_idx] < 4:
                    retrieval_count[mem_idx] += 1

            # Periodic review
            if (global_qi + 1) % T == 0:
                for mi in list(available):
                    if quality[mi] >= 4:
                        continue
                    if retrieval_count[mi] <= S:
                        quality[mi] += 1
                        if quality[mi] >= 4:
                            available.discard(mi)
                # Counts accumulate, never reset

        queries_so_far += n_test
        surviving = sum(1 for mi in available if quality[mi] < 4)
        hr = compute_hit_rate(rmap, n_test, quality, available)
        freq_res["memory_sizes"].append(surviving)
        freq_res["hit_rates"].append(hr)
        freq_res["batch_labels"].append(f"B{b}")
        print(f"  Batch {b}: size={surviving}, hit_rate={hr:.1f}%")
    results["freq_adaptive"] = freq_res

    # ============================================================
    # Strategy 5: LRU (fixed capacity)
    # ============================================================
    print("\n--- LRU ---")
    lru_res = {"memory_sizes": [], "hit_rates": [], "batch_labels": []}
    quality = [4] * n_memories
    last_access_lru = [-1] * n_memories
    available = set()
    queries_so_far = 0
    for b in range(N_BATCHES):
        for idx in batches[b]:
            available.add(idx)
            quality[idx] = 0
            last_access_lru[idx] = queries_so_far

        # Simulate queries, update last access
        for qi in range(n_test):
            for mem_idx, sim in rmap.get(qi, []):
                if mem_idx in available and quality[mem_idx] < 4:
                    last_access_lru[mem_idx] = queries_so_far + qi
        queries_so_far += n_test

        # Evict LRU if over capacity
        if len(available) > FIFO_CAPACITY:
            sorted_mems = sorted(available, key=lambda i: last_access_lru[i], reverse=True)
            keep_set = set(sorted_mems[:FIFO_CAPACITY])
            for mi in list(available):
                if mi not in keep_set:
                    available.discard(mi)
                    quality[mi] = 4

        hr = compute_hit_rate(rmap, n_test, quality, available)
        lru_res["memory_sizes"].append(len(available))
        lru_res["hit_rates"].append(hr)
        lru_res["batch_labels"].append(f"B{b}")
        print(f"  Batch {b}: size={len(available)}, hit_rate={hr:.1f}%")
    results["lru"] = lru_res

    # Save results
    out_path = OUTPUT_DIR / "streaming_results.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
