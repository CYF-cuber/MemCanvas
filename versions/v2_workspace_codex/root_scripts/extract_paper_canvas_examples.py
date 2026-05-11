#!/usr/bin/env python3
"""
Extract representative canvas examples from each benchmark for paper figures.

Outputs to /home/cyf/codex/paper_canvas_examples/
"""

import io
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

from PIL import Image

# Need MemoryEntry class for unpickling ScienceQA memory index
sys.path.insert(0, "/home/cyf/memory/memory_canvas/experiments")
from scienceqa_qwen25vl_full_experiment import MemoryEntry, ScienceQADataLoader

OUTPUT_DIR = Path("/home/cyf/codex/paper_canvas_examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_image_bytes(data: bytes, path: Path):
    """Save image bytes to file."""
    img = Image.open(io.BytesIO(data))
    img.save(str(path))
    w, h = img.size
    print(f"  Saved: {path.name} ({w}x{h}, {len(data)/1024:.1f} KB)")


# =============================================================================
# 1. ScienceQA
# =============================================================================
def extract_scienceqa():
    print("\n=== ScienceQA ===")
    out = OUTPUT_DIR / "scienceqa"
    out.mkdir(exist_ok=True)

    pkl_path = "/home/cyf/memory/experiments/scienceqa_qwen25vl_full/memory_index_qwen25vl_full.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    memories = data["memories"]
    print(f"  Total memories: {len(memories)}")

    # Load train data to find good examples with diverse subjects
    loader = ScienceQADataLoader()
    train_data = loader.get_split("train")

    # Find examples with different subjects and that have images
    subjects_seen = set()
    selected = []

    # Priority subjects for diversity
    priority_subjects = ["natural science", "social science", "language science"]

    for mem in memories:
        pid = mem.pid
        if pid not in train_data:
            continue
        problem = train_data[pid]
        subject = problem.get("subject", "unknown")
        topic = problem.get("topic", "unknown")
        has_image = problem.get("image") is not None and problem.get("image") != ""

        if subject not in subjects_seen and mem.canvas_image_bytes and has_image:
            subjects_seen.add(subject)
            selected.append((mem, subject, topic, pid))
            if len(selected) >= 3:
                break

    # If we didn't get 3, grab any remaining
    if len(selected) < 3:
        for mem in memories[:100]:
            if mem.canvas_image_bytes and mem not in [s[0] for s in selected]:
                pid = mem.pid
                problem = train_data.get(pid, {})
                subject = problem.get("subject", "unknown")
                topic = problem.get("topic", "unknown")
                selected.append((mem, subject, topic, pid))
                if len(selected) >= 3:
                    break

    for i, (mem, subject, topic, pid) in enumerate(selected):
        fname = f"scienceqa_{i+1}_{subject.replace(' ', '_')}_{topic.replace(' ', '_')}.png"
        save_image_bytes(mem.canvas_image_bytes, out / fname)
        print(f"    Subject: {subject}, Topic: {topic}, PID: {pid}")


# =============================================================================
# 2. OK-VQA
# =============================================================================
def extract_okvqa():
    print("\n=== OK-VQA ===")
    out = OUTPUT_DIR / "okvqa"
    out.mkdir(exist_ok=True)

    pkl_path = "/home/cyf/codex/okvqa_data/canvases.pkl"
    with open(pkl_path, "rb") as f:
        canvases = pickle.load(f)
    print(f"  Total canvases: {len(canvases)}")

    # canvases is a dict: {question_id: {"canvas_bytes": ..., "question": ..., ...}}
    # or list of tuples. Let me inspect the structure first.
    if isinstance(canvases, dict):
        items = list(canvases.items())
    elif isinstance(canvases, list):
        items = [(i, c) for i, c in enumerate(canvases)]
    else:
        print(f"  Unknown canvases type: {type(canvases)}")
        return

    print(f"  Structure: {type(canvases)}, first key type: {type(items[0][0])}")

    # Inspect first item structure
    first_val = items[0][1]
    if isinstance(first_val, dict):
        print(f"  First item keys: {list(first_val.keys())[:10]}")
    elif isinstance(first_val, (bytes,)):
        print(f"  First item: raw bytes ({len(first_val)} bytes)")
    elif hasattr(first_val, '__dict__'):
        print(f"  First item attrs: {list(first_val.__dict__.keys())[:10]}")
    else:
        print(f"  First item type: {type(first_val)}")

    # Save 3 examples
    count = 0
    for key, val in items:
        if count >= 3:
            break
        # Extract bytes based on structure
        if isinstance(val, dict):
            canvas_bytes = val.get("canvas_bytes") or val.get("image_bytes") or val.get("canvas")
        elif isinstance(val, bytes):
            canvas_bytes = val
        elif hasattr(val, 'canvas_image_bytes'):
            canvas_bytes = val.canvas_image_bytes
        elif hasattr(val, 'canvas_bytes'):
            canvas_bytes = val.canvas_bytes
        else:
            # Try treating the val itself
            canvas_bytes = None

        if canvas_bytes and len(canvas_bytes) > 100:
            count += 1
            save_image_bytes(canvas_bytes, out / f"okvqa_{count}_qid{key}.png")


# =============================================================================
# 3. MSR-VTT (already PNG files)
# =============================================================================
def extract_msrvtt():
    print("\n=== MSR-VTT ===")
    out = OUTPUT_DIR / "msrvtt"
    out.mkdir(exist_ok=True)

    canvas_dir = Path("/home/cyf/codex/msrvtt_experiment_20260225_171750/canvas")
    files = sorted(canvas_dir.glob("*.png"))
    print(f"  Total canvases: {len(files)}")

    # Pick 3 diverse examples
    # Choose videos from different parts of the test set
    indices = [0, len(files)//3, 2*len(files)//3]
    for i, idx in enumerate(indices):
        src = files[idx]
        dst = out / f"msrvtt_{i+1}_{src.stem}.png"
        shutil.copy2(src, dst)
        img = Image.open(src)
        w, h = img.size
        fsize = src.stat().st_size / 1024
        print(f"  Saved: {dst.name} ({w}x{h}, {fsize:.1f} KB)")


# =============================================================================
# 4. LoCoMo (sample PNGs exist)
# =============================================================================
def extract_locomo():
    print("\n=== LoCoMo ===")
    out = OUTPUT_DIR / "locomo"
    out.mkdir(exist_ok=True)

    samples_dir = Path("/home/cyf/codex/locomo_canvas_experiment_20260225_174553/samples")
    files = sorted(samples_dir.glob("*.png"))
    print(f"  Available samples: {len(files)}")

    # Pick 3 diverse examples from different conversations
    convs_seen = set()
    selected = []
    for f in files:
        conv = f.stem.rsplit("_", 1)[0]  # e.g. "conv-30" from "conv-30_4"
        if conv not in convs_seen:
            convs_seen.add(conv)
            selected.append(f)
        if len(selected) >= 3:
            break

    # If not enough unique conversations, just take first 3
    if len(selected) < 3:
        selected = files[:3]

    for i, src in enumerate(selected):
        dst = out / f"locomo_{i+1}_{src.stem}.png"
        shutil.copy2(src, dst)
        img = Image.open(src)
        w, h = img.size
        fsize = src.stat().st_size / 1024
        print(f"  Saved: {dst.name} ({w}x{h}, {fsize:.1f} KB)")


# =============================================================================
# 5. MMQA
# =============================================================================
def extract_mmqa():
    print("\n=== MMQA ===")
    out = OUTPUT_DIR / "mmqa"
    out.mkdir(exist_ok=True)

    pkl_path = "/home/cyf/codex/mmqa_data/canvases.pkl"
    with open(pkl_path, "rb") as f:
        canvases = pickle.load(f)
    print(f"  Total canvases: {len(canvases)}")

    # Inspect structure
    if isinstance(canvases, dict):
        items = list(canvases.items())
    elif isinstance(canvases, list):
        items = [(i, c) for i, c in enumerate(canvases)]
    else:
        print(f"  Unknown type: {type(canvases)}")
        return

    first_val = items[0][1]
    if isinstance(first_val, dict):
        print(f"  Keys: {list(first_val.keys())[:10]}")
    elif isinstance(first_val, tuple):
        print(f"  Tuple len: {len(first_val)}, types: {[type(x).__name__ for x in first_val[:5]]}")
    else:
        print(f"  Type: {type(first_val)}")

    # Load MMQA data to find diverse question types
    import gzip
    mmqa_dev = []
    with gzip.open("/home/cyf/codex/mmqa_data/MMQA_dev.jsonl.gz", "rt") as f:
        for line in f:
            mmqa_dev.append(json.loads(line))

    # Build qid -> question type mapping
    qtype_map = {}
    for item in mmqa_dev:
        qid = item["qid"]
        qtype = item.get("metadata", {}).get("type", "unknown")
        qtype_map[qid] = qtype

    # We want examples with different modality types:
    # TextQ, TableQ, ImageQ, Compose(TextQ,TableQ), Compose(ImageQ,TableQ), etc.
    target_types = ["TextQ", "TableQ", "ImageQ", "Compose(TextQ,TableQ)", "Compose(ImageQ,TableQ)"]

    type_examples = {}
    for key, val in items:
        if len(type_examples) >= 3:
            break

        # Determine question ID
        if isinstance(val, dict):
            qid = val.get("qid", key)
            canvas_bytes = val.get("canvas_bytes") or val.get("image_bytes")
        elif isinstance(val, tuple):
            qid = val[0] if len(val) > 0 else key
            # Try different tuple positions for bytes
            canvas_bytes = None
            for elem in val:
                if isinstance(elem, bytes) and len(elem) > 100:
                    canvas_bytes = elem
                    break
        elif isinstance(val, bytes):
            qid = key
            canvas_bytes = val
        else:
            continue

        if not canvas_bytes or len(canvas_bytes) < 100:
            continue

        qtype = qtype_map.get(str(qid), qtype_map.get(qid, "unknown"))

        # Pick diverse types
        if qtype not in type_examples:
            type_examples[qtype] = (key, val, canvas_bytes, qtype)

    # Select up to 3 diverse examples
    selected = list(type_examples.values())[:3]

    # If not enough, just grab first 3 with valid bytes
    if len(selected) < 3:
        count = len(selected)
        for key, val in items:
            if count >= 3:
                break
            if isinstance(val, dict):
                canvas_bytes = val.get("canvas_bytes") or val.get("image_bytes")
            elif isinstance(val, bytes):
                canvas_bytes = val
            else:
                continue
            if canvas_bytes and len(canvas_bytes) > 100:
                already = False
                for s in selected:
                    if s[0] == key:
                        already = True
                        break
                if not already:
                    selected.append((key, val, canvas_bytes, "unknown"))
                    count += 1

    for i, (key, val, canvas_bytes, qtype) in enumerate(selected):
        fname = f"mmqa_{i+1}_{qtype.replace('(', '_').replace(')', '_').replace(',', '_')}.png"
        save_image_bytes(canvas_bytes, out / fname)
        print(f"    Type: {qtype}, Key: {key}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print(f"Output directory: {OUTPUT_DIR}")

    extract_scienceqa()
    extract_okvqa()
    extract_msrvtt()
    extract_locomo()
    extract_mmqa()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Paper Canvas Examples")
    print("="*60)
    for subdir in sorted(OUTPUT_DIR.iterdir()):
        if subdir.is_dir():
            files = list(subdir.glob("*.png"))
            print(f"  {subdir.name}: {len(files)} examples")
            for f in sorted(files):
                print(f"    - {f.name}")
    print(f"\nAll saved to: {OUTPUT_DIR}")
