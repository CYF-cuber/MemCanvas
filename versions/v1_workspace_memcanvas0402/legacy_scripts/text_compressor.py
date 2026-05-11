#!/usr/bin/env python3
"""
Compress HotpotQA text passages using Qwen2.5-VL-3B.
Two compression levels: light (key sentences) and heavy (bullet summary).
"""
import json, os, pickle, sys, time
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm

MODEL_PATH = "/home/cyf/Qwen2.5-VL-3B-Instruct"
DATA_PATH = "/home/cyf/codex/hotpotqa_data/hotpotqa_meta.pkl"
OUTPUT_DIR = Path("/home/cyf/memcanvas0402")
DEFAULT_GPU = "cuda:0"

PROMPTS = {
    "light": (
        "Extract only the key factual sentences from this passage. "
        "Remove filler, background info, and keep only essential facts. "
        "Be concise. Output key sentences directly.\n\n"
        "Passage:\n{text}\n\nKey sentences:"
    ),
    "heavy": (
        "Compress this passage into the shortest possible form. "
        "Use entity: fact format. Maximum 2 lines. Example: 'Paris: capital of France, pop 2.1M'\n\n"
        "Passage:\n{text}\n\nCompressed:"
    ),
}


def load_model(device="cuda:0"):
    print(f"Loading Qwen2.5-VL-3B on {device}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=device,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    print("Model loaded.")
    return model, processor


def compress_text(model, processor, text, level, max_new_tokens=256):
    prompt = PROMPTS[level].format(text=text[:800])  # cap input to avoid OOM
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    txt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[txt], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    result = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return result.strip()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True, choices=["light", "heavy"])
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    args = parser.parse_args()

    level = args.level
    out_dir = OUTPUT_DIR / f"hotpotqa_{level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / "compressed_texts.pkl"

    # Load data
    print(f"Loading HotpotQA data...")
    meta = pickle.load(open(DATA_PATH, "rb"))
    train = meta["train"][:args.max_samples]
    print(f"  {len(train)} samples")

    # Load checkpoint if exists
    if cache_file.exists():
        compressed = pickle.load(open(cache_file, "rb"))
        print(f"  Resuming from checkpoint: {len(compressed)} samples done")
    else:
        compressed = {}

    # Load model
    device = f"cuda:{args.gpu}"
    model, processor = load_model(device)

    # Compress supporting paragraphs
    stats = {"input_chars": [], "output_chars": []}
    t0 = time.time()
    for i in tqdm(range(len(train)), desc=f"Compress ({level})"):
        if i in compressed:
            continue
        s = train[i]
        sf_titles = set(t for t, _ in s.get("supporting_facts", []))

        paras_compressed = {}
        for p in s["paragraphs"]:
            title = p["title"]
            text = p["text"]
            if title in sf_titles and len(text) > 50:
                c = compress_text(model, processor, text, level)
                paras_compressed[title] = c
                stats["input_chars"].append(len(text))
                stats["output_chars"].append(len(c))
            else:
                # Keep short texts and distractors as-is
                paras_compressed[title] = text

        compressed[i] = {
            "id": s["id"],
            "question": s["question"],
            "answer": s["answer"],
            "type": s.get("type", ""),
            "level": s.get("level", ""),
            "supporting_facts": s.get("supporting_facts", []),
            "paragraphs": [
                {"title": p["title"], "text": paras_compressed[p["title"]]}
                for p in s["paragraphs"]
            ],
        }

        # Save checkpoint every 500 samples
        if (i + 1) % 500 == 0:
            pickle.dump(compressed, open(cache_file, "wb"))
            elapsed = time.time() - t0
            avg_in = sum(stats["input_chars"]) / max(len(stats["input_chars"]), 1)
            avg_out = sum(stats["output_chars"]) / max(len(stats["output_chars"]), 1)
            ratio = avg_out / max(avg_in, 1)
            print(f"  [{i+1}/{len(train)}] {elapsed:.0f}s | "
                  f"avg input={avg_in:.0f} → output={avg_out:.0f} chars ({ratio:.1%})")

    # Final save
    pickle.dump(compressed, open(cache_file, "wb"))

    # Print stats
    if stats["input_chars"]:
        import numpy as np
        avg_in = np.mean(stats["input_chars"])
        avg_out = np.mean(stats["output_chars"])
        print(f"\n=== Compression Stats ({level}) ===")
        print(f"  Samples: {len(compressed)}")
        print(f"  Compressed passages: {len(stats['input_chars'])}")
        print(f"  Avg input: {avg_in:.0f} chars")
        print(f"  Avg output: {avg_out:.0f} chars")
        print(f"  Compression ratio: {avg_out/avg_in:.1%}")
        print(f"  Saved: {cache_file}")


if __name__ == "__main__":
    main()
