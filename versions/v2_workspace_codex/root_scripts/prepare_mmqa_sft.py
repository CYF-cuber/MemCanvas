#!/usr/bin/env python3
"""
Prepare MMQA SFT datasets for LLaMA-Factory fine-tuning of Qwen2.5-VL-7B.

Reads the MMQA parsed pickle data and creates:
  - /home/cyf/LLaMA-Factory-main/data/mmqa_sft_train.json  (from train split)
  - /home/cyf/LLaMA-Factory-main/data/mmqa_sft_val.json    (from dev split)

Format: ShareGPT with messages + optional images, matching the existing
scienceqa/infovqa/okvqa SFT datasets already in LLaMA-Factory.
"""

import json
import os
import pickle
from collections import Counter
from pathlib import Path

# Paths
MMQA_PKL = "/home/cyf/codex/mmqa_data/mmqa_parsed.pkl"
IMAGES_DIR = "/home/cyf/codex/mmqa_data/final_dataset_images"
OUT_TRAIN = "/home/cyf/LLaMA-Factory-main/data/mmqa_sft_train.json"
OUT_VAL = "/home/cyf/LLaMA-Factory-main/data/mmqa_sft_val.json"


def table_to_markdown(table_data, max_rows=15):
    """Convert MMQA table dict to markdown string."""
    header = [h["column_name"] for h in table_data["header"]]
    rows = [[cell["text"] for cell in row] for row in table_data["table_rows"]]

    # Filter out empty column names for cleaner rendering
    non_empty_cols = [i for i, h in enumerate(header) if h.strip()]
    if non_empty_cols:
        header = [header[i] for i in non_empty_cols]
        rows = [[row[i] if i < len(row) else "" for i in non_empty_cols] for row in rows]

    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[:max_rows]:
        cells = [c[:60] for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows)} rows total, showing first {max_rows})")
    return "\n".join(lines)


def convert_sample(sample, tables, texts, images_meta, images_dir):
    """
    Convert one MMQA sample into a ShareGPT-format dict.

    Returns: dict with 'messages' and optionally 'images', or None if skipped.
    """
    question = sample["question"]
    answers = sample.get("answers", [])
    if not answers:
        return None
    # Use the first answer (primary answer)
    answer_text = str(answers[0]["answer"])
    if not answer_text.strip():
        return None

    context_parts = []
    image_paths = []

    for ctx in sample.get("supporting_context", []):
        doc_id = ctx["doc_id"]
        doc_part = ctx["doc_part"]

        if doc_part == "text" and doc_id in texts:
            text_doc = texts[doc_id]
            title = text_doc.get("title", "")
            passage = text_doc.get("text", "")
            # Truncate very long passages
            if len(passage) > 800:
                passage = passage[:800] + "..."
            if title:
                context_parts.append(f"[Text: {title}]\n{passage}")
            else:
                context_parts.append(f"[Text]\n{passage}")

        elif doc_part == "table" and doc_id in tables:
            table_doc = tables[doc_id]
            title = table_doc.get("title", "")
            table_content = table_doc.get("table", {})
            md = table_to_markdown(table_content, max_rows=15)
            if title:
                context_parts.append(f"[Table: {title}]\n{md}")
            else:
                context_parts.append(f"[Table]\n{md}")

        elif doc_part == "image" and doc_id in images_meta:
            img_info = images_meta[doc_id]
            img_path_rel = img_info.get("path", "")
            img_path_full = os.path.join(images_dir, img_path_rel)
            title = img_info.get("title", "")
            if os.path.exists(img_path_full):
                image_paths.append(img_path_full)
                if title:
                    context_parts.append(f"[Image: {title}]")
                else:
                    context_parts.append("[Image]")
            else:
                # Image file missing, include title as text context
                if title:
                    context_parts.append(f"[Image: {title}] (image not available)")

    # Build the user prompt
    prompt_lines = []

    # Add <image> tags for each image (before text content)
    if image_paths:
        for _ in image_paths:
            prompt_lines.append("<image>")

    # Add context
    if context_parts:
        prompt_lines.append("Use the following context to answer the question.\n")
        prompt_lines.append("\n\n".join(context_parts))
        prompt_lines.append("")

    prompt_lines.append(f"Question: {question}")
    prompt_lines.append("Answer concisely:")

    user_content = "\n".join(prompt_lines)

    entry = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer_text}
        ]
    }

    if image_paths:
        entry["images"] = image_paths

    return entry


def main():
    print(f"Loading MMQA data from {MMQA_PKL}...")
    with open(MMQA_PKL, "rb") as f:
        data = pickle.load(f)

    train_data = data["train"]
    dev_data = data["dev"]
    tables = data["tables"]
    texts = data["texts"]
    images_meta = data["images"]

    print(f"  Train: {len(train_data)}, Dev: {len(dev_data)}")
    print(f"  Tables: {len(tables)}, Texts: {len(texts)}, Images: {len(images_meta)}")

    # Show modality distribution
    for split_name, split_data in [("Train", train_data), ("Dev", dev_data)]:
        mod_counter = Counter()
        type_counter = Counter()
        for s in split_data:
            mod_counter[tuple(sorted(s["metadata"]["modalities"]))] += 1
            type_counter[s["metadata"]["type"]] += 1
        print(f"\n  {split_name} modalities: {dict(mod_counter)}")
        print(f"  {split_name} types: {dict(type_counter)}")

    # Convert train split
    print(f"\nConverting train split ({len(train_data)} samples)...")
    train_sft = []
    train_skipped = 0
    train_with_images = 0
    for sample in train_data:
        entry = convert_sample(sample, tables, texts, images_meta, IMAGES_DIR)
        if entry is not None:
            train_sft.append(entry)
            if "images" in entry:
                train_with_images += 1
        else:
            train_skipped += 1

    print(f"  Train SFT: {len(train_sft)} samples ({train_with_images} with images, {train_skipped} skipped)")

    # Convert dev split
    print(f"\nConverting dev split ({len(dev_data)} samples)...")
    val_sft = []
    val_skipped = 0
    val_with_images = 0
    for sample in dev_data:
        entry = convert_sample(sample, tables, texts, images_meta, IMAGES_DIR)
        if entry is not None:
            val_sft.append(entry)
            if "images" in entry:
                val_with_images += 1
        else:
            val_skipped += 1

    print(f"  Val SFT: {len(val_sft)} samples ({val_with_images} with images, {val_skipped} skipped)")

    # Write output files
    print(f"\nWriting {OUT_TRAIN}...")
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        json.dump(train_sft, f, ensure_ascii=False, indent=2)
    train_size_mb = os.path.getsize(OUT_TRAIN) / (1024 * 1024)
    print(f"  Written: {len(train_sft)} samples, {train_size_mb:.1f} MB")

    print(f"Writing {OUT_VAL}...")
    with open(OUT_VAL, "w", encoding="utf-8") as f:
        json.dump(val_sft, f, ensure_ascii=False, indent=2)
    val_size_mb = os.path.getsize(OUT_VAL) / (1024 * 1024)
    print(f"  Written: {len(val_sft)} samples, {val_size_mb:.1f} MB")

    # Show a few example entries
    print("\n=== Example entries ===")
    shown = {"text_only": False, "with_image": False, "with_table": False}
    for entry in train_sft[:200]:
        has_img = "images" in entry
        content = entry["messages"][0]["content"]
        has_table = "[Table" in content
        has_text = "[Text" in content

        if has_img and not shown["with_image"]:
            print(f"\n--- With Image ---")
            print(f"User (first 300 chars): {content[:300]}...")
            print(f"Assistant: {entry['messages'][1]['content']}")
            print(f"Images: {entry.get('images', [])}")
            shown["with_image"] = True
        elif has_table and not has_img and not shown["with_table"]:
            print(f"\n--- With Table (no image) ---")
            print(f"User (first 300 chars): {content[:300]}...")
            print(f"Assistant: {entry['messages'][1]['content']}")
            shown["with_table"] = True
        elif has_text and not has_img and not has_table and not shown["text_only"]:
            print(f"\n--- Text Only ---")
            print(f"User (first 300 chars): {content[:300]}...")
            print(f"Assistant: {entry['messages'][1]['content']}")
            shown["text_only"] = True

        if all(shown.values()):
            break

    print("\nDone!")


if __name__ == "__main__":
    main()
