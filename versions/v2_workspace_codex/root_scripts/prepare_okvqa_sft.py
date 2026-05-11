#!/usr/bin/env python3
"""Generate OK-VQA SFT dataset for LLaMA-Factory"""
import pickle
import json
import os
from collections import Counter

data = pickle.load(open("/home/cyf/codex/okvqa_data/okvqa_cached.pkl", "rb"))

def make_okvqa_samples(split_data, max_samples=None):
    samples = []
    skipped = 0
    for item in split_data:
        if max_samples and len(samples) >= max_samples:
            break

        question = item['question']
        answers = item['answers']
        image_path = item['image_path']

        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            skipped += 1
            continue

        # Most common answer
        top_answer = Counter(answers).most_common(1)[0][0]

        user_content = f"<image>Look at the image and answer the question.\nQuestion: {question}\nAnswer concisely:"
        sample = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": top_answer}
            ],
            "images": [image_path]
        }
        samples.append(sample)

    print(f"  Generated {len(samples)} samples, skipped {skipped} (missing images)")
    return samples

# Train
print("Processing OK-VQA train split...")
train_samples = make_okvqa_samples(data['train'])

# Val (first 500 from test)
print("Processing OK-VQA test split (first 500 for val)...")
val_samples = make_okvqa_samples(data['test'], max_samples=500)

# Save
train_path = "/home/cyf/LLaMA-Factory-main/data/okvqa_sft_train.json"
val_path = "/home/cyf/LLaMA-Factory-main/data/okvqa_sft_val.json"

with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_samples, f, ensure_ascii=False, indent=2)
print(f"Saved train: {train_path} ({len(train_samples)} samples)")

with open(val_path, 'w', encoding='utf-8') as f:
    json.dump(val_samples, f, ensure_ascii=False, indent=2)
print(f"Saved val: {val_path} ({len(val_samples)} samples)")
