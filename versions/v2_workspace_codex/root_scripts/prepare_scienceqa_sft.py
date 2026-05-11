#!/usr/bin/env python3
"""Generate ScienceQA SFT dataset for LLaMA-Factory"""
import sys
sys.path.insert(0, "/home/cyf/memory/memory_canvas/experiments")
from scienceqa_qwen25vl_full_experiment import ScienceQADataLoader
import json
import os

loader = ScienceQADataLoader()
choice_labels = ['A', 'B', 'C', 'D', 'E', 'F']

def make_sft_samples(data_dict, split_name):
    samples = []
    skipped_no_img_file = 0
    with_image = 0
    without_image = 0
    for pid, prob in data_dict.items():
        question = prob.get('question', '')
        choices = prob.get('choices', [])
        answer_idx = prob.get('answer', 0)
        correct_letter = choice_labels[answer_idx] if answer_idx < len(choice_labels) else 'A'

        # Build choices string
        choices_str = ', '.join([f"{choice_labels[i]}. {c}" for i, c in enumerate(choices) if i < len(choice_labels)])

        has_image = bool(prob.get('image', ''))

        if has_image:
            img_path = f"/home/cyf/memory/ScienceQA/data/scienceqa/images/{split_name}/{pid}/image.png"
            if not os.path.exists(img_path):
                skipped_no_img_file += 1
                has_image = False

        if has_image:
            user_content = f"<image>Look at the image and answer the question.\nQuestion: {question}\nChoices: {choices_str}\nAnswer with the letter only:"
            sample = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": correct_letter}
                ],
                "images": [img_path]
            }
            with_image += 1
        else:
            user_content = f"Look at the image and answer the question.\nQuestion: {question}\nChoices: {choices_str}\nAnswer with the letter only:"
            sample = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": correct_letter}
                ]
            }
            without_image += 1

        samples.append(sample)

    print(f"  {split_name}: {len(samples)} samples ({with_image} with image, {without_image} without image, {skipped_no_img_file} missing image files)")
    return samples

# Train
print("Processing ScienceQA train split...")
train_data = loader.get_split("train")
train_samples = make_sft_samples(train_data, "train")

print("Processing ScienceQA val split...")
val_data = loader.get_split("val")
val_samples = make_sft_samples(val_data, "val")

# Save
train_path = "/home/cyf/LLaMA-Factory-main/data/scienceqa_sft_train.json"
val_path = "/home/cyf/LLaMA-Factory-main/data/scienceqa_sft_val.json"

with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_samples, f, ensure_ascii=False, indent=2)
print(f"Saved train: {train_path} ({len(train_samples)} samples)")

with open(val_path, 'w', encoding='utf-8') as f:
    json.dump(val_samples, f, ensure_ascii=False, indent=2)
print(f"Saved val: {val_path} ({len(val_samples)} samples)")
