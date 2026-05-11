import pickle
import json
import os

with open('/home/cyf/codex/infographicvqa_data/infographicvqa_meta.pkl', 'rb') as f:
    data = pickle.load(f)

def convert_split(samples, split_name):
    converted = []
    skipped = 0
    for s in samples:
        img_path = s['image_path']
        if not os.path.exists(img_path):
            skipped += 1
            continue
        entry = {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>Look at the infographic and answer the question.\nQuestion: {}\nAnswer concisely:".format(s['question'])
                },
                {
                    "role": "assistant",
                    "content": s['answers'][0]
                }
            ],
            "images": [img_path]
        }
        converted.append(entry)
    print(f"{split_name}: {len(converted)} converted, {skipped} skipped (missing images)")
    return converted

train_data = convert_split(data['train'], 'train')
val_data = convert_split(data['val'], 'val')

with open('/home/cyf/LLaMA-Factory-main/data/infovqa_sft_train.json', 'w') as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open('/home/cyf/LLaMA-Factory-main/data/infovqa_sft_val.json', 'w') as f:
    json.dump(val_data, f, indent=2, ensure_ascii=False)

print("Train file size:", os.path.getsize('/home/cyf/LLaMA-Factory-main/data/infovqa_sft_train.json'))
print("Val file size:", os.path.getsize('/home/cyf/LLaMA-Factory-main/data/infovqa_sft_val.json'))
print("Done!")
