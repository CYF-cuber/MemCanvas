#!/usr/bin/env python3
"""
Agent SFT Training — Train Qwen3-4B with LoRA on oracle-labeled canvas management data.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n qwen python -u agent_sft_train.py
"""

import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "/home/cyf/Qwen3-4B"
DATA_DIR = Path("/home/cyf/codex/agent_sft_data")
OUTPUT_DIR = Path("/home/cyf/codex/agent_sft_output")

LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET = ["q_proj", "v_proj", "k_proj", "o_proj"]
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
MAX_LENGTH = 4096
WARMUP_RATIO = 0.05


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sft_data():
    """Load combined SFT data (write + read)."""
    combined_file = DATA_DIR / "combined_sft.json"
    print(f"Loading SFT data from {combined_file}...")
    with open(combined_file, "r") as f:
        data = json.load(f)
    print(f"  Total samples: {len(data)}")
    return data


def tokenize_sample(sample, tokenizer, max_length=MAX_LENGTH):
    """Tokenize a single SFT sample into input_ids + labels."""
    messages = [
        {"role": "system", "content": sample["instruction"]},
        {"role": "user", "content": sample["input"]},
        {"role": "assistant", "content": sample["output"]},
    ]

    # Tokenize full conversation
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    full_ids = tokenizer(full_text, truncation=True, max_length=max_length)["input_ids"]

    # Tokenize without assistant response to find the prompt length
    prompt_messages = [
        {"role": "system", "content": sample["instruction"]},
        {"role": "user", "content": sample["input"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]

    # Build labels: mask prompt tokens with -100
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    # Ensure same length
    labels = labels[:len(full_ids)]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    raw_data = load_sft_data()

    # Tokenize
    print("Tokenizing...")
    tokenized = []
    skipped = 0
    for sample in raw_data:
        tok = tokenize_sample(sample, tokenizer)
        if len(tok["input_ids"]) < MAX_LENGTH:
            tokenized.append(tok)
        else:
            skipped += 1
    print(f"  Tokenized: {len(tokenized)}, Skipped (too long): {skipped}")

    dataset = Dataset.from_list(tokenized)
    # Split 95/5
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET,
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    # Train
    print(f"\n{'='*60}")
    print("Starting Agent SFT Training")
    print(f"  LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}")
    print(f"  LR={LEARNING_RATE}, epochs={NUM_EPOCHS}")
    print(f"  Effective batch={BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"{'='*60}\n")

    trainer.train()

    # Save
    print("Saving model...")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    # Eval
    metrics = trainer.evaluate()
    print(f"\nFinal eval metrics: {metrics}")

    with open(OUTPUT_DIR / "train_results.json", "w") as f:
        json.dump({"train_args": training_args.to_dict(), "eval_metrics": metrics}, f, indent=2, default=str)

    print(f"\nDone! Model saved to {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
