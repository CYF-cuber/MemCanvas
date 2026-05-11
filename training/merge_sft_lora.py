#!/usr/bin/env python3
"""
Merge SFT LoRA adapter into base Qwen3-4B model.

verl's LoRA support creates fresh LoRA layers via get_peft_model() and cannot
load an existing adapter.  We merge the SFT adapter into the base weights so
that verl can treat the result as a new "base" and apply its own RL LoRA on top.

Input:
  base_model  = /home/cyf/Qwen3-4B
  lora_path   = /home/cyf/codex/agent_sft_output/final
Output:
  merged_path = /home/cyf/codex/agent_sft_merged/

Usage:
  python merge_sft_lora.py [--output /home/cyf/codex/agent_sft_merged]
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge SFT LoRA into base model")
    parser.add_argument(
        "--base-model",
        type=str,
        default="/home/cyf/Qwen3-4B",
        help="Path to Qwen3-4B base model",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="/home/cyf/codex/agent_sft_output/final",
        help="Path to SFT LoRA adapter",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/cyf/codex/agent_sft_merged",
        help="Output path for merged model",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    if output_path.exists() and (output_path / "config.json").exists():
        print(f"Merged model already exists at {output_path}, skipping.")
        return

    print("=" * 60)
    print("Merging SFT LoRA into base model")
    print("=" * 60)
    print(f"  Base model : {args.base_model}")
    print(f"  LoRA path  : {args.lora_path}")
    print(f"  Output     : {args.output}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Load base model
    print("Loading base model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # merge on CPU to avoid GPU OOM
        trust_remote_code=True,
    )

    # Load and merge LoRA
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.lora_path)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    # Save merged model
    print(f"Saving merged model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    # Verify
    saved_files = list(output_path.glob("*.safetensors"))
    config_file = output_path / "config.json"
    print(f"\nVerification:")
    print(f"  config.json exists: {config_file.exists()}")
    print(f"  Safetensors files:  {len(saved_files)}")
    total_size = sum(f.stat().st_size for f in saved_files)
    print(f"  Total size:         {total_size / 1e9:.2f} GB")
    print(f"\nDone! Merged model saved to {output_path}")


if __name__ == "__main__":
    main()
