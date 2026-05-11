#!/usr/bin/env python3
"""
Merge RL LoRA adapter from verl checkpoint into SFT-merged model.

verl saves LoRA adapters at:
  checkpoints/.../global_step_XXX/actor/lora_adapter/

This script loads the SFT-merged base + RL LoRA → merged RL model.

Usage:
  python merge_rl_lora.py --step 250
  python merge_rl_lora.py --step best  # auto-pick best from log
"""

import argparse
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


CKPT_DIR = Path("/home/cyf/verl-main/checkpoints/memcanvas_read_grpo/qwen3_4b_read_grpo_lora")
SFT_MERGED = "/home/cyf/codex/agent_sft_merged"
OUTPUT_DIR = Path("/home/cyf/codex/agent_rl_merged")
LOG_FILE = Path("/home/cyf/codex/grpo_train.log")


def find_best_step() -> int:
    """Parse training log to find the step with highest validation reward."""
    pattern = re.compile(
        r"val-core/memcanvas_read/reward/mean@1:([\d.]+).*?training/global_step:(\d+)"
    )
    # The log line has both val reward and global_step in the same line
    # but they come from different grep patterns. Let's parse step lines directly.

    # Each validation is logged every 10 steps. The val reward line contains
    # the step info in the same log line.
    best_reward = -1.0
    best_step = 250  # fallback

    with open(LOG_FILE, "r") as f:
        for line in f:
            if "val-core/memcanvas_read/reward/mean@1:" not in line:
                continue
            # Extract reward
            m_reward = re.search(r"val-core/memcanvas_read/reward/mean@1:([\d.]+)", line)
            m_step = re.search(r"training/global_step:(\d+)", line)
            if m_reward and m_step:
                reward = float(m_reward.group(1))
                step = int(m_step.group(1))
                if reward > best_reward:
                    best_reward = reward
                    best_step = step

    # Round to nearest saved checkpoint (every 50 steps)
    saved_step = round(best_step / 50) * 50
    if saved_step == 0:
        saved_step = 50

    # Verify checkpoint exists
    ckpt_path = CKPT_DIR / f"global_step_{saved_step}" / "actor" / "lora_adapter"
    if not ckpt_path.exists():
        # Try the exact step or nearby
        for delta in [0, -50, 50, -10, 10]:
            alt = CKPT_DIR / f"global_step_{best_step + delta}" / "actor" / "lora_adapter"
            if alt.exists():
                saved_step = best_step + delta
                break

    print(f"Best validation reward: {best_reward:.4f} at step {best_step}")
    print(f"Using checkpoint: global_step_{saved_step}")
    return saved_step


def main():
    parser = argparse.ArgumentParser(description="Merge RL LoRA into SFT-merged model")
    parser.add_argument("--step", type=str, default="best",
                        help="Checkpoint step number, or 'best' to auto-select")
    parser.add_argument("--base-model", type=str, default=SFT_MERGED,
                        help="SFT-merged model path (RL base)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Output path for merged RL model")
    args = parser.parse_args()

    # Determine step
    if args.step == "best":
        step = find_best_step()
    else:
        step = int(args.step)

    lora_path = CKPT_DIR / f"global_step_{step}" / "actor" / "lora_adapter"
    output_path = Path(args.output)

    print("=" * 60)
    print("Merging RL LoRA into SFT-merged model")
    print("=" * 60)
    print(f"  Base model  : {args.base_model}")
    print(f"  RL LoRA     : {lora_path}")
    print(f"  Output      : {output_path}")

    if not lora_path.exists():
        print(f"\nERROR: LoRA path does not exist: {lora_path}")
        print(f"Available checkpoints:")
        for d in sorted(CKPT_DIR.glob("global_step_*")):
            print(f"  {d.name}")
        return

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Load base model (SFT-merged)
    print("Loading SFT-merged base model (bf16, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Load RL LoRA
    print(f"Loading RL LoRA adapter (step {step})...")
    model = PeftModel.from_pretrained(model, str(lora_path))

    # Merge
    print("Merging RL LoRA weights...")
    model = model.merge_and_unload()

    # Save
    print(f"Saving merged RL model to {output_path}...")
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
    print(f"\nDone! Merged RL model saved to {output_path}")


if __name__ == "__main__":
    main()
