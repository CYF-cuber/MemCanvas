#!/bin/bash
source /home/cyf/miniconda3/etc/profile.d/conda.sh
conda activate qwen
cd /home/cyf/LLaMA-Factory-main
CUDA_VISIBLE_DEVICES=0,1 nohup llamafactory-cli train examples/train_lora/infovqa_lora_sft.yaml > /home/cyf/codex/infovqa_sft_train.log 2>&1 &
echo "PID: $!"
