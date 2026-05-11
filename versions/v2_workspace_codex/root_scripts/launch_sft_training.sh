#!/bin/bash
# Launch ScienceQA and OK-VQA SFT training jobs
# ScienceQA on GPU 0, OK-VQA on GPU 1, with unique MASTER_PORTs

cd /home/cyf/LLaMA-Factory-main

LLAMA_CLI=/home/cyf/miniconda3/envs/qwen/bin/llamafactory-cli

echo "Launching ScienceQA SFT training on GPU 0..."
CUDA_VISIBLE_DEVICES=0 MASTER_PORT=40478 nohup $LLAMA_CLI train examples/train_lora/scienceqa_lora_sft.yaml > /home/cyf/codex/scienceqa_sft_train.log 2>&1 &
SCIENCEQA_PID=$!
echo "  ScienceQA PID: $SCIENCEQA_PID"

echo "Launching OK-VQA SFT training on GPU 1..."
CUDA_VISIBLE_DEVICES=1 MASTER_PORT=40479 nohup $LLAMA_CLI train examples/train_lora/okvqa_lora_sft.yaml > /home/cyf/codex/okvqa_sft_train.log 2>&1 &
OKVQA_PID=$!
echo "  OK-VQA PID: $OKVQA_PID"

echo ""
echo "Both jobs launched. Monitor with:"
echo "  tail -f /home/cyf/codex/scienceqa_sft_train.log"
echo "  tail -f /home/cyf/codex/okvqa_sft_train.log"
echo "  nvidia-smi"
