#!/bin/bash
# GRPO v2 training for Text Compressor with VLM Readability Reward (Single GPU)
#
# Key difference from v1: adds VLM readability reward (20% weight)
#   R = 0.35*answer + 0.25*facts + 0.20*conciseness + 0.20*vlm_readability
#
# The VLM (Qwen2.5-VL-3B) reads rendered canvas images to verify text legibility.
# During training, the actor model is offloaded during reward phase, freeing GPU
# for the reward VLM.
#
# Prerequisites:
#   1. python prepare_compress_v2_data.py  → compress_v2_data/
#   2. SFT-merged Qwen3-4B base at /home/cyf/codex/agent_sft_merged/
#   3. pip install python-Levenshtein
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash run_compress_grpo_v2.sh
#   # To disable VLM reward (faster, text-only):
#   VLM_REWARD_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 bash run_compress_grpo_v2.sh

set -x

# Activate verl conda environment, skip user site-packages to avoid torch conflicts
export PATH=/home/cyf/miniconda3/envs/verl/bin:$PATH
export PYTHONNOUSERSITE=1

# Prevent Ray from clearing CUDA_VISIBLE_DEVICES for actors with num_gpus=0
# (needed so TaskRunner can load VLM reward model on GPU)
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Override reward function to v2
export REWARD_SCRIPT=/home/cyf/codex/compress_reward_v2.py

# VLM readability settings
export VLM_READABILITY_MODEL=/home/cyf/Qwen2.5-VL-3B-Instruct
export VLM_REWARD_DEVICE=cuda  # GPU is free during reward phase (param_offload)

# Disable VLM if env says so (for faster debugging)
if [ "${VLM_REWARD_DISABLE}" = "1" ]; then
    export VLM_REWARD_DISABLE=1
fi

BASE_MODEL=/home/cyf/codex/agent_sft_merged

nproc_per_gpu=16
mini_batch_size=${nproc_per_gpu}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/cyf/codex/compress_v2_data/compress_train.parquet \
    data.val_files=/home/cyf/codex/compress_v2_data/compress_val.parquet \
    data.train_batch_size=${nproc_per_gpu} \
    data.val_batch_size=${nproc_per_gpu} \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='[console]' \
    trainer.project_name=memcanvas_compress_grpo_v2 \
    trainer.experiment_name=qwen3_4b_compress_grpo_v2_vlm_reward \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 \
    "$@"
