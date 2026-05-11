#!/bin/bash
# GRPO training for VLM Layout Agent (Single GPU)
#
# Trains Qwen2.5-VL-3B with LoRA using GRPO.
# Reward = format(0.1) + quality(0.3) + qa_accuracy(0.6)
#
# Prerequisites:
#   1. python -m layout_agent.reward_table --phase all
#   2. python -m layout_agent.prepare_data
#
# Usage:
#   bash layout_agent/run_grpo.sh
#   bash layout_agent/run_grpo.sh trainer.total_epochs=5

set -x

# Activate verl environment
source /home/cyf/miniconda3/etc/profile.d/conda.sh
conda activate verl

export CUDA_VISIBLE_DEVICES=0
export RAY_memory_monitor_refresh_ms=0

# Conservative memory settings to avoid OOM on 44GB GPU
nproc_per_gpu=8
mini_batch_size=${nproc_per_gpu}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/cyf/codex/layout_agent/rl_data/layout_train.parquet \
    data.val_files=/home/cyf/codex/layout_agent/rl_data/layout_val.parquet \
    data.train_batch_size=${nproc_per_gpu} \
    data.val_batch_size=${nproc_per_gpu} \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path=/home/cyf/Qwen2.5-VL-3B-Instruct \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.max_model_len=640 \
    actor_rollout_ref.rollout.max_num_batched_tokens=640 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=layout_agent/layout_reward.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='[console]' \
    trainer.project_name=memcanvas_layout_agent \
    trainer.experiment_name=qwen25vl_3b_layout_grpo_lora \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=3 \
    "$@"
