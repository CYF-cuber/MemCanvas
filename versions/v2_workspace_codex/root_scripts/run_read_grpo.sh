#!/bin/bash
# GRPO training for MemCanvas Read Agent (Single GPU)
#
# Trains a LoRA adapter on the SFT-merged Qwen3-4B model using GRPO.
# Uses vLLM uniprocess executor (patched) to avoid Ray scheduling deadlocks.
#
# Prerequisites:
#   1. python merge_sft_lora.py   → /home/cyf/codex/agent_sft_merged/
#   2. python prepare_read_rl_data.py → /home/cyf/codex/agent_rl_data/
#
# Usage:
#   bash run_read_grpo.sh
#   bash run_read_grpo.sh trainer.total_epochs=5

set -x

export CUDA_VISIBLE_DEVICES=0,1

# Batch sizing: single GPU, smaller batches to avoid OOM on 4B model
# batch=16, micro=2 for training (backward pass); micro=4 for log_prob (forward only)
nproc_per_gpu=16
mini_batch_size=${nproc_per_gpu}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/cyf/codex/agent_rl_data/read_train.parquet \
    data.val_files=/home/cyf/codex/agent_rl_data/read_val.parquet \
    data.train_batch_size=${nproc_per_gpu} \
    data.val_batch_size=${nproc_per_gpu} \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path=/home/cyf/codex/agent_sft_merged \
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
    actor_rollout_ref.rollout.max_model_len=1280 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1280 \
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
    trainer.project_name=memcanvas_read_grpo \
    trainer.experiment_name=qwen3_4b_read_grpo_lora \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 \
    "$@"
