#!/bin/bash
# Modified in Code-R1 Enhanced, 2026-11-24; see CHANGELOG-ENHANCED.md.
# The config is optimized for 8xH200
set -eo pipefail

# Enhanced: environment-driven settings and a no-GPU configuration check.

if [ -n "${GPUS_PER_NODE:-}" ]; then
    :
elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

# Jiawei's notes for 4xA100 PCIe (@Yifeng):
# - Becasue of PCIe, prefer gradient checkpointing over offloading
# - If offloading, prefer optimizer offloading (zero1) over param offloading
# - The code execution concurrency is $TOTAL_SAMPLES - nice to make it larger than $(nproc) to maximize CPUs
# - Try to make the #steps as long as possible: e.g., increasing epochs / reducing batches...
# - Set save_freq to a large number as I guess Colossus has little space left
# - If you are short of VRAM, consider removing reference policy. To do so, you need to go to
#    main_ppo.py:main_task - and comment "Role.RefPolicy..." in "role_worker_mapping = ".

# MAIN CONFIG
MAX_EPOCHS=${MAX_EPOCHS:-6}
DATASET=${DATASET:-code-r1-12k}
MODEL_PATH=${MODEL_PATH:-./models/Qwen2.5-7B-Instruct-1M}
ROLLOUT_N_SAMPLE=${ROLLOUT_N_SAMPLE:-16}
ROLLOUT_N_QUERY=${ROLLOUT_N_QUERY:-8}
MICRO_BATCH_PER_GPU=${MICRO_BATCH_PER_GPU:-16} # * GPUS_PER_NODE -> GLOBAL_BATCH_SIZE
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
for value in "$GPUS_PER_NODE" "$MAX_EPOCHS" "$ROLLOUT_N_SAMPLE" "$ROLLOUT_N_QUERY" "$MICRO_BATCH_PER_GPU" "$GRAD_ACC_STEPS"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then echo "Invalid positive integer: $value" >&2; exit 2; fi
done
GLOBAL_BATCH_SIZE=$(($(($GPUS_PER_NODE * $MICRO_BATCH_PER_GPU)) * $GRAD_ACC_STEPS))

# assert ROLLOUT_N_QUERY * ROLLOUT_N_SAMPLE % GLOBAL_BATCH_SIZE == 0
TOTAL_SAMPLES=$(( ROLLOUT_N_QUERY * ROLLOUT_N_SAMPLE ))
if (( TOTAL_SAMPLES % GLOBAL_BATCH_SIZE != 0 )); then
    echo "Error: (ROLLOUT_N_QUERY * ROLLOUT_N_SAMPLE) must be divisible by GLOBAL_BATCH_SIZE."
    echo "Currently, ${TOTAL_SAMPLES} is not divisible by ${GLOBAL_BATCH_SIZE}."
    exit 1
else
    echo "Assertion passed: ${TOTAL_SAMPLES} is divisible by ${GLOBAL_BATCH_SIZE}."
fi

TRAIN_FILE=${TRAIN_FILE:-data/$DATASET/train.parquet}
VAL_FILE=${VAL_FILE:-data/$DATASET/test.parquet}
if [ -f "data/$DATASET/validation.parquet" ] && [ "$VAL_FILE" = "data/$DATASET/test.parquet" ]; then
    VAL_FILE="data/$DATASET/validation.parquet"
fi
if [ "${CODER1_DRY_RUN:-0}" = 1 ]; then
    printf 'GPUs=%s total_samples=%s global_batch=%s train=%s validation=%s\n' "$GPUS_PER_NODE" "$TOTAL_SAMPLES" "$GLOBAL_BATCH_SIZE" "$TRAIN_FILE" "$VAL_FILE"
    exit 0
fi
for file in "$TRAIN_FILE" "$VAL_FILE"; do
    if [ ! -f "$file" ]; then echo "Missing dataset: $file" >&2; exit 2; fi
done
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

python3 -m verl.trainer.main_ppo \
    "data.train_files=$TRAIN_FILE" \
    "data.val_files=$VAL_FILE" \
    data.train_batch_size=$ROLLOUT_N_QUERY \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$GLOBAL_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=256 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N_SAMPLE \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size=256 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    "critic.model.path=$MODEL_PATH" \
    critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_PER_GPU \
    critic.model.enable_gradient_checkpointing=True \
    \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='code-r1' \
    trainer.experiment_name=${DATASET}-ppo \
    trainer.nnodes=1 \
    trainer.default_local_dir=./models/${DATASET}-ppo \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.save_freq=32 \
    trainer.test_freq=8 \
    trainer.resume_mode=auto \
    trainer.total_epochs=$MAX_EPOCHS \
    reward_model.reward_manager=prime "$@" 2>&1 | tee ppo.log
