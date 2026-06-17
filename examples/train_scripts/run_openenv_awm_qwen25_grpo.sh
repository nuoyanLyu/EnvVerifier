#!/usr/bin/env bash
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export AGENTFLY_OPENENV_AWM_BASE_URL="${AGENTFLY_OPENENV_AWM_BASE_URL:-http://127.0.0.1:8899}"
export AGENTFLY_OPENENV_AWM_VERIFIER_MODE="${AGENTFLY_OPENENV_AWM_VERIFIER_MODE:-${OPENENV_AWM_VERIFIER_MODE:-code}}"
export AGENTFLY_OPENENV_AWM_KEEP_SESSION="${AGENTFLY_OPENENV_AWM_KEEP_SESSION:-0}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
TRAIN_DATASET="${TRAIN_DATASET:-${REPO_ROOT}/data/awm/awm_train.json}"
VAL_DATASET="${VAL_DATASET:-${TRAIN_DATASET}}"
SYSTEM_PROMPT_CONFIG="${SYSTEM_PROMPT_CONFIG:-${REPO_ROOT}/src/agentfly/configs/prompts/system_prompt_openenv_awm.yaml}"

N_GPUS="${N_GPUS:-8}"
N_NODES="${N_NODES:-1}"
RAY_PORT="${RAY_PORT:-6379}"
START_RAY="${START_RAY:-1}"

LR="${LR:-4e-7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32000}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-2048}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"

NUM_CHAINS="${NUM_CHAINS:-16}"
MAX_TURNS="${MAX_TURNS:-20}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10}"

ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
KL_COEF="${KL_COEF:-0.001}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-mse}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
TEMPERATURE="${TEMPERATURE:-1.0}"

PROJECT_NAME="${PROJECT_NAME:-AgentFly-OpenEnv-AWM}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25-openenv-awm-grpo}"
LOGGER="${LOGGER:-['console','wandb']}"

if [[ "${START_RAY}" == "1" ]]; then
  ray stop || true
  rm -rf /tmp/ray/ray_current_cluster
  HEAD_NODE_IP="$(hostname --ip-address)"
  ray start --head --node-ip-address="${HEAD_NODE_IP}" --port="${RAY_PORT}" --num-gpus="${N_GPUS}"
fi

python -m agentfly.cli train \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  data.train_files="${TRAIN_DATASET}" \
  data.val_files="${VAL_DATASET}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  agent.use_agent=True \
  agent.init_config.agent_type=hf \
  agent.init_config.max_model_len="${MAX_MODEL_LEN}" \
  agent.init_config.template=qwen2.5 \
  agent.init_config.tool_parser_name=hermes \
  agent.init_config.backend=async_verl \
  agent.init_config.tools='[list_tools,call_tool]' \
  agent.init_config.reward_name=openenv_awm_verifier_reward \
  agent.init_config.system_prompt_config_path="${SYSTEM_PROMPT_CONFIG}" \
  agent.init_config.model_name_or_path="${MODEL_PATH}" \
  agent.generation_config.max_tokens="${MAX_RESPONSE_TOKENS}" \
  agent.generation_config.temperature="${TEMPERATURE}" \
  agent.max_turns="${MAX_TURNS}" \
  agent.num_chains="${NUM_CHAINS}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
  actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE}" \
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.model.path="${MODEL_PATH}" \
  critic.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  critic.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
  trainer.critic_warmup=0 \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes="${N_NODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.val_before_train=False
