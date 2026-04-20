#!/usr/bin/env bash
# Server-only WebShop smoke training script.
#
# Download the WebShop RL data on the server before running:
#   mkdir -p /data1/lvnuoyan/dataset/webshop
#   huggingface-cli download Agent-One/AgentFly-Train \
#     --repo-type dataset \
#     --include "webshop_goals_*.json" \
#     --local-dir /data1/lvnuoyan/dataset/webshop
#
# Useful server-side checks:
#   bash -n examples/train_scripts/run_webshop_agent_swanlab_smoke.sh
#   python -c "import swanlab, ray, hydra, agentfly"
#   ls -lh /data1/lvnuoyan/dataset/webshop/webshop_goals_*.json
#   PRINT_HYDRA_CONFIG=1 bash examples/train_scripts/run_webshop_agent_swanlab_smoke.sh
#
# Minimal server smoke run:
#   TOTAL_TRAINING_STEPS=1 TRAIN_BATCH_SIZE=2 VAL_BATCH_SIZE=2 NUM_CHAINS=1 SAVE_FREQ=1 \
#     bash examples/train_scripts/run_webshop_agent_swanlab_smoke.sh

set -euo pipefail
set -x

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODEL_PATH="${MODEL_PATH:-/data1/lvnuoyan/llm_model/Qwen2.5-1.5B-Instruct}"
DATA_DIR="${DATA_DIR:-/data1/lvnuoyan/dataset/AgentFly-Train}"
TRAIN_DATASET="${TRAIN_DATASET:-${DATA_DIR}/webshop_goals_train.json}"
EVAL_DATASET="${EVAL_DATASET:-${DATA_DIR}/webshop_goals_val.json}"
CKPT_DIR="${CKPT_DIR:-/data1/lvnuoyan/llm_model/agentfly/webshop}"

N_GPUS="${N_GPUS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
RAY_PORT="${RAY_PORT:-6379}"
RESTART_RAY="${RESTART_RAY:-1}"
PRINT_HYDRA_CONFIG="${PRINT_HYDRA_CONFIG:-0}"

SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-10}"
RESUME_MODE="${RESUME_MODE:-auto}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
ACTOR_MICRO_BATCH_PER_GPU="${ACTOR_MICRO_BATCH_PER_GPU:-1}"
CRITIC_PPO_MINI_BATCH_SIZE="${CRITIC_PPO_MINI_BATCH_SIZE:-8}"
CRITIC_MICRO_BATCH_PER_GPU="${CRITIC_MICRO_BATCH_PER_GPU:-1}"

LR="${LR:-4e-7}"
NUM_CHAINS="${NUM_CHAINS:-2}"
MAX_TURNS="${MAX_TURNS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NEW_TOKENS_PER_TURN="${MAX_NEW_TOKENS_PER_TURN:-384}"
KL_COEF="${KL_COEF:-0.001}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.0001}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-mse}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"

ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
LOG_PROB_MICRO_BATCH_PER_GPU="${LOG_PROB_MICRO_BATCH_PER_GPU:-2}"

TEMPLATE="${TEMPLATE:-qwen2.5}"
TOOL_PARSER_NAME="${TOOL_PARSER_NAME:-hermes}"
TOOLS="${TOOLS:-[webshop_browser]}"
REWARD_NAME="${REWARD_NAME:-webshop_reward}"
AGENT_TYPE="${AGENT_TYPE:-hf}"
AGENT_BACKEND="${AGENT_BACKEND:-async_verl}"

PROJECT_NAME="${PROJECT_NAME:-AgentFly}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-webshop_qwen25_1p5b_swanlab_smoke}"
LOGGER="${LOGGER:-[\"console\",\"swanlab\"]}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-/data1/lvnuoyan/llm_model/agentfly/webshop/swanlog}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model path does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${TRAIN_DATASET}" || ! -f "${EVAL_DATASET}" ]]; then
    echo "WebShop data is missing. Expected:" >&2
    echo "  ${TRAIN_DATASET}" >&2
    echo "  ${EVAL_DATASET}" >&2
    echo "Download on the server with:" >&2
    echo "  mkdir -p ${DATA_DIR}" >&2
    echo "  huggingface-cli download Agent-One/AgentFly-Train --repo-type dataset --include 'webshop_goals_*.json' --local-dir ${DATA_DIR}" >&2
    exit 1
fi

mkdir -p "${CKPT_DIR}" "${SWANLAB_LOG_DIR}"

head_node_ip="${HEAD_NODE_IP:-$(hostname --ip-address | awk '{print $1}')}"

hydra_config_args=()
if [[ "${PRINT_HYDRA_CONFIG}" == "1" ]]; then
    hydra_config_args=(--cfg job)
fi

if [[ "${RESTART_RAY}" == "1" && "${PRINT_HYDRA_CONFIG}" != "1" ]]; then
    ray stop || true
    rm -rf /tmp/ray/ray_current_cluster
    ray start --head \
        --node-ip-address="${head_node_ip}" \
        --port="${RAY_PORT}" \
        --num-cpus="${RAY_NUM_CPUS}" \
        --num-gpus="${N_GPUS}"
fi

python -m agentfly.cli train "${hydra_config_args[@]}" \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    data.train_files="${TRAIN_DATASET}" \
    data.val_files="${EVAL_DATASET}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    agent.use_agent=True \
    agent.init_config.agent_type="${AGENT_TYPE}" \
    agent.init_config.max_model_len="${MAX_MODEL_LEN}" \
    agent.init_config.template="${TEMPLATE}" \
    agent.init_config.tool_parser_name="${TOOL_PARSER_NAME}" \
    agent.init_config.backend="${AGENT_BACKEND}" \
    agent.init_config.tools="${TOOLS}" \
    agent.init_config.reward_name="${REWARD_NAME}" \
    agent.init_config.model_name_or_path="${MODEL_PATH}" \
    agent.generation_config.max_tokens="${MAX_NEW_TOKENS_PER_TURN}" \
    agent.max_turns="${MAX_TURNS}" \
    agent.num_chains="${NUM_CHAINS}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_MICRO_BATCH_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
    actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE}" \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_PER_GPU}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_PER_GPU}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path="${MODEL_PATH}" \
    critic.ppo_mini_batch_size="${CRITIC_PPO_MINI_BATCH_SIZE}" \
    critic.ppo_micro_batch_size_per_gpu="${CRITIC_MICRO_BATCH_PER_GPU}" \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.val_before_train=False \
    ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}"
