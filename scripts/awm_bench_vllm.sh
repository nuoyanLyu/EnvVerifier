#!/usr/bin/env bash
set -euo pipefail

# Run AWM benchmark/evaluation through the mcp-adapted-bench harness.
# This can launch local vLLM or reuse an existing OpenAI-compatible endpoint.

usage() {
  cat <<'USAGE'
Usage:
  HF_MODEL_PATH=/path/or/hf/repo CUDA_DEVICES=0 BENCH_MODE=bfcl LIMIT=32 \
    bash ~/EnvVerifier/scripts/awm_bench_vllm.sh

  SKIP_VLLM=1 OPENAI_BASE_URL=http://127.0.0.1:8000/v1 MODEL=/path/or/hf/repo \
    BENCH_MODE=bfcl LIMIT=32 bash ~/EnvVerifier/scripts/awm_bench_vllm.sh

  HF_MODEL_PATH=/path/or/hf/repo bash ~/EnvVerifier/scripts/awm_bench_vllm.sh --dry-run

Benchmark modes:
  bfcl
  tau2
  mcp_universe

Core overrides:
  REPO_ROOT                  default: ~/EnvVerifier
  AWM_ROOT                   default: ~/EnvVerifier/agent-world-model
  UV_BIN                     default: /home/lvnuoyan/.local/bin/uv
  BENCH_MODE                 default: bfcl
  OUTPUT_DIR                 default: ~/EnvVerifier/agent-world-model/outputs/<mode>_<timestamp>
  LIMIT                      optional small-scale subset size
  NUM_ROLLOUTS               default: 1
  MAX_CONCURRENCY            default: 32
  MAX_TURNS                  default: 20, paper-training strict setting; set 30 for harness default
  HISTORY_LIMIT              default: 10, paper evaluation long-context setting
  MAX_COMPLETION_TOKENS      default: 2048
  TEMPERATURE                default: 0.6
  TOP_P                      default: 0.95
  TOP_K                      default: 20
  ENABLE_THINKING            default: true
  EARLY_STOP                 default: false
  TOLERANT_MODE              default: true
  RESUME                     default: false
  NOTE                       optional run note
  BENCH_EXTRA_ARGS           extra awm bench args, e.g. '--bfcl.skip_list_tools false'

Local vLLM overrides:
  HF_MODEL_PATH              required unless SKIP_VLLM=1
  MODEL                      default: HF_MODEL_PATH; should be the HF repo/path for tokenizer accounting
  CUDA_DEVICES               default: 0
  VLLM_BIN                   default: vllm
  VLLM_HOST                  default: 127.0.0.1
  VLLM_PORT                  default: 8000
  MAX_MODEL_LEN              default: 131072
  TENSOR_PARALLEL_SIZE       default: 1
  VLLM_EXTRA_ARGS            extra vLLM args
  VLLM_LOG_DIR               default: ~/EnvVerifier/agent-world-model/outputs/bench_logs
  VLLM_START_TIMEOUT         default: 600 seconds

Existing endpoint overrides:
  SKIP_VLLM=1
  OPENAI_BASE_URL            default with local vLLM: http://127.0.0.1:8000/v1
  OPENAI_API_KEY             default: EMPTY
  MODEL                      required with SKIP_VLLM=1 unless AWM_SYN_OVERRIDE_MODEL is already set

Tau2 user simulator overrides:
  TAU2_USER_SIM_LLM_BASE_URL
  TAU2_USER_SIM_LLM_API_KEY
  TAU2_USER_SIM_LLM_MODEL
USAGE
}

DRY_RUN=0
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "Unknown option: $1" >&2
  usage >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-$HOME/EnvVerifier}"
AWM_ROOT="${AWM_ROOT:-${REPO_ROOT}/agent-world-model}"
UV_BIN="${UV_BIN:-/home/lvnuoyan/.local/bin/uv}"
BENCH_MODE="${BENCH_MODE:-bfcl}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${AWM_ROOT}/outputs/${BENCH_MODE}_${RUN_ID}}"
LIMIT="${LIMIT:-}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
MAX_TURNS="${MAX_TURNS:-20}"
HISTORY_LIMIT="${HISTORY_LIMIT:-10}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
EARLY_STOP="${EARLY_STOP:-false}"
TOLERANT_MODE="${TOLERANT_MODE:-true}"
RESUME="${RESUME:-false}"
NOTE="${NOTE:-}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"

SKIP_VLLM="${SKIP_VLLM:-0}"
HF_MODEL_PATH="${HF_MODEL_PATH:-}"
MODEL="${MODEL:-${AWM_SYN_OVERRIDE_MODEL:-${HF_MODEL_PATH}}}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-${AWM_ROOT}/outputs/bench_logs}"
VLLM_START_TIMEOUT="${VLLM_START_TIMEOUT:-600}"

if [[ ! -d "${AWM_ROOT}" ]]; then
  echo "AWM_ROOT not found: ${AWM_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${UV_BIN}" ]]; then
  echo "UV_BIN is not executable: ${UV_BIN}" >&2
  exit 1
fi
case "${BENCH_MODE}" in
  bfcl|tau2|mcp_universe) ;;
  *)
    echo "Invalid BENCH_MODE: ${BENCH_MODE}" >&2
    exit 2
    ;;
esac

if [[ "${SKIP_VLLM}" != "1" && -z "${HF_MODEL_PATH}" ]]; then
  echo "HF_MODEL_PATH is required unless SKIP_VLLM=1." >&2
  exit 1
fi
if [[ -z "${MODEL}" ]]; then
  echo "MODEL is required. For local vLLM it defaults to HF_MODEL_PATH; for SKIP_VLLM=1 set MODEL explicitly." >&2
  exit 1
fi

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export AWM_SYN_LLM_PROVIDER="${AWM_SYN_LLM_PROVIDER:-openai}"
export AWM_SYN_OVERRIDE_MODEL="${MODEL}"

bench_args=(
  bench
  --mode "${BENCH_MODE}"
  --output_dir "${OUTPUT_DIR}"
  --api_url "${OPENAI_BASE_URL}"
  --model "${MODEL}"
  --num_rollouts "${NUM_ROLLOUTS}"
  --max_concurrency "${MAX_CONCURRENCY}"
  --max_turns "${MAX_TURNS}"
  --history_limit "${HISTORY_LIMIT}"
  --max_completion_tokens "${MAX_COMPLETION_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --top_k "${TOP_K}"
  --enable_thinking "${ENABLE_THINKING}"
  --early_stop "${EARLY_STOP}"
  --tolerant_mode "${TOLERANT_MODE}"
  --resume "${RESUME}"
  --note "${NOTE}"
  --bfcl.data_dir "${AWM_ROOT}/mcp-adapted-bench/third_party/bfcl-original/bfcl_eval/data"
  --tau2.tau2_root "${AWM_ROOT}/mcp-adapted-bench/third_party/tau2-bench-verified"
  --mcp_universe.mcp_universe_root "${AWM_ROOT}/mcp-adapted-bench/third_party/MCP-Universe"
)
if [[ -n "${LIMIT}" ]]; then
  bench_args+=(--limit "${LIMIT}")
fi
if [[ -n "${TAU2_USER_SIM_LLM_BASE_URL:-}" ]]; then
  bench_args+=(--tau2.user_sim_api_url "${TAU2_USER_SIM_LLM_BASE_URL}")
fi
if [[ -n "${TAU2_USER_SIM_LLM_API_KEY:-}" ]]; then
  bench_args+=(--tau2.user_sim_api_key "${TAU2_USER_SIM_LLM_API_KEY}")
fi
if [[ -n "${TAU2_USER_SIM_LLM_MODEL:-}" ]]; then
  bench_args+=(--tau2.user_sim_model "${TAU2_USER_SIM_LLM_MODEL}")
fi
if [[ -n "${BENCH_EXTRA_ARGS}" ]]; then
  read -r -a parsed_bench_extra_args <<< "${BENCH_EXTRA_ARGS}"
  bench_args+=("${parsed_bench_extra_args[@]}")
fi

vllm_args=(
  serve "${HF_MODEL_PATH}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)
if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
  read -r -a parsed_vllm_extra_args <<< "${VLLM_EXTRA_ARGS}"
  vllm_args+=("${parsed_vllm_extra_args[@]}")
fi

mask_key() {
  local key="${OPENAI_API_KEY:-}"
  local len="${#key}"
  if (( len <= 8 )); then
    printf '<set:%s chars>' "${len}"
  else
    printf '%s...%s (%s chars)' "${key:0:4}" "${key: -4}" "${len}"
  fi
}

print_config() {
  echo "AWM benchmark configuration:"
  echo "  AWM_ROOT=${AWM_ROOT}"
  echo "  BENCH_MODE=${BENCH_MODE}"
  echo "  OUTPUT_DIR=${OUTPUT_DIR}"
  echo "  LIMIT=${LIMIT:-<unset>}"
  echo "  MODEL=${MODEL}"
  echo "  OPENAI_BASE_URL=${OPENAI_BASE_URL}"
  echo "  OPENAI_API_KEY=$(mask_key)"
  echo "  SKIP_VLLM=${SKIP_VLLM}"
  if [[ "${SKIP_VLLM}" != "1" ]]; then
    echo "  HF_MODEL_PATH=${HF_MODEL_PATH}"
    echo "  CUDA_DEVICES=${CUDA_DEVICES}"
    echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
    echo "  TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
  fi
  printf "  BENCH_COMMAND="
  printf "%q " "${UV_BIN}" run awm "${bench_args[@]}"
  printf "\n"
  if [[ "${SKIP_VLLM}" != "1" ]]; then
    printf "  VLLM_COMMAND="
    printf "%q " env "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}" "${VLLM_BIN}" "${vllm_args[@]}"
    printf "\n"
  fi
}

wait_for_vllm() {
  local deadline=$((SECONDS + VLLM_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "${OPENAI_BASE_URL%/}/models" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "${VLLM_PID:-}" ]] && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "vLLM exited early. Log: ${VLLM_LOG}" >&2
      tail -n 160 "${VLLM_LOG}" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "vLLM did not become ready before timeout: ${OPENAI_BASE_URL%/}/models" >&2
  tail -n 160 "${VLLM_LOG}" >&2 || true
  return 1
}

VLLM_PID=""
cleanup() {
  local status=$?
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[cleanup] stopping vLLM pid=${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
      kill -9 "${VLLM_PID}" 2>/dev/null || true
    fi
  fi
  echo "BENCH_STATUS=${status}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
}
print_config
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run only; vLLM/evaluation not started."
  exit 0
fi

mkdir -p "${OUTPUT_DIR}" "${VLLM_LOG_DIR}"
trap cleanup EXIT

if [[ "${SKIP_VLLM}" != "1" ]]; then
  VLLM_LOG="${VLLM_LOG_DIR}/vllm_${BENCH_MODE}_${RUN_ID}.log"
  echo "Starting vLLM. log=${VLLM_LOG}"
  (
    cd "${AWM_ROOT}"
    exec env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${VLLM_BIN}" "${vllm_args[@]}"
  ) >"${VLLM_LOG}" 2>&1 &
  VLLM_PID="$!"
  wait_for_vllm
fi

cd "${AWM_ROOT}"
"${UV_BIN}" run awm "${bench_args[@]}"
