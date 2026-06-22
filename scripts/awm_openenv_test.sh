#!/usr/bin/env bash
set -euo pipefail

# Unified entry for OpenEnv Agent-World-Model correctness and stress checks.

usage() {
  cat <<'USAGE'
Usage:
  bash ~/EnvVerifier/scripts/awm_openenv_test.sh [MODE]
  bash ~/EnvVerifier/scripts/awm_openenv_test.sh --dry-run [MODE]

Modes:
  full      Run the full OpenEnv/AWM correctness chain, with 32 tasks x 8 rollouts stress enabled.
  stress    Run only the 32 x 8 concurrent environment stress test.
  long      Run 5 rounds of the 32 x 8 stress test, sleeping 20 seconds between rounds.
  sql       Run a 32 x 8 stress test with SQL verifier enabled; loads the LLM judge env config.

Default MODE: full

Common overrides:
  REPO_ROOT                  default: ~/EnvVerifier
  PYTHON_BIN                 default: /home/lvnuoyan/anaconda3/envs/agent/bin/python
  SERVER_PYTHON_BIN          default: ~/EnvVerifier/OpenEnv/.venv/bin/python
  AWM_DATA_DIR               default: ~/EnvVerifier/agent-world-model/AgentWorldModel-1K
  NUM_TASKS                  default: 32
  ROLLOUTS_PER_TASK          default: 8
  MAX_CONCURRENCY            default: 256
  ROUNDS                     default: 5, long mode only
  SLEEP_SECONDS              default: 20, long mode only
  SQL_MODEL                  default: deepseek-v4-flash, sql mode only
  AWM_LLM_ENV_SCRIPT         default: ~/EnvVerifier/my_test/set_openenv_awm_llm_env.sh, sql mode only
  LOG_DIR                    optional explicit log directory
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

MODE="${1:-${MODE:-full}}"
REPO_ROOT="${REPO_ROOT:-$HOME/EnvVerifier}"
NUM_TASKS="${NUM_TASKS:-32}"
ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-8}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"
ROUNDS="${ROUNDS:-5}"
SLEEP_SECONDS="${SLEEP_SECONDS:-20}"
SQL_MODEL="${SQL_MODEL:-deepseek-v4-flash}"
AWM_LLM_ENV_SCRIPT="${AWM_LLM_ENV_SCRIPT:-${REPO_ROOT}/my_test/set_openenv_awm_llm_env.sh}"

case "${MODE}" in
  full)
    cmd=("${REPO_ROOT}/my_test/run_awm_openenv_full_check.sh")
    envs=(
      "STRESS_NUM_TASKS=${NUM_TASKS}"
      "STRESS_ROLLOUTS_PER_TASK=${ROLLOUTS_PER_TASK}"
      "STRESS_MAX_CONCURRENCY=${MAX_CONCURRENCY}"
      "SESSION_NUM_TASK_GROUPS=${NUM_TASKS}"
      "SESSION_ROLLOUTS_PER_TASK=${ROLLOUTS_PER_TASK}"
      "SESSION_MAX_CONCURRENCY=${MAX_CONCURRENCY}"
    )
    ;;
  stress)
    cmd=("${REPO_ROOT}/my_test/run_openenv_awm_stress_32x8.sh")
    envs=(
      "NUM_TASKS=${NUM_TASKS}"
      "ROLLOUTS_PER_TASK=${ROLLOUTS_PER_TASK}"
      "MAX_CONCURRENCY=${MAX_CONCURRENCY}"
      "TOOL_CALL_MODE=${TOOL_CALL_MODE:-auto}"
      "VERIFIER_MODE=${VERIFIER_MODE:-none}"
    )
    ;;
  long)
    cmd=("${REPO_ROOT}/my_test/run_openenv_awm_stress_32x8_5rounds.sh")
    envs=(
      "ROUNDS=${ROUNDS}"
      "SLEEP_SECONDS=${SLEEP_SECONDS}"
      "NUM_TASKS=${NUM_TASKS}"
      "ROLLOUTS_PER_TASK=${ROLLOUTS_PER_TASK}"
      "MAX_CONCURRENCY=${MAX_CONCURRENCY}"
      "TOOL_CALL_MODE=${TOOL_CALL_MODE:-auto}"
      "VERIFIER_MODE=${VERIFIER_MODE:-none}"
    )
    ;;
  sql)
    if [[ ! -x "${AWM_LLM_ENV_SCRIPT}" ]]; then
      echo "LLM env script is missing or not executable: ${AWM_LLM_ENV_SCRIPT}" >&2
      exit 1
    fi
    eval "$("${AWM_LLM_ENV_SCRIPT}" --print-exports)"
    export OPENENV_AWM_LLM_MODEL="${SQL_MODEL}"
    cmd=("${REPO_ROOT}/my_test/run_openenv_awm_stress_32x8.sh")
    envs=(
      "NUM_TASKS=${NUM_TASKS}"
      "ROLLOUTS_PER_TASK=${ROLLOUTS_PER_TASK}"
      "MAX_CONCURRENCY=${MAX_CONCURRENCY}"
      "TOOL_CALL_MODE=${TOOL_CALL_MODE:-auto}"
      "VERIFIER_MODE=sql"
    )
    ;;
  *)
    echo "Unknown MODE: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! -x "${cmd[0]}" ]]; then
  echo "Target script is not executable: ${cmd[0]}" >&2
  exit 1
fi

echo "OpenEnv/AWM test configuration:"
echo "  MODE=${MODE}"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  NUM_TASKS=${NUM_TASKS}"
echo "  ROLLOUTS_PER_TASK=${ROLLOUTS_PER_TASK}"
echo "  MAX_CONCURRENCY=${MAX_CONCURRENCY}"
if [[ "${MODE}" == "long" ]]; then
  echo "  ROUNDS=${ROUNDS}"
  echo "  SLEEP_SECONDS=${SLEEP_SECONDS}"
fi
if [[ "${MODE}" == "sql" ]]; then
  echo "  OPENENV_AWM_LLM_BASE_URL=${OPENENV_AWM_LLM_BASE_URL:-}"
  echo "  OPENENV_AWM_LLM_MODEL=${OPENENV_AWM_LLM_MODEL:-}"
fi
printf "  COMMAND="
printf "%q " env "${envs[@]}" "${cmd[@]}"
printf "\n"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run only; test not started."
  exit 0
fi

exec env "${envs[@]}" "${cmd[@]}"
