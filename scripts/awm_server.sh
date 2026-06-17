#!/usr/bin/env bash
set -euo pipefail

# Start the OpenEnv Agent-World-Model server with SQL-judge LLM env configured.
#
# Usage:
#   bash ~/EnvVerifier/scripts/awm_server.sh
#   bash ~/EnvVerifier/scripts/awm_server.sh --dry-run
#
# Optional overrides:
#   AWM_SERVER_HOST=0.0.0.0
#   AWM_SERVER_PORT=8899
#   AWM_DATA_DIR=/path/to/AgentWorldModel-1K
#   AWM_LLM_ENV_SCRIPT=/path/to/set_openenv_awm_llm_env.sh

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash ~/EnvVerifier/scripts/awm_server.sh
  bash ~/EnvVerifier/scripts/awm_server.sh --dry-run

Starts OpenEnv's agent_world_model_env server after exporting:
  AWM_DATA_DIR
  PYTHONPATH
  OPENENV_AWM_LLM_BASE_URL
  OPENENV_AWM_LLM_MODEL
  OPENENV_AWM_LLM_API_KEY

Environment overrides:
  AWM_SERVER_HOST       default: 0.0.0.0
  AWM_SERVER_PORT       default: 8899
  AWM_DATA_DIR          default: ~/EnvVerifier/agent-world-model/AgentWorldModel-1K
  AWM_LLM_ENV_SCRIPT    default: ~/EnvVerifier/my_test/set_openenv_awm_llm_env.sh
EOF
  exit 0
fi

REPO_ROOT="${REPO_ROOT:-$HOME/EnvVerifier}"
OPENENV_DIR="$REPO_ROOT/OpenEnv"
AWM_DATA_DIR="${AWM_DATA_DIR:-$REPO_ROOT/agent-world-model/AgentWorldModel-1K}"
AWM_LLM_ENV_SCRIPT="${AWM_LLM_ENV_SCRIPT:-$REPO_ROOT/my_test/set_openenv_awm_llm_env.sh}"
AWM_SERVER_HOST="${AWM_SERVER_HOST:-0.0.0.0}"
AWM_SERVER_PORT="${AWM_SERVER_PORT:-8899}"

if [[ ! -d "$OPENENV_DIR" ]]; then
  echo "OpenEnv directory not found: $OPENENV_DIR" >&2
  exit 1
fi

if [[ ! -d "$AWM_DATA_DIR" ]]; then
  echo "AWM_DATA_DIR not found: $AWM_DATA_DIR" >&2
  exit 1
fi

if [[ ! -x "$AWM_LLM_ENV_SCRIPT" ]]; then
  echo "LLM env script is missing or not executable: $AWM_LLM_ENV_SCRIPT" >&2
  exit 1
fi

export AWM_DATA_DIR
export PYTHONPATH="src:envs${PYTHONPATH:+:$PYTHONPATH}"

eval "$("$AWM_LLM_ENV_SCRIPT" --print-exports)"

mask_key() {
  local key="${OPENENV_AWM_LLM_API_KEY:-}"
  local len="${#key}"
  if (( len <= 8 )); then
    printf '<set:%s chars>' "$len"
  else
    printf '%s...%s (%s chars)' "${key:0:4}" "${key: -4}" "$len"
  fi
}

echo "OpenEnv AWM server configuration:"
echo "  OPENENV_DIR=$OPENENV_DIR"
echo "  AWM_DATA_DIR=$AWM_DATA_DIR"
echo "  PYTHONPATH=$PYTHONPATH"
echo "  OPENENV_AWM_LLM_BASE_URL=$OPENENV_AWM_LLM_BASE_URL"
echo "  OPENENV_AWM_LLM_MODEL=$OPENENV_AWM_LLM_MODEL"
echo "  OPENENV_AWM_LLM_API_KEY=$(mask_key)"
echo "  HOST=$AWM_SERVER_HOST"
echo "  PORT=$AWM_SERVER_PORT"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "Dry run only; server not started."
  exit 0
fi

if [[ $# -gt 0 ]]; then
  echo "Unknown option: $1" >&2
  exit 2
fi

cd "$OPENENV_DIR"
exec python -m uvicorn envs.agent_world_model_env.server.app:app \
  --host "$AWM_SERVER_HOST" \
  --port "$AWM_SERVER_PORT" \
  --workers 8
