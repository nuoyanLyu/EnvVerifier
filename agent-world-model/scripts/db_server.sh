#!/usr/bin/env bash
export AWM_SYN_LLM_PROVIDER=dmx
export AWM_SYN_OVERRIDE_MODEL=gpt-5
export OPENAI_BASE_URL=https://www.dmxapi.cn/v1
# export EMBEDDING_MODEL=text-embedding-3-large

# Reset databases to initial state
awm env reset_db \
    --input_db outputs/gen_db.jsonl \
    --input_sample outputs/gen_sample.jsonl

# Start MCP server for a scenario
awm env start \
    --scenario "etsy" \
    --envs_load_path outputs/gen_envs.jsonl \
    --port 8001