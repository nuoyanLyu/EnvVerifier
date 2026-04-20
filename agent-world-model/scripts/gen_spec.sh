#!/usr/bin/env bash
export AWM_SYN_LLM_PROVIDER=dmx
export AWM_SYN_OVERRIDE_MODEL=gpt-5
export OPENAI_BASE_URL=https://www.dmxapi.cn/v1

# API spec (interface schema)
awm gen spec \
    --input_task outputs/gen_tasks.jsonl \
    --input_db outputs/gen_db.jsonl \
    --output outputs/gen_spec.jsonl