#!/usr/bin/env bash
export AWM_SYN_LLM_PROVIDER=dmx
export AWM_SYN_OVERRIDE_MODEL=gpt-5
export OPENAI_BASE_URL=https://www.dmxapi.cn/v1
export EMBEDDING_MODEL=text-embedding-3-large

# run classify commands

awm gen scenario \
    --input_path outputs/gen_scenario_20.jsonl \
    --output_path outputs/gen_scenario_20.jsonl \
    --classify_only True