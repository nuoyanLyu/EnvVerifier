#!/usr/bin/env bash
export AWM_SYN_LLM_PROVIDER=dmx
export AWM_SYN_OVERRIDE_MODEL=gpt-5
export OPENAI_BASE_URL=https://www.dmxapi.cn/v1
# export EMBEDDING_MODEL=text-embedding-3-large

awm gen verifier \
    --mode sql \
    --input_task outputs/gen_tasks.jsonl \
    --output outputs/gen_verifier.jsonl