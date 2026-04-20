#!/usr/bin/env bash
export AWM_SYN_LLM_PROVIDER=dmx
export AWM_SYN_OVERRIDE_MODEL=gpt-5
export OPENAI_BASE_URL=https://www.dmxapi.cn/v1

awm env check_all --input outputs/gen_envs.jsonl