
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Agent World Model (AWM)** is a synthetic environment generation pipeline that creates executable, SQL-backed tool-use environments for multi-turn agentic reinforcement learning. Each environment is exposed via the MCP (Model Context Protocol) and backed by a SQLite database.

## Setup

```bash
uv sync   # Install dependencies (Python 3.12+ required)
```

Configure `.env` with LLM provider credentials:
```
AWM_SYN_LLM_PROVIDER=openai|azure|deepseek|dmx
OPENAI_API_KEY=...
OPENAI_BASE_URL=...           # Optional, for custom endpoints
EMBEDDING_OPENAI_API_KEY=...  # For scenario diversity checking
AWM_SYN_OVERRIDE_MODEL=...    # Optional model override
```

## Common Commands

```bash
# Full synthesis pipeline (all 7 stages)
awm gen all --input outputs/seed_scenario.jsonl --output_dir outputs --target_count 1000 --num_tasks 10

# Individual synthesis stages
awm gen scenario   # Stage 1: generate diverse scenarios
awm gen task       # Stage 2: generate tasks per scenario
awm gen db         # Stage 3: synthesize DB schemas
awm gen sample     # Stage 4: generate sample data
awm gen spec       # Stage 5: generate API specs
awm gen env        # Stage 6: generate MCP server code
awm gen verifier   # Stage 7: generate verification code

# Environment management
awm env start --scenario <name> --envs_load_path outputs/gen_envs.jsonl --port 8001
awm env check --url http://localhost:8001/mcp
awm env check_all --input outputs/gen_envs.jsonl
awm env reset_db --input_db outputs/gen_db.jsonl --input_sample outputs/gen_sample.jsonl

# Agent & verification
awm agent --task "..." --mcp_url http://localhost:8001/mcp --api_url http://localhost:8000/v1 --model <model>
awm verify --input outputs/agents/<timestamp> --mode sql  # or --mode code
```

## Architecture

### 7-Stage Synthesis Pipeline

```
seed_scenario.jsonl
    → [1] scenario.py   → gen_scenario.jsonl   (1K diverse scenarios)
    → [2] task.py       → gen_tasks.jsonl       (10K tasks)
    → [3] db.py         → gen_db.jsonl          (SQLite schemas)
    → [4] sample.py     → gen_sample.jsonl      (initial data)
    → [5] spec.py       → gen_spec.jsonl        (API specs)
    → [6] env.py        → gen_envs.jsonl        (FastAPI+MCP server code)
    → [7] verifier.py   → gen_verifier.jsonl    (verification logic)
```

Each stage reads the previous stage's JSONL output and writes to its own JSONL file in `outputs/`.

### Key Modules

| Module | Role |
|--------|------|
| `awm/cli.py` | CLI entry point; routes `awm <cmd> <subcmd>` using `TopCmd`/`GenCmd`/`EnvCmd` enums |
| `awm/gpt.py` | `GPTClient` — async LLM client with retry and batch completion; supports OpenAI/Azure/deepseek/dmx |
| `awm/prompts.py` | All LLM system/user prompts for every synthesis stage |
| `awm/tools.py` | Shared utilities: JSONL I/O, robust JSON parsing, token counting, port management |
| `awm/core/pipeline.py` | Orchestrates all 7 stages end-to-end |
| `awm/core/server.py` | Starts FastAPI+MCP server subprocess; manages `initial.db` / `final.db` lifecycle |
| `awm/core/agent.py` | Tool-use agent that interacts with live MCP servers via `mcp-agent` |
| `awm/core/verify.py` | Verifies agent trajectories using SQL-based or code-based verification |
| `awm/core/scenario.py` | Semantic-diversity-checked scenario generation using embeddings |

### Output Artifacts

```
outputs/
├── gen_scenario.jsonl       # 1K scenarios
├── gen_tasks.jsonl          # 10K tasks
├── gen_db.jsonl             # DB schemas
├── gen_sample.jsonl         # Sample data
├── gen_spec.jsonl           # API specs
├── gen_envs.jsonl           # MCP server code
├── gen_verifier.jsonl       # Verification code
├── databases/               # SQLite files (one per scenario)
├── servers/<ts>_<scenario>/ # Per-run server artifacts (initial.db, final.db, server_code.py, server.log)
└── agents/<ts>/             # Agent trajectory outputs (trajectory.json, initial.db, final.db)
```

### Key Dependencies

- `fastapi` + `fastapi-mcp` — generated MCP server runtime
- `mcp-agent` — agent framework for tool-use interaction
- `openai` — LLM API client (used for all providers via base URL override)
- `sqlalchemy` — database interaction in generated servers
- `json-repair` — robust parsing of LLM JSON outputs
- `tiktoken` — token counting for prompt management
