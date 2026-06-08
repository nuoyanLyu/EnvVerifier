#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def yaml_block_scalar(value: str) -> str:
    return "system_prompt: |2-\n" + "\n".join("  " + line if line else "" for line in value.splitlines()) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Sync AgentFly AWM system prompt from agent-world-model.")
    parser.add_argument("--awm-root", default=str(repo_root / "agent-world-model"))
    parser.add_argument("--uv-bin", default=os.environ.get("UV_BIN", "/home/lvnuoyan/.local/bin/uv"))
    parser.add_argument(
        "--output",
        default=str(repo_root / "src/agentfly/configs/prompts/system_prompt_awm.yaml"),
    )
    args = parser.parse_args()

    awm_root = Path(args.awm_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not awm_root.exists():
        raise FileNotFoundError(f"AWM root does not exist: {awm_root}")

    code = """
import json
from awm.core.agent import get_system_prompt
print(json.dumps(get_system_prompt()))
"""
    result = subprocess.run(
        [args.uv_bin, "run", "python", "-c", code],
        cwd=awm_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    prompt = json.loads(result.stdout)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml_block_scalar(prompt), encoding="utf-8")
    print(f"Synced AWM system prompt to {output}")


if __name__ == "__main__":
    main()
