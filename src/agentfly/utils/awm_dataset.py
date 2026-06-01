from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .awm import (
    DEFAULT_AWM_DB_SCHEMA_PATH,
    DEFAULT_AWM_ENVS_PATH,
    DEFAULT_AWM_SAMPLE_PATH,
    DEFAULT_AWM_TASKS_PATH,
    DEFAULT_AWM_VERIFIER_CODE_PATH,
    ensure_awm_importable,
    load_jsonl,
    resolve_path,
)


def build_awm_records(
    *,
    tasks_path: str | Path,
    envs_path: str | Path,
    verifier_path: str | Path,
    db_schema_path: str | Path,
    sample_path: str | Path,
    output_root: str | Path,
    verifier_mode: str = "code",
    max_scenarios: int | None = None,
    max_tasks_per_scenario: int | None = None,
) -> list[dict[str, Any]]:
    tasks_entries = load_jsonl(tasks_path)
    verifier_entries = load_jsonl(verifier_path)
    verifier_pairs = {
        (entry["scenario"], entry.get("task_idx", entry.get("task_id"))): entry
        for entry in verifier_entries
    }

    records: list[dict[str, Any]] = []
    for scenario_idx, entry in enumerate(tasks_entries):
        if max_scenarios is not None and scenario_idx >= max_scenarios:
            break
        scenario = entry["scenario"]
        tasks = entry.get("tasks", [])
        for task_id, task in enumerate(tasks):
            if max_tasks_per_scenario is not None and task_id >= max_tasks_per_scenario:
                break
            if (scenario, task_id) not in verifier_pairs:
                continue
            records.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": task}],
                        }
                    ],
                    "prompt": task,
                    "task": task,
                    "scenario": scenario,
                    "task_id": task_id,
                    "data_source": "awm",
                    "reward_model": {"ground_truth": task},
                    "extra_info": {
                        "scenario": scenario,
                        "task_id": task_id,
                        "task": task,
                        "awm_envs_path": str(resolve_path(envs_path, DEFAULT_AWM_ENVS_PATH)),
                        "awm_tasks_path": str(resolve_path(tasks_path, DEFAULT_AWM_TASKS_PATH)),
                        "awm_verifier_path": str(resolve_path(verifier_path, DEFAULT_AWM_VERIFIER_CODE_PATH)),
                        "awm_verifier_mode": verifier_mode,
                        "awm_db_schema_path": str(resolve_path(db_schema_path, DEFAULT_AWM_DB_SCHEMA_PATH)),
                        "awm_sample_path": str(resolve_path(sample_path, DEFAULT_AWM_SAMPLE_PATH)),
                        "awm_output_root": str(Path(output_root).expanduser().resolve()),
                    },
                }
            )
    return records


def write_json_records(records: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    ensure_awm_importable()
    parser = argparse.ArgumentParser(description="Build a minimal AgentFly RL dataset from AWM tasks.")
    parser.add_argument("--tasks-path", default=str(DEFAULT_AWM_TASKS_PATH))
    parser.add_argument("--envs-path", default=str(DEFAULT_AWM_ENVS_PATH))
    parser.add_argument("--verifier-path", default=str(DEFAULT_AWM_VERIFIER_CODE_PATH))
    parser.add_argument("--db-schema-path", default=str(DEFAULT_AWM_DB_SCHEMA_PATH))
    parser.add_argument("--sample-path", default=str(DEFAULT_AWM_SAMPLE_PATH))
    parser.add_argument("--output-root", default="/tmp/agentfly_awm_sessions")
    parser.add_argument("--output", required=True, help="Output JSON file for AgentFly training.")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-tasks-per-scenario", type=int, default=None)
    parser.add_argument("--verifier-mode", default="code", choices=["code", "sql"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_awm_records(
        tasks_path=args.tasks_path,
        envs_path=args.envs_path,
        verifier_path=args.verifier_path,
        db_schema_path=args.db_schema_path,
        sample_path=args.sample_path,
        output_root=args.output_root,
        verifier_mode=args.verifier_mode,
        max_scenarios=args.max_scenarios,
        max_tasks_per_scenario=args.max_tasks_per_scenario,
    )
    write_json_records(records, args.output)
    print(f"Wrote {len(records)} AWM samples to {args.output}")


if __name__ == "__main__":
    main()
