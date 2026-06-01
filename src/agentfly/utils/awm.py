from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
AWM_ROOT = Path(os.getenv("AGENTFLY_AWM_ROOT", REPO_ROOT / "agent-world-model"))
AWM_OUTPUTS_ROOT = Path(os.getenv("AGENTFLY_AWM_OUTPUTS_ROOT", AWM_ROOT / "outputs"))
DEFAULT_AWM_ENVS_PATH = AWM_OUTPUTS_ROOT / "gen_envs.jsonl"
DEFAULT_AWM_TASKS_PATH = AWM_OUTPUTS_ROOT / "gen_tasks.jsonl"
DEFAULT_AWM_VERIFIER_CODE_PATH = AWM_OUTPUTS_ROOT / "gen_verifier.pure_code.jsonl"
DEFAULT_AWM_VERIFIER_SQL_PATH = AWM_OUTPUTS_ROOT / "gen_verifier.jsonl"
DEFAULT_AWM_DB_SCHEMA_PATH = AWM_OUTPUTS_ROOT / "gen_db.jsonl"
DEFAULT_AWM_SAMPLE_PATH = AWM_OUTPUTS_ROOT / "gen_sample.jsonl"
DEFAULT_AWM_SESSION_ROOT = Path(
    os.getenv("AGENTFLY_AWM_SESSION_ROOT", Path("/tmp") / "agentfly_awm_sessions")
)


def ensure_awm_importable() -> Path:
    awm_root = AWM_ROOT.resolve()
    if not awm_root.exists():
        raise FileNotFoundError(
            f"AgentWorldModel root not found: {awm_root}. "
            "Set AGENTFLY_AWM_ROOT or place the submodule at repo_root/agent-world-model."
        )
    awm_root_str = str(awm_root)
    if awm_root_str not in sys.path:
        sys.path.insert(0, awm_root_str)
    return awm_root


def resolve_path(
    value: str | os.PathLike[str] | None,
    default: str | os.PathLike[str],
) -> Path:
    raw = Path(value if value is not None else default).expanduser()
    if raw.is_absolute():
        return raw
    return (REPO_ROOT / raw).resolve()


def load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            entries.append(json.loads(stripped))
    return entries


def normalize_scenario_name(scenario: str) -> str:
    ensure_awm_importable()
    from awm.tools import normalize_scenario_name as awm_normalize_scenario_name

    return awm_normalize_scenario_name(scenario)


def find_task_text(
    tasks_path: str | os.PathLike[str],
    scenario: str,
    task_id: int,
) -> str:
    normalized = normalize_scenario_name(scenario)
    for entry in load_jsonl(tasks_path):
        if normalize_scenario_name(entry["scenario"]) != normalized:
            continue
        tasks = entry.get("tasks", [])
        if task_id < 0 or task_id >= len(tasks):
            raise IndexError(
                f"task_id {task_id} out of range for scenario {scenario} in {tasks_path}"
            )
        task = tasks[task_id]
        if not isinstance(task, str) or not task.strip():
            raise ValueError(
                f"Empty task text for scenario={scenario}, task_id={task_id} in {tasks_path}"
            )
        return task
    raise KeyError(f"Scenario {scenario} not found in {tasks_path}")


def find_verifier_entry(
    verifier_path: str | os.PathLike[str],
    scenario: str,
    task_id: int,
) -> dict[str, Any] | None:
    normalized = normalize_scenario_name(scenario)
    for entry in load_jsonl(verifier_path):
        if normalize_scenario_name(entry["scenario"]) != normalized:
            continue
        entry_task_id = entry.get("task_idx", entry.get("task_id"))
        if entry_task_id == task_id:
            return entry
    return None


def _schema_to_lines(schema: dict[str, Any], indent: int = 6) -> list[str]:
    lines: list[str] = []
    if not isinstance(schema, dict):
        return lines
    properties = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    for name, info in properties.items():
        if not isinstance(info, dict):
            continue
        type_name = info.get("type", "unknown")
        suffix = "required" if name in required else "optional"
        lines.append(f'{" " * indent}- {name}: {type_name} ({suffix})')
        description = info.get("description")
        if description:
            lines.append(f'{" " * (indent + 2)}Description: {description}')
        enums = info.get("enum")
        if enums:
            lines.append(f'{" " * (indent + 2)}Allowed values: {enums}')
    return lines


def format_mcp_tools(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return "No MCP tools are available in the current AWM session."

    lines = [
        f"Available MCP tools ({len(tools)} total):",
        "Call them with awm_call_tool(tool_name=..., arguments={...}).",
        "",
    ]
    for idx, tool in enumerate(tools, start=1):
        name = tool.get("name", "")
        description = tool.get("description", "") or "No description."
        lines.append(f"{idx}. {name}")
        lines.append(f"   Description: {description}")
        schema_lines = _schema_to_lines(tool.get("inputSchema") or tool.get("input_schema") or {})
        if schema_lines:
            lines.append("   Parameters:")
            lines.extend(schema_lines)
        else:
            lines.append("   Parameters: None")
        lines.append("")
    return "\n".join(lines).strip()


def parse_jsonish_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"arguments must be valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("arguments JSON must decode to an object")
        return parsed
    raise ValueError(f"arguments must be a dict or JSON string, got {type(arguments).__name__}")


def extract_text_from_message(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text") or "")
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part).strip()
    return str(content)


SERVER_ERROR_PATTERNS = [
    r"\binternal server error\b",
    r"\bserver failed\b",
    r"\bservice unavailable\b",
    r"\bbad gateway\b",
    r"\bgateway timeout\b",
    r"\bconnection refused\b",
    r"\bfailed to start\b",
    r"\bmcp server failed\b",
    r"\btimed out after\b",
]
AGENT_ERROR_PATTERNS = [
    r"\bhallucinated tool\b",
    r"\bunknown tool\b",
    r"\binvalid argument\b",
    r"\binvalid input\b",
    r"\bmust be valid json\b",
    r"\barguments must be valid json\b",
]


def _contains_any_pattern(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def classify_trajectory_issue(trajectory: list[dict[str, Any]]) -> str | None:
    combined_chunks = [extract_text_from_message(message) for message in trajectory]
    combined = "\n".join(chunk for chunk in combined_chunks if chunk)
    if _contains_any_pattern(combined, SERVER_ERROR_PATTERNS):
        return "server_error"
    if _contains_any_pattern(combined, AGENT_ERROR_PATTERNS):
        return "agent_error"
    return None


def flatten_awm_env_args(env_args: dict[str, Any] | None) -> dict[str, Any]:
    if env_args is None:
        return {}
    flattened = dict(env_args)
    extra_info = flattened.get("extra_info")
    if isinstance(extra_info, dict):
        for key, value in extra_info.items():
            flattened.setdefault(key, value)
    return flattened


def coerce_task_id(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("task_id must be an integer, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        return int(value)
    raise TypeError(f"task_id must be an integer-compatible value, got {value!r}")
