from __future__ import annotations

import json
import os
import inspect
import re
import sqlite3
import sys
from textwrap import dedent
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
AWM_ROOT = Path(os.getenv("AGENTFLY_AWM_ROOT", REPO_ROOT / "agent-world-model"))
_DEFAULT_AWM_OUTPUTS_ROOT = (
    AWM_ROOT / "AgentWorldModel-1K" if (AWM_ROOT / "AgentWorldModel-1K").exists() else AWM_ROOT / "outputs"
)
AWM_OUTPUTS_ROOT = Path(os.getenv("AGENTFLY_AWM_OUTPUTS_ROOT", _DEFAULT_AWM_OUTPUTS_ROOT))
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
    normalized = scenario.lower()
    normalized = re.sub(r"[^a-z0-9_]", "_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_").strip()


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
        "Call them with call_tool(tool_name=..., arguments={...}).",
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
    r"^\s*error\s*:",
    r"\binternal server error\b",
    r"\bserver failed\b",
    r"\bservice unavailable\b",
    r"\bbad gateway\b",
    r"\bgateway timeout\b",
    r"\bconnection refused\b",
    r"\bfailed to start\b",
    r"\bmcp server failed\b",
    r"\btimed out after\b",
    r"\btool '.+?' timed out\b",
    r"\btool '.+?' failed\b",
    r"\bgeneration timed out\b",
    r"\bgeneration failed\b",
    r"\btraceback\b",
    r"\bexception\b",
]
AGENT_ERROR_PATTERNS = [
    r"\bhallucinated tool\b",
    r"\bunknown tool\b",
    r"\binvalid tool\b",
    r"\binvalid argument\b",
    r"\binvalid input\b",
    r"\bmust be valid json\b",
    r"\barguments must be valid json\b",
    r"\barguments json must decode to an object\b",
    r"\barguments must be a dict or json string\b",
]


def _contains_any_pattern(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def classify_trajectory_issue(trajectory: list[dict[str, Any]]) -> str | None:
    combined_chunks: list[str] = []
    for message in trajectory:
        combined_chunks.append(extract_text_from_message(message))
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                combined_chunks.append("invalid tool call function")
                continue
            function_name = function.get("name", "")
            if function_name not in {"awm_list_tools", "awm_call_tool", "list_tools", "call_tool"}:
                combined_chunks.append(f"invalid tool name: {function_name}")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    combined_chunks.append("arguments must be valid JSON")
                    continue
                if not isinstance(parsed_arguments, dict):
                    combined_chunks.append("arguments JSON must decode to an object")

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



def _call_verifier_func(verify_func: Any, initial_db_path: str, final_db_path: str, final_answer: str | None) -> Any:
    signature = inspect.signature(verify_func)
    params = signature.parameters
    kwargs: dict[str, Any] = {}
    if "initial_db_path" in params:
        kwargs["initial_db_path"] = initial_db_path
    if "final_db_path" in params:
        kwargs["final_db_path"] = final_db_path
    if "final_answer" in params:
        kwargs["final_answer"] = final_answer or ""
    if kwargs:
        return verify_func(**kwargs)

    args = [initial_db_path, final_db_path]
    if len(params) >= 3:
        args.append(final_answer or "")
    return verify_func(*args)


def _sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in value]
    return str(value)


def _execute_verifier_code(
    verifier_code: str,
    function_name: str,
    verifier_mode: str,
    initial_db_path: str,
    final_db_path: str,
    final_answer: str | None,
) -> dict[str, Any]:
    if not os.path.exists(initial_db_path):
        return {
            "result": "others",
            "execution_status": "error",
            "error_message": f"Initial DB not found: {initial_db_path}",
        }
    if not os.path.exists(final_db_path):
        return {
            "result": "others",
            "execution_status": "error",
            "error_message": f"Final DB not found: {final_db_path}",
        }

    original_modes: dict[str, int] = {}
    for path in [initial_db_path, final_db_path]:
        try:
            original_modes[path] = os.stat(path).st_mode
            os.chmod(path, 0o444)
        except OSError:
            pass

    try:
        namespace = {
            "sqlite3": sqlite3,
            "json": json,
            "os": os,
            "re": re,
            "__builtins__": __builtins__,
        }
        exec(verifier_code, namespace)
        verify_func = namespace.get(function_name)
        if verify_func is None:
            return {"result": "others", "execution_status": "error", "error_message": f"Function '{function_name}' not found"}

        result = _call_verifier_func(verify_func, initial_db_path, final_db_path, final_answer)
        try:
            json.dumps(result)
        except TypeError:
            result = _sanitize_for_json(result)

        if verifier_mode == "code":
            if not isinstance(result, dict) or "result" not in result:
                return {
                    "result": "others",
                    "execution_status": "error",
                    "error_message": f"Invalid return format: {type(result).__name__}",
                }
            result_value = result.get("result", "others")
            if result_value not in ("complete", "others"):
                result_value = "others"
            return {"result": result_value, "execution_status": "success", "raw_result": result}

        return result if isinstance(result, dict) else {"raw_result": result}
    except Exception as exc:  # noqa: BLE001
        return {"result": "others", "execution_status": "error", "error_message": f"Execution error: {exc}"}
    finally:
        for path, mode in original_modes.items():
            try:
                os.chmod(path, mode)
            except OSError:
                pass


def run_awm_verifier(
    verifier_entry: dict[str, Any],
    verifier_mode: str,
    initial_db_path: str,
    final_db_path: str,
    final_answer: str | None = None,
) -> tuple[str, dict[str, Any]]:
    verification = verifier_entry.get("verification", {})
    code = verification.get("code", "")
    if not isinstance(code, str) or len(code.strip()) < 10:
        return "judge_error", {"error": "No valid verifier code found"}

    mode = verifier_mode.lower()
    function_name = "verify_task_completion" if mode == "code" else "verify_task"
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("def verify_") and "(" in stripped:
            function_name = stripped.split("(", 1)[0].replace("def ", "").strip()
            break

    result = _execute_verifier_code(code, function_name, mode, initial_db_path, final_db_path, final_answer)
    if result.get("execution_status") == "error":
        return "judge_error", result
    if mode == "code":
        return str(result.get("result", "others")), result
    return "incomplete", result



def _normalize_azure_url(url: str) -> str:
    if "openai.azure.com" not in url:
        return url
    if "/openai/deployments/" in url:
        return url
    return url.rstrip("/") + "/openai/v1"


def _load_dmx_api_key(api_keys_path: str | os.PathLike[str] | None = None) -> str:
    path = Path(api_keys_path or os.environ.get("AWM_API_KEYS_PATH", AWM_ROOT / "api-keys.json")).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    keys = data.get("dmx")
    if not isinstance(keys, list) or not keys or not isinstance(keys[0], str) or not keys[0].strip():
        raise ValueError(f"No usable dmx API key found in {path}")
    return keys[0].strip()


def resolve_awm_llm_config(
    api_url_override: str | None = None,
    model_override: str | None = None,
    require_model: bool = False,
) -> tuple[str, str, str]:
    provider = os.environ.get("AWM_SYN_LLM_PROVIDER", "dmx").lower()

    if provider == "azure":
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "EMPTY")
    elif provider == "dmx":
        api_key = _load_dmx_api_key()
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

    if api_url_override:
        api_url = _normalize_azure_url(api_url_override)
    elif provider == "azure":
        azure_endpoint = os.environ.get("AZURE_ENDPOINT_URL", "")
        if not azure_endpoint:
            raise ValueError(
                "AZURE_ENDPOINT_URL not set. "
                "Please set AWM_SYN_LLM_PROVIDER=azure, AZURE_ENDPOINT_URL, and AZURE_OPENAI_API_KEY."
            )
        api_url = _normalize_azure_url(azure_endpoint)
    elif provider == "dmx":
        api_url = os.environ.get("OPENAI_BASE_URL", "https://www.dmxapi.cn/v1")
    elif os.environ.get("OPENAI_BASE_URL"):
        api_url = os.environ["OPENAI_BASE_URL"]
    else:
        raise ValueError("No LLM API URL provided. Set OPENAI_BASE_URL, use AWM_SYN_LLM_PROVIDER=dmx, or set Azure endpoint variables.")

    model = model_override or os.environ.get("AWM_SYN_OVERRIDE_MODEL", "")
    if not model and require_model:
        raise ValueError("AWM_SYN_OVERRIDE_MODEL not set and no model override provided")
    return api_url, api_key, model


async def run_awm_llm_judge(
    *,
    task: str,
    verifier_result: dict[str, Any],
    trajectory: list[dict[str, Any]],
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_model: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run AWM's SQL-mode LLM judge without importing AWM runtime dependencies."""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        return "judge_error", {"error": f"openai package is required for sql LLM judge: {exc}"}

    try:
        if not llm_base_url or not llm_model:
            llm_base_url, resolved_key, llm_model = resolve_awm_llm_config(require_model=True)
            if llm_api_key is None:
                llm_api_key = resolved_key
        if not llm_base_url or not llm_model:
            return "judge_error", {"error": "LLM endpoint not configured for sql verifier mode"}

        client = AsyncOpenAI(base_url=llm_base_url, api_key=llm_api_key or "EMPTY")
        system_prompt = dedent("""\
            You are an impartial evaluator of automated agent task results with access to database verification. Based on the provided JSON trajectory AND the Python verification results from querying the database, decide the task outcome. This trajectory is generated by an MCP agent on a simulated simplified environment. The environment provides a set of MCP tools to help the agent complete the task.

            Input:
                task_json: dict containing task fields such as user task and agent execution_history.
                verification_json: dict containing Python code, reasoning, success_criteria, failure_criteria, and execution results that verified the database state from the Python function

            Output:
                You only output UTF-8 encoded string, avoid any emoji or special characters. You only output English text.

            Classification categories:
                - complete: all required steps and closure actions were successfully executed, AND the database state confirms the task was completed
                - incomplete: partial progress or the database state shows the task is not fully completed
                - server_error: the agent is blocked by MCP server/environment error, e.g., 5xx errors such as "Internal Server Error". Or the MCP server cannot process the valid tool call and return valid results. This can block the agent from completing the task.
                - agent_error: the agent made mistakes, invalid parameters, or missing required data without recovery, failed to complete the user's instruction.

            Priority order for classification:
                1) complete (trajectory shows success AND database confirms it)
                2) server_error (due to the MCP server/environment error)
                3) agent_error (agent-side issue, e.g., invalid mcp_tool_call arguments, hallucination, agent mistakes)
                4) incomplete (everything else unfinished or database state doesn't match expected outcome)

            Key considerations:
            - The verification_json contains checks performed on the database states. You can read the verification code to understand what checks were performed on the database states.
            - The verification_json contains the execution results of the verification code. You can use the execution results to help you judge the task completion.
            - The verification results may be empty or error, or even the verification code itself is inaccurate. You should not fully rely on the verification results. You need to comprehensively consider the trajectory information to help you judge the task completion.

            Output format (must be valid JSON, no markdown fences, no additional commentary):
                {
                  "reasoning": "<concise explanation considering both trajectory and verification code execution results>",
                  "confidence_score": [0-100, 0-100, 0-100, 0-100] for complete, incomplete, server_error, agent_error respectively,
                  "classification": "<one_of_[complete, incomplete, server_error, agent_error]>",
                  "evidence": {
                    "status": "<original result.status>",
                    "iterations": <int>,
                    "error_signals": ["<important error messages or codes>"],
                    "last_actions": ["<summaries of last few actions>"],
                    "database_verification": "<summary of what the database state changed based on code execution results of verification>"
                  }
                }""")

        task_payload = {
            "user_task": task,
            "actual_execution_steps": len(trajectory),
            "trajectory": trajectory,
        }
        verification_json = _sanitize_for_json({"code_execution_result": verifier_result})
        user_prompt = (
            "task_json:\n"
            f"{json.dumps(task_payload, ensure_ascii=False, indent=2, default=str)}\n\n"
            "verification_json:\n"
            f"{json.dumps(verification_json, ensure_ascii=False, indent=2, default=str)}"
        )
        response = await client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1.0,
            max_completion_tokens=4096,
        )
        content = response.choices[0].message.content or ""
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                return "judge_error", {"error": f"Failed to parse LLM response: {content}"}
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                return "judge_error", {"error": f"Failed to parse LLM response: {content}"}

        classification = str(result.get("classification", "judge_error")).lower().strip()
        if classification not in {"complete", "incomplete", "server_error", "agent_error", "judge_error"}:
            classification = "judge_error"
        return classification, result
    except Exception as exc:  # noqa: BLE001
        return "judge_error", {"error": str(exc)}
