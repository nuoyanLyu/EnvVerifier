from __future__ import annotations

import json
import re
from typing import Any

from ..envs.awm_session_env import AWMSessionEnv
from ..utils.awm import extract_text_from_message, flatten_awm_env_args, run_awm_llm_judge
from .reward_base import reward


THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
RAW_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

VALID_AWM_TOOL_NAMES = {"list_tools", "call_tool"}
PARTIAL_SUCCESS_RESULTS = {"partial", "partial_success", "partially_complete", "partially_successful"}
SERVER_ERROR_PATTERNS = [
    r"^\s*error\s*:",
    r"\bfailed\b",
    r"\bexception\b",
    r"\btraceback\b",
    r"\btimed out\b",
    r"\btimeout\b",
    r"\binternal server error\b",
    r"\bserver failed\b",
    r"\bservice unavailable\b",
    r"\bbad gateway\b",
    r"\bgateway timeout\b",
    r"\bconnection refused\b",
]


def _count_tool_messages(trajectory: list[dict[str, Any]]) -> int:
    return sum(1 for message in trajectory if message.get("role") == "tool")



def _loads_json_object(value: Any) -> dict[str, Any] | None:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _tool_call_function(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    function = tool_call.get("function", {})
    if not isinstance(function, dict):
        return "", None
    name = str(function.get("name", "") or "")
    arguments = _loads_json_object(function.get("arguments", {}))
    return name, arguments


def _raw_tool_call_function(raw_tool_call: str) -> tuple[str, dict[str, Any] | None]:
    try:
        data = json.loads(raw_tool_call.strip())
    except json.JSONDecodeError:
        return "", None
    if isinstance(data, list):
        if not data or not isinstance(data[0], dict):
            return "", None
        data = data[0]
    if not isinstance(data, dict):
        return "", None
    name = str(data.get("name", "") or "")
    arguments = _loads_json_object(data.get("arguments", {}))
    return name, arguments


def _extract_mcp_tool_names(list_tools_observation: str) -> set[str]:
    names: set[str] = set()
    for line in list_tools_observation.splitlines():
        match = re.match(r"\s*\d+\.\s+([A-Za-z_][A-Za-z0-9_.:-]*)\s*$", line)
        if match:
            names.add(match.group(1))
    return names


def _is_error_observation(observation: str) -> bool:
    lowered = observation.lower()
    return any(re.search(pattern, lowered) for pattern in SERVER_ERROR_PATTERNS)


def _assistant_messages(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [message for message in trajectory if message.get("role") == "assistant"]


def _validate_think_format(trajectory: list[dict[str, Any]]) -> tuple[bool, str | None]:
    for index, message in enumerate(_assistant_messages(trajectory)):
        text = extract_text_from_message(message)
        match = THINK_PATTERN.search(text)
        if not match:
            return False, f"assistant_message_{index}_missing_think"
        if not match.group(1).strip():
            return False, f"assistant_message_{index}_empty_think"
    return True, None


def _collect_tool_calls_and_observations(
    trajectory: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    calls: list[dict[str, Any]] = []
    observations: list[str] = []
    for message in trajectory:
        if message.get("role") == "assistant":
            for tool_call in message.get("tool_calls") or []:
                calls.append(tool_call)
        elif message.get("role") == "tool":
            observations.append(extract_text_from_message(message))
    return calls, observations


def _collect_raw_tool_calls_and_observations(
    trajectory: list[dict[str, Any]],
) -> tuple[list[str], list[str], str | None]:
    raw_calls: list[str] = []
    observations: list[str] = []
    for index, message in enumerate(trajectory):
        if message.get("role") == "assistant":
            text = extract_text_from_message(message)
            matches = RAW_TOOL_CALL_PATTERN.findall(text)
            parsed_count = len(message.get("tool_calls") or [])
            if len(matches) != parsed_count:
                return [], observations, f"assistant_message_{index}_malformed_tool_call_json"
            raw_calls.extend(matches)
        elif message.get("role") == "tool":
            observations.append(extract_text_from_message(message))
    return raw_calls, observations, None


def _validate_format_and_server(
    trajectory: list[dict[str, Any]],
    *,
    require_think: bool,
) -> dict[str, Any]:
    if require_think:
        valid_think, think_error = _validate_think_format(trajectory)
        if not valid_think:
            return {"classification": "format_error", "reason": think_error, "valid_think": 0.0}
    else:
        valid_think = None

    raw_calls, observations, raw_error = _collect_raw_tool_calls_and_observations(trajectory)
    if raw_error is not None:
        return {
            "classification": "format_error",
            "reason": raw_error,
            "valid_think": valid_think,
        }

    if not raw_calls:
        return {
            "classification": "format_error",
            "reason": "list_tools_not_called_exactly_once",
            "valid_think": valid_think,
        }

    native_calls: list[tuple[str, dict[str, Any]]] = []
    for index, raw_tool_call in enumerate(raw_calls):
        name, arguments = _raw_tool_call_function(raw_tool_call)
        if name not in VALID_AWM_TOOL_NAMES:
            return {
                "classification": "format_error",
                "reason": f"tool_call_{index}_invalid_tool_name",
                "valid_think": valid_think,
            }
        if arguments is None:
            return {
                "classification": "format_error",
                "reason": f"tool_call_{index}_invalid_json_arguments",
                "valid_think": valid_think,
            }
        native_calls.append((name, arguments))

    if native_calls[0][0] != "list_tools":
        return {
            "classification": "format_error",
            "reason": "first_tool_call_not_list_tools",
            "valid_think": valid_think,
        }

    list_tool_count = sum(1 for name, _ in native_calls if name == "list_tools")
    if list_tool_count != 1:
        return {
            "classification": "format_error",
            "reason": "list_tools_not_called_exactly_once",
            "valid_think": valid_think,
        }

    if len(observations) < len(native_calls):
        return {
            "classification": "server_error",
            "reason": "tool_call_missing_server_response",
            "valid_think": valid_think,
        }

    if any(_is_error_observation(observation) for observation in observations[: len(native_calls)]):
        return {
            "classification": "server_error",
            "reason": "tool_call_error_server_response",
            "valid_think": valid_think,
        }

    available_mcp_tools = _extract_mcp_tool_names(observations[0]) if observations else set()
    for index, (name, arguments) in enumerate(native_calls):
        if name != "call_tool":
            continue
        mcp_tool_name = arguments.get("tool_name")
        if not isinstance(mcp_tool_name, str) or not mcp_tool_name.strip():
            return {
                "classification": "format_error",
                "reason": f"tool_call_{index}_missing_mcp_tool_name",
                "valid_think": valid_think,
            }
        inner_arguments = arguments.get("arguments", {})
        if isinstance(inner_arguments, str):
            inner_arguments = _loads_json_object(inner_arguments)
        if not isinstance(inner_arguments, dict):
            return {
                "classification": "format_error",
                "reason": f"tool_call_{index}_invalid_mcp_arguments_json",
                "valid_think": valid_think,
            }
        if available_mcp_tools and mcp_tool_name not in available_mcp_tools:
            return {
                "classification": "format_error",
                "reason": f"tool_call_{index}_mcp_tool_not_listed",
                "valid_think": valid_think,
            }

    assistant_turns = len(_assistant_messages(trajectory))
    successful_call_tool_after_list = any(
        name == "call_tool"
        for name, _ in native_calls[1:]
    )
    if assistant_turns > 1 and not successful_call_tool_after_list:
        return {
            "classification": "format_error",
            "reason": "multi_turn_without_successful_call_tool_after_list_tools",
            "valid_think": valid_think,
        }

    return {"classification": None, "reason": None, "valid_think": valid_think}


def _extract_final_answer(final_response: str) -> str | None:
    match = ANSWER_PATTERN.search(final_response)
    if not match:
        return None
    return match.group(1).strip()


def _analyze_think_trajectory(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    assistant_texts = [
        extract_text_from_message(message)
        for message in trajectory
        if message.get("role") == "assistant"
    ]
    if not assistant_texts:
        return {
            "valid_think": False,
            "final_has_answer": False,
            "final_answer": None,
        }

    valid_think = True
    for text in assistant_texts:
        match = THINK_PATTERN.search(text)
        if not match or not match.group(1).strip():
            valid_think = False
            break

    final_text = assistant_texts[-1]
    final_answer = _extract_final_answer(final_text)
    return {
        "valid_think": valid_think,
        "final_has_answer": final_answer is not None,
        "final_answer": final_answer,
    }


def _build_reward_payload(
    *,
    classification: str,
    reward_value: float,
    acc: float,
    verifier_details: dict[str, Any],
    tool_call_count: int,
    valid_think: float | None = None,
    format_error_reason: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "reward": reward_value,
        "acc": acc,
        "classification": classification,
        "tool_call_count": float(tool_call_count),
        "complete": 1.0 if classification == "complete" else 0.0,
        "incomplete": 1.0 if classification == "incomplete" else 0.0,
        "partial_success": 1.0 if classification == "partial_success" else 0.0,
        "format_error": 1.0 if classification == "format_error" else 0.0,
        "agent_error": 1.0 if classification == "agent_error" else 0.0,
        "server_error": 1.0 if classification == "server_error" else 0.0,
        "judge_error": 1.0 if classification == "judge_error" else 0.0,
    }
    if valid_think is not None:
        payload["valid_think"] = valid_think
    if format_error_reason:
        payload["format_error_reason"] = format_error_reason

    execution_status = verifier_details.get("execution_status")
    if execution_status is not None:
        payload["verifier_execution_status"] = str(execution_status)
    raw_result = verifier_details.get("raw_result")
    if raw_result is not None:
        try:
            payload["verifier_raw_result"] = json.dumps(raw_result, ensure_ascii=False)
        except TypeError:
            payload["verifier_raw_result"] = str(raw_result)
    llm_judge = verifier_details.get("llm_judge")
    if llm_judge is not None:
        try:
            payload["llm_judge"] = json.dumps(llm_judge, ensure_ascii=False)
        except TypeError:
            payload["llm_judge"] = str(llm_judge)
    error_message = verifier_details.get("error") or verifier_details.get("error_message")
    if error_message:
        payload["verifier_error"] = str(error_message)

    return payload


def _classification_from_verifier(
    verifier_result: Any,
    trajectory: list[dict[str, Any]],
    env: AWMSessionEnv,
) -> str:
    normalized_result = str(verifier_result or "").strip().lower()
    if normalized_result == "complete":
        return "complete"
    if normalized_result in PARTIAL_SUCCESS_RESULTS:
        return "partial_success"
    if normalized_result == "judge_error":
        return "judge_error"
    return env.classify_trajectory_issue(trajectory) or "incomplete"


def _reward_from_classification(classification: str) -> tuple[float, float]:
    if classification == "format_error":
        return -1.0, 0.0
    if classification == "complete":
        return 1.0, 1.0
    if classification == "partial_success":
        return 0.1, 0.0
    return 0.0, 0.0


async def _classify_with_verifier(
    *,
    verifier_result: Any,
    verifier_details: dict[str, Any],
    trajectory: list[dict[str, Any]],
    env: AWMSessionEnv,
) -> tuple[str, dict[str, Any]]:
    if str(verifier_result or "").strip().lower() == "judge_error":
        return "judge_error", verifier_details
    if env.current_config is not None and env.current_config.verifier_mode == "sql":
        classification, judge_result = await run_awm_llm_judge(
            task=env.current_config.task,
            verifier_result=verifier_details,
            trajectory=trajectory,
        )
        details = dict(verifier_details)
        details["llm_judge"] = judge_result
        return classification, details
    return _classification_from_verifier(verifier_result, trajectory, env), verifier_details


async def _ensure_env_ready(
    env: AWMSessionEnv,
    scenario: str | None,
    task_id: int | None,
    task: str | None,
    extra_info: dict[str, Any] | None,
) -> None:
    if env.runtime is not None:
        return
    env_args = flatten_awm_env_args(extra_info)
    if scenario is not None:
        env_args["scenario"] = scenario
    if task_id is not None:
        env_args["task_id"] = task_id
    if task is not None:
        env_args["task"] = task
    await env.reset(env_args)


@reward(name="awm_verifier_reward", env_cls=AWMSessionEnv, pool_size=8)
async def awm_verifier_reward(
    final_response: str,
    trajectory: list[dict[str, Any]],
    env: AWMSessionEnv,
    scenario: str | None = None,
    task_id: int | None = None,
    task: str | None = None,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tool_call_count = _count_tool_messages(trajectory)
    validation = _validate_format_and_server(trajectory, require_think=False)
    if validation["classification"] is not None:
        reward_value, acc = _reward_from_classification(validation["classification"])
        return _build_reward_payload(
            classification=validation["classification"],
            reward_value=reward_value,
            acc=acc,
            verifier_details={},
            tool_call_count=tool_call_count,
            format_error_reason=validation["reason"] if validation["classification"] == "format_error" else None,
        )

    await _ensure_env_ready(env, scenario, task_id, task, extra_info)
    verifier_result, verifier_details = await env.run_verifier(final_response)
    classification, verifier_details = await _classify_with_verifier(
        verifier_result=verifier_result,
        verifier_details=verifier_details,
        trajectory=trajectory,
        env=env,
    )

    reward_value, acc = _reward_from_classification(classification)
    return _build_reward_payload(
        classification=classification,
        reward_value=reward_value,
        acc=acc,
        verifier_details=verifier_details,
        tool_call_count=tool_call_count,
    )


@reward(name="awm_verifier_reward_think", env_cls=AWMSessionEnv, pool_size=8)
async def awm_verifier_reward_think(
    final_response: str,
    trajectory: list[dict[str, Any]],
    env: AWMSessionEnv,
    scenario: str | None = None,
    task_id: int | None = None,
    task: str | None = None,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tool_call_count = _count_tool_messages(trajectory)
    validation = _validate_format_and_server(trajectory, require_think=True)
    if validation["classification"] is not None:
        reward_value, acc = _reward_from_classification(validation["classification"])
        return _build_reward_payload(
            classification=validation["classification"],
            reward_value=reward_value,
            acc=acc,
            verifier_details={},
            tool_call_count=tool_call_count,
            valid_think=validation["valid_think"],
            format_error_reason=validation["reason"] if validation["classification"] == "format_error" else None,
        )

    await _ensure_env_ready(env, scenario, task_id, task, extra_info)
    verifier_result, verifier_details = await env.run_verifier(final_response)
    classification, verifier_details = await _classify_with_verifier(
        verifier_result=verifier_result,
        verifier_details=verifier_details,
        trajectory=trajectory,
        env=env,
    )

    think_analysis = _analyze_think_trajectory(trajectory)
    valid_think = think_analysis["valid_think"]
    if not valid_think:
        classification = "format_error"

    reward_value, acc = _reward_from_classification(classification)

    return _build_reward_payload(
        classification=classification,
        reward_value=reward_value,
        acc=acc,
        verifier_details=verifier_details,
        tool_call_count=tool_call_count,
        valid_think=1.0 if valid_think else 0.0,
    )
