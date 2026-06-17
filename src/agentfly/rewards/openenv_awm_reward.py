from __future__ import annotations

import copy
import json
import re
from typing import Any

from ..envs.openenv_awm_session_env import OpenEnvAWMSessionEnv
from ..utils.awm import flatten_awm_env_args
from .awm_reward import (
    RAW_TOOL_CALL_PATTERN,
    _build_reward_payload,
    _count_tool_messages,
    _reward_from_classification,
    _validate_format_and_server,
)
from .reward_base import reward


OPENENV_TO_NATIVE_TOOL_NAMES = {
    "openenv_awm_list_tools": "list_tools",
    "openenv_awm_call_tool": "call_tool",
}
OPENENV_REWARD_CLASSIFICATIONS = {
    "complete",
    "incomplete",
    "format_error",
    "agent_error",
    "server_error",
    "judge_error",
}


def _classification_from_openenv_verifier(verifier_result: Any) -> str:
    normalized_result = str(verifier_result or "judge_error").strip().lower()
    if normalized_result in OPENENV_REWARD_CLASSIFICATIONS:
        return normalized_result
    return "incomplete"


def _normalize_tool_call_name(name: Any) -> Any:
    return OPENENV_TO_NATIVE_TOOL_NAMES.get(name, name)


def _normalize_raw_tool_call(raw: str) -> str:
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        return raw

    def normalize_item(item: Any) -> Any:
        if isinstance(item, dict):
            copied = dict(item)
            if "name" in copied:
                copied["name"] = _normalize_tool_call_name(copied["name"])
            return copied
        return item

    if isinstance(data, list):
        data = [normalize_item(item) for item in data]
    else:
        data = normalize_item(data)
    return json.dumps(data, ensure_ascii=False)


def _normalize_openenv_trajectory(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(trajectory)
    for message in normalized:
        if message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = RAW_TOOL_CALL_PATTERN.sub(
                    lambda match: f"<tool_call>\n{_normalize_raw_tool_call(match.group(1))}\n</tool_call>",
                    content,
                )
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function")
                if isinstance(function, dict) and "name" in function:
                    function["name"] = _normalize_tool_call_name(function["name"])
    return normalized


def openenv_awm_early_termination_validator(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_trajectory = _normalize_openenv_trajectory(trajectory)
    return _validate_format_and_server(normalized_trajectory, require_think=False)


def openenv_awm_think_early_termination_validator(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_trajectory = _normalize_openenv_trajectory(trajectory)
    return _validate_format_and_server(normalized_trajectory, require_think=True)


async def _ensure_env_ready(
    env: OpenEnvAWMSessionEnv,
    scenario: str | None,
    task_id: int | None,
    task_idx: int | None,
    task: str | None,
    extra_info: dict[str, Any] | None,
) -> None:
    if env.runtime is not None:
        return
    env_args = flatten_awm_env_args(extra_info)
    if scenario is not None:
        env_args["scenario"] = scenario
    if task_idx is not None:
        env_args["task_idx"] = task_idx
    elif task_id is not None:
        env_args["task_id"] = task_id
    if task is not None:
        env_args["task"] = task
    await env.reset(env_args)


def _is_sql_verifier_mode(env: OpenEnvAWMSessionEnv, verifier_details: dict[str, Any]) -> bool:
    current_config = getattr(env, "current_config", None)
    mode = getattr(current_config, "verifier_mode", None) or verifier_details.get("verifier_mode")
    return str(mode or "").strip().lower() == "sql"


def _openenv_reward_from_details(
    classification: str,
    verifier_details: dict[str, Any],
    *,
    partial_credit_for_incomplete: bool = False,
) -> tuple[float, float]:
    reward_value, _ = _reward_from_classification(
        classification,
        partial_credit_for_incomplete=partial_credit_for_incomplete,
    )
    acc = 1.0 if classification == "complete" else 0.0
    return reward_value, acc


def _details_for_payload(verifier_details: dict[str, Any]) -> dict[str, Any]:
    details = dict(verifier_details)
    if "raw_result" not in details and "verify_result" in details:
        details["raw_result"] = details["verify_result"]
    error = details.get("error")
    if error and "error_message" not in details:
        details["error_message"] = error
    return details


def _build_openenv_reward_payload(
    *,
    classification: str,
    reward_value: float,
    acc: float,
    verifier_details: dict[str, Any],
    tool_call_count: int,
    valid_think: float | None = None,
    format_error_reason: str | None = None,
) -> dict[str, Any]:
    payload = _build_reward_payload(
        classification=classification,
        reward_value=reward_value,
        acc=acc,
        verifier_details=verifier_details,
        tool_call_count=tool_call_count,
        valid_think=valid_think,
        format_error_reason=format_error_reason,
    )
    payload.setdefault("valid_think", -1.0)
    payload.setdefault("reason", format_error_reason or payload.get("verifier_error", ""))
    payload.setdefault("format_error_reason", format_error_reason or "")
    payload.setdefault("verifier_execution_status", "")
    payload.setdefault("verifier_raw_result", "")
    payload.setdefault("verifier_error", "")
    return payload


@reward(name="openenv_awm_verifier_reward", env_cls=OpenEnvAWMSessionEnv, pool_size=8)
async def openenv_awm_verifier_reward(
    final_response: str,
    trajectory: list[dict[str, Any]],
    env: OpenEnvAWMSessionEnv,
    scenario: str | None = None,
    task_id: int | None = None,
    task_idx: int | None = None,
    task: str | None = None,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tool_call_count = _count_tool_messages(trajectory)
    normalized_trajectory = _normalize_openenv_trajectory(trajectory)
    validation = _validate_format_and_server(normalized_trajectory, require_think=False)
    if validation["classification"] is not None:
        reward_value, acc = _reward_from_classification(validation["classification"])
        return _build_openenv_reward_payload(
            classification=validation["classification"],
            reward_value=reward_value,
            acc=acc,
            verifier_details={},
            tool_call_count=tool_call_count,
            format_error_reason=validation["reason"] if validation["classification"] == "format_error" else None,
        )

    await _ensure_env_ready(env, scenario, task_id, task_idx, task, extra_info)
    verifier_result, verifier_details = await env.run_verifier(final_response)
    classification = _classification_from_openenv_verifier(verifier_result)
    reward_value, acc = _openenv_reward_from_details(
        classification,
        verifier_details,
        partial_credit_for_incomplete=_is_sql_verifier_mode(env, verifier_details),
    )
    return _build_openenv_reward_payload(
        classification=classification,
        reward_value=reward_value,
        acc=acc,
        verifier_details=_details_for_payload(verifier_details),
        tool_call_count=tool_call_count,
    )


@reward(name="openenv_awm_verifier_reward_think", env_cls=OpenEnvAWMSessionEnv, pool_size=8)
async def openenv_awm_verifier_reward_think(
    final_response: str,
    trajectory: list[dict[str, Any]],
    env: OpenEnvAWMSessionEnv,
    scenario: str | None = None,
    task_id: int | None = None,
    task_idx: int | None = None,
    task: str | None = None,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tool_call_count = _count_tool_messages(trajectory)
    normalized_trajectory = _normalize_openenv_trajectory(trajectory)
    validation = _validate_format_and_server(normalized_trajectory, require_think=True)
    if validation["classification"] is not None:
        reward_value, acc = _reward_from_classification(validation["classification"])
        return _build_openenv_reward_payload(
            classification=validation["classification"],
            reward_value=reward_value,
            acc=acc,
            verifier_details={},
            tool_call_count=tool_call_count,
            valid_think=validation["valid_think"],
            format_error_reason=validation["reason"] if validation["classification"] == "format_error" else None,
        )

    await _ensure_env_ready(env, scenario, task_id, task_idx, task, extra_info)
    verifier_result, verifier_details = await env.run_verifier(final_response)
    classification = _classification_from_openenv_verifier(verifier_result)
    reward_value, acc = _openenv_reward_from_details(
        classification,
        verifier_details,
        partial_credit_for_incomplete=_is_sql_verifier_mode(env, verifier_details),
    )
    return _build_openenv_reward_payload(
        classification=classification,
        reward_value=reward_value,
        acc=acc,
        verifier_details=_details_for_payload(verifier_details),
        tool_call_count=tool_call_count,
        valid_think=1.0,
    )

openenv_awm_verifier_reward.early_termination_validator = openenv_awm_early_termination_validator
openenv_awm_verifier_reward.early_termination_classifications = {"format_error", "server_error"}
openenv_awm_verifier_reward_think.early_termination_validator = openenv_awm_think_early_termination_validator
openenv_awm_verifier_reward_think.early_termination_classifications = {"format_error", "server_error"}
