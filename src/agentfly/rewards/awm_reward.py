from __future__ import annotations

import json
import re
from typing import Any

from ..envs.awm_session_env import AWMSessionEnv
from ..utils.awm import extract_text_from_message, flatten_awm_env_args
from .reward_base import reward


THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _count_tool_messages(trajectory: list[dict[str, Any]]) -> int:
    return sum(1 for message in trajectory if message.get("role") == "tool")


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
        if not THINK_PATTERN.search(text):
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
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "reward": reward_value,
        "acc": acc,
        "classification": classification,
        "tool_call_count": float(tool_call_count),
        "complete": 1.0 if classification == "complete" else 0.0,
        "incomplete": 1.0 if classification == "incomplete" else 0.0,
        "agent_error": 1.0 if classification == "agent_error" else 0.0,
        "server_error": 1.0 if classification == "server_error" else 0.0,
        "judge_error": 1.0 if classification == "judge_error" else 0.0,
    }
    if valid_think is not None:
        payload["valid_think"] = valid_think

    execution_status = verifier_details.get("execution_status")
    if execution_status is not None:
        payload["verifier_execution_status"] = str(execution_status)
    raw_result = verifier_details.get("raw_result")
    if raw_result is not None:
        try:
            payload["verifier_raw_result"] = json.dumps(raw_result, ensure_ascii=False)
        except TypeError:
            payload["verifier_raw_result"] = str(raw_result)
    error_message = verifier_details.get("error") or verifier_details.get("error_message")
    if error_message:
        payload["verifier_error"] = str(error_message)

    return payload


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
    await _ensure_env_ready(env, scenario, task_id, task, extra_info)
    tool_call_count = _count_tool_messages(trajectory)
    verifier_result, verifier_details = await env.run_verifier(final_response)
    if verifier_result == "complete":
        classification = "complete"
    elif verifier_result == "judge_error":
        classification = "judge_error"
    else:
        classification = env.classify_trajectory_issue(trajectory) or "incomplete"

    reward_value = 1.0 if classification == "complete" else 0.0
    acc = 1.0 if classification == "complete" else 0.0
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
    await _ensure_env_ready(env, scenario, task_id, task, extra_info)
    tool_call_count = _count_tool_messages(trajectory)
    verifier_result, verifier_details = await env.run_verifier(final_response)
    if verifier_result == "complete":
        classification = "complete"
    elif verifier_result == "judge_error":
        classification = "judge_error"
    else:
        classification = env.classify_trajectory_issue(trajectory) or "incomplete"

    think_analysis = _analyze_think_trajectory(trajectory)
    valid_think = think_analysis["valid_think"]
    if classification == "complete" and valid_think:
        reward_value = 1.0
        acc = 1.0
    elif classification == "complete" and not valid_think:
        reward_value = 0.5
        acc = 1.0
    elif classification == "incomplete" and valid_think:
        reward_value = 0.1
        acc = 0.0
    else:
        reward_value = 0.0
        acc = 0.0

    return _build_reward_payload(
        classification=classification,
        reward_value=reward_value,
        acc=acc,
        verifier_details=verifier_details,
        tool_call_count=tool_call_count,
        valid_think=1.0 if valid_think else 0.0,
    )
