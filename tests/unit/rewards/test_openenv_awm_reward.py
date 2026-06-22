import asyncio
from types import SimpleNamespace

import pytest

from agentfly.rewards.openenv_awm_reward import (
    _normalize_openenv_trajectory,
    openenv_awm_verifier_reward,
)


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": [{}] if "<tool_call>" in content else []}


def _tool(content: str) -> dict:
    return {"role": "tool", "content": content}


class FakeOpenEnv:
    runtime = object()
    current_config = SimpleNamespace(verifier_mode="sql")

    async def run_verifier(self, final_answer=None):
        return "incomplete", {
            "reward": 0.0,
            "verifier_mode": "sql",
            "verify_result": {"result": "others"},
        }


def test_openenv_awm_reward_sql_incomplete_gets_partial_reward():
    trajectory = [
        _assistant('<tool_call>{"name":"list_tools","arguments":null}</tool_call>'),
        _tool("Available MCP tools (1 total):\n\n1. search_products\n   Parameters: None"),
        _assistant(
            '<tool_call>{"name":"call_tool","arguments":{"tool_name":"search_products","arguments":"{\\"query\\":\\"headphones\\"}"}}</tool_call>'
        ),
        _tool("[]"),
    ]

    async def run_case():
        return await openenv_awm_verifier_reward._func(
            final_response="done",
            trajectory=trajectory,
            env=FakeOpenEnv(),
        )

    result = asyncio.run(run_case())

    assert result["classification"] == "incomplete"
    assert result["reward"] == 0.1
    assert result["acc"] == 0.0
    assert result["verifier_raw_result"] == '{"result": "others"}'



def test_openenv_awm_reward_maps_others_to_incomplete_metrics():
    class OthersOpenEnv:
        runtime = object()
        current_config = SimpleNamespace(verifier_mode="code")

        async def run_verifier(self, final_answer=None):
            return "others", {
                "reward": 0.0,
                "verifier_mode": "code",
                "verify_result": {"result": "others"},
            }

    trajectory = [
        _assistant('<tool_call>{"name":"list_tools","arguments":null}</tool_call>'),
        _tool("Available MCP tools (1 total):\n\n1. search_products\n   Parameters: None"),
        _assistant(
            '<tool_call>{"name":"call_tool","arguments":{"tool_name":"search_products","arguments":"{\\"query\\":\\"headphones\\"}"}}</tool_call>'
        ),
        _tool("[]"),
    ]

    async def run_case():
        return await openenv_awm_verifier_reward._func(
            final_response="done",
            trajectory=trajectory,
            env=OthersOpenEnv(),
        )

    result = asyncio.run(run_case())

    assert result["classification"] == "incomplete"
    assert result["reward"] == 0.0
    assert result["acc"] == 0.0
    assert result["incomplete"] == 1.0
    assert result["complete"] == 0.0
    assert result["verifier_raw_result"] == '{"result": "others"}'


def test_openenv_awm_reward_maps_partial_label_to_incomplete():
    class PartialOpenEnv:
        runtime = object()
        current_config = SimpleNamespace(verifier_mode="code")

        async def run_verifier(self, final_answer=None):
            return "partial_success", {
                "reward": 0.0,
                "verifier_mode": "code",
                "verify_result": {"result": "partial_success"},
            }

    trajectory = [
        _assistant('<tool_call>{"name":"list_tools","arguments":null}</tool_call>'),
        _tool("Available MCP tools (1 total):\n\n1. search_products\n   Parameters: None"),
        _assistant(
            '<tool_call>{"name":"call_tool","arguments":{"tool_name":"search_products","arguments":"{\\"query\\":\\"headphones\\"}"}}</tool_call>'
        ),
        _tool("[]"),
    ]

    async def run_case():
        return await openenv_awm_verifier_reward._func(
            final_response="done",
            trajectory=trajectory,
            env=PartialOpenEnv(),
        )

    result = asyncio.run(run_case())

    assert result["classification"] == "incomplete"
    assert result["reward"] == 0.0
    assert "partial_success" not in result
    assert result["incomplete"] == 1.0


def test_openenv_awm_reward_agent_tool_error_precedes_server_error():
    class ShouldNotVerifyOpenEnv:
        runtime = object()

        async def run_verifier(self, final_answer=None):
            raise AssertionError("format validation should short-circuit verifier")

    trajectory = [
        _assistant('<tool_call>{"name":"list_tools","arguments":null}</tool_call>'),
        _tool("Available MCP tools (1 total):\n\n1. search_products\n   Parameters: None"),
        _assistant(
            '<tool_call>{"name":"call_tool","arguments":{"tool_name":"add_product_to_cart","arguments":"{\\"product_id\\":1}"}}</tool_call>'
        ),
        _tool("Error: Unknown tool: add_product_to_cart"),
    ]

    async def run_case():
        return await openenv_awm_verifier_reward._func(
            final_response="done",
            trajectory=trajectory,
            env=ShouldNotVerifyOpenEnv(),
        )

    result = asyncio.run(run_case())

    assert result["classification"] == "format_error"
    assert result["reward"] == -1.0
    assert result["format_error"] == 1.0
    assert result["server_error"] == 0.0
    assert result["format_error_reason"] == "tool_call_1_mcp_tool_not_listed"



def test_openenv_awm_reward_invalid_args_error_is_format_error():
    class ShouldNotVerifyOpenEnv:
        runtime = object()

        async def run_verifier(self, final_answer=None):
            raise AssertionError("format validation should short-circuit verifier")

    trajectory = [
        _assistant('<tool_call>{"name":"list_tools","arguments":null}</tool_call>'),
        _tool("Available MCP tools (1 total):\n\n1. search_products\n   Parameters: None"),
        {
            "role": "assistant",
            "content": '<tool_call>{"name":"call_tool","arguments":{"tool_name":"search_products","arguments":{"query":{"not":"a string"}}}}</tool_call>',
            "tool_calls": [
                {
                    "function": {
                        "name": "call_tool",
                        "arguments": '{"tool_name":"search_products","arguments":{"query":{"not":"a string"}}}',
                    }
                }
            ],
        },
        _tool("Error: Input validation error: {'not': 'a string'} is not of type 'string'"),
    ]

    async def run_case():
        return await openenv_awm_verifier_reward._func(
            final_response="done",
            trajectory=trajectory,
            env=ShouldNotVerifyOpenEnv(),
        )

    result = asyncio.run(run_case())

    assert result["classification"] == "format_error"
    assert result["reward"] == -1.0
    assert result["format_error"] == 1.0
    assert result["server_error"] == 0.0
    assert result["format_error_reason"] == "tool_call_format_error_server_response"

def test_openenv_awm_reward_format_error_still_penalized():
    async def run_case():
        return await openenv_awm_verifier_reward._func(
            final_response="done",
            trajectory=[_assistant("final answer without tool calls")],
            env=FakeOpenEnv(),
        )

    result = asyncio.run(run_case())

    assert result["classification"] == "format_error"
    assert result["reward"] == -1.0
    assert result["format_error"] == 1.0



def test_openenv_awm_reward_prefers_structured_tool_call_ok_over_error_text():
    class OthersOpenEnv:
        runtime = object()
        current_config = SimpleNamespace(verifier_mode="code")

        async def run_verifier(self, final_answer=None):
            return "others", {
                "reward": 0.0,
                "verifier_mode": "code",
                "verify_result": {"result": "others"},
            }

    trajectory = [
        _assistant('<tool_call>{"name":"list_tools","arguments":null}</tool_call>'),
        _tool("Available MCP tools (1 total):\n\n1. search_products\n   Parameters: None"),
        _assistant(
            '<tool_call>{"name":"call_tool","arguments":{"tool_name":"search_products","arguments":"{\\"query\\":\\"missing\\"}"}}</tool_call>'
        ),
        {
            "role": "tool",
            "content": "Error: no product matched the query; search failed",
            "info": {
                "openenv_reward_type": "tool_call_ok",
                "openenv_error": None,
                "openenv_observation": {"reward_type": "tool_call_ok"},
            },
        },
    ]

    async def run_case():
        return await openenv_awm_verifier_reward._func(
            final_response="done",
            trajectory=trajectory,
            env=OthersOpenEnv(),
        )

    result = asyncio.run(run_case())

    assert result["classification"] == "incomplete"
    assert result["reward"] == 0.0
    assert result["server_error"] == 0.0
    assert result["incomplete"] == 1.0

def test_openenv_awm_reward_server_error_returns_zero_without_verifier():
    class ShouldNotVerifyOpenEnv:
        runtime = object()
        verifier_called = False

        async def run_verifier(self, final_answer=None):
            self.verifier_called = True
            raise AssertionError("server_error validation should short-circuit verifier")

    env = ShouldNotVerifyOpenEnv()
    trajectory = [
        _assistant('<tool_call>{"name":"list_tools","arguments":null}</tool_call>'),
        _tool("Error: Internal Server Error while processing request"),
    ]

    async def run_case():
        return await openenv_awm_verifier_reward._func(
            final_response="done",
            trajectory=trajectory,
            env=env,
        )

    result = asyncio.run(run_case())

    assert result["classification"] == "server_error"
    assert result["reward"] == 0.0
    assert result["acc"] == 0.0
    assert result["server_error"] == 1.0
    assert result["format_error"] == 0.0
    assert env.verifier_called is False


def test_openenv_awm_reward_missing_server_response_returns_zero():
    trajectory = [
        _assistant('<tool_call>{"name":"list_tools","arguments":null}</tool_call>'),
    ]

    async def run_case():
        return await openenv_awm_verifier_reward._func(
            final_response="done",
            trajectory=trajectory,
            env=FakeOpenEnv(),
        )

    result = asyncio.run(run_case())

    assert result["classification"] == "server_error"
    assert result["reward"] == 0.0
    assert result["server_error"] == 1.0


def test_normalize_openenv_tool_names_for_backward_compatibility():
    trajectory = [
        _assistant('<tool_call>{"name":"openenv_awm_list_tools","arguments":null}</tool_call>'),
    ]

    normalized = _normalize_openenv_trajectory(trajectory)

    assert "openenv_awm_list_tools" not in normalized[0]["content"]
    assert '"name": "list_tools"' in normalized[0]["content"]
