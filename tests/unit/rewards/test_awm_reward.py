import pytest

from agentfly.rewards.awm_reward import (
    _classify_error_observation,
    _reward_from_classification,
    awm_early_termination_validator,
    awm_think_early_termination_validator,
)


def _assistant(content: str) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"function": {"name": "dummy", "arguments": "{}"}}] if "<tool_call>" in content else [],
    }


def _tool(content: str, info: dict | None = None) -> dict:
    message = {"role": "tool", "content": content}
    if info is not None:
        message["info"] = info
    return message


LIST_CALL = '<tool_call>{"name":"list_tools","arguments":null}</tool_call>'
LIST_OBS = "Available MCP tools (1 total):\n\n1. search_products\n   Parameters: None"


def test_reward_values_match_awm_definition():
    assert _reward_from_classification("complete") == (1.0, 1.0)
    assert _reward_from_classification("format_error") == (-1.0, 0.0)
    assert _reward_from_classification("server_error") == (0.0, 0.0)
    assert _reward_from_classification("incomplete") == (0.0, 0.0)
    assert _reward_from_classification("incomplete", partial_credit_for_incomplete=True) == (0.1, 0.0)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"openenv_reward_type": "tool_not_found"}, "format_error"),
        ({"openenv_reward_type": "invalid_args"}, "format_error"),
        ({"openenv_observation": {"reward_type": "timeout"}}, "server_error"),
        ({"openenv_reward_type": "tool_call_ok"}, None),
    ],
)
def test_structured_openenv_metadata_precedes_text_fallback(metadata, expected):
    assert _classify_error_observation("Error: business operation failed", metadata) == expected


def test_text_failed_without_error_prefix_does_not_trigger_server_error():
    assert _classify_error_observation("business operation failed but returned normally") is None


def test_tool_name_format_error_precedes_error_observation():
    trajectory = [
        _assistant(LIST_CALL),
        _tool(LIST_OBS),
        _assistant(
            '<tool_call>{"name":"call_tool","arguments":{"tool_name":"missing_tool","arguments":{}}}</tool_call>'
        ),
        _tool("Error: Internal Server Error while handling missing tool"),
    ]

    result = awm_early_termination_validator(trajectory)

    assert result["classification"] == "format_error"
    assert result["reason"] == "tool_call_1_mcp_tool_not_listed"


def test_server_error_returns_zero_classification_when_structured():
    trajectory = [
        _assistant(LIST_CALL),
        _tool(
            "Error: Sub-environment is not running. Call reset() first.",
            info={"openenv_reward_type": "server_error"},
        ),
    ]

    result = awm_early_termination_validator(trajectory)

    assert result["classification"] == "server_error"
    assert _reward_from_classification(result["classification"]) == (0.0, 0.0)


def test_think_validator_requires_non_empty_think():
    trajectory = [
        _assistant(LIST_CALL),
        _tool(LIST_OBS),
    ]

    result = awm_think_early_termination_validator(trajectory)

    assert result["classification"] == "format_error"
    assert result["reason"] == "assistant_message_0_missing_think"
    assert result["valid_think"] == 0.0


def test_think_validator_accepts_non_empty_think_before_tool_call():
    trajectory = [
        _assistant('<think>inspect tools</think>\n<tool_call>{"name":"list_tools","arguments":null}</tool_call>'),
        _tool(LIST_OBS),
    ]

    result = awm_think_early_termination_validator(trajectory)

    assert result["classification"] is None
    assert result["valid_think"] is True
