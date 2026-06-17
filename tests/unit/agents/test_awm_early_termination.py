from agentfly.agents.chain.chain_base import ChainRollout, Node
from agentfly.agents.utils.messages import Messages
from agentfly.rewards.openenv_awm_reward import openenv_awm_verifier_reward


class _OtherReward:
    name = "math_reward"


class _CustomEarlyTerminationReward:
    name = "custom_reward"
    early_termination_classifications = {"custom_stop"}

    @staticmethod
    def early_termination_validator(trajectory):
        return {"classification": "custom_stop", "reason": "custom_reason"}


def _node_with_tool_error() -> Node:
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": '<tool_call>{"name":"list_tools","arguments":null}</tool_call>',
            "tool_calls": [
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "list_tools", "arguments": "null"},
                }
            ],
        },
        {
            "role": "tool",
            "content": "Available MCP tools (1 total):\n\n1. search_products\n   Parameters: None",
            "tool_name": "list_tools",
        },
        {
            "role": "assistant",
            "content": '<tool_call>{"name":"call_tool","arguments":{"tool_name":"search_products","arguments":"{\\"query\\":\\"headphones\\",\\"max_price\\":null}"}}</tool_call>',
            "tool_calls": [
                {
                    "id": None,
                    "type": "function",
                    "function": {
                        "name": "call_tool",
                        "arguments": '{"tool_name":"search_products","arguments":"{\\"query\\":\\"headphones\\",\\"max_price\\":null}"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "Error: Input validation error: None is not of type 'number'",
            "tool_name": "call_tool",
        },
    ]
    return Node(messages=Messages({"messages": messages}))


def test_openenv_awm_reward_declares_early_termination_policy():
    assert callable(openenv_awm_verifier_reward.early_termination_validator)
    assert openenv_awm_verifier_reward.early_termination_classifications == {"format_error", "server_error"}


def test_reward_declared_early_termination_stops_after_invalid_args_format_error():
    rollout = ChainRollout()
    rollout._reward_fn = openenv_awm_verifier_reward
    node = _node_with_tool_error()

    rollout._apply_reward_early_termination(node, chain_id="chain", group_id="group", depth=1)

    assert node.is_terminal is True
    assert node.observation_code == "format_error"


def test_custom_reward_can_declare_early_termination_policy():
    rollout = ChainRollout()
    rollout._reward_fn = _CustomEarlyTerminationReward()
    node = _node_with_tool_error()

    rollout._apply_reward_early_termination(node, chain_id="chain", group_id="group", depth=1)

    assert node.is_terminal is True
    assert node.observation_code == "custom_stop"


def test_reward_without_policy_does_not_use_early_termination():
    rollout = ChainRollout()
    rollout._reward_fn = _OtherReward()
    node = _node_with_tool_error()

    rollout._apply_reward_early_termination(node, chain_id="chain", group_id="group", depth=1)

    assert node.is_terminal is False
    assert node.observation_code is None
