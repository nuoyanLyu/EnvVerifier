import json
import asyncio

from agentfly.agents.agent_base import BaseAgent
from agentfly.agents.auto import AutoAgent
from agentfly.agents.specialized.hf_agent import HFAgent, drop_awm_xml_prompt_tools


def test_awm_xml_parser_maps_native_list_tools_to_agentfly_tool():
    agent = HFAgent.__new__(HFAgent)
    agent._use_awm_xml_tool_parser = True

    payload = {"name": "list_tools", "arguments": None}
    messages = agent.parse([f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"])

    assert messages[0]["tool_calls"] == [
        {
            "id": "awm_xml_call_0_0",
            "type": "function",
            "function": {"name": "awm_list_tools", "arguments": "{}"},
        }
    ]
    assert messages[0]["status"] == "continue"


def test_awm_xml_parser_maps_native_call_tool_arguments_to_agentfly_tool():
    agent = HFAgent.__new__(HFAgent)
    agent._use_awm_xml_tool_parser = True

    payload = {
        "name": "call_tool",
        "arguments": {
            "tool_name": "search_products",
            "arguments": json.dumps({"query": "laptop"}),
        },
    }
    messages = agent.parse([f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"])

    assert messages[0]["tool_calls"] == [
        {
            "id": "awm_xml_call_0_0",
            "type": "function",
            "function": {
                "name": "awm_call_tool",
                "arguments": '{"tool_name": "search_products", "arguments": {"query": "laptop"}}',
            },
        }
    ]
    assert messages[0]["status"] == "continue"


def test_awm_xml_generation_kwargs_drop_template_tools():
    assert drop_awm_xml_prompt_tools({"tools": ["schema"], "max_tokens": 16}) == {"max_tokens": 16}


def test_awm_prompt_mode_no_think_resolves_native_xml_prompt_defaults():
    config = AutoAgent._apply_awm_prompt_mode_defaults(
        {
            "agent_type": "code",
            "tools": ["google_search", "answer"],
            "backend": "async_verl",
            "awm_prompt_mode": "no_think",
        }
    )

    assert config["agent_type"] == "hf"
    assert config["tools"] == ["awm_list_tools", "awm_call_tool"]
    assert config["tool_parser_name"] == "awm_xml"
    assert config["reward_name"] == "awm_verifier_reward"
    prompt = AutoAgent._resolve_system_prompt(config)
    assert "<tool_call>" in prompt
    assert "list_tools" in prompt
    assert "<think>" not in prompt


def test_awm_prompt_mode_think_resolves_native_xml_prompt_defaults():
    config = AutoAgent._apply_awm_prompt_mode_defaults(
        {
            "agent_type": "code",
            "tools": ["google_search", "answer"],
            "backend": "async_verl",
            "awm_prompt_mode": "think",
        }
    )

    assert config["agent_type"] == "hf"
    assert config["tools"] == ["awm_list_tools", "awm_call_tool"]
    assert config["tool_parser_name"] == "awm_xml"
    assert config["reward_name"] == "awm_verifier_reward_think"
    prompt = AutoAgent._resolve_system_prompt(config)
    assert "<tool_call>" in prompt
    assert "list_tools" in prompt
    assert "<think>" in prompt


def test_awm_xml_generate_async_drops_template_tools(monkeypatch):
    observed_kwargs = {}

    async def fake_generate_async(self, messages_list_or_inputs, **kwargs):
        observed_kwargs.update(kwargs)
        return ["ok"]

    monkeypatch.setattr(BaseAgent, "generate_async", fake_generate_async)
    agent = HFAgent.__new__(HFAgent)
    agent._use_awm_xml_tool_parser = True

    result = asyncio.run(HFAgent.generate_async(agent, [], tools=["schema"], max_tokens=16))

    assert result == ["ok"]
    assert observed_kwargs == {"max_tokens": 16}
