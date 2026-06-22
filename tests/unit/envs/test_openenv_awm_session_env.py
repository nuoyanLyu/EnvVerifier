import json
import asyncio

import pytest

from agentfly.envs.openenv_awm_session_env import OpenEnvAWMSessionEnv, OpenEnvAWMSessionRuntime


class FakeWebSocket:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []
        self.closed = False

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def recv(self):
        return json.dumps(self.responses.pop(0))

    async def close(self):
        self.closed = True


def test_openenv_awm_runtime_payloads(monkeypatch):
    responses = [
        {"type": "observation", "data": {"observation": {"reward_type": "reset_ok", "task": "demo", "num_tools": 1}}},
        {
            "type": "observation",
            "data": {
                "observation": {
                    "tools": [
                        {
                            "name": "list_product_categories",
                            "description": "List categories",
                            "input_schema": {"type": "object", "properties": {}},
                        }
                    ]
                }
            },
        },
        {"type": "observation", "data": {"observation": {"reward_type": "tool_call_ok", "tool_result": "categories"}}},
        {"type": "observation", "data": {"reward": 0.0, "observation": {"reward_type": "tool_call_ok", "tool_result": "categories"}}},
        {
            "type": "observation",
            "data": {
                "reward": 0.1,
                "observation": {
                    "reward_type": "incomplete",
                    "verify_result": {"result": "others"},
                },
            },
        },
        {"type": "observation", "data": {"done": True, "observation": {"reward_type": "episode_done"}}},
        {"type": "observation", "data": {}},
    ]
    fake_ws = FakeWebSocket(responses)

    async def fake_connect(*args, **kwargs):
        return fake_ws

    import websockets

    monkeypatch.setattr(websockets, "connect", fake_connect)

    async def run_case():
        env = OpenEnvAWMSessionEnv()
        await env.reset({"scenario": "e_commerce_33", "task_id": 3, "openenv_awm_base_url": "http://server:8899"})
        assert env.current_config.task_idx == 3

        tools_text = await env.list_tools_text()
        assert "list_product_categories" in tools_text
        assert "call_tool" in tools_text
        assert "openenv_awm_call_tool" not in tools_text

        result = await env.call_tool("list_product_categories", {})
        assert result == "categories"

        details = await env.call_tool_with_details("list_product_categories", {})
        assert details["observation"] == "categories"
        assert details["openenv_reward_type"] == "tool_call_ok"
        assert details["openenv_observation"]["reward_type"] == "tool_call_ok"

        verifier_result, verifier_details = await env.run_verifier("done")
        assert verifier_result == "incomplete"
        assert verifier_details["reward"] == 0.1
        assert verifier_details["verify_result"] == {"result": "others"}

        await env.aclose()

    asyncio.run(run_case())

    assert fake_ws.sent[0] == {
        "type": "reset",
        "data": {"scenario": "e_commerce_33", "task_idx": 3},
    }
    assert fake_ws.sent[1] == {"type": "step", "data": {"type": "list_tools"}}
    assert fake_ws.sent[2] == {
        "type": "step",
        "data": {
            "type": "call_tool",
            "tool_name": "list_product_categories",
            "arguments": {},
        },
    }
    assert fake_ws.sent[3] == {
        "type": "step",
        "data": {
            "type": "call_tool",
            "tool_name": "list_product_categories",
            "arguments": {},
        },
    }
    assert fake_ws.sent[4] == {
        "type": "step",
        "data": {
            "type": "call_tool",
            "tool_name": "verify",
            "arguments": {"verifier_mode": "code", "final_answer": "done"},
        },
    }
    assert fake_ws.sent[5]["data"]["tool_name"] == "done"
    assert fake_ws.closed is True


def test_openenv_awm_runtime_rejects_websocket_errors():
    runtime = OpenEnvAWMSessionRuntime.__new__(OpenEnvAWMSessionRuntime)
    runtime._ws = FakeWebSocket([{"type": "error", "data": {"message": "boom"}}])
    runtime.config = type("Config", (), {"message_timeout_s": 1})()

    async def run_case():
        with pytest.raises(RuntimeError, match="boom"):
            await runtime._request({"type": "state"})

    asyncio.run(run_case())

def _reset_ok_responses():
    return [
        {"type": "observation", "data": {"observation": {"reward_type": "reset_ok", "task": "demo", "num_tools": 1}}},
        {
            "type": "observation",
            "data": {"observation": {"tools": [
                {"name": "list_product_categories", "description": "List", "input_schema": {"type": "object", "properties": {}}}
            ]}},
        },
    ]


def test_openenv_awm_reset_retries_on_startup_race(monkeypatch):
    # First connection: reset ok, but list_tools reports the startup race.
    # Second connection: clean start. reset() should transparently retry.
    conn1 = FakeWebSocket([
        {"type": "observation", "data": {"observation": {"reward_type": "reset_ok", "num_tools": 0}}},
        {"type": "observation", "data": {"observation": {"tools": [], "error": "Sub-environment is not running. Call reset() first."}}},
        {"type": "observation", "data": {"observation": {"reward_type": "episode_done"}}},
        {"type": "observation", "data": {}},
    ])
    conn2 = FakeWebSocket(_reset_ok_responses())
    conns = [conn1, conn2]

    async def fake_connect(*args, **kwargs):
        return conns.pop(0)

    import websockets
    monkeypatch.setattr(websockets, "connect", fake_connect)

    async def run_case():
        env = OpenEnvAWMSessionEnv()
        await env.reset({
            "scenario": "e_commerce_33", "task_id": 3,
            "openenv_awm_reset_retries": 3, "openenv_awm_reset_retry_backoff_s": 0,
        })
        tools_text = await env.list_tools_text()
        assert "list_product_categories" in tools_text
        await env.aclose()

    asyncio.run(run_case())
    assert conn1.closed is True
    assert not conns


def test_openenv_awm_reset_does_not_retry_non_startup_error(monkeypatch):
    # A genuine reset error (not the startup race) must propagate without retry.
    conn1 = FakeWebSocket([
        {"type": "observation", "data": {"observation": {"reward_type": "reset_error", "error": "Scenario not found"}}},
        {"type": "observation", "data": {"observation": {"reward_type": "episode_done"}}},
        {"type": "observation", "data": {}},
    ])
    conns = [conn1]

    async def fake_connect(*args, **kwargs):
        return conns.pop(0)

    import websockets
    monkeypatch.setattr(websockets, "connect", fake_connect)

    async def run_case():
        env = OpenEnvAWMSessionEnv()
        with pytest.raises(RuntimeError, match="reset failed|Scenario not found"):
            await env.reset({
                "scenario": "missing", "task_id": 0,
                "openenv_awm_reset_retries": 3, "openenv_awm_reset_retry_backoff_s": 0,
            })

    asyncio.run(run_case())
    assert not conns
