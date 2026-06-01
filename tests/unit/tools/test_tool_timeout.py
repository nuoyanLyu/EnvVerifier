import asyncio
import os
import sys
import time
import types

import pytest

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/agentfly-test-cache")


def _install_test_stubs() -> None:
    class _BaseEnv:
        async def start(self):
            return None

        async def reset(self):
            return None

        async def step(self, action: str):
            return action

        async def aclose(self):
            return None

        def close(self):
            return None

        @staticmethod
        async def acquire():
            return _BaseEnv()

    class _EnvironmentManager:
        @classmethod
        async def start(cls, *args, **kwargs):
            return None

        @classmethod
        async def acquire(cls, *args, **kwargs):
            return _BaseEnv()

        @classmethod
        async def release(cls, *args, **kwargs):
            return None

        @classmethod
        async def reset(cls, *args, **kwargs):
            return None

    envs_module = types.ModuleType("agentfly.envs")
    env_base_module = types.ModuleType("agentfly.envs.env_base")
    env_base_module.BaseEnv = _BaseEnv
    manager_module = types.ModuleType("agentfly.envs.manager")
    env_manager_module = types.ModuleType("agentfly.envs.manager.env_manager")
    env_manager_module.EnvironmentManager = _EnvironmentManager

    sys.modules.setdefault("agentfly.envs", envs_module)
    sys.modules.setdefault("agentfly.envs.env_base", env_base_module)
    sys.modules.setdefault("agentfly.envs.manager", manager_module)
    sys.modules.setdefault("agentfly.envs.manager.env_manager", env_manager_module)

    async def _async_stub(*args, **kwargs):
        return "stub"

    def _sync_stub(*args, **kwargs):
        return "stub"

    class _CodeInterpreterTool:
        pass

    tool_module_stubs = {
        "agentfly.tools.src.alfworld.tools": {
            "alfworld_get_admissible_commands": _async_stub,
            "alfworld_get_task_objective": _async_stub,
            "alfworld_reset": _async_stub,
            "alfworld_step": _async_stub,
        },
        "agentfly.tools.src.chess.tools": {
            "chess_get_legal_moves": _async_stub,
            "chess_get_state": _async_stub,
            "chess_move": _async_stub,
        },
        "agentfly.tools.src.code.tools": {
            "CodeInterpreterTool": _CodeInterpreterTool,
            "code_interpreter": _async_stub,
        },
        "agentfly.tools.src.react.tools": {
            "answer_math": _sync_stub,
            "answer_qa": _sync_stub,
        },
        "agentfly.tools.src.scienceworld.tools": {
            "scienceworld_explorer": _async_stub,
        },
        "agentfly.tools.src.search.async_dense_retriever": {
            "asyncdense_retrieve": _async_stub,
        },
        "agentfly.tools.src.search.dense_retriever": {
            "dense_retrieve": _async_stub,
        },
        "agentfly.tools.src.search.google_search": {
            "google_search_serper": _async_stub,
        },
        "agentfly.tools.src.ui.tools": {
            "pyautogui_code_generator": _sync_stub,
        },
        "agentfly.tools.src.webshop.tools": {
            "webshop_browser": _async_stub,
        },
    }

    for module_name, attrs in tool_module_stubs.items():
        module = types.ModuleType(module_name)
        for attr_name, attr_value in attrs.items():
            setattr(module, attr_name, attr_value)
        sys.modules.setdefault(module_name, module)


_install_test_stubs()

from agentfly.tools import tool
from agentfly.tools.tool_base import submit_tool_call


def _collect_events(storage):
    def hook(event, **kwargs):
        storage.append((event, kwargs))

    return hook


def test_sync_tool_timeout_uses_subprocess_and_recovers():
    @tool(name="sleep_tool_timeout_test", description="sleep for timeout tests")
    def sleep_tool(seconds: float):
        time.sleep(seconds)
        return f"slept {seconds}"

    @tool(name="fast_tool_after_timeout_test", description="fast follow-up tool")
    def fast_tool(value: str):
        return value

    timeout_events = []
    async def run_timeout_call():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                submit_tool_call(
                    "sleep_tool_timeout_test",
                    {"seconds": 5},
                    diag_hook=_collect_events(timeout_events),
                ),
                timeout=0.5,
            )

    asyncio.run(run_timeout_call())

    assert ("tool_dispatch", {"exec_mode": "sync_subprocess"}) in timeout_events
    assert any(event == "tool_subprocess_start" for event, _ in timeout_events)
    assert any(event == "tool_subprocess_terminated" for event, _ in timeout_events)

    recovery_events = []
    async def run_recovery_call():
        return await submit_tool_call(
            "fast_tool_after_timeout_test",
            {"value": "ok"},
            diag_hook=_collect_events(recovery_events),
        )

    result = asyncio.run(run_recovery_call())
    assert result["observation"] == "ok"
    assert ("tool_dispatch", {"exec_mode": "sync_subprocess"}) in recovery_events


def test_calculator_timeout_can_be_cancelled():
    events = []

    async def run_call():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                submit_tool_call(
                    "calculator",
                    {"expression": "2^(2**100 - 1)/2"},
                    diag_hook=_collect_events(events),
                ),
                timeout=1.0,
            )

    asyncio.run(run_call())

    assert ("tool_dispatch", {"exec_mode": "sync_subprocess"}) in events
    assert any(event == "tool_subprocess_terminated" for event, _ in events)


def test_async_tool_stays_on_direct_path():
    @tool(name="async_echo_timeout_test", description="async direct execution test")
    async def async_echo(value: str):
        await asyncio.sleep(0.01)
        return value

    events = []

    async def run_call():
        return await submit_tool_call(
            "async_echo_timeout_test",
            {"value": "pong"},
            diag_hook=_collect_events(events),
        )

    result = asyncio.run(run_call())

    assert result["observation"] == "pong"
    assert ("tool_dispatch", {"exec_mode": "async_direct"}) in events
    assert not any(event.startswith("tool_subprocess_") for event, _ in events)
