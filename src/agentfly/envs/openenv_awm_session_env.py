from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from .env_base import BaseEnv
from ..utils.awm import coerce_task_id, flatten_awm_env_args, format_mcp_tools, parse_jsonish_arguments


DEFAULT_OPENENV_AWM_BASE_URL = "http://127.0.0.1:8899"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_ws_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    url = url.replace("http://", "ws://").replace("https://", "wss://")
    if not url.endswith("/ws"):
        url += "/ws"
    return url


# Substrings identifying the OpenEnv reset->list_tools startup race: the sub-env
# subprocess reports ready (TCP port up) but its MCP server is not yet usable, so
# the first list_tools/call fails. A fresh session (new subprocess) almost always
# succeeds. This is a client-side workaround; the underlying OpenEnv readiness
# check (_wait_for_ready) is not modified. See plan/ server checklist.
_RETRYABLE_STARTUP_MARKERS = (
    "sub-environment is not running",
    "call reset() first",
    "list_tools returned no tools",
    "connectionclosed",
    "mcp connection not established",
)


def _is_retryable_startup_error(message: str) -> bool:
    lowered = str(message).lower()
    return any(marker in lowered for marker in _RETRYABLE_STARTUP_MARKERS)


@dataclass
class OpenEnvAWMSessionConfig:
    scenario: str
    task_idx: int
    task: str | None
    base_url: str
    verifier_mode: str = "code"
    keep_session: bool = False
    connect_timeout_s: float = 20.0
    message_timeout_s: float = 60.0
    max_message_mb: int = 100
    reset_retries: int = 2
    reset_retry_backoff_s: float = 0.5


class OpenEnvAWMSessionRuntime:
    def __init__(self, config: OpenEnvAWMSessionConfig):
        self.config = config
        self.ws_url = _to_ws_url(config.base_url)
        self._ws: Any = None
        self._tools_cache: list[dict[str, Any]] | None = None
        self._tools_text_cache: str | None = None
        self.reset_observation: dict[str, Any] | None = None
        self.last_verify_observation: dict[str, Any] | None = None
        self.last_done_observation: dict[str, Any] | None = None

    async def start(self) -> None:
        try:
            import websockets
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "OpenEnv AWM client requires the 'websockets' package in the AgentFly training environment."
            ) from exc

        self._ws = await websockets.connect(
            self.ws_url,
            open_timeout=self.config.connect_timeout_s,
            max_size=self.config.max_message_mb * 1024 * 1024,
        )

        reset_payload: dict[str, Any] = {
            "type": "reset",
            "data": {
                "scenario": self.config.scenario,
                "task_idx": self.config.task_idx,
            },
        }
        if self.config.task:
            reset_payload["data"]["task"] = self.config.task
        reset_result = await self._request(reset_payload)
        reset_obs = self._observation(reset_result)
        self.reset_observation = reset_obs
        if reset_obs.get("error"):
            raise RuntimeError(f"OpenEnv AWM reset failed: {reset_obs.get('error')}")

    async def list_tools(self) -> list[dict[str, Any]]:
        if self._tools_cache is None:
            result = await self._request({"type": "step", "data": {"type": "list_tools"}})
            obs = self._observation(result)
            if obs.get("error"):
                raise RuntimeError(f"OpenEnv AWM list_tools failed: {obs.get('error')}")
            tools = obs.get("tools", [])
            if not isinstance(tools, list):
                raise RuntimeError(f"OpenEnv AWM list_tools returned unexpected observation: {obs}")
            self._tools_cache = [tool for tool in tools if isinstance(tool, dict)]
        return list(self._tools_cache)

    async def list_tools_text(self) -> str:
        if self._tools_text_cache is None:
            text = format_mcp_tools(await self.list_tools())
            self._tools_text_cache = text.replace("awm_call_tool", "call_tool")
        return self._tools_text_cache

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | str | None) -> str:
        normalized_args = parse_jsonish_arguments(arguments)
        result = await self._request(
            {
                "type": "step",
                "data": {
                    "type": "call_tool",
                    "tool_name": tool_name,
                    "arguments": normalized_args,
                },
            }
        )
        obs = self._observation(result)
        if obs.get("error"):
            return f"Error: {obs.get('error')}"
        tool_result = obs.get("tool_result")
        if isinstance(tool_result, str):
            return tool_result
        return json.dumps(tool_result, ensure_ascii=False, default=str)

    async def run_verifier(self, final_answer: str | None = None) -> tuple[str, dict[str, Any]]:
        args: dict[str, Any] = {"verifier_mode": self.config.verifier_mode}
        if final_answer is not None:
            args["final_answer"] = final_answer
        result = await self._request(
            {
                "type": "step",
                "data": {
                    "type": "call_tool",
                    "tool_name": "verify",
                    "arguments": args,
                },
            }
        )
        obs = self._observation(result)
        self.last_verify_observation = obs
        reward_type = str(obs.get("reward_type") or "judge_error")
        details = {
            "reward": result.get("reward"),
            "reward_type": reward_type,
            "verifier_mode": self.config.verifier_mode,
            "verify_result": obs.get("verify_result"),
            "openenv_observation": obs,
        }
        if obs.get("error"):
            details["error"] = obs.get("error")
        return reward_type, details

    async def close(self) -> None:
        if self._ws is None:
            return
        try:
            done_result = await self._request(
                {
                    "type": "step",
                    "data": {
                        "type": "call_tool",
                        "tool_name": "done",
                        "arguments": {"keep_session": self.config.keep_session},
                    },
                }
            )
            self.last_done_observation = self._observation(done_result)
        except Exception:
            pass
        try:
            await self._ws.send(json.dumps({"type": "close"}))
        except Exception:
            pass
        try:
            await self._ws.close()
        except Exception:
            pass
        self._ws = None

    async def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("OpenEnv AWM websocket is not connected.")
        await self._ws.send(json.dumps(payload, ensure_ascii=False))
        raw = await asyncio.wait_for(self._ws.recv(), timeout=self.config.message_timeout_s)
        message = json.loads(raw)
        if message.get("type") == "error":
            raise RuntimeError(f"OpenEnv websocket error: {message.get('data')}")
        data = message.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected OpenEnv websocket response: {message}")
        return data

    @staticmethod
    def _observation(result: dict[str, Any]) -> dict[str, Any]:
        obs = result.get("observation", {})
        return obs if isinstance(obs, dict) else {}


class OpenEnvAWMSessionEnv(BaseEnv):
    close_on_release = True

    def __init__(self):
        super().__init__()
        self.runtime: OpenEnvAWMSessionRuntime | None = None
        self.current_config: OpenEnvAWMSessionConfig | None = None

    async def start(self) -> None:
        return None

    async def reset(self, env_args: dict | None = None) -> str:
        flattened = flatten_awm_env_args(env_args)
        scenario = flattened.get("scenario")
        task_idx_raw = flattened.get("task_idx", flattened.get("task_id"))
        if not scenario or task_idx_raw is None:
            await self._close_runtime()
            return "OpenEnv AWM session is idle until scenario and task_idx/task_id are provided."

        base_url = (
            flattened.get("openenv_awm_base_url")
            or os.getenv("AGENTFLY_OPENENV_AWM_BASE_URL")
            or DEFAULT_OPENENV_AWM_BASE_URL
        )
        verifier_mode = (
            flattened.get("openenv_awm_verifier_mode")
            or os.getenv("AGENTFLY_OPENENV_AWM_VERIFIER_MODE")
            or "code"
        )
        keep_session = _truthy(
            flattened.get(
                "openenv_awm_keep_session",
                os.getenv("AGENTFLY_OPENENV_AWM_KEEP_SESSION", "0"),
            )
        )
        config = OpenEnvAWMSessionConfig(
            scenario=str(scenario),
            task_idx=coerce_task_id(task_idx_raw),
            task=flattened.get("task") if isinstance(flattened.get("task"), str) else None,
            base_url=str(base_url),
            verifier_mode=str(verifier_mode).lower(),
            keep_session=keep_session,
            connect_timeout_s=float(flattened.get("openenv_awm_connect_timeout_s", 20.0)),
            message_timeout_s=float(flattened.get("openenv_awm_message_timeout_s", 60.0)),
            max_message_mb=int(flattened.get("openenv_awm_max_message_mb", 100)),
            reset_retries=int(
                flattened.get(
                    "openenv_awm_reset_retries",
                    os.getenv("AGENTFLY_OPENENV_AWM_RESET_RETRIES", 2),
                )
            ),
            reset_retry_backoff_s=float(
                flattened.get(
                    "openenv_awm_reset_retry_backoff_s",
                    os.getenv("AGENTFLY_OPENENV_AWM_RESET_RETRY_BACKOFF_S", 0.5),
                )
            ),
        )
        if config.verifier_mode not in {"code", "sql"}:
            raise ValueError("openenv_awm_verifier_mode must be 'code' or 'sql'")

        await self._close_runtime()

        # Start the session with bounded retries for the OpenEnv startup race.
        # Each attempt builds a fresh runtime (new websocket session -> new sub-env
        # subprocess), and prewarms list_tools so a half-initialized sub-env is
        # detected here (during reset) rather than mid-rollout. Only the known
        # startup-race errors are retried; all other errors propagate immediately.
        max_attempts = max(1, config.reset_retries + 1)
        runtime: OpenEnvAWMSessionRuntime | None = None
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            candidate = OpenEnvAWMSessionRuntime(config)
            try:
                await candidate.start()
                # Prewarm: forces the first list_tools, surfacing the race now.
                await candidate.list_tools()
                runtime = candidate
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                try:
                    await candidate.close()
                except Exception:
                    pass
                if attempt + 1 < max_attempts and _is_retryable_startup_error(str(exc)):
                    await asyncio.sleep(config.reset_retry_backoff_s * (attempt + 1))
                    continue
                raise
        if runtime is None:  # pragma: no cover - defensive
            raise RuntimeError(
                f"OpenEnv AWM session failed to start after {max_attempts} attempts: {last_exc}"
            )
        self.current_config = config
        self.runtime = runtime
        return (
            f"Started OpenEnv AWM session for scenario={config.scenario}, "
            f"task_idx={config.task_idx}, verifier_mode={config.verifier_mode}. "
            f"WebSocket URL: {runtime.ws_url}"
        )

    async def step(self, action: str) -> str:
        if self.runtime is None:
            return "Error: OpenEnv AWM session is not initialized."
        if action == "list_tools":
            return await self.runtime.list_tools_text()
        return (
            "Error: OpenEnvAWMSessionEnv.step only supports the action 'list_tools'. "
            "Use env.call_tool(...) for tool execution."
        )

    async def list_tools_text(self) -> str:
        if self.runtime is None:
            return "Error: OpenEnv AWM session is not initialized."
        return await self.runtime.list_tools_text()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | str | None) -> str:
        if self.runtime is None:
            return "Error: OpenEnv AWM session is not initialized."
        return await self.runtime.call_tool(tool_name, arguments)

    async def run_verifier(self, final_answer: str | None = None) -> tuple[str, dict[str, Any]]:
        if self.runtime is None:
            return "judge_error", {"error": "OpenEnv AWM session is not initialized"}
        return await self.runtime.run_verifier(final_answer)

    async def aclose(self) -> None:
        await self._close_runtime()

    def close(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
            return
        loop.create_task(self.aclose())

    @staticmethod
    async def acquire():
        env = OpenEnvAWMSessionEnv()
        await env.start()
        return env

    async def _close_runtime(self) -> None:
        if self.runtime is not None:
            await self.runtime.close()
        self.runtime = None
        self.current_config = None
