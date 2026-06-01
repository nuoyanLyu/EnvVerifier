from __future__ import annotations

import asyncio
import contextlib
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .env_base import BaseEnv
from ..utils.awm import (
    DEFAULT_AWM_DB_SCHEMA_PATH,
    DEFAULT_AWM_ENVS_PATH,
    DEFAULT_AWM_SAMPLE_PATH,
    DEFAULT_AWM_SESSION_ROOT,
    DEFAULT_AWM_TASKS_PATH,
    DEFAULT_AWM_VERIFIER_CODE_PATH,
    classify_trajectory_issue,
    coerce_task_id,
    ensure_awm_importable,
    find_task_text,
    find_verifier_entry,
    flatten_awm_env_args,
    format_mcp_tools,
    parse_jsonish_arguments,
    resolve_path,
)


@dataclass
class AWMSessionConfig:
    scenario: str
    task_id: int
    task: str
    envs_path: Path
    tasks_path: Path
    verifier_path: Path
    verifier_mode: str
    db_schema_path: Path
    sample_path: Path
    session_root: Path
    server_timeout_s: float = 60.0
    tool_timeout_s: float = 60.0


class AWMSessionRuntime:
    def __init__(self, config: AWMSessionConfig):
        self.config = config
        self.output_dir = (
            config.session_root
            / f"{time.strftime('%Y%m%d_%H%M%S')}_{config.scenario}_task_{config.task_id}_{uuid.uuid4().hex[:8]}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.server_proc: subprocess.Popen | None = None
        self.mcp_url: str | None = None
        self.initial_db_path: Path | None = None
        self.final_db_path: Path | None = None
        self.db_file_path: Path | None = None
        self.verifier_entry: dict[str, Any] | None = None
        self._mcp = None
        self._tools_cache: list[dict[str, Any]] | None = None
        self._tools_text_cache: str | None = None

    async def start(self) -> None:
        ensure_awm_importable()
        from awm.core.agent import MCPToolExecutor
        from awm.core.server import (
            Config as ServerConfig,
            _prepare_database,
            start_server_process,
        )
        from awm.tools import async_wait_for_server, get_random_available_port

        server_cfg = ServerConfig(
            scenario=self.config.scenario,
            envs_load_path=str(self.config.envs_path),
            db_schema_path=str(self.config.db_schema_path),
            sample_path=str(self.config.sample_path),
            output_dir=str(self.output_dir),
        )
        db_file_path = Path(_prepare_database(server_cfg, str(self.output_dir))).resolve()
        self.db_file_path = db_file_path
        self.initial_db_path = (self.output_dir / "initial.db").resolve()
        self.final_db_path = db_file_path
        self.verifier_entry = find_verifier_entry(
            self.config.verifier_path,
            self.config.scenario,
            self.config.task_id,
        )

        port = get_random_available_port()
        self.server_proc = start_server_process(
            self.config.scenario,
            str(self.config.envs_path),
            str(db_file_path),
            port,
            output_dir=str(self.output_dir),
        )
        if not await async_wait_for_server(port, timeout=self.config.server_timeout_s):
            stderr = ""
            if self.server_proc and self.server_proc.stderr:
                with contextlib.suppress(Exception):
                    stderr = self.server_proc.stderr.read().decode()
            await self.close()
            raise RuntimeError(
                f"AWM MCP server failed to start for scenario={self.config.scenario}, "
                f"task_id={self.config.task_id}. stderr={stderr[:1000]}"
            )

        self.mcp_url = f"http://127.0.0.1:{port}/mcp"
        self._mcp = MCPToolExecutor(self.mcp_url, timeout=self.config.tool_timeout_s)

    async def list_tools(self) -> list[dict[str, Any]]:
        if self._mcp is None:
            raise RuntimeError("AWM session runtime has not been started.")
        if self._tools_cache is None:
            self._tools_cache = await self._mcp.list_tools()
        return list(self._tools_cache)

    async def list_tools_text(self) -> str:
        if self._tools_text_cache is None:
            self._tools_text_cache = format_mcp_tools(await self.list_tools())
        return self._tools_text_cache

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | str | None) -> str:
        if self._mcp is None:
            raise RuntimeError("AWM session runtime has not been started.")
        normalized_args = parse_jsonish_arguments(arguments)
        return await self._mcp.call_tool(tool_name, normalized_args)

    async def close(self) -> None:
        if self.server_proc is not None and self.server_proc.poll() is None:
            self.server_proc.terminate()
            try:
                await asyncio.to_thread(self.server_proc.wait, 5)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()
                await asyncio.to_thread(self.server_proc.wait)
        self.server_proc = None
        if self.db_file_path and self.final_db_path and self.db_file_path.exists():
            if self.db_file_path != self.final_db_path:
                shutil.copy2(self.db_file_path, self.final_db_path)


class AWMSessionEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.runtime: AWMSessionRuntime | None = None
        self.current_config: AWMSessionConfig | None = None
        self.session_root = resolve_path(None, DEFAULT_AWM_SESSION_ROOT)
        self.session_root.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        self.session_root.mkdir(parents=True, exist_ok=True)

    async def reset(self, env_args: dict | None = None) -> str:
        flattened = flatten_awm_env_args(env_args)
        scenario = flattened.get("scenario")
        task_id_raw = flattened.get("task_id")

        if not scenario or task_id_raw is None:
            await self._close_runtime()
            return "AWM session is idle until scenario and task_id are provided."

        task_id = coerce_task_id(task_id_raw)
        envs_path = resolve_path(flattened.get("awm_envs_path"), DEFAULT_AWM_ENVS_PATH)
        tasks_path = resolve_path(flattened.get("awm_tasks_path"), DEFAULT_AWM_TASKS_PATH)
        verifier_mode = str(flattened.get("awm_verifier_mode", "code")).lower()
        verifier_default = DEFAULT_AWM_VERIFIER_CODE_PATH
        if verifier_mode == "sql":
            from ..utils.awm import DEFAULT_AWM_VERIFIER_SQL_PATH

            verifier_default = DEFAULT_AWM_VERIFIER_SQL_PATH
        verifier_path = resolve_path(flattened.get("awm_verifier_path"), verifier_default)
        db_schema_path = resolve_path(flattened.get("awm_db_schema_path"), DEFAULT_AWM_DB_SCHEMA_PATH)
        sample_path = resolve_path(flattened.get("awm_sample_path"), DEFAULT_AWM_SAMPLE_PATH)
        session_root = resolve_path(flattened.get("awm_output_root"), self.session_root)
        session_root.mkdir(parents=True, exist_ok=True)

        task = flattened.get("task")
        if not isinstance(task, str) or not task.strip():
            task = find_task_text(tasks_path, scenario, task_id)

        config = AWMSessionConfig(
            scenario=scenario,
            task_id=task_id,
            task=task,
            envs_path=envs_path,
            tasks_path=tasks_path,
            verifier_path=verifier_path,
            verifier_mode=verifier_mode,
            db_schema_path=db_schema_path,
            sample_path=sample_path,
            session_root=session_root,
            server_timeout_s=float(flattened.get("awm_server_timeout_s", 60.0)),
            tool_timeout_s=float(flattened.get("awm_tool_timeout_s", 60.0)),
        )

        await self._close_runtime()
        runtime = AWMSessionRuntime(config)
        await runtime.start()
        self.current_config = config
        self.runtime = runtime
        return (
            f"Started AWM session for scenario={config.scenario}, task_id={config.task_id}. "
            f"MCP URL: {runtime.mcp_url}"
        )

    async def step(self, action: str) -> str:
        if self.runtime is None:
            return "Error: AWM session is not initialized."
        if action == "list_tools":
            return await self.runtime.list_tools_text()
        return (
            "Error: AWMSessionEnv.step only supports the action 'list_tools'. "
            "Use env.call_tool(...) for tool execution."
        )

    async def list_tools_text(self) -> str:
        if self.runtime is None:
            return "Error: AWM session is not initialized."
        return await self.runtime.list_tools_text()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | str | None) -> str:
        if self.runtime is None:
            return "Error: AWM session is not initialized."
        return await self.runtime.call_tool(tool_name, arguments)

    async def run_verifier(self, final_answer: str | None = None) -> tuple[str, dict[str, Any]]:
        if self.runtime is None or self.current_config is None:
            return "judge_error", {"error": "AWM session is not initialized"}
        if self.runtime.verifier_entry is None:
            return "judge_error", {"error": "Verifier entry not found"}
        if self.runtime.initial_db_path is None or self.runtime.final_db_path is None:
            return "judge_error", {"error": "Database snapshots are unavailable"}

        ensure_awm_importable()
        from awm.core.verify import run_verifier

        return run_verifier(
            self.runtime.verifier_entry,
            self.current_config.verifier_mode,
            str(self.runtime.initial_db_path),
            str(self.runtime.final_db_path),
            final_answer=final_answer,
        )

    def classify_trajectory_issue(self, trajectory: list[dict[str, Any]]) -> str | None:
        return classify_trajectory_issue(trajectory)

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
        env = AWMSessionEnv()
        await env.start()
        return env

    async def _close_runtime(self) -> None:
        if self.runtime is not None:
            await self.runtime.close()
        self.runtime = None
        self.current_config = None
