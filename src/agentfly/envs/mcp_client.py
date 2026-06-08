from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable


class MCPToolExecutor:
    """Small streamable-HTTP MCP client used by AgentFly environments.

    The AWM project already ships a MCP executor, but importing it pulls AWM
    runtime dependencies into the RL training environment. This wrapper keeps
    AgentFly coupled only to the MCP protocol.
    """

    def __init__(self, mcp_url: str, timeout: float = 60.0):
        self.mcp_url = mcp_url
        self.timeout = timeout
        self._tools: list[dict[str, Any]] = []

    async def _with_session(self, operation: Callable[[Any], Awaitable[Any]]) -> Any:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:  # pragma: no cover - depends on external env setup
            raise RuntimeError(
                "The Python package 'mcp' is required to connect to external MCP servers. "
                "Install AgentFly dependencies or run: python -m pip install mcp"
            ) from exc

        async with streamablehttp_client(self.mcp_url, timeout=self.timeout) as streams:
            read_stream, write_stream = streams[0], streams[1]
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=self.timeout)
                return await asyncio.wait_for(operation(session), timeout=self.timeout)

    async def list_tools(self) -> list[dict[str, Any]]:
        async def _list(session: Any) -> Any:
            return await session.list_tools()

        result = await self._with_session(_list)
        tools: list[dict[str, Any]] = []
        for tool in result.tools:
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": getattr(tool, "inputSchema", None)
                    or getattr(tool, "input_schema", None)
                    or {},
                }
            )
        self._tools = tools
        return list(tools)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        async def _call(session: Any) -> Any:
            return await session.call_tool(tool_name, arguments)

        result = await self._with_session(_call)
        parts: list[str] = []
        for content in result.content:
            text = getattr(content, "text", None)
            parts.append(text if text is not None else str(content))

        text = "\n".join(parts)
        if getattr(result, "isError", False):
            return f"Error: {text}"
        return text


__all__ = ["MCPToolExecutor"]
