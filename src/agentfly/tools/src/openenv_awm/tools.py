import os
from typing import Any

from ....envs.openenv_awm_session_env import OpenEnvAWMSessionEnv
from ...decorator import tool


OPENENV_AWM_POOL_SIZE = int(os.getenv("AGENTFLY_OPENENV_AWM_POOL_SIZE", "32"))


@tool(
    env_cls=OpenEnvAWMSessionEnv,
    name="openenv_awm_list_tools",
    description=(
        "List the dynamic MCP tools available in the current OpenEnv AgentWorldModel session. "
        "Use this near the start of the task before calling openenv_awm_call_tool."
    ),
    stateful=True,
    pool_size=OPENENV_AWM_POOL_SIZE,
)
async def openenv_awm_list_tools(env: OpenEnvAWMSessionEnv) -> str:
    """
    List the dynamic MCP tools available in the current OpenEnv AWM episode.

    Args:
        env: The active OpenEnv AWM session environment.

    Returns:
        A formatted description of all current MCP tools and their parameters.
    """
    return await env.list_tools_text()


@tool(
    env_cls=OpenEnvAWMSessionEnv,
    name="list_tools",
    description="List all available MCP tools for the current environment.",
    stateful=True,
    pool_size=OPENENV_AWM_POOL_SIZE,
)
async def list_tools(env: OpenEnvAWMSessionEnv) -> str:
    """
    List all available MCP tools for the current environment.

    Args:
        env: The active OpenEnv AWM session environment.

    Returns:
        A formatted description of all current MCP tools and their parameters.
    """
    return await env.list_tools_text()


@tool(
    env_cls=OpenEnvAWMSessionEnv,
    name="openenv_awm_call_tool",
    description=(
        "Call one MCP tool inside the current OpenEnv AgentWorldModel session. "
        "Pass the tool_name exactly as returned by openenv_awm_list_tools and provide arguments as a JSON string."
    ),
    stateful=True,
    pool_size=OPENENV_AWM_POOL_SIZE,
)
async def openenv_awm_call_tool(
    tool_name: str,
    arguments: str,
    env: OpenEnvAWMSessionEnv,
) -> dict[str, Any]:
    """
    Call one MCP tool inside the current OpenEnv AWM episode.

    Args:
        tool_name: The MCP tool name returned by openenv_awm_list_tools.
        arguments: JSON string of arguments for the MCP tool. Empty JSON string if no arguments are required.
        env: The active OpenEnv AWM session environment.

    Returns:
        A tool response dict containing the text observation plus OpenEnv metadata.
    """
    return await env.call_tool_with_details(tool_name, arguments)


@tool(
    env_cls=OpenEnvAWMSessionEnv,
    name="call_tool",
    description="Call a MCP environment-specific tool. Pass arguments as a valid JSON string.",
    stateful=True,
    pool_size=OPENENV_AWM_POOL_SIZE,
)
async def call_tool(
    tool_name: str,
    arguments: str,
    env: OpenEnvAWMSessionEnv,
) -> str:
    """
    Call a MCP environment-specific tool.

    Args:
        tool_name: The MCP tool name returned by list_tools.
        arguments: Valid JSON string of arguments for the MCP tool.
        env: The active OpenEnv AWM session environment.

    Returns:
        A tool response dict containing the text observation plus OpenEnv metadata.
    """
    return await env.call_tool_with_details(tool_name, arguments)
