from ....envs.awm_session_env import AWMSessionEnv
from ...decorator import tool


@tool(
    env_cls=AWMSessionEnv,
    name="awm_list_tools",
    description=(
        "List the dynamic MCP tools available in the current AgentWorldModel session. "
        "Use this near the start of the task before calling awm_call_tool."
    ),
    stateful=True,
    pool_size=8,
)
async def awm_list_tools(env: AWMSessionEnv) -> str:
    """
    List the dynamic MCP tools available in the current AWM episode.

    Args:
        env: The active AWM session environment.

    Returns:
        A formatted description of all current MCP tools and their parameters.
    """
    return await env.list_tools_text()


@tool(
    env_cls=AWMSessionEnv,
    name="awm_call_tool",
    description=(
        "Call one MCP tool inside the current AgentWorldModel session. "
        "Pass the tool_name exactly as returned by awm_list_tools and provide arguments as a JSON object."
    ),
    stateful=True,
    pool_size=8,
)
async def awm_call_tool(
    tool_name: str,
    arguments: dict,
    env: AWMSessionEnv,
) -> str:
    """
    Call one MCP tool inside the current AWM episode.

    Args:
        tool_name: The MCP tool name returned by awm_list_tools.
        arguments: JSON object of arguments for the MCP tool. Empty object if no arguments are required.
        env: The active AWM session environment.

    Returns:
        The text observation returned by the MCP tool.
    """
    return await env.call_tool(tool_name, arguments)
