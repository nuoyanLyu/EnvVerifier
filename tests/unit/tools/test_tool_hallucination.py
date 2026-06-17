import asyncio

from agentfly.tools import hallucination_tool  # noqa: F401 - ensure tool registration
from agentfly.tools.tool_base import submit_tool_call


def test_submit_tool_call_preserves_original_hallucinated_tool_name():
    async def run_case():
        return await submit_tool_call(
            "made_up_tool",
            "{}",
            id="hallucination-case",
            allowed_tool_names=["list_tools", "call_tool"],
        )

    result = asyncio.run(run_case())

    assert result["name"] == "hallucination_tool"
    assert result["arguments"] == {"tool_name": "made_up_tool"}
    assert result["observation"] == "Hallucinated tool: made_up_tool does not exist."
