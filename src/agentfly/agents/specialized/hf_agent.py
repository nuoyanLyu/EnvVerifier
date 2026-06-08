import json
import logging
import re
from typing import Dict, List, Optional

from ..agent_base import BaseAgent

logger = logging.getLogger(__file__)

AWM_XML_TOOL_PARSER_NAMES = {"awm_xml", "awm_xml_tool_call", "xml_tool_call"}
AWM_XML_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _loads_json_object(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_awm_xml_tool_call(data: dict, call_id: str) -> dict | None:
    name = data.get("name", "")
    arguments = data.get("arguments", {})

    if name == "list_tools":
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": "awm_list_tools", "arguments": "{}"},
        }

    if name.startswith("mcp_tool_"):
        name = "call_tool"
        arguments = {
            "tool_name": data.get("name", "")[len("mcp_tool_") :],
            "arguments": arguments if arguments else {},
        }

    if name != "call_tool":
        return None

    parsed_arguments = _loads_json_object(arguments)
    if not isinstance(parsed_arguments, dict):
        return None

    tool_name = parsed_arguments.get("tool_name", "")
    if isinstance(tool_name, str) and tool_name.startswith("mcp_tool_"):
        tool_name = tool_name[len("mcp_tool_") :]

    inner_args = parsed_arguments.get("arguments", {})
    if isinstance(inner_args, str):
        inner_args = _loads_json_object(inner_args)
    if inner_args is None:
        inner_args = {}
    if not isinstance(inner_args, dict):
        return None

    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "awm_call_tool",
            "arguments": json.dumps(
                {"tool_name": tool_name, "arguments": inner_args},
                ensure_ascii=False,
            ),
        },
    }


def drop_awm_xml_prompt_tools(kwargs: dict) -> dict:
    filtered_kwargs = dict(kwargs)
    filtered_kwargs.pop("tools", None)
    return filtered_kwargs


def parse_awm_xml_tool_calls(response: str, response_index: int = 0) -> list[dict]:
    tool_calls = []
    for match_index, match in enumerate(AWM_XML_TOOL_CALL_PATTERN.findall(response)):
        try:
            data = json.loads(match.strip())
        except json.JSONDecodeError:
            continue

        if isinstance(data, list):
            if not data or not isinstance(data[0], dict):
                continue
            data = data[0]
        if not isinstance(data, dict):
            continue

        tool_call = _normalize_awm_xml_tool_call(
            data,
            call_id=f"awm_xml_call_{response_index}_{match_index}",
        )
        if tool_call is not None:
            tool_calls.append(tool_call)
    return tool_calls


class HFAgent(BaseAgent):
    def __init__(
        self,
        model_name_or_path: str,
        tool_parser_name: Optional[str] = "hermes",
        **kwargs,
    ):
        self._use_awm_xml_tool_parser = tool_parser_name in AWM_XML_TOOL_PARSER_NAMES
        super().__init__(
            model_name_or_path,
            tool_parser_name=None if self._use_awm_xml_tool_parser else tool_parser_name,
            **kwargs,
        )

    async def generate_async(self, messages_list_or_inputs, **kwargs):
        if getattr(self, "_use_awm_xml_tool_parser", False):
            kwargs = drop_awm_xml_prompt_tools(kwargs)
        return await super().generate_async(messages_list_or_inputs, **kwargs)

    async def generate_streaming(self, messages_list_or_inputs, **kwargs):
        if getattr(self, "_use_awm_xml_tool_parser", False):
            kwargs = drop_awm_xml_prompt_tools(kwargs)
        async for chunk in super().generate_streaming(messages_list_or_inputs, **kwargs):
            yield chunk

    def parse(self, responses: List[str], tools=None, **kwargs) -> List[Dict]:
        if not getattr(self, "_use_awm_xml_tool_parser", False):
            return super().parse(responses, tools=tools, **kwargs)

        new_messages_list = []
        for response_index, response in enumerate(responses):
            formatted_tool_calls = parse_awm_xml_tool_calls(response, response_index=response_index)
            new_messages_list.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                    "tool_calls": formatted_tool_calls,
                    "loss": True,
                    "status": "continue" if formatted_tool_calls else "terminal",
                }
            )
        return new_messages_list
