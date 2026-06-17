from .decorator import tool
from .registry import get_tool_from_name, get_tools_from_names, register_tool
from .src.alfworld.tools import (
    alfworld_get_admissible_commands,
    alfworld_get_task_objective,
    alfworld_reset,
    alfworld_step,
)
from .src.awm.tools import awm_call_tool, awm_list_tools
from .src.calculate.tools import calculator
from .src.code.tools import CodeInterpreterTool, code_interpreter
from .src.openenv_awm.tools import call_tool, list_tools, openenv_awm_call_tool, openenv_awm_list_tools
from .src.react.tools import answer_math, answer_qa
from .src.ui.tools import pyautogui_code_generator
from .tool_base import BaseTool

try:  # pragma: no cover - optional tools may depend on extra packages
    from .src.chess.tools import chess_get_legal_moves, chess_get_state, chess_move
except Exception:  # noqa: BLE001
    chess_get_legal_moves = None
    chess_get_state = None
    chess_move = None

try:  # pragma: no cover
    from .src.scienceworld.tools import scienceworld_explorer
except Exception:  # noqa: BLE001
    scienceworld_explorer = None

try:  # pragma: no cover
    from .src.search.async_dense_retriever import asyncdense_retrieve
    from .src.search.dense_retriever import dense_retrieve
    from .src.search.google_search import google_search_serper
except Exception:  # noqa: BLE001
    asyncdense_retrieve = None
    dense_retrieve = None
    google_search_serper = None

try:  # pragma: no cover
    from .src.webshop.tools import webshop_browser
except Exception:  # noqa: BLE001
    webshop_browser = None


@tool()
def hallucination_tool(tool_name):
    return f"Hallucinated tool: {tool_name} does not exist."


@tool()
def invalid_input_tool(tool_input):
    return f"Invalid input: {tool_input}, input must be a valid JSON object."


__all__ = [
    "tool",
    "BaseTool",
    "code_interpreter",
    "CodeInterpreterTool",
    "alfworld_step",
    "alfworld_get_task_objective",
    "alfworld_get_admissible_commands",
    "alfworld_reset",
    "awm_list_tools",
    "awm_call_tool",
    "openenv_awm_list_tools",
    "openenv_awm_call_tool",
    "list_tools",
    "call_tool",
    "calculator",
    "google_search_serper",
    "dense_retrieve",
    "asyncdense_retrieve",
    "scienceworld_explorer",
    "webshop_browser",
    "answer_qa",
    "answer_math",
    "pyautogui_code_generator",
    "hallucination_tool",
    "invalid_input_tool",
    "get_tool_from_name",
    "get_tools_from_names",
    "register_tool",
    "chess_move",
    "chess_get_state",
    "chess_get_legal_moves",
]
