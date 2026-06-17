from .awm_reward import awm_verifier_reward, awm_verifier_reward_think
from .math_reward import (
    math_equal_reward,
    math_equal_reward_think,
    math_equal_reward_think_no_tool_bonus,
    math_equal_reward_think_tool,
    math_equal_reward_tool,
    math_string_equal_reward_tool,
)
from .qa_reward import qa_f1_reward, qa_f1_reward_tool
from .reward_base import (
    BaseReward,
    get_reward_from_name,
    get_rewards_from_names,
    list_available_rewards,
    register_reward,
    reward,
)
from .openenv_awm_reward import openenv_awm_verifier_reward, openenv_awm_verifier_reward_think

try:  # pragma: no cover - optional rewards may depend on environment-specific packages
    from .alfworld_reward import alfworld_episode_reward
except Exception:  # noqa: BLE001
    alfworld_episode_reward = None

try:  # pragma: no cover
    from .chess_reward import chess_puzzle_reward, chess_puzzle_reward_simple
except Exception:  # noqa: BLE001
    chess_puzzle_reward = None
    chess_puzzle_reward_simple = None

try:  # pragma: no cover
    from .code_reward import code_reward_test
except Exception:  # noqa: BLE001
    code_reward_test = None

try:  # pragma: no cover
    from .gui_reward import gui_reward
except Exception:  # noqa: BLE001
    gui_reward = None

try:  # pragma: no cover
    from .scienceworld_reward import scienceworld_reward
except Exception:  # noqa: BLE001
    scienceworld_reward = None

try:  # pragma: no cover
    from .webshop_reward import webshop_reward
except Exception:  # noqa: BLE001
    webshop_reward = None

__all__ = [
    "BaseReward",
    "get_reward_from_name",
    "get_rewards_from_names",
    "list_available_rewards",
    "register_reward",
    "reward",
    "awm_verifier_reward",
    "awm_verifier_reward_think",
    "openenv_awm_verifier_reward",
    "openenv_awm_verifier_reward_think",
    "qa_f1_reward",
    "qa_f1_reward_tool",
    "math_equal_reward",
    "math_equal_reward_tool",
    "math_equal_reward_think",
    "math_equal_reward_think_tool",
    "math_equal_reward_think_no_tool_bonus",
    "math_string_equal_reward_tool",
    "webshop_reward",
    "alfworld_episode_reward",
    "scienceworld_reward",
    "gui_reward",
    "code_reward_test",
    "chess_puzzle_reward",
    "chess_puzzle_reward_simple",
]
