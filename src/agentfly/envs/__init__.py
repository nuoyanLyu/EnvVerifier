from .awm_session_env import AWMSessionEnv
from .manager import EnvironmentManager, WarmPool, clear_enroot_containers, from_env
from .openenv_awm_session_env import OpenEnvAWMSessionEnv
from .python_env import PythonSandboxEnv

try:  # pragma: no cover - optional environments may have extra dependencies
    from .alfworld_env import ALFWorldEnv
except Exception:  # noqa: BLE001
    ALFWorldEnv = None

try:  # pragma: no cover
    from .chess_env import ChessPuzzleEnv
except Exception:  # noqa: BLE001
    ChessPuzzleEnv = None

try:  # pragma: no cover
    from .redis_env import RedisEnv
except Exception:  # noqa: BLE001
    RedisEnv = None

try:  # pragma: no cover
    from .scienceworld_env import ScienceWorldEnv
except Exception:  # noqa: BLE001
    ScienceWorldEnv = None

try:  # pragma: no cover
    from .webshop_text_env import WebAgentTextEnv
except Exception:  # noqa: BLE001
    WebAgentTextEnv = None

__all__ = [
    "PythonSandboxEnv",
    "ALFWorldEnv",
    "AWMSessionEnv",
    "OpenEnvAWMSessionEnv",
    "WebAgentTextEnv",
    "ScienceWorldEnv",
    "RedisEnv",
    "ChessPuzzleEnv",
    "from_env",
    "WarmPool",
    "EnvironmentManager",
    "clear_enroot_containers",
]
