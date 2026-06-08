
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from ..rewards.reward_base import get_reward_from_name
from ..tools import get_tools_from_names
from .agent_base import BaseAgent
from .react.react_agent import ReactAgent
from .specialized.code_agent import CodeAgent
from .specialized.gui_agent import GUIAgent
from .specialized.hf_agent import HFAgent
from .specialized.openai_agent import OpenAIAgent
from .specialized.think_agent import ThinkAgent


class AutoAgent:
    """
    AutoAgent is a class that automatically handles agent initialization based on configuration.

    Built-in agent types:
    - 'react': ReactAgent for ReAct-style reasoning and tool use
    - 'code': CodeAgent for code generation and execution

    These agents are registered automatically. Additional custom agents can be
    registered using the register_agent method.
    """

    AGENT_MAPPING = {}
    PROMPTS_DIR = Path(__file__).resolve().parents[1] / "configs" / "prompts"
    DEFAULT_SYSTEM_PROMPT_CONFIGS = {
        "think": PROMPTS_DIR / "system_prompt_think.yaml",
    }
    AWM_PROMPT_MODE_CONFIGS = {
        "no_think": {
            "agent_type": "hf",
            "template": "qwen2.5",
            "tools": ["awm_list_tools", "awm_call_tool"],
            "tool_parser_name": "awm_xml",
            "reward_name": "awm_verifier_reward",
            "system_prompt_config_path": PROMPTS_DIR / "system_prompt_awm.yaml",
        },
        "think": {
            "agent_type": "hf",
            "template": "qwen2.5",
            "tools": ["awm_list_tools", "awm_call_tool"],
            "tool_parser_name": "awm_xml",
            "reward_name": "awm_verifier_reward_think",
            "system_prompt_config_path": PROMPTS_DIR / "system_prompt_awm_think.yaml",
        },
    }
    _AWM_PROMPT_MODE_ALIASES = {
        "native": "no_think",
        "no-think": "no_think",
        "nothink": "no_think",
        "no_think": "no_think",
        "think": "think",
    }
    _DEFAULT_AWM_REPLACEMENTS = {
        "agent_type": {"code"},
        "tool_parser_name": {"hermes"},
        "reward_name": {"qa_f1_reward"},
    }
    _DEFAULT_TOOLS = ["google_search", "answer"]

    @classmethod
    def _normalize_awm_prompt_mode(cls, mode: Any) -> str | None:
        if mode is None or mode == "":
            return None
        normalized = str(mode).strip().lower()
        prompt_mode = cls._AWM_PROMPT_MODE_ALIASES.get(normalized)
        if prompt_mode is None:
            valid_modes = ", ".join(sorted(cls._AWM_PROMPT_MODE_CONFIGS))
            raise ValueError(f"Unknown awm_prompt_mode: {mode!r}. Expected one of: {valid_modes}")
        return prompt_mode

    @classmethod
    def _should_apply_awm_default(cls, key: str, current_value: Any) -> bool:
        if current_value is None or current_value == "":
            return True
        if key == "tools":
            try:
                return list(current_value) == cls._DEFAULT_TOOLS
            except TypeError:
                return False
        default_values = cls._DEFAULT_AWM_REPLACEMENTS.get(key)
        if default_values is None:
            return False
        return str(current_value).lower() in default_values

    @classmethod
    def _apply_awm_prompt_mode_defaults(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        updated_config = dict(config)
        prompt_mode = cls._normalize_awm_prompt_mode(updated_config.get("awm_prompt_mode"))
        if prompt_mode is None:
            return updated_config

        for key, value in cls.AWM_PROMPT_MODE_CONFIGS[prompt_mode].items():
            if cls._should_apply_awm_default(key, updated_config.get(key)):
                updated_config[key] = value
        updated_config["awm_prompt_mode"] = prompt_mode
        return updated_config

    @classmethod
    def _load_system_prompt_from_yaml(cls, prompt_config_path: str | Path) -> str:
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise ImportError(
                "Loading system prompt from yaml requires omegaconf. Use the training environment with hydra/omegaconf installed."
            ) from exc

        path = Path(prompt_config_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"System prompt config file not found: {path}")

        prompt_config = OmegaConf.load(path)
        system_prompt = OmegaConf.select(prompt_config, "system_prompt")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError(f"Missing non-empty 'system_prompt' in config file: {path}")

        return system_prompt

    @classmethod
    def _resolve_system_prompt(cls, config: Dict[str, Any]) -> str | None:
        config = cls._apply_awm_prompt_mode_defaults(config)
        explicit_system_prompt = config.get("system_prompt")
        if explicit_system_prompt:
            return explicit_system_prompt

        agent_type = str(config["agent_type"]).lower()
        prompt_config_path = config.get("system_prompt_config_path")
        if not prompt_config_path:
            prompt_config_path = cls.DEFAULT_SYSTEM_PROMPT_CONFIGS.get(agent_type)

        if not prompt_config_path:
            return None

        return cls._load_system_prompt_from_yaml(prompt_config_path)


    @classmethod
    def register_agent(cls, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register a new agent type in the AGENT_MAPPING.

        Args:
            agent_type: The name identifier for the agent type (e.g., 'react', 'code')
            agent_class: The agent class to instantiate for this type
        """
        cls.AGENT_MAPPING[agent_type.lower()] = agent_class

    @classmethod
    def _get_agent_class(cls, agent_type: str) -> Type[BaseAgent]:
        """
        Get the agent class for a given agent type.

        Args:
            agent_type: Type of agent ('react', 'code', etc.)

        Returns:
            The agent class

        Raises:
            ValueError: If the agent type is not registered
        """
        agent_type = agent_type.lower()


        if agent_type not in cls.AGENT_MAPPING:
            available_types = list(cls.AGENT_MAPPING.keys())
            raise ValueError(
                f"Unknown agent type: '{agent_type}'. Available types: {available_types}"
            )

        return cls.AGENT_MAPPING[agent_type]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseAgent:
        """
        Create an agent from a configuration dictionary.

        Args:
            config: A dictionary containing the agent configuration.
                Required keys:
                    - agent_type: Type of agent ('react', 'code', etc.)
                    - model_name_or_path: Model name or path
                    - template: Conversation template
                Optional keys:
                    - tools: List of tool objects
                    - vllm: Whether to use vLLM for inference (default: False)
                    - debug: Whether to enable debug logging (default: False)
                    - log_file: Log file name (default: "agent")
                    - task_info: Task-specific information (for ReactAgent)
                    - reward_function: Reward function to use (default: None)
                    - reward_name: Name of registered reward function to use
                    - reward_args: Arguments to pass to the reward function

        Returns:
            An initialized agent instance.
        """
        # Extract and validate required parameters
        if config is None:
            raise ValueError("Config could not be None")

        config = cls._apply_awm_prompt_mode_defaults(config)

        # construct a copy for agent_kwargs
        agent_kwargs = {}
        for k, v in config.items():
            agent_kwargs[k] = v

        required_params = ["agent_type", "tools", "backend"]
        missing_params = [param for param in required_params if not config.get(param)]

        if missing_params:
            raise ValueError(
                f"Missing required parameters: {', '.join(missing_params)}"
            )

        agent_type = config["agent_type"]
        agent_kwargs.pop("agent_type")
        agent_kwargs.pop("system_prompt_config_path", None)
        agent_kwargs.pop("awm_prompt_mode", None)

        tools = get_tools_from_names(config["tools"])
        agent_class = cls._get_agent_class(agent_type)
        reward_name = config.get("reward_name")
        if reward_name is not None:
            reward_fn = get_reward_from_name(reward_name)
            agent_kwargs.pop("reward_name")
        else:
            reward_fn = None

        agent_kwargs["tools"] = tools
        agent_kwargs["reward_fn"] = reward_fn
        resolved_system_prompt = cls._resolve_system_prompt(config)
        if resolved_system_prompt is not None:
            agent_kwargs["system_prompt"] = resolved_system_prompt


        if "use_agent" in agent_kwargs:
            agent_kwargs.pop("use_agent")

        agent = agent_class(**agent_kwargs)

        return agent

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        agent_type: str,
        template: str,
        tools: Optional[List] = None,
        debug: bool = False,
        reward_fn: Optional[Callable] = None,
        **kwargs,
    ) -> BaseAgent:
        """
        Create an agent directly from a model name/path and agent type.

        Args:
            model_name_or_path: Pretrained model name or path
            agent_type: Type of agent ('react', 'code', etc.)
            template: Conversation template name
            tools: List of tool objects
            vllm: Whether to use vLLM for inference
            debug: Whether to enable debug logging
            log_file: Log file name
            wrapper: Whether to use the agent as a wrapper
            reward_function: Reward function instance to use (takes precedence)
            reward_name: Name of registered reward function to use
            reward_args: Arguments to pass to the reward function constructor
            **kwargs: Additional arguments specific to the agent type

        Returns:
            An initialized agent instance.
        """
        # Create config dictionary and reuse from_config logic
        config = {
            "agent_type": agent_type,
            "model_name_or_path": model_name_or_path,
            "template": template,
            "tools": tools or [],
            "debug": debug,
            "reward_fn": reward_fn,
            **kwargs,
        }

        return cls.from_config(config)


# Auto-register built-in agent types
AutoAgent.register_agent("react", ReactAgent)
AutoAgent.register_agent("code", CodeAgent)
AutoAgent.register_agent("openai", OpenAIAgent)
AutoAgent.register_agent("think", ThinkAgent)
AutoAgent.register_agent("gui", GUIAgent)
AutoAgent.register_agent("hf", HFAgent)
