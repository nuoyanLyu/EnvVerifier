import argparse
import json
import random
import sys
import types
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _install_awm_tool_shims() -> None:
    if "orjson" not in sys.modules:
        orjson_module = types.ModuleType("orjson")
        orjson_module.dumps = lambda obj, *args, **kwargs: json.dumps(obj).encode("utf-8")
        orjson_module.loads = lambda data, *args, **kwargs: json.loads(
            data.decode("utf-8") if isinstance(data, bytes) else data
        )
        sys.modules["orjson"] = orjson_module

    if "json_repair" not in sys.modules:
        json_repair_module = types.ModuleType("json_repair")
        json_repair_module.repair_json = lambda data, return_objects=False: (
            json.loads(data) if return_objects else data
        )
        sys.modules["json_repair"] = json_repair_module

    if "tiktoken" not in sys.modules:
        tiktoken_module = types.ModuleType("tiktoken")

        class _DummyEncoding:
            def encode(self, text: str) -> list[int]:
                return list(text.encode("utf-8"))

        tiktoken_module.encoding_for_model = lambda model: _DummyEncoding()
        sys.modules["tiktoken"] = tiktoken_module

    if "loguru" not in sys.modules:
        loguru_module = types.ModuleType("loguru")

        class _DummyLogger:
            def warning(self, *args, **kwargs) -> None:
                return None

            def error(self, *args, **kwargs) -> None:
                return None

            def info(self, *args, **kwargs) -> None:
                return None

        loguru_module.logger = _DummyLogger()
        sys.modules["loguru"] = loguru_module

    if "mcp_agent" not in sys.modules:
        mcp_agent_module = types.ModuleType("mcp_agent")
        app_module = types.ModuleType("mcp_agent.app")
        agents_module = types.ModuleType("mcp_agent.agents")
        agent_module = types.ModuleType("mcp_agent.agents.agent")
        config_module = types.ModuleType("mcp_agent.config")

        class _Dummy:
            pass

        app_module.MCPApp = _Dummy
        agent_module.Agent = _Dummy
        config_module.Settings = _Dummy
        config_module.MCPSettings = _Dummy
        config_module.MCPServerSettings = _Dummy
        config_module.LoggerSettings = _Dummy

        sys.modules["mcp_agent"] = mcp_agent_module
        sys.modules["mcp_agent.app"] = app_module
        sys.modules["mcp_agent.agents"] = agents_module
        sys.modules["mcp_agent.agents.agent"] = agent_module
        sys.modules["mcp_agent.config"] = config_module


# _install_awm_tool_shims()

from awm.tools import tools_json_save, tools_jsonl_load, tools_jsonl_save


DEFAULT_INPUT_DIR = Path("AgentWorldModel-1K")
DEFAULT_OUTPUT_DIR = Path("extract")
DEFAULT_SAMPLE_SIZE = 25
DEFAULT_SEED = 12345

TASKS_FILE = "gen_tasks.jsonl"
TARGET_FILES = [
    "gen_envs.jsonl",
    "gen_sample.jsonl",
    "gen_db.jsonl",
    "gen_verifier.jsonl",
    "gen_verifier.pure_code.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a reproducible scenario subset from AgentWorldModel-1K."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Source directory containing jsonl files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for extracted files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of scenarios to sample. Default: {DEFAULT_SAMPLE_SIZE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible sampling. Default: {DEFAULT_SEED}",
    )
    return parser.parse_args()


def require_existing_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {path}")


def load_jsonl_with_scenario(path: Path) -> list[dict]:
    require_existing_file(path)
    data = tools_jsonl_load(str(path))
    # for idx, item in enumerate(data):
    #     if "scenario" not in item:
    #         raise KeyError(f"Missing 'scenario' field in {path} at row {idx}")
    return data


def sample_scenarios(tasks_path: Path, num_scenarios: int, seed: int) -> list[str]:
    task_items = load_jsonl_with_scenario(tasks_path)
    scenarios = [item["scenario"] for item in task_items]
    unique_scenarios = list(dict.fromkeys(scenarios))

    if num_scenarios > len(unique_scenarios):
        raise ValueError(
            f"Requested {num_scenarios} scenarios, but only {len(unique_scenarios)} are available."
        )

    rng = random.Random(seed)
    return rng.sample(unique_scenarios, num_scenarios)


def filter_rows_by_scenarios(data: list[dict], selected_scenarios: list[str]) -> tuple[list[dict], list[str]]:
    selected_set = set(selected_scenarios)
    filtered = [item for item in data if item["scenario"] in selected_set]
    present_scenarios = {item["scenario"] for item in filtered}
    missing = [scenario for scenario in selected_scenarios if scenario not in present_scenarios]
    return filtered, missing


def write_scenarios_json(selected_scenarios: list[str], output_dir: Path) -> None:
    output_path = output_dir / "scenarios.json"
    tools_json_save(selected_scenarios, str(output_path))


def main() -> None:
    args = parse_args()

    if args.num_scenarios <= 0:
        raise ValueError("--num-scenarios must be a positive integer.")

    tasks_path = args.input_dir / TASKS_FILE
    selected_scenarios = sample_scenarios(tasks_path, args.num_scenarios, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_scenarios_json(selected_scenarios, args.output_dir)

    print(
        f"Selected {len(selected_scenarios)} scenarios from {tasks_path} "
        f"with seed={args.seed}."
    )
    print(f"Saved scenario manifest to {args.output_dir / 'scenarios.json'}")

    for filename in TARGET_FILES:
        input_path = args.input_dir / filename
        output_path = args.output_dir / filename

        data = load_jsonl_with_scenario(input_path)
        filtered, missing = filter_rows_by_scenarios(data, selected_scenarios)
        tools_jsonl_save(filtered, str(output_path))

        print(f"{filename}: kept {len(filtered)} rows -> {output_path}")
        if missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            print(
                f"{filename}: missing {len(missing)} selected scenarios: {preview}{suffix}"
            )


if __name__ == "__main__":
    main()
