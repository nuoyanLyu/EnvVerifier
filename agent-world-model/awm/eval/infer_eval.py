"""
Inference + evaluation entry point for agent-world-model.

This is a minimal bridge around `mcp-adapted-bench`.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

_HERE = Path(__file__).resolve()
_AWM_ROOT = _HERE.parent.parent.parent
_CANDIDATE_PATHS = [
    _AWM_ROOT / "mcp-adapted-bench",
    _AWM_ROOT.parent / "mcp-adapted-bench",
]
for _path in _CANDIDATE_PATHS:
    if (_path / "mcp_adapted_bench").exists():
        _BENCH_ROOT = _path
        sys.path.insert(0, str(_path))
        break
else:
    _BENCH_ROOT = _AWM_ROOT / "mcp-adapted-bench"

from awm.tools import tools_json_save, tools_robust_json_loads

from mcp_adapted_bench.bfcl.runner import run_bfcl_evaluation
from mcp_adapted_bench.common.infer_config import EvalMode, InferEvalConfig
from mcp_adapted_bench.common.llm_client import LLMCaller, build_openai_client
from mcp_adapted_bench.mcp_universe.runner import run_mcp_universe_evaluation
from mcp_adapted_bench.tau2.runner import run_tau2_evaluation


logger.remove()
logger.add(sys.stdout, level="INFO")


def _read_completed_task_ids(output_dir: Path) -> set[str]:
    """Return task ids with non-error trajectories already written."""
    traj_dir = output_dir / "agentfly_trajectories"
    if not traj_dir.exists():
        return set()

    completed: set[str] = set()
    for path in traj_dir.glob("*.json"):
        try:
            data = tools_robust_json_loads(path.as_posix())
        except Exception:
            continue
        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict) or data.get("error"):
            continue
        task_id = data.get("task_id")
        if task_id is not None:
            completed.add(str(task_id))
    return completed


def _setup_resume(config: InferEvalConfig) -> None:
    if not config.resume or not config.output_dir:
        return

    output_dir = Path(config.output_dir)
    if not output_dir.exists():
        logger.warning(f"--resume but {output_dir} does not exist; starting fresh")
        return

    previous_config = output_dir / "config.json"
    if previous_config.exists():
        try:
            previous = tools_robust_json_loads(previous_config.as_posix())
            if previous.get("mode") != config.mode.value:
                logger.warning(
                    f"--resume: previous run mode={previous.get('mode')} != {config.mode.value}. "
                    "Ignoring stale config; existing trajectories may be skipped anyway."
                )
        except Exception:
            pass

    logger.info(f"Resume: found {len(_read_completed_task_ids(output_dir))} completed trajectories in {output_dir}")


def _save_config(config: InferEvalConfig) -> None:
    if not config.output_dir:
        return

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "mode": config.mode.value,
        "model": config.model,
        "api_url": config.api_url,
        "limit": config.limit,
        "num_rollouts": config.num_rollouts,
        "max_turns": config.max_turns,
        "history_limit": config.history_limit,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "min_p": config.min_p,
        "max_completion_tokens": config.max_completion_tokens,
        "max_concurrency": config.max_concurrency,
        "tau2_domains": list(config.tau2.domains),
        "mcp_universe_categories": list(config.mcp_universe.categories),
        "bfcl_test_categories": list(config.bfcl.test_categories),
        "note": config.note,
        "saved_at": datetime.now().isoformat(),
    }
    tools_json_save(snapshot, (output_dir / "config.json").as_posix())


def _normalize_benchmark_paths(config: InferEvalConfig) -> None:
    """Resolve benchmark third_party defaults for either supported submodule layout."""
    bfcl_default = "/third_party/bfcl-original/bfcl_eval/data"
    bfcl_data_dir = Path(config.bfcl.data_dir or "")
    if config.bfcl.data_dir == bfcl_default or not bfcl_data_dir.exists():
        config.bfcl.data_dir = str(_BENCH_ROOT / "third_party" / "bfcl-original" / "bfcl_eval" / "data")

    tau2_root = Path(config.tau2.tau2_root)
    if config.tau2.tau2_root == "./third_party/tau2-bench-verified" or not tau2_root.exists():
        config.tau2.tau2_root = str(_BENCH_ROOT / "third_party" / "tau2-bench-verified")

    mcp_root = Path(config.mcp_universe.mcp_universe_root)
    if config.mcp_universe.mcp_universe_root == "third_party/MCP-Universe" or not mcp_root.exists():
        config.mcp_universe.mcp_universe_root = str(_BENCH_ROOT / "third_party" / "MCP-Universe")


async def _run_async(config: InferEvalConfig) -> dict:
    _normalize_benchmark_paths(config)

    client, model, api_url = build_openai_client(api_url=config.api_url, model_override=config.model)
    config.api_url = api_url
    config.model = model
    logger.info(f"[agent LLM] model={model} base_url={api_url}")

    caller = LLMCaller(
        client=client,
        model=model,
        api_url=api_url,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        min_p=config.min_p,
        min_tokens=config.min_tokens,
        repetition_penalty=config.repetition_penalty,
        add_generation_prompt=config.add_generation_prompt,
        enable_thinking=config.enable_thinking,
        max_completion_tokens=config.max_completion_tokens,
        max_concurrency=config.max_concurrency,
        tolerant_mode=config.tolerant_mode,
    )

    _setup_resume(config)
    _save_config(config)

    if config.mode == EvalMode.bfcl:
        return await run_bfcl_evaluation(caller, config)
    if config.mode == EvalMode.tau2:
        return await run_tau2_evaluation(caller, config)
    if config.mode == EvalMode.mcp_universe:
        return await run_mcp_universe_evaluation(caller, config)
    raise ValueError(f"Unknown eval mode: {config.mode}")


def run(config: InferEvalConfig) -> None:
    summary = asyncio.run(_run_async(config))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tools_json_save(summary, (output_dir / "summary.json").as_posix())
    logger.info(f"Wrote summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    from simpleArgParser import parse_args

    run(parse_args(InferEvalConfig))
