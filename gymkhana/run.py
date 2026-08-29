"""CLI entry point for Gymkhana dataset generation."""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from dotenv import load_dotenv
from pydantic import BaseModel
import yaml
load_dotenv()

from gymkhana.envs import get_environment
from gymkhana.envs.config import EnvConfig, InferenceConfig, LLMClientType
from gymkhana.core.services import ServiceContainer
from gymkhana.core.services.sandboxes import REPLSandbox


def _merge_config_overrides(
    model: BaseModel,
    overrides: Mapping[str, Any],
    *,
    prefix: str = "",
) -> Dict[str, Any]:
    """Merge a partial YAML mapping over a config tree, returning a plain dict.

    Nested models recurse, dict fields merge key-wise, everything else is replaced.
    The caller re-validates the whole model once, so cross-field rules (for
    example ``generation.target_language`` referring to ``generation.languages``)
    see the final state regardless of YAML key order.
    """
    merged: Dict[str, Any] = model.model_dump()
    for key, value in overrides.items():
        field_name = f"{prefix}.{key}" if prefix else key
        if key not in type(model).model_fields:
            raise ValueError(f"unknown configuration field: {field_name}")
        current = getattr(model, key)
        if isinstance(current, BaseModel) and isinstance(value, Mapping):
            merged[key] = _merge_config_overrides(current, value, prefix=field_name)
        elif isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = {**current, **value}
        else:
            merged[key] = value
    return merged


def _apply_config_overrides(model: BaseModel, overrides: Mapping[str, Any]) -> BaseModel:
    """Apply a partial YAML mapping to a validated Pydantic config tree."""
    return type(model).model_validate(_merge_config_overrides(model, overrides))


def load_environment_config(
    config_path: Optional[Path] = None,
    *,
    environment_override: Optional[str] = None,
) -> tuple[str, EnvConfig]:
    """Load an environment's defaults and overlay a partial YAML config."""
    overrides: dict[str, Any] = {}
    if config_path is not None:
        with config_path.expanduser().open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"config must contain a YAML mapping: {config_path}")
        overrides = payload

    env_name = environment_override or overrides.get("name") or "math-python"
    env_cls = get_environment(env_name)
    if not hasattr(env_cls, "default_config") or env_cls.default_config is None:
        raise ValueError(
            f"environment {env_name!r} does not expose a default_config"
        )
    config = _apply_config_overrides(env_cls.default_config, overrides)
    if environment_override is not None:
        config.name = environment_override
    return env_name, config


def main():
    parser = argparse.ArgumentParser(
        description="Generate verified training trajectories and SFT datasets"
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Partial YAML config (e.g., "
            "configs/multi_turn_qa/nepali_textbooks.yaml)"
        ),
    )
    parser.add_argument(
        "--env",
        choices=[
            "math-python",
            "oolong",
            "code",
            "hotpotqa",
            "swe",
            "ifeval",
            "tool-use-singleturn",
            "romanized-nepali",
            "english-sharegpt-to-nepali",
            "multi-turn-qa",
        ],
        default=None,
        help="Task environment (overrides config)"
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Hugging Face dataset name or local JSON/JSONL path (overrides config)",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional Hugging Face dataset configuration (overrides config)",
    )
    parser.add_argument(
        "--dataset-split",
        default=None,
        help="Dataset split (overrides config)",
    )
    parser.add_argument(
        "--dataset-backend",
        choices=["auto", "local", "huggingface", "huggingface-rows"],
        default=None,
        help="Dataset loader backend (overrides config)",
    )
    parser.add_argument(
        "--dataset-offset",
        type=int,
        default=None,
        help="Number of source rows to skip before applying --limit",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Main model (overrides config)"
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Semantic judge model (overrides config)",
    )
    parser.add_argument(
        "--questioner-model",
        default=None,
        help="Questioner model for multi-agent QA generation (overrides config)",
    )
    parser.add_argument(
        "--qa-profile",
        default=None,
        help="QA domain profile such as textbook, legal, health, or finance",
    )
    parser.add_argument(
        "--qa-turns",
        type=int,
        default=None,
        help="Number of generated QA turns; use 1 for single-turn generation",
    )
    parser.add_argument(
        "--target-language",
        default=None,
        help=(
            "Target language/script code for QA generation (built-in: en, ne-Deva, "
            "ne-Latn; more via generation.languages in the config)"
        ),
    )
    parser.add_argument(
        "--client",
        choices=["openai", "anthropic", "litellm"],
        default=None,
        help="LLM client (overrides config)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of tasks (overrides config)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Max REPL turns per task (overrides config)"
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="REPL server URL (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--output-basename",
        default=None,
        help="Basename for JSONL and summary artifacts (overrides config)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path (defaults to OUTPUT_DIR/run.log)",
    )
    parser.add_argument(
        "--database",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable PostgreSQL persistence; defaults to enabled when "
            "DB_* variables are configured"
        ),
    )
    parser.add_argument(
        "--mask-observations",
        action="store_true",
        help="Add loss_weight=0 to observations in ShareGPT output"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of parallel tasks (overrides config)"
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=None,
        help="Rollouts per task (best-of-N, GRPO; overrides config.dataset.num_rollouts)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode"
    )
    parser.add_argument(
        "--enable-reasoning",
        action="store_true",
        default=None,
        help="Enable <think> blocks for chain-of-thought reasoning (overrides config default)"
    )
    parser.add_argument(
        "--enable-correction-feedback",
        action="store_true",
        default=None,
        help="Add correction prompts when model responds without code (default: False for RL training)"
    )

    args = parser.parse_args()

    # Load environment defaults, then overlay YAML before applying CLI flags.
    env_name, config = load_environment_config(
        Path(args.config) if args.config else None,
        environment_override=args.env,
    )
    env_cls = get_environment(env_name)

    # Apply CLI overrides
    client_map = {
        "openai": LLMClientType.OPENAI,
        "anthropic": LLMClientType.ANTHROPIC,
        "litellm": LLMClientType.LITELLM,
    }

    # Handle model/client overrides (support both new and legacy config)
    if config.llm is None and (args.model or args.client):
        config.llm = InferenceConfig(
            model=args.model or getattr(config, "main_model", None) or "openai:gpt-4.1-mini",
            client=client_map[args.client] if args.client else LLMClientType.OPENAI,
        )
    elif args.model:
        config.llm.model = args.model
    if args.client and config.llm is not None:
        config.llm.client = client_map[args.client]
    if args.judge_model:
        if config.llm_judge is None:
            raise ValueError(
                "--judge-model requires an environment with LLM judge settings"
            )
        config.llm_judge.model = args.judge_model
    if args.questioner_model:
        if not hasattr(config, "questioner_llm"):
            raise ValueError("--questioner-model requires a multi-agent environment")
        config.questioner_llm.model = args.questioner_model
    if args.qa_profile:
        if not hasattr(config, "generation"):
            raise ValueError("--qa-profile requires a QA generation environment")
        config.generation.profile = args.qa_profile
    if args.qa_turns is not None:
        if not hasattr(config, "generation"):
            raise ValueError("--qa-turns requires a QA generation environment")
        config.generation.turns = args.qa_turns
    if args.target_language:
        if not hasattr(config, "generation"):
            raise ValueError("--target-language requires a QA generation environment")
        config.generation.target_language = args.target_language

    if args.limit is not None:
        config.dataset.limit = args.limit
    if args.dataset_name:
        config.dataset.dataset_name = args.dataset_name
    if args.dataset_config:
        config.dataset.dataset_config = args.dataset_config
    if args.dataset_split:
        config.dataset.dataset_split = args.dataset_split
    if args.dataset_backend:
        config.dataset.dataset_backend = args.dataset_backend
    if args.dataset_offset is not None:
        config.dataset.dataset_offset = args.dataset_offset
    if args.max_turns:
        # Only set if RLM mode
        if hasattr(config, 'repl'):
            config.repl.max_turns = args.max_turns
    if args.server_url:
        if hasattr(config, 'repl'):
            config.repl.server_url = args.server_url
    if args.output_dir:
        config.dataset.output_dir = args.output_dir
    if args.output_basename:
        config.dataset.output_basename = args.output_basename
    if args.mask_observations:
        config.dataset.mask_observations = True
    if args.num_rollouts is not None:
        config.dataset.num_rollouts = args.num_rollouts
    if args.debug:
        config.debug = True
    if args.enable_reasoning:
        config.enable_reasoning = True
    if args.enable_correction_feedback:
        # Only applies to RLM mode
        mode_config = config.get_mode_config()
        from gymkhana.envs.config import RLMModeSettings
        if isinstance(mode_config, RLMModeSettings):
            mode_config.enable_correction_feedback = True

    output_dir = Path(config.dataset.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        Path(args.log_file).expanduser()
        if args.log_file
        else output_dir / "run.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if config.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )

    # Get LLM config for display
    llm_config = config.get_llm_config()

    print("=" * 60)
    print("Gymkhana Dataset Generation")
    print("=" * 60)
    print(f"Environment:  {env_name}")
    print(f"Dataset:      {config.dataset.dataset_name or 'default'}")
    print(f"Split:        {config.dataset.dataset_split}")
    model_identifier = getattr(llm_config, "model_identifier", llm_config.model)
    client_name = getattr(llm_config.client, "value", str(llm_config.client))
    print(f"Model:        {model_identifier} ({client_name})")
    if config.llm_judge:
        print(f"Judge model:  {config.llm_judge.model}")
    print(f"Mode:         {config.interaction_mode.value}")

    # Display mode-specific settings
    mode_config = config.get_mode_config()
    if mode_config:
        from gymkhana.envs.config import RLMModeSettings, ToolUseModeSettings, ChatModeSettings
        if isinstance(mode_config, RLMModeSettings):
            print(f"Server:       {mode_config.repl.server_url}")
            print(f"Max turns:    {mode_config.repl.max_turns}")
            print(f"Corrections:  {'enabled' if mode_config.enable_correction_feedback else 'disabled'} (correction prompts)")
        elif isinstance(mode_config, (ToolUseModeSettings, ChatModeSettings)):
            print(f"Max turns:    {mode_config.max_turns}")

    # Display common features (apply to all modes)
    print(f"Reasoning:    {'enabled' if config.enable_reasoning else 'disabled'} (<think> blocks)")

    limit_label = (
        config.dataset.limit if config.dataset.limit is not None else "all"
    )
    print(f"Limit:        {limit_label}")
    print(f"Batch size:   {args.batch_size or config.dataset.batch_size} (max concurrent tasks)")
    print(f"Num rollouts: {getattr(config.dataset, 'num_rollouts', 1)} (per task)")
    print(f"Output:       {config.dataset.output_dir}")
    print(f"Log:          {log_path}")
    print("=" * 60)

    # Initialize Database if configured
    data_inserter = None
    try:
        import os
        from gymkhana.core.services.storage.env_storage import EnvStorageService

        db_is_configured = any(
            os.getenv(name)
            for name in ("DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT")
        )
        if args.database is False or (args.database is None and not db_is_configured):
            print("Database insertion: DISABLED")
        else:
            # Map env vars to database service fields
            db_args = {}
            if os.getenv("DB_NAME"): db_args["db_name"] = os.getenv("DB_NAME")
            if os.getenv("DB_USER"): db_args["user"] = os.getenv("DB_USER")
            if os.getenv("DB_PASSWORD"): db_args["password"] = os.getenv("DB_PASSWORD")
            if os.getenv("DB_HOST"): db_args["host"] = os.getenv("DB_HOST")
            if os.getenv("DB_PORT"): db_args["port"] = int(os.getenv("DB_PORT", 5432))

            data_inserter = EnvStorageService(**db_args)
            print(f"Database insertion: ENABLED (User: {data_inserter.user}, DB: {data_inserter.db_name})")
    except Exception as e:
        print(f"Database insertion: DISABLED ({e})")

    async def run_pipeline():
        nonlocal data_inserter
        if data_inserter:
            try:
                await data_inserter.initialize()
            except OSError as e:
                print(f"Database unavailable ({e}). Running without persistence.")
                data_inserter = None
            except Exception as e:
                if "Connect" in str(e) or "Connection refused" in str(e) or "5432" in str(e) or "5433" in str(e):
                    print(f"Database unavailable ({e}). Running without persistence.")
                    data_inserter = None
                else:
                    raise

        # Async sandbox required for RLM mode only
        sandbox = None
        mode_config = config.get_mode_config()
        if mode_config:
            from gymkhana.envs.config import RLMModeSettings
            if isinstance(mode_config, RLMModeSettings):
                timeout_seconds = mode_config.repl.timeout_seconds
                sandbox = REPLSandbox(server_url=mode_config.repl.server_url, timeout_seconds=timeout_seconds)

        services = ServiceContainer(sandbox=sandbox)

        # max_parallel_rollouts = max concurrent tasks (semaphore). num_rollouts = rollouts per task (G sessions per task; no separate worker pool).
        env = env_cls(
            config=config,
            data_inserter=data_inserter,
            services=services,
            max_parallel_rollouts=args.batch_size or config.dataset.batch_size,
            num_rollouts=getattr(config.dataset, "num_rollouts", 1),
        )
        try:
            summary = await env.run(limit=config.dataset.limit)
            for artifact_name, artifact_path in summary.artifacts.items():
                print(f"Artifact:     {artifact_name}={artifact_path}")
        finally:
            await env.finalize()
            # Close REPL aiohttp client to avoid "Unclosed client session" warnings
            sandbox = getattr(getattr(env, "services", None), "sandbox", None)
            if sandbox and hasattr(sandbox, "client"):
                client = sandbox.client
                if hasattr(client, "close"):
                    await client.close()

    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
