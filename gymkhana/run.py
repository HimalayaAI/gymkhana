"""CLI entry point for Gymkhana dataset generation."""

import asyncio
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from gymkhana.envs import get_environment
from gymkhana.envs.config import EnvConfig, InferenceConfig, LLMClientType
from gymkhana.core.services import ServiceContainer
from gymkhana.core.services.sandboxes import REPLSandbox


def main():
    parser = argparse.ArgumentParser(
        description="Generate interleaved reasoning + code training data"
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Config YAML file (e.g., configs/default_config.yaml)"
    )
    parser.add_argument(
        "--env",
        choices=["math-python", "oolong", "code", "hotpotqa", "swe", "ifeval", "tool-use-singleturn", "romanized-nepali"],
        default=None,
        help="Task environment (overrides config)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Main model (overrides config)"
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

    # Get environment and its default config
    env_name = args.env or "math-python"
    env_cls = get_environment(env_name)

    # Get default config - handle both class attribute and lazy loading
    if hasattr(env_cls, 'default_config') and env_cls.default_config is not None:
        config = env_cls.default_config.model_copy(deep=True)
    else:
        # For environments with lazy config loading, instantiate to get config
        temp_env = env_cls()
        config = temp_env.config.model_copy(deep=True)

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

    if args.limit:
        config.dataset.limit = args.limit
    if args.max_turns:
        # Only set if RLM mode
        if hasattr(config, 'repl'):
            config.repl.max_turns = args.max_turns
    if args.server_url:
        if hasattr(config, 'repl'):
            config.repl.server_url = args.server_url
    if args.output_dir:
        config.dataset.output_dir = args.output_dir
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

    print(f"Limit:        {config.dataset.limit or 'all'}")
    print(f"Batch size:   {args.batch_size or config.dataset.batch_size} (max concurrent tasks)")
    print(f"Num rollouts: {getattr(config.dataset, 'num_rollouts', 1)} (per task)")
    print(f"Output:       {config.dataset.output_dir}")
    print("=" * 60)

    # Initialize Database if configured
    data_inserter = None
    try:
        import os
        from gymkhana.core.services.storage.env_storage import EnvStorageService

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
        await env.setup()
        try:
            await env.run(limit=config.dataset.limit)
        finally:
            # Close REPL aiohttp client to avoid "Unclosed client session" warnings
            sandbox = getattr(getattr(env, "services", None), "sandbox", None)
            if sandbox and hasattr(sandbox, "client"):
                client = sandbox.client
                if hasattr(client, "close"):
                    await client.close()

    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
