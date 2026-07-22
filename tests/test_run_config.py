from pathlib import Path

from gymkhana.envs.english_sharegpt_nepali import EnglishShareGPTToNepaliEnv
from gymkhana.run import load_environment_config


def test_yaml_config_overrides_environment_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "openhermes.yaml"
    config_path.write_text(
        """
name: english-sharegpt-to-nepali
llm:
  model: anthropic:claude-sonnet-4-6
  client: anthropic
llm_judge:
  model: anthropic:claude-haiku-4-5
  client: anthropic
dataset:
  dataset_name: teknium/OpenHermes-2.5
  dataset_backend: huggingface-rows
  dataset_offset: 250
  limit: 100
  output_dir: outputs/openhermes
  output_basename: openhermes_nepali
""".strip(),
        encoding="utf-8",
    )

    env_name, config = load_environment_config(config_path)

    assert env_name == "english-sharegpt-to-nepali"
    assert isinstance(
        EnglishShareGPTToNepaliEnv.default_config,
        type(config),
    )
    assert config.llm.model == "anthropic:claude-sonnet-4-6"
    assert config.llm.client.value == "anthropic"
    assert config.llm_judge.model == "anthropic:claude-haiku-4-5"
    assert config.dataset.dataset_backend == "huggingface-rows"
    assert config.dataset.dataset_offset == 250
    assert config.dataset.output_basename == "openhermes_nepali"


def test_cli_environment_override_wins_over_yaml_name(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "name: english-sharegpt-to-nepali\ndataset:\n  limit: 3\n",
        encoding="utf-8",
    )

    env_name, config = load_environment_config(
        config_path,
        environment_override="english-sharegpt-to-nepali",
    )

    assert env_name == "english-sharegpt-to-nepali"
    assert config.dataset.limit == 3
