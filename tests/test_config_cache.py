import pytest

from blune_cli.config_cache import get_config, list_curated


def test_list_curated_finds_shipped_configs():
    curated = list_curated()
    assert "mlx-community/gpt-oss-20b-OptiQ-4bit" in curated
    assert "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit" in curated


def test_get_config_reads_curated_cache_without_network():
    config = get_config("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", offline=True)
    assert "model_type" in config


def test_get_config_offline_miss_raises():
    with pytest.raises(FileNotFoundError):
        get_config("some-org/definitely-not-cached-repo", offline=True)
