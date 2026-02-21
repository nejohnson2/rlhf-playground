"""YAML config loading with hierarchical merging."""

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> DictConfig:
    """Load a YAML config file.

    Args:
        config_path: Path to YAML file.

    Returns:
        OmegaConf DictConfig object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    logger.info("Loaded config from %s", config_path)
    return cfg


def merge_configs(*configs: DictConfig | dict) -> DictConfig:
    """Merge multiple configs, later ones override earlier ones.

    Args:
        *configs: Config dicts or DictConfig objects.

    Returns:
        Merged DictConfig.
    """
    return OmegaConf.merge(*configs)


def apply_cli_overrides(config: DictConfig, overrides: dict) -> DictConfig:
    """Apply CLI argument overrides to a config.

    Args:
        config: Base config.
        overrides: Dict of key=value overrides (dot-notation keys supported).

    Returns:
        Updated DictConfig.
    """
    override_cfg = OmegaConf.from_dotlist(
        [f"{k}={v}" for k, v in overrides.items() if v is not None]
    )
    return OmegaConf.merge(config, override_cfg)
