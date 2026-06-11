"""Configuration and optional API client helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def env_or_config(name: str, config: dict[str, Any], default: str | None = None) -> str | None:
    return os.getenv(name) or config.get(name.lower()) or config.get(name) or default


def load_api_config(path: str | Path | None = None) -> dict[str, Any]:
    config = load_yaml(path) if path else {}
    return {
        "openai_api_key": env_or_config("OPENAI_API_KEY", config),
        "openai_base_url": env_or_config("OPENAI_BASE_URL", config),
        "anthropic_api_key": env_or_config("ANTHROPIC_API_KEY", config),
        "dashscope_api_key": env_or_config("DASHSCOPE_API_KEY", config),
        "dashscope_base_url": env_or_config("DASHSCOPE_BASE_URL", config),
        "hf_token": env_or_config("HF_TOKEN", config),
    }
