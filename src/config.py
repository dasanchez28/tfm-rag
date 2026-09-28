"""Centralised configuration loading (.env + config/config.yaml)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Project root (two levels above this file)
ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")


def load_config(path: str | None = None) -> dict:
    """Load the YAML configuration file."""
    cfg_path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Environment variables (defaults match the reported experiments) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "local")
EMBEDDINGS_MODEL = os.getenv(
    "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

CONFIG = load_config()

# Language in which the generator writes its answers
ANSWER_LANGUAGE = CONFIG.get("generation", {}).get("answer_language", "Spanish")

# Absolute paths derived from the configuration
RAW_DIR = ROOT / CONFIG["data"]["raw_dir"]
PROCESSED_DIR = ROOT / CONFIG["data"]["processed_dir"]
FIGURES_DIR = ROOT / CONFIG["figures_dir"]
RESULTS_DIR = ROOT / CONFIG["evaluation"]["results_dir"]

for _d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix the seeds of the random number generators used by the pipeline."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
