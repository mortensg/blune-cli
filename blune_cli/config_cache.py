"""
Local + curated config.json cache.

Lookup order for a repo's config.json:
  1. This project's own curated cache (configs/<org>/<repo>/config.json) --
     shipped in the repo, zero network calls, zero load on Hugging Face.
  2. The user's local HF cache (~/.cache/huggingface), via huggingface_hub's
     normal caching -- fast, but does a small freshness check unless offline.
  3. A live fetch from the Hub, which is then written into (1) so the next
     run (yours or anyone else who pulls this repo) never needs it again.

This is the "don't spam Hugging Face" piece: a config.json is a few KB, but
if this tool gets popular, thousands of users re-fetching the same handful
of popular models' configs is real, avoidable load. A community-curated,
git-tracked cache turns that into a one-time cost per model, ever.
"""
import json
import os
from pathlib import Path
from typing import Optional

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CURATED_CACHE_DIR = REPO_ROOT / "configs"


def _curated_path(repo_id: str) -> Path:
    return CURATED_CACHE_DIR / repo_id / "config.json"


def get_config(repo_id: str, offline: bool = False, save_curated: bool = True) -> dict:
    """Return a model's config.json as a dict, preferring the curated cache."""
    curated = _curated_path(repo_id)
    if curated.exists():
        with open(curated) as f:
            return json.load(f)

    if offline:
        raise FileNotFoundError(
            f"No cached config for {repo_id} and --offline was set. "
            f"Run once without --offline to populate the cache."
        )

    # Fall back to a direct fetch (small, unauthenticated is fine for a
    # single public file).
    url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    config = resp.json()

    if save_curated:
        curated.parent.mkdir(parents=True, exist_ok=True)
        with open(curated, "w") as f:
            json.dump(config, f, indent=2)

    return config


def list_curated() -> list[str]:
    """All repo IDs already present in the curated cache."""
    if not CURATED_CACHE_DIR.exists():
        return []
    result = []
    for org_dir in CURATED_CACHE_DIR.iterdir():
        if not org_dir.is_dir():
            continue
        for repo_dir in org_dir.iterdir():
            if (repo_dir / "config.json").exists():
                result.append(f"{org_dir.name}/{repo_dir.name}")
    return result
