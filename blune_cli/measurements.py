"""Load and query the real, measured-not-estimated tokens/sec dataset."""
import json
from pathlib import Path
from typing import Optional

_DATA_PATH = Path(__file__).parent / "measurements.json"


def load_measurements() -> list[dict]:
    with open(_DATA_PATH) as f:
        return json.load(f)["measurements"]


def _normalize_library(library: str) -> str:
    """'vllm (metal)' and 'vllm' should match the same probe request --
    the parenthetical is just a display detail about the backend."""
    return library.split(" ")[0].strip().lower()


def find_real_measurement(
    repo_id: str, library: str, machine: Optional[str] = None
) -> Optional[dict]:
    """Return a real measurement for this exact repo+library(+machine) if
    one has been recorded, else None. Prefer this over any estimate."""
    for m in load_measurements():
        if m["repo_id"] != repo_id:
            continue
        if _normalize_library(m["library"]) != _normalize_library(library):
            continue
        if machine is not None and m["machine"] != machine:
            continue
        return m
    return None
