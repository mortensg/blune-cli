"""
Auto-detect the machine this tool is running on: chip name, total memory,
and (if we have it) real memory bandwidth -- without the user typing
anything in.
"""
import platform
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

from .hardware_data import BY_ID, ChipSpec


@dataclass
class DetectedMachine:
    os_name: str
    chip_name: str
    total_ram_gb: float
    chip_spec: Optional[ChipSpec]  # None if we don't recognize this exact chip

    @property
    def bandwidth_gbs(self) -> Optional[float]:
        return self.chip_spec.memory_bandwidth_gbs if self.chip_spec else None

    @property
    def sustained_bandwidth_gbs(self) -> Optional[float]:
        """Realistic bandwidth under continuous inference load, not the
        spec-sheet peak -- see ChipSpec.sustained_bandwidth_ratio."""
        if not self.chip_spec:
            return None
        return self.chip_spec.memory_bandwidth_gbs * self.chip_spec.sustained_bandwidth_ratio


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return ""


def _detect_macos() -> DetectedMachine:
    brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    mem_bytes = _run(["sysctl", "-n", "hw.memsize"])
    total_ram_gb = round(int(mem_bytes) / (1024**3)) if mem_bytes.isdigit() else 0

    # sysctl brand_string on Apple Silicon looks like "Apple M4 Pro"
    chip_spec = None
    if brand:
        # normalize "Apple M4 Pro" -> "m4_pro", "Apple M2 Max" -> "m2_max"
        m = re.search(r"Apple (M\d+)\s*(Pro|Max|Ultra)?", brand)
        if m:
            gen = m.group(1).lower()
            tier = (m.group(2) or "").lower()
            key = f"{gen}_{tier}" if tier else gen
            chip_spec = BY_ID.get(key)

    return DetectedMachine(
        os_name="macOS",
        chip_name=brand or "unknown Apple Silicon",
        total_ram_gb=total_ram_gb,
        chip_spec=chip_spec,
    )


def _detect_nvidia_gpu() -> Optional[tuple[str, float, int]]:
    """Returns (name, bandwidth_gbs, vram_gb) via nvidia-smi if present."""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return None
    line = out.splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return None
    name = parts[0]
    vram_mb = float(parts[1])
    # nvidia-smi doesn't report bandwidth directly; nvidia-ml-py / deviceQuery
    # would, but isn't always installed. We report VRAM + name and let the
    # static fallback table supply bandwidth if we recognize the model.
    return name, vram_mb / 1024


def _detect_linux_or_windows() -> DetectedMachine:
    os_name = platform.system()
    total_ram_gb = 0
    try:
        if os_name == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        total_ram_gb = round(kb / (1024**2))
                        break
    except Exception:
        pass

    nv = _detect_nvidia_gpu()
    chip_spec = None
    chip_name = platform.processor() or "unknown CPU"
    if nv:
        name, vram_gb = nv
        chip_name = name
        # try to match against our small fallback table by substring
        for c in BY_ID.values():
            if c.vendor == "NVIDIA" and c.name.lower() in name.lower():
                chip_spec = c
                break

    return DetectedMachine(
        os_name=os_name,
        chip_name=chip_name,
        total_ram_gb=total_ram_gb,
        chip_spec=chip_spec,
    )


def detect_machine() -> DetectedMachine:
    """Entry point: figure out what machine we're on, no user input needed."""
    if platform.system() == "Darwin":
        return _detect_macos()
    return _detect_linux_or_windows()
