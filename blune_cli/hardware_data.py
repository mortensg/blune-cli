"""
Static hardware specification database.

Bandwidth figures are real, measured/published specs (GB/s), not vendor
marketing peaks where a real-world figure is known to differ. Apple Silicon
entries explicitly include the "Pro" tier chips that most public GPU
databases (built for discrete-GPU PCs) omit -- this project started because
of exactly that gap (see README "Why this exists").
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChipSpec:
    id: str
    name: str
    vendor: str
    memory_bandwidth_gbs: float
    max_unified_or_vram_gb: Optional[int] = None
    tflops16: Optional[float] = None
    notes: str = ""


APPLE_SILICON: list[ChipSpec] = [
    ChipSpec("m1", "Apple M1", "Apple", 68.25, 16),
    ChipSpec("m1_pro", "Apple M1 Pro", "Apple", 200, 32),
    ChipSpec("m1_max", "Apple M1 Max", "Apple", 400, 64),
    ChipSpec("m1_ultra", "Apple M1 Ultra", "Apple", 800, 128),
    ChipSpec("m2", "Apple M2", "Apple", 100, 24),
    ChipSpec("m2_pro", "Apple M2 Pro", "Apple", 200, 32),
    ChipSpec("m2_max", "Apple M2 Max", "Apple", 400, 96),
    ChipSpec("m2_ultra", "Apple M2 Ultra", "Apple", 800, 192),
    ChipSpec("m3", "Apple M3", "Apple", 100, 24),
    ChipSpec("m3_pro", "Apple M3 Pro", "Apple", 150, 36),
    ChipSpec("m3_max", "Apple M3 Max", "Apple", 400, 128),
    ChipSpec("m3_ultra", "Apple M3 Ultra", "Apple", 819, 512),
    ChipSpec("m4", "Apple M4", "Apple", 120, 32),
    # The chip this whole project started with: missing from every public
    # GPU calculator we checked (llm-vram-calculator, quantprobe's Apple path).
    ChipSpec("m4_pro", "Apple M4 Pro", "Apple", 273, 48, 18.43),
    ChipSpec("m4_max", "Apple M4 Max", "Apple", 546, 128, 21.2),
    ChipSpec("m4_ultra", "Apple M4 Ultra", "Apple", 820, 192, 39.6),
]

# NVIDIA/AMD entries kept intentionally small -- for discrete GPUs we prefer
# live detection via nvidia-smi/rocm-smi (see detect_gpu_bandwidth below)
# over a static table that goes stale. These are fallbacks only.
DISCRETE_GPU_FALLBACK: list[ChipSpec] = [
    ChipSpec("rtx4090", "RTX 4090", "NVIDIA", 1008, 24, 82.6),
    ChipSpec("rtx4080", "RTX 4080", "NVIDIA", 717, 16, 48.7),
    ChipSpec("rtx3090", "RTX 3090", "NVIDIA", 936, 24, 35.6),
    ChipSpec("a100_80", "A100 80GB", "NVIDIA", 2039, 80, 312),
    ChipSpec("h100_sxm", "H100 SXM5", "NVIDIA", 3350, 80, 989),
    ChipSpec("rx7900xtx", "RX 7900 XTX", "AMD", 960, 24, 61.4),
]

ALL_CHIPS = APPLE_SILICON + DISCRETE_GPU_FALLBACK
BY_ID = {c.id: c for c in ALL_CHIPS}
