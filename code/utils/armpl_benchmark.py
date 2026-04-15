"""
ArmPL BLAS and FFT benchmark utilities for Lab 2.

Provides synthetic DGEMM throughput curves, roofline models, and FFT scaling data.
All synthetic data is explicitly labeled as illustrative, based on known Neoverse V3
cache hierarchy and ArmPL performance characteristics.

Hardware assumptions (not verified — illustrative):
  - L1 data cache: 32 KB per core
  - L2 cache: 512 KB per core
  - L3 cache: 8 MB per core (typical Neoverse V3)
  - Peak throughput (Neoverse V3 @ 3.5 GHz with SVE2): ~2560 GFLOP/s

No published Neoverse V3 ArmPL vs MKL benchmark found in public corpus as of 2026-04-10.
ArmPL throughput advantage (15-25% on large matrices) is inferred from SVE2 benefits
and published SVE performance studies.
"""

import json
from pathlib import Path

import pandas as pd

from code.utils.config import BENCHMARK_CACHE_CONFIG


def generate_dgemm_throughput(
    sizes: list[int], library: str = "numpy"
) -> pd.DataFrame:
    """
    Generate synthetic DGEMM throughput curves with cache-aware behavior.

    For 'numpy', generates realistic numpy.matmul() curve showing cache knees.
    For 'armpl', generates synthetic curve 15-25% higher on large matrices
    (demonstrating SVE2 benefits).

    Args:
        sizes: List of matrix sizes (NxN square matrices).
        library: 'numpy' or 'armpl'.

    Returns:
        pd.DataFrame with columns: matrix_size, throughput_gflops, library.
            Each row represents one matrix size with estimated throughput.

    Raises:
        ValueError: if library not in ('numpy', 'armpl').
    """
    if library not in ("numpy", "armpl"):
        raise ValueError(f"library must be 'numpy' or 'armpl', got {library}")

    # Cache boundaries and performance constants (from config)
    L1_SIZE_BYTES = BENCHMARK_CACHE_CONFIG["L1_SIZE_BYTES"]
    L2_SIZE_BYTES = BENCHMARK_CACHE_CONFIG["L2_SIZE_BYTES"]
    L3_SIZE_BYTES = BENCHMARK_CACHE_CONFIG["L3_SIZE_BYTES"]
    DRAM_BW_GBPS = BENCHMARK_CACHE_CONFIG["DRAM_BW_GBPS"]
    PEAK_GFLOPS = BENCHMARK_CACHE_CONFIG["PEAK_GFLOPS"]

    throughputs = []

    for size in sizes:
        # Memory footprint: 3 NxN matrices (A, B, C) with 8 bytes per element
        matrix_bytes = 3 * size * size * 8

        # Arithmetic intensity: (N^3 ops * 2 FLOP/mul) / (N^2 * 8 bytes per elem * 3)
        #                     = (2*N) / 24 FLOP/byte
        ai = (2 * size) / 24.0  # FLOPs per byte

        # Roofline: throughput = min(peak_gflops, arithmetic_intensity * bandwidth)
        bandwidth_limited = ai * DRAM_BW_GBPS

        # Cache-aware penalty based on working set size vs cache levels
        if matrix_bytes < L1_SIZE_BYTES:
            # Fits in L1: ~95% of peak
            throughput = min(PEAK_GFLOPS, bandwidth_limited) * 0.95
        elif matrix_bytes < L2_SIZE_BYTES:
            # L2: ~70% of peak (L1 evictions)
            throughput = min(PEAK_GFLOPS, bandwidth_limited) * 0.70
        elif matrix_bytes < L3_SIZE_BYTES:
            # L3: ~40% of peak (L2 evictions)
            throughput = min(PEAK_GFLOPS, bandwidth_limited) * 0.40
        else:
            # DRAM: ~10% of peak (NUMA/bandwidth limited)
            throughput = min(PEAK_GFLOPS, bandwidth_limited) * 0.10

        throughputs.append(throughput)

    # ArmPL curve: 15-25% higher on large matrices due to SVE2 benefits
    if library == "armpl":
        max_size = max(sizes) if sizes else 1
        throughputs = [
            t * (1.0 + 0.25 * min(1.0, (size / max_size) ** 0.8))  # Scales ~0% (small) to ~25% (large)
            for t, size in zip(throughputs, sizes)
        ]

    df = pd.DataFrame(
        {
            "matrix_size": sizes,
            "throughput_gflops": throughputs,
            "library": library,
        }
    )

    return df


def load_library_comparison(data_path: Path) -> pd.DataFrame:
    """
    Load pre-generated DGEMM comparison data from JSON.

    Args:
        data_path: Path to JSON file (e.g., code/data/dgemm_comparison.json).

    Returns:
        pd.DataFrame with columns: matrix_size, numpy_gflops, armpl_gflops.

    Raises:
        FileNotFoundError: if data_path does not exist.
        ValueError: if JSON structure is invalid.
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path) as f:
        data = json.load(f)

    if "data" not in data:
        raise ValueError("JSON must contain 'data' key with list of records")

    df = pd.DataFrame(data["data"])

    required_cols = {"matrix_size", "numpy_gflops", "armpl_gflops"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    return df


def compute_roofline_model(
    peak_gflops: float, peak_bandwidth_gbs: float, sizes: list[int]
) -> pd.DataFrame:
    """
    Compute roofline model for DGEMM workloads.

    The roofline model bounds throughput as:
        throughput = min(peak_gflops, arithmetic_intensity × bandwidth)

    For DGEMM on NxN matrices:
        arithmetic_intensity = (2*N) / 24 FLOP/byte

    Args:
        peak_gflops: Peak theoretical throughput (GFLOP/s). Must be > 0.
        peak_bandwidth_gbs: Memory bandwidth (GB/s). Must be > 0.
        sizes: List of matrix sizes. Each must be > 0.

    Returns:
        pd.DataFrame with columns: matrix_size, roofline_gflops.
            Each row is the roofline ceiling for that matrix size.

    Raises:
        ValueError: if peak_gflops <= 0, peak_bandwidth_gbs <= 0, or any size <= 0.
    """
    if peak_gflops <= 0:
        raise ValueError(f"peak_gflops must be > 0, got {peak_gflops}")
    if peak_bandwidth_gbs <= 0:
        raise ValueError(f"peak_bandwidth_gbs must be > 0, got {peak_bandwidth_gbs}")
    if any(s <= 0 for s in sizes):
        raise ValueError(f"All sizes must be > 0, got {sizes}")

    rooflines = []

    for size in sizes:
        # Arithmetic intensity for DGEMM: (2*N) / (3*N^2*8 bytes / N^2 elements)
        ai = (2 * size) / 24.0

        # Roofline: min of compute-bound and bandwidth-bound limits
        roofline = min(peak_gflops, ai * peak_bandwidth_gbs)
        rooflines.append(roofline)

    df = pd.DataFrame({"matrix_size": sizes, "roofline_gflops": rooflines})

    return df


def load_fft_scaling(data_path: Path) -> pd.DataFrame:
    """
    Load FFT throughput scaling data from JSON.

    Args:
        data_path: Path to JSON file (e.g., code/data/fft_scaling.json).

    Returns:
        pd.DataFrame with columns: fft_size, armpl_gflops, numpy_fft_gflops.

    Raises:
        FileNotFoundError: if data_path does not exist.
        ValueError: if JSON structure is invalid.
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path) as f:
        data = json.load(f)

    if "data" not in data:
        raise ValueError("JSON must contain 'data' key with list of records")

    df = pd.DataFrame(data["data"])

    required_cols = {"fft_size", "armpl_gflops", "numpy_fft_gflops"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    return df
