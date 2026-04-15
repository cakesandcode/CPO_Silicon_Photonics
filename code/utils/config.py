"""
Shared configuration for ARM SW Stack Series notebooks and tools.

This module defines global constants for styling, color schemes, CPU data,
and metadata used across all Post 2 notebooks.
"""

# ISA tier color scheme (per MANIFEST UI standard)
ISA_COLORS = {
    "NEON": "#0066CC",      # Blue
    "SVE2": "#00CC66",      # Green
    "SME2": "#FFAA00",      # Amber
    "ArmPL": "#9933FF",     # Purple
    "KleidiAI": "#FF3333",  # Red
}

# Plotly dark theme template (black background, white text)
PLOTLY_DARK_TEMPLATE = {
    "paper_bgcolor": "#000000",
    "plot_bgcolor": "#111111",
    "font": {"color": "white", "family": "Arial, sans-serif", "size": 12},
    "title": {"font": {"size": 16, "color": "white"}},
    "xaxis": {"title": {"font": {"size": 12, "color": "white"}}, "tickfont": {"color": "white"}},
    "yaxis": {"title": {"font": {"size": 12, "color": "white"}}, "tickfont": {"color": "white"}},
    "hovermode": "closest",
}

# Neoverse CPU configurations: CPU family, ISA level, extension support, core count
NEOVERSE_CPUS = [
    {
        "cpu": "neoverse-v3",
        "isa_level": "armv9.2-a",
        "sve": True,
        "sve2": True,
        "sme": True,
        "sme2": True,
        "neon": True,
        "cores_per_socket": 128,
        "notes": "128-bit SVE width, streaming SVE (SME2), highest performance",
    },
    {
        "cpu": "neoverse-v2",
        "isa_level": "armv9.1-a",
        "sve": True,
        "sve2": True,
        "sme": False,
        "sme2": False,
        "neon": True,
        "cores_per_socket": 128,
        "notes": "128-bit SVE width, no SME2",
    },
    {
        "cpu": "neoverse-n3",
        "isa_level": "armv9.2-a",
        "sve": True,
        "sve2": True,
        "sme": False,
        "sme2": False,
        "neon": True,
        "cores_per_socket": 128,
        "notes": "General-purpose, SVE2 capable, no SME",
    },
    {
        "cpu": "neoverse-n2",
        "isa_level": "armv9.0-a",
        "sve": True,
        "sve2": False,
        "sme": False,
        "sme2": False,
        "neon": True,
        "cores_per_socket": 128,
        "notes": "General-purpose, SVE only (no SVE2)",
    },
    {
        "cpu": "neoverse-n1",
        "isa_level": "armv8.2-a",
        "sve": False,
        "sve2": False,
        "sme": False,
        "sme2": False,
        "neon": True,
        "cores_per_socket": 64,
        "notes": "NEON only, no SVE",
    },
    {
        "cpu": "cortex-a78",
        "isa_level": "armv8.2-a",
        "sve": False,
        "sve2": False,
        "sme": False,
        "sme2": False,
        "neon": True,
        "cores_per_socket": 8,
        "notes": "Mobile/edge, NEON only",
    },
]

# Notebook series and version metadata
NOTEBOOK_METADATA = {
    "series": "ARM SW Stack Series",
    "post": 2,
    "title": "Building Your Arm HPC Stack: Compilers, Math Libraries, and Cluster Tools",
    "version": "2026-04-10",
    "author": "Aruna Kumar",
}

# Benchmark cache hierarchy and performance constants (Neoverse V3 typical)
BENCHMARK_CACHE_CONFIG = {
    "L1_SIZE_BYTES": 32 * 1024,       # 32 KB per core
    "L2_SIZE_BYTES": 512 * 1024,      # 512 KB per core
    "L3_SIZE_BYTES": 8 * 1024 * 1024, # 8 MB per core (typical V3)
    "DRAM_BW_GBPS": 200,              # Approximate for Neoverse V3
    "PEAK_GFLOPS": 2560,              # V3 @ 3.5 GHz (2 FMA × 128 lanes / 64-bit)
}

# Slurm binding configuration
SLURM_BINDING_CONFIG = {
    "valid_cpu_binds": ["mask_cpu", "rank", "none"],
    "valid_mem_binds": ["local", "none"],
}
