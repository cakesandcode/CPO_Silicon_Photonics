"""
Linaro Forge MAP profile parser and flamechart generator.

Parses synthetic or real Linaro MAP profile data in JSON format, generates
flamechart-equivalent visualizations, and identifies performance optimization
opportunities through heuristic analysis.

Design: production utilities for Lab 4 (Linaro Forge MAP Walkthrough).
"""

from code.utils.config import ISA_COLORS


def parse_map_file(data: dict) -> dict:
    """
    Parse MAP profile data and normalize to standard structure.

    Validates and extracts metadata, time breakdown, and hotspot information
    from a dict representing MAP profile output. Intended for synthetic JSON
    data shaped like real MAP binary output structure.

    Args:
        data: Dict with keys 'metadata', 'time_breakdown', 'hotspots'.
            metadata: 'tool', 'generated_by', 'app', 'ranks', 'runtime_s'
            time_breakdown: 'cpu_pct', 'mpi_pct', 'io_pct', 'thread_overhead_pct'
            hotspots: list of dicts with 'function', 'file', 'line', 'cpu_pct', 'mpi_pct'

    Returns:
        Dict with keys: total_runtime_s, ranks, samples, cpu_time_pct, mpi_time_pct,
        io_time_pct, thread_overhead_pct, hotspots (list of dicts).

    Raises:
        KeyError: if required keys missing from input data.
        ValueError: if percentages do not normalize or runtime <= 0.
    """
    if "metadata" not in data or "time_breakdown" not in data or "hotspots" not in data:
        raise KeyError(
            "Input data must contain 'metadata', 'time_breakdown', 'hotspots' keys"
        )

    metadata = data["metadata"]
    time_breakdown = data["time_breakdown"]
    hotspots = data["hotspots"]

    runtime_s = metadata.get("runtime_s")
    if runtime_s is None or runtime_s <= 0:
        raise ValueError(f"Invalid runtime_s: {runtime_s}")

    ranks = metadata.get("ranks", 1)
    if ranks < 1:
        raise ValueError(f"Invalid ranks: {ranks}")

    cpu_pct = time_breakdown.get("cpu_pct", 0.0)
    mpi_pct = time_breakdown.get("mpi_pct", 0.0)
    io_pct = time_breakdown.get("io_pct", 0.0)
    thread_overhead_pct = time_breakdown.get("thread_overhead_pct", 0.0)

    total_pct = cpu_pct + mpi_pct + io_pct + thread_overhead_pct
    if abs(total_pct - 100.0) > 1.0:
        raise ValueError(
            f"Time breakdown percentages sum to {total_pct}%, expected ~100%"
        )

    normalized_hotspots = []
    for hs in hotspots:
        normalized_hotspots.append(
            {
                "function_name": hs.get("function", "unknown"),
                "file": hs.get("file", "unknown"),
                "line": hs.get("line", 0),
                "pct_cpu_time": hs.get("cpu_pct", 0.0),
                "pct_mpi_time": hs.get("mpi_pct", 0.0),
            }
        )

    return {
        "total_runtime_s": runtime_s,
        "ranks": ranks,
        "samples": len(normalized_hotspots),
        "cpu_time_pct": cpu_pct,
        "mpi_time_pct": mpi_pct,
        "io_time_pct": io_pct,
        "thread_overhead_pct": thread_overhead_pct,
        "hotspots": normalized_hotspots,
    }


def generate_flamechart_data(profile: dict) -> list[dict]:
    """
    Convert profile hotspot data to flamechart-compatible format.

    Maps hotspots to horizontal bar chart data with positioning, coloring by
    workload category (compute, MPI, I/O, other). Uses ISA tier color scheme:
    - compute (high cpu_pct, low mpi_pct) → SVE2 green
    - MPI (high mpi_pct) → NEON blue
    - I/O (tagged) → amber
    - other → gray

    Args:
        profile: Dict returned by parse_map_file().

    Returns:
        List of dicts: x_start (float), x_end (float), y_level (int), label (str),
        color (str hex), pct (float).
        x_start/x_end: position in runtime window (0–100%).
        y_level: vertical position (rank index).
        label: function name.
        color: hex code from ISA tier scheme.
        pct: CPU time percentage.
    """
    if not profile.get("hotspots"):
        return []

    hotspots = profile["hotspots"]
    hotspots_sorted = sorted(
        hotspots, key=lambda h: h["pct_cpu_time"], reverse=True
    )

    flamechart_data = []
    cumulative_x = 0.0

    for rank_idx, hs in enumerate(hotspots_sorted):
        cpu_pct = hs["pct_cpu_time"]
        mpi_pct = hs["pct_mpi_time"]
        func_name = hs["function_name"]

        if cpu_pct <= 0:
            continue

        x_start = cumulative_x
        x_end = cumulative_x + cpu_pct
        cumulative_x = x_end

        if cpu_pct > 20.0 and mpi_pct < 5.0:
            color = ISA_COLORS["SVE2"]  # Green: compute-bound
        elif mpi_pct > 10.0:
            color = ISA_COLORS["NEON"]  # Blue: MPI-bound
        elif "MPI_" in func_name or "allreduce" in func_name.lower():
            color = ISA_COLORS["NEON"]  # Blue: MPI communication
        else:
            color = "#CCCCCC"  # Gray: other

        flamechart_data.append(
            {
                "x_start": x_start,
                "x_end": x_end,
                "y_level": rank_idx,
                "label": func_name,
                "color": color,
                "pct": cpu_pct,
            }
        )

    return flamechart_data


def analyze_profile_for_optimization(profile: dict) -> list[dict]:
    """
    Identify optimization opportunities from profile metrics.

    Applies heuristic rules to profile time breakdown and hotspots to suggest
    optimization strategies:
    - MPI time > 20%: overlap or non-blocking collectives
    - I/O time > 10%: MPI-IO or buffered writes
    - CPU > 70% + concentrated hotspot > 30%: vectorize or use ArmPL

    Args:
        profile: Dict returned by parse_map_file().

    Returns:
        List of dicts: priority (HIGH/MEDIUM/LOW), category (str), suggestion (str),
        metric_value (float). Sorted by priority descending.
    """
    recommendations = []

    mpi_pct = profile.get("mpi_time_pct", 0.0)
    io_pct = profile.get("io_time_pct", 0.0)
    cpu_pct = profile.get("cpu_time_pct", 0.0)
    hotspots = profile.get("hotspots", [])

    if mpi_pct > 20.0:
        recommendations.append(
            {
                "priority": "HIGH",
                "category": "MPI",
                "suggestion": "MPI time dominates ({:.1f}%). Consider overlapping MPI collectives with computation or use non-blocking MPI calls.".format(
                    mpi_pct
                ),
                "metric_value": mpi_pct,
            }
        )

    if io_pct > 10.0:
        recommendations.append(
            {
                "priority": "HIGH",
                "category": "I/O",
                "suggestion": "I/O time significant ({:.1f}%). Consider buffering, MPI-IO, or asynchronous writes.".format(
                    io_pct
                ),
                "metric_value": io_pct,
            }
        )

    if cpu_pct > 70.0 and hotspots:
        top_hotspot = max(hotspots, key=lambda h: h["pct_cpu_time"])
        top_pct = top_hotspot["pct_cpu_time"]
        if top_pct > 30.0:
            func_name = top_hotspot["function_name"]
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "Compute",
                    "suggestion": "Computational hotspot in {} ({:.1f}% of total runtime). Candidate for SVE2 loop optimization or ArmPL BLAS (DGEMM, etc.).".format(
                        func_name, top_pct
                    ),
                    "metric_value": top_pct,
                }
            )

    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recommendations.sort(key=lambda r: priority_order.get(r["priority"], 3))

    return recommendations
