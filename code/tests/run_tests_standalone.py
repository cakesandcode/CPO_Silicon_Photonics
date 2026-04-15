"""
Standalone test runner for ARM SW Stack Series — Post 2 Labs.

Runs all utility module tests without pytest. Use this in sandboxed environments.
On your Mac with full dependencies, run: pytest code/tests/ -v --tb=short

Requirements for this runner: numpy, pandas (no plotly needed for utils).
"""

import sys
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PASS = 0
FAIL = 0
ERRORS: list[str] = []


def check(name: str, condition: bool, msg: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}" + (f" — {msg}" if msg else ""))
        FAIL += 1
        ERRORS.append(name)


def run_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# config.py
# ─────────────────────────────────────────────────────────────
run_section("config.py")
try:
    from code.utils.config import ISA_COLORS, PLOTLY_DARK_TEMPLATE, NEOVERSE_CPUS, NOTEBOOK_METADATA
    check("ISA_COLORS has NEON", "NEON" in ISA_COLORS)
    check("ISA_COLORS has SVE2", "SVE2" in ISA_COLORS)
    check("ISA_COLORS has SME2", "SME2" in ISA_COLORS)
    check("ISA_COLORS has ArmPL", "ArmPL" in ISA_COLORS)
    check("ISA_COLORS has KleidiAI", "KleidiAI" in ISA_COLORS)
    check("PLOTLY_DARK_TEMPLATE has paper_bgcolor", "paper_bgcolor" in PLOTLY_DARK_TEMPLATE)
    check("NEOVERSE_CPUS is non-empty list", isinstance(NEOVERSE_CPUS, list) and len(NEOVERSE_CPUS) > 0)
except Exception as e:
    print(f"  ERROR  config import failed: {e}")
    FAIL += 1


# ─────────────────────────────────────────────────────────────
# acfl_flags.py
# ─────────────────────────────────────────────────────────────
run_section("acfl_flags.py")
try:
    from code.utils.acfl_flags import parse_acfl_command, mcpu_to_extensions, load_vectorization_report

    r = parse_acfl_command("armclang -O3 -mcpu=neoverse-v3 file.c")
    check("parse basic: compiler=armclang", r.get("compiler") == "armclang")
    check("parse basic: opt_level=-O3", r.get("opt_level") == "-O3")
    check("parse basic: mcpu=neoverse-v3", r.get("mcpu") == "neoverse-v3")
    check("parse basic: SVE2 in inferred_extensions", "SVE2" in r.get("inferred_extensions", []))

    r2 = parse_acfl_command("armclang -march=armv9.2-a+sve2 file.c")
    check("parse -march: SVE2 inferred", "SVE2" in r2.get("inferred_extensions", []))

    e = mcpu_to_extensions("neoverse-v3")
    check("neoverse-v3: sve2=True", e.get("sve2") is True)
    check("neoverse-v3: sme2=True", e.get("sme2") is True)

    e2 = mcpu_to_extensions("neoverse-n3")
    check("neoverse-n3: sve2=True", e2.get("sve2") is True)

    try:
        mcpu_to_extensions("not-a-real-cpu")
        check("KeyError on unknown cpu", False, "no exception raised")
    except KeyError:
        check("KeyError on unknown cpu", True)

    mock_path = Path("code/data/acfl_sve2_mock_output.txt")
    report, lines = load_vectorization_report(mock_path)
    check("vectorization report: vectorized_loops key", "vectorized_loops" in report)
    check("vectorization report: lines is list", isinstance(lines, list))

    try:
        load_vectorization_report(Path("nonexistent_file.txt"))
        check("FileNotFoundError on missing file", False, "no exception raised")
    except FileNotFoundError:
        check("FileNotFoundError on missing file", True)

except Exception as e:
    print(f"  ERROR  acfl_flags: {e}")
    traceback.print_exc()
    FAIL += 1


# ─────────────────────────────────────────────────────────────
# armpl_benchmark.py
# ─────────────────────────────────────────────────────────────
run_section("armpl_benchmark.py")
try:
    from code.utils.armpl_benchmark import (
        generate_dgemm_throughput,
        compute_roofline_model,
        load_library_comparison,
        load_fft_scaling,
    )

    df = generate_dgemm_throughput([64, 256, 1024], "numpy")
    check("dgemm shape: correct columns", set(["matrix_size", "throughput_gflops", "library"]).issubset(df.columns))
    check("dgemm shape: 3 rows for 3 sizes", len(df) == 3)

    df2 = generate_dgemm_throughput([64, 256], "numpy")
    df3 = generate_dgemm_throughput([64, 256], "numpy")
    check("dgemm deterministic: same output", df2["throughput_gflops"].tolist() == df3["throughput_gflops"].tolist())

    try:
        generate_dgemm_throughput([64], "invalid_library")
        check("ValueError on bad library", False, "no exception raised")
    except ValueError:
        check("ValueError on bad library", True)

    rf = compute_roofline_model(100.0, 50.0, [64, 512, 2048])
    check("roofline bounded by peak_gflops", all(rf["roofline_gflops"] <= 100.0))
    check("roofline non-negative", all(rf["roofline_gflops"] >= 0.0))

    cmp = load_library_comparison(Path("code/data/dgemm_comparison.json"))
    check("library comparison: matrix_size column", "matrix_size" in cmp.columns)

    fft = load_fft_scaling(Path("code/data/fft_scaling.json"))
    check("fft scaling: non-empty", len(fft) > 0)

except Exception as e:
    print(f"  ERROR  armpl_benchmark: {e}")
    traceback.print_exc()
    FAIL += 1


# ─────────────────────────────────────────────────────────────
# slurm_tools.py
# ─────────────────────────────────────────────────────────────
run_section("slurm_tools.py")
try:
    from code.utils.slurm_tools import (
        parse_sbatch_script,
        generate_numa_topology,
        validate_binding_flags,
        suggest_binding_strategy,
    )

    script = "#!/bin/bash\n#SBATCH --nodes=4\n#SBATCH --ntasks-per-node=128\n#SBATCH --mem-bind=local\n#SBATCH --cpu-bind=mask_cpu\n"
    result = parse_sbatch_script(script)
    check("parse basic: nodes=4", result.get("nodes") == "4")
    check("parse basic: ntasks-per-node=128", result.get("ntasks_per_node") == "128")

    script2 = "#!/bin/bash\n#SBATCH --unknown-directive=somevalue\n"
    result2 = parse_sbatch_script(script2)
    check("parse unknown: goes to other list", "other" in result2 and isinstance(result2["other"], list))

    topo_v3 = generate_numa_topology("neoverse-v3")
    check("topology v3: cores_per_socket=128", topo_v3.get("cores_per_socket") == 128)
    check("topology v3: has confidence key", "confidence" in topo_v3)

    topo_n3 = generate_numa_topology("neoverse-n3")
    check("topology n3: creates valid topology", "cores_per_socket" in topo_n3)

    try:
        generate_numa_topology("x86-garbage")
        check("ValueError on unknown model", False, "no exception raised")
    except ValueError:
        check("ValueError on unknown model", True)

    warnings = validate_binding_flags({"cpu_bind": "mask_cpu", "mem_bind": "local"})
    check("validate valid flags: no warnings", warnings == [])

    warnings2 = validate_binding_flags({"cpu_bind": "bad_value", "mem_bind": "local"})
    check("validate invalid cpu_bind: warning emitted", len(warnings2) > 0)

    strategy = suggest_binding_strategy(topo_v3, mpi_ranks=128)
    check("strategy v3 128 ranks: cpu_bind_flag present", "cpu_bind_flag" in strategy)
    check("strategy v3 128 ranks: mem_bind_flag present", "mem_bind_flag" in strategy)
    check("strategy v3 128 ranks: rationale present", "rationale" in strategy)

except Exception as e:
    print(f"  ERROR  slurm_tools: {e}")
    traceback.print_exc()
    FAIL += 1


# ─────────────────────────────────────────────────────────────
# map_profiler.py
# ─────────────────────────────────────────────────────────────
run_section("map_profiler.py")
try:
    from code.utils.map_profiler import parse_map_file, generate_flamechart_data, analyze_profile_for_optimization

    with open("code/data/mock_map_profile.json") as f:
        raw = json.load(f)

    profile = parse_map_file(raw)
    required = {"total_runtime_s", "ranks", "cpu_time_pct", "mpi_time_pct", "io_time_pct", "hotspots"}
    check("parse: all required keys present", required.issubset(set(profile.keys())))
    check("parse: total_runtime_s is float", isinstance(profile["total_runtime_s"], float))
    check("parse: hotspots is list", isinstance(profile["hotspots"], list))

    fc = generate_flamechart_data(profile)
    check("flamechart: returns list", isinstance(fc, list))
    check("flamechart: non-empty for non-empty hotspots", len(fc) > 0)
    if fc:
        required_fc = {"x_start", "x_end", "y_level", "label", "color", "pct"}
        check("flamechart: all keys in first entry", required_fc.issubset(set(fc[0].keys())))

    # MPI-heavy: sum=100
    mpi_raw = json.loads(json.dumps(raw))
    mpi_raw["time_breakdown"] = {"cpu_pct": 40.0, "mpi_pct": 35.0, "io_pct": 20.0, "thread_overhead_pct": 5.0}
    p_mpi = parse_map_file(mpi_raw)
    recs_mpi = analyze_profile_for_optimization(p_mpi)
    cats_mpi = [r["category"] for r in recs_mpi]
    check("MPI-heavy: MPI suggestion triggered", "MPI" in cats_mpi)

    # Compute-heavy: sum=100
    cpu_raw = json.loads(json.dumps(raw))
    cpu_raw["time_breakdown"] = {"cpu_pct": 80.0, "mpi_pct": 12.0, "io_pct": 5.0, "thread_overhead_pct": 3.0}
    p_cpu = parse_map_file(cpu_raw)
    recs_cpu = analyze_profile_for_optimization(p_cpu)
    cats_cpu = [r["category"] for r in recs_cpu]
    check("Compute-heavy: compute suggestion triggered", any(c in cats_cpu for c in ["Compute", "CPU", "ArmPL", "SVE2"]))

    # I/O-heavy: sum=100
    io_raw = json.loads(json.dumps(raw))
    io_raw["time_breakdown"] = {"cpu_pct": 50.0, "mpi_pct": 15.0, "io_pct": 30.0, "thread_overhead_pct": 5.0}
    p_io = parse_map_file(io_raw)
    recs_io = analyze_profile_for_optimization(p_io)
    cats_io = [r["category"] for r in recs_io]
    check("I/O-heavy: I/O suggestion triggered", "I/O" in cats_io)

except Exception as e:
    print(f"  ERROR  map_profiler: {e}")
    traceback.print_exc()
    FAIL += 1


# ─────────────────────────────────────────────────────────────
# notebook JSON validity
# ─────────────────────────────────────────────────────────────
run_section("Notebook JSON validity")
import json
notebook_dir = Path("code/notebooks")
for nb_file in sorted(notebook_dir.glob("*.ipynb")):
    try:
        with open(nb_file) as f:
            nb = json.load(f)
        cells = nb.get("cells", [])
        code_cells = [c for c in cells if c["cell_type"] == "code"]
        md_cells = [c for c in cells if c["cell_type"] == "markdown"]
        check(f"{nb_file.name}: valid JSON, {len(cells)} cells ({len(code_cells)} code, {len(md_cells)} md)",
              len(cells) > 0 and len(code_cells) > 0 and len(md_cells) > 0)
    except Exception as e:
        check(f"{nb_file.name}: valid JSON", False, str(e))


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
if ERRORS:
    print(f"  Failed: {', '.join(ERRORS)}")
print(f"{'='*60}")
sys.exit(0 if FAIL == 0 else 1)
