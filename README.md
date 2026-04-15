# ARM Software Stack Series

A hands-on, 4-part tutorial series covering the full Arm AI/HPC software stack — from Neoverse silicon (V3, N3, AGI CPU) through compilers, math libraries, cluster tooling, and ML frameworks.

Each post includes working Jupyter Lab notebooks, utility modules, and unit tests. Every technical claim is verified against at least 2 primary sources.

**Author:** Aruna Kumar — Senior SoC Firmware Architect, ex-Intel Sr. Director

## What's Here

```
code/
  notebooks/       4 Jupyter Labs (ACfL flags, ArmPL BLAS, Slurm jobs, Linaro Forge MAP)
  utils/           5 modules (config, acfl_flags, armpl_benchmark, slurm_tools, map_profiler)
  tests/           84 unit tests + standalone runner
  data/            Mock SVE2 compiler output, benchmarks, profiles
Research/
  topics/          11 verified research topics
  verified/        Fact-checked sources (≥2 per claim)
```

## Series

| # | Title | Status |
|---|-------|--------|
| 1 | **Silicon + Intro: The CPU Tier That Changes Everything** | Published (Substack draft) |
| 2 | **HPC Compilers + Cluster: Building Your Arm HPC Stack** | Published (Substack draft) |
| 3 | ML Frameworks: PyTorch, TF, ONNX on Arm CPU | Planned |
| 4 | Edge + Full Picture: ExecuTorch, AVH, and the Complete Stack | Planned |

## Tests

```bash
cd code && python tests/run_tests_standalone.py
```

84 tests across 4 test files:
- `test_acfl_flags.py` (31) — Arm Compiler for Linux flag parsing
- `test_armpl_benchmark.py` (19) — ArmPL BLAS benchmark validation
- `test_map_profiler.py` (18) — Linaro Forge MAP profile parsing
- `test_slurm_tools.py` (16) — Slurm job script generation

## Topics Covered

- Neoverse V3/N3 microarchitecture and AGI CPU positioning
- SVE2 → SME2 transition and matrix acceleration
- Arm Compiler for Linux (ACfL) 25.04 flag exploration
- ArmPL 25.04 BLAS/LAPACK/FFT benchmarking
- Slurm cluster job scripting for Neoverse nodes
- Linaro Forge MAP profiling and hotspot analysis
- KleidiAI, ONNX Runtime ACL EP, ExecuTorch on Arm
- PyTorch 2.9 Arm CPU optimizations

## Related

- [Silicon Photonics for the System Architect](../CPO_Silicon_Photonics) — CPO platform architecture + Arm Neoverse firmware integration
- [armfirmware.substack.com](https://armfirmware.substack.com) — Published articles

## License

MIT
