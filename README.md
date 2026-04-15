# Silicon Photonics for the System Architect

A layered tutorial and research project covering silicon photonics from physical-layer link budgets through Co-Packaged Optics (CPO) platform architecture, with a focus on firmware integration for Arm Neoverse platforms.

**Author:** Aruna Kumar — Senior SoC Firmware Architect, ex-Intel Sr. Director

## Thesis

> An Arm Neoverse CSS SoC can serve as the host for a Silicon Photonics CPO module, with the SCP owning the optical control plane via SCMI extensions, CMIS as the management interface, and a defined RAS model that prevents laser failures from causing system-level crashes. This reduces rack-level I/O power by >50% vs. pluggable OSFP for AI fabric bandwidth >=3.2 Tb/s per node.

## What's Here

```
code/
  notebooks/       Link budget calculator (Jupyter Lab) + architecture spec docs
  utils/           optical_math.py — 500+ lines of photonics math
  tests/           44 unit tests (power conversions, coupling, waveguide, modulation, link budget)
  data/            Simulation cache directory
Modules/           5 learning layers (L1-L5), structured curriculum
Research/
  topics/          Link budget, optical coupling, SOI platform, waveguide loss
  sources/         Primary source material
```

## 5-Layer Learning Map

| Layer | Title | Status |
|-------|-------|--------|
| **L1** | Foundations — Link budget for 400G-LR8 | Code + tests complete |
| L2 | Components — Modulator/detector/WDM selection | In progress |
| L3 | CPO System — EIC/PIC interface, thermal, package co-design | Planned |
| L4 | Arm FW Integration — SCP ownership, SCMI, CMIS, RAS | Planned |
| L5 | Proposal — OCP-format proposal draft | Planned |

## Tests

```bash
cd code && python -m pytest tests/test_link_budget.py -v
```

44 tests covering:
- Power conversions (dBm <-> mW): 8 tests
- Optical coupling losses (grating coupler vs. edge coupler): 8 tests
- Waveguide attenuation (SOI platform): 4 tests
- Modulation insertion loss (MZM, ring resonator): 3 tests
- Full link budget closure (400G-LR8): 14 tests
- Regression + corpus validation: 7 tests

## Key Insight: The Open Firmware Surface Problem

CPO-integrated Arm platforms have unresolved firmware ownership questions:
- **SCMI** defines no optical domain primitives (no laser power, modulator bias, or PD responsivity messages)
- **TF-A** has no optical engine driver class
- **CMIS** state machine ownership is unresolved for Neoverse SCP/AP firmware split

This is the gap this project aims to define and propose solutions for.

## Tutorial

The 30-page structured tutorial is available as:
- `silicon photonics architect tutorial.pdf` (312 KB)
- `silicon_photonics_architect_tutorial.docx`

## Related

- [ARM Software Stack Series](../ARM_SW_Stack) — Arm Neoverse HPC/AI software stack tutorials
- [armfirmware.substack.com](https://armfirmware.substack.com) — Published articles

## License

MIT
