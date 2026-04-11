# Architecture Validation: L1 Link Budget Calculator

**Agent:** sip-arch-validator
**Date:** 2026-04-11
**Decision:** APPROVE

---

## Findings

### Critical (must fix before coding)
None.

### Major (must fix before coding)

1. **`tx_power_dBm` upper bound** — architecture states range −5 to +10 dBm as "configurable, no corpus constraint." The link-budget corpus does cite 24.5 dBm as required ELS output (MEDIUM confidence, ECOC 2024). The widget max of +10 dBm would prevent the user from exploring the ELS scenario. **Required change:** Extend slider max to +25 dBm. Add corpus annotation: `# ELS output reference: 24.5 dBm (MEDIUM — ECOC 2024 Intel, single source)`.

2. **`mux_demux_il_dB` provisional labeling** — architecture correctly flags this. **Required change (enforced at code stage):** The constant definition in Cell 2 must include a `# PROVISIONAL` comment and an `assert` or widget tooltip noting this. Not blocking architecture approval but must be enforced by sip-code-writer.

### Minor (fix during code review)
1. Architecture does not specify units for the `budget_breakdown` dict values — should state `(dB)` in the docstring return block. Minor; fix in implementation.
2. `compute_link_margin` returns a dict — consider naming it `compute_link_budget` to match the `link-budget` slug convention. Minor naming suggestion.

---

## Approved interface contracts

The following function signatures are locked. sip-code-writer must implement these exactly:

```python
def dBm_to_mW(power_dBm: float) -> float
def mW_to_dBm(power_mW: float) -> float  # raises ValueError if power_mW <= 0
def compute_on_chip_loss(coupler_loss_dB, waveguide_loss_dBcm, waveguide_length_cm, modulator_il_dB, mux_demux_il_dB) -> float
def compute_fiber_loss(fiber_attenuation_dBkm, fiber_length_km) -> float
def compute_link_margin(tx_power_dBm, on_chip_loss_dB, fiber_loss_dB, rx_sensitivity_dBm, system_margin_dB) -> dict
def compute_budget_breakdown(coupler_loss_dB, waveguide_loss_dBcm, waveguide_length_cm, modulator_il_dB, mux_demux_il_dB, fiber_attenuation_dBkm, fiber_length_km) -> dict
```

All approved with the following additional constraints:
- `compute_on_chip_loss`: must add `2 ×` coupler_loss in the function body with an explicit comment explaining the two-facet assumption
- `compute_link_margin`: must include `assert total_loss_dB >= 0` with message "Total loss must be positive — check for unit mixing (dB vs dBm)"

---

## Domain accuracy verdict

**PASS.** The link budget equation `rx_power = tx_power - total_loss` is correct in the dB domain. Loss addition is additive (correct). The KP4 BER threshold of 2.4×10⁻⁴ and OMA sensitivity of −18.3 dBm are both HIGH-confidence corpus values. The two-facet coupling assumption is physically correct for a standard fiber-in/fiber-out CPO configuration and is appropriately flagged as an assumption.

The 400G-LR8 total budget (8.4 dB with penalties; 6.3 dB passive) is not directly modeled here — this notebook models the component-level budget. A note should be added in Cell 0 markdown explaining the relationship: the passive fiber budget of 6.3 dB must be subtracted from the total link loss when the external fiber span is included. **Enforce at code stage.**

---

## Conditions for re-review
Not applicable — APPROVE issued. Major finding #1 (slider max) to be addressed by sip-code-writer without re-validation.
