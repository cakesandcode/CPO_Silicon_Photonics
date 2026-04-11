# Architecture: L1 Link Budget Calculator

**Agent:** sip-sw-architect
**Date:** 2026-04-11
**Artifact type:** Jupyter notebook + shared utility module
**Target files:**
- `code/notebooks/L1_link_budget_calculator.ipynb`
- `code/utils/optical_math.py`

---

## Purpose

Compute the end-to-end optical link budget for a 400GBASE-LR8 channel, covering both the CPO on-chip path (fiber-to-chip coupling, waveguide routing, modulator, MUX/DEMUX) and the external fiber span (up to 10 km on OS2 SMF). The user varies component parameters interactively and the notebook determines whether the link closes at the KP4 FEC BER threshold.

**Competency gate (per _MANIFEST.md L1):** Can calculate a complete link budget for a 400G-LR8 channel.
**Feeds proposal section:** §2 (Physical layer constraints), §3 (Coupling loss), §8 (Power analysis).

---

## Inputs

All ranges sourced from corpus. No invented bounds.

| Parameter | Type | Units | Valid range | Default | Editable | Corpus source |
|-----------|------|-------|-------------|---------|----------|---------------|
| `tx_power_dBm` | float | dBm | −5.0 to +10.0 | 0.0 | Yes | Practical ELS range; no direct corpus constraint — flagged as configurable |
| `coupler_loss_dB` | float | dB | 0.08 to 4.5 | 2.0 | Yes | optical-coupling: GC standard 2–4.5 dB (HIGH); TSMC COUPE 0.08 dB (MEDIUM) |
| `waveguide_loss_dBcm` | float | dB/cm | 0.1 to 4.0 | 0.4 | Yes | link-budget: rib 0.27 dB/cm (HIGH); strip 3.6 dB/cm (HIGH); ST PIC100 0.4 dB/cm (MEDIUM) |
| `waveguide_length_cm` | float | cm | 0.1 to 5.0 | 1.0 | Yes | Architectural estimate; configurable |
| `modulator_il_dB` | float | dB | 0.2 to 4.0 | 1.5 | Yes | link-budget: MZM 1.29–1.8 dB (MEDIUM); ring off-resonance <0.2 dB (HIGH) |
| `mux_demux_il_dB` | float | dB | 0.5 to 3.0 | 1.0 | Yes | Typical CWDM4/LWDM MUX; corpus: wdm-grid topic (PLANNED — use 1.0 dB provisional) |
| `fiber_length_km` | float | km | 0.0 to 10.0 | 10.0 | Yes | link-budget: 400G-LR8 reach 10 km (HIGH) |
| `fiber_attenuation_dBkm` | float | dB/km | fixed | 0.43 | No | link-budget: 0.43 dB/km @1295 nm, ITU-T G.695, IEEE 802.3bs (HIGH) |
| `rx_sensitivity_dBm` | float | dBm | −25.0 to −10.0 | −18.3 | Yes | link-budget: OMA sensitivity −17.8 to −18.8 dBm at KP4 BER (HIGH) |
| `system_margin_dB` | float | dB | 0.0 to 3.0 | 1.0 | Yes | Engineering margin; configurable |

**Corpus risk flags:**
- `mux_demux_il_dB` — topic `wdm-grid` is PLANNED, not ACTIVE. Default 1.0 dB is provisional and must be labeled as such in code comments.
- `tx_power_dBm` — no direct corpus constraint on ELS output. Must be treated as fully user-configurable; do not imply it is corpus-sourced.

---

## Outputs

| Output | Type | Units | Formula | Corpus source |
|--------|------|-------|---------|---------------|
| `coupler_loss_total_dB` | float | dB | `2 × coupler_loss_dB` | optical-coupling: per-facet values (2 facets: TX + RX) |
| `waveguide_loss_total_dB` | float | dB | `waveguide_loss_dBcm × waveguide_length_cm` | link-budget: waveguide propagation loss |
| `on_chip_loss_dB` | float | dB | `coupler_loss_total + waveguide_loss_total + modulator_il_dB + mux_demux_il_dB` | link-budget: sum of on-chip elements |
| `fiber_loss_dB` | float | dB | `fiber_attenuation_dBkm × fiber_length_km` | link-budget: 0.43 dB/km × distance |
| `total_loss_dB` | float | dB | `on_chip_loss_dB + fiber_loss_dB` | link-budget: total link loss |
| `rx_power_dBm` | float | dBm | `tx_power_dBm − total_loss_dB` | link-budget: received power at PD |
| `link_margin_dB` | float | dB | `rx_power_dBm − rx_sensitivity_dBm − system_margin_dB` | link-budget: margin above floor |
| `link_status` | str | — | `"PASS" if link_margin_dB >= 0 else "FAIL"` | link-budget: KP4 BER threshold |
| `budget_breakdown` | dict | dB | component-keyed losses | — |

**Visualization outputs (4 charts):**
1. Waterfall bar — loss budget breakdown per component (dB each, running total)
2. Line — link margin (dB) vs. coupler loss (0.08–4.5 dB); reference line at 0 dB margin
3. Line — link margin (dB) vs. fiber length (0–10 km); reference line at 0 dB margin
4. Styled table — all parameters, values, units, PASS/FAIL

---

## Module structure

### `code/utils/optical_math.py`
Pure functions only. No I/O. Importable by tests and notebook.

### `code/notebooks/L1_link_budget_calculator.ipynb`
```
Cell 0:  [Markdown] Title, layer, competency gate, description
Cell 1:  [Code] Imports
Cell 2:  [Code] Constants (all corpus-sourced, cited)
Cell 3:  [Code] rcParams block (from sip-ui-architect spec — unmodified)
Cell 4:  [Code] Import optical_math functions
Cell 5:  [Code] ipywidgets parameter sliders
Cell 6:  [Code] Compute all outputs (calls optical_math functions)
Cell 7:  [Code] Chart 1 — Waterfall loss breakdown
Cell 8:  [Code] Chart 2 — Margin vs. coupler loss sweep
Cell 9:  [Code] Chart 3 — Margin vs. fiber length sweep
Cell 10: [Code] Summary table (pandas DataFrame, styled)
Cell 11: [Markdown] References
```

---

## Interface contracts

### `code/utils/optical_math.py`

```python
def dBm_to_mW(power_dBm: float) -> float:
    """
    Convert optical power from dBm to milliwatts.
    Args:
        power_dBm: Optical power (dBm). Any finite float.
    Returns:
        Power in milliwatts (mW). Always positive.
    Source: Standard conversion — no corpus citation needed.
    """

def mW_to_dBm(power_mW: float) -> float:
    """
    Convert optical power from milliwatts to dBm.
    Args:
        power_mW: Optical power (mW). Must be > 0.
    Returns:
        Power in dBm.
    Raises:
        ValueError: if power_mW <= 0.
    Source: Standard conversion.
    """

def compute_on_chip_loss(
    coupler_loss_dB: float,
    waveguide_loss_dBcm: float,
    waveguide_length_cm: float,
    modulator_il_dB: float,
    mux_demux_il_dB: float,
) -> float:
    """
    Compute total on-chip insertion loss for one optical path.
    Accounts for two coupling facets (TX input + RX output).
    Args:
        coupler_loss_dB: Per-facet coupling loss (dB). Range: 0.08–4.5.
            Source: Research/topics/optical-coupling.md
        waveguide_loss_dBcm: Propagation loss (dB/cm). Range: 0.1–4.0.
            Source: Research/topics/link-budget.md
        waveguide_length_cm: On-chip routing length (cm). Range: 0.1–5.0.
        modulator_il_dB: Modulator insertion loss (dB). Range: 0.2–4.0.
            Source: Research/topics/link-budget.md
        mux_demux_il_dB: MUX/DEMUX insertion loss per port (dB). Range: 0.5–3.0.
            NOTE: wdm-grid topic PLANNED — default 1.0 dB is provisional.
    Returns:
        Total on-chip loss (dB). Always positive.
    Raises:
        ValueError: if any parameter is outside its valid range.
    """

def compute_fiber_loss(
    fiber_attenuation_dBkm: float,
    fiber_length_km: float,
) -> float:
    """
    Compute optical loss over an SMF fiber span.
    Args:
        fiber_attenuation_dBkm: Fiber attenuation (dB/km). Corpus value: 0.43.
            Source: Research/topics/link-budget.md — IEEE 802.3bs, ITU-T G.695 (HIGH)
        fiber_length_km: Span length (km). Range: 0.0–10.0.
    Returns:
        Total fiber loss (dB). Always >= 0.
    Raises:
        ValueError: if fiber_length_km < 0 or fiber_attenuation_dBkm <= 0.
    """

def compute_link_margin(
    tx_power_dBm: float,
    on_chip_loss_dB: float,
    fiber_loss_dB: float,
    rx_sensitivity_dBm: float,
    system_margin_dB: float,
) -> dict:
    """
    Compute full link budget and margin.
    Args:
        tx_power_dBm: Transmitter launch power (dBm).
        on_chip_loss_dB: Total on-chip loss from compute_on_chip_loss() (dB).
        fiber_loss_dB: Fiber span loss from compute_fiber_loss() (dB).
        rx_sensitivity_dBm: Receiver sensitivity floor (dBm).
            Corpus value: −18.3 dBm. Source: Research/topics/link-budget.md (HIGH)
        system_margin_dB: Required engineering margin (dB). Default: 1.0.
    Returns:
        dict with keys:
            total_loss_dB (float): on_chip_loss_dB + fiber_loss_dB
            rx_power_dBm (float): tx_power_dBm - total_loss_dB
            link_margin_dB (float): rx_power - sensitivity - margin
            link_status (str): "PASS" or "FAIL"
    Raises:
        ValueError: if system_margin_dB < 0.
    """

def compute_budget_breakdown(
    coupler_loss_dB: float,
    waveguide_loss_dBcm: float,
    waveguide_length_cm: float,
    modulator_il_dB: float,
    mux_demux_il_dB: float,
    fiber_attenuation_dBkm: float,
    fiber_length_km: float,
) -> dict:
    """
    Return per-component loss breakdown for waterfall chart.
    Returns:
        dict mapping component name to loss in dB:
        {
            'Fiber-chip coupling (×2)': float,
            'Waveguide routing': float,
            'Modulator IL': float,
            'MUX/DEMUX': float,
            'Fiber span': float,
            'Total': float,
        }
    Source: All components from Research/topics/link-budget.md and
            Research/topics/optical-coupling.md
    """
```

---

## Dependency decisions

| Library | Version | Justification |
|---------|---------|---------------|
| `numpy` | any | Array ops for sweep calculations |
| `pandas` | any | Summary table styling |
| `matplotlib` | any | All charts — per project standard |
| `ipywidgets` | any | Interactive sliders |
| `pytest` | any | Test file only |

No non-standard libraries required.

---

## Open questions / risks

1. **`mux_demux_il_dB` default is provisional** — `wdm-grid` topic not yet researched. Code must label this with `# PROVISIONAL: wdm-grid topic PLANNED — verify when wdm-grid research complete`. Default 1.0 dB is conservative; MUX IL for LWDM AWG is typically 0.8–1.5 dB in practice.

2. **`tx_power_dBm` not corpus-sourced** — ELS output power range depends on laser architecture (on-chip vs. remote), which is not resolved until L3 (cpo-architecture topic). Parameter must be fully user-configurable; notebook must note the 24.5 dBm ELS requirement from link-budget topic as an architectural reference point.

3. **Back-reflection penalty not modeled** — grating coupler back-reflection (−10 to −15 dB) causes RIN-induced BER penalty not accounted for in this budget. This is a known omission; the notebook must include a note: "Back-reflection penalty from GC (−10 to −15 dB) not included. See optical-coupling Open Question 4."

4. **Two-facet assumption** — model assumes 2 coupling events (fiber in, fiber out). For CPO with remote laser via FAU, the count may differ. Notebook must label the assumption explicitly.
