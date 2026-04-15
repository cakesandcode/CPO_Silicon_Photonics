"""
optical_math.py — Core optical link budget calculations.

Pure functions only. No I/O, no side effects, no global state.
All constants used by callers must be sourced from Research/topics/link-budget.md
or Research/topics/optical-coupling.md.

Layer: L1
Competency gate: Can calculate a complete link budget for a 400G-LR8 channel.
"""

import math
from typing import Dict


# ── Valid ranges (from corpus) ────────────────────────────────────────────────
# Source: Research/topics/optical-coupling.md — GC standard 2–4.5 dB (HIGH),
#         TSMC COUPE 0.08 dB (MEDIUM). Use 0.08 as absolute floor.
COUPLER_LOSS_MIN_DB: float = 0.08
COUPLER_LOSS_MAX_DB: float = 4.5

# Source: Research/topics/link-budget.md — rib 0.27 dB/cm (HIGH), strip 3.6 dB/cm (HIGH)
WAVEGUIDE_LOSS_MIN_DBCM: float = 0.1
WAVEGUIDE_LOSS_MAX_DBCM: float = 4.0

# Source: Research/topics/link-budget.md — ring <0.2 dB (HIGH), MZM 1.29–1.8 dB (MEDIUM)
MODULATOR_IL_MIN_DB: float = 0.1
MODULATOR_IL_MAX_DB: float = 4.0

# PROVISIONAL: wdm-grid topic PLANNED — 1.0 dB default is an estimate.
# Verify when Research/topics/wdm-grid.md is complete.
MUX_DEMUX_IL_MIN_DB: float = 0.5
MUX_DEMUX_IL_MAX_DB: float = 3.0

# Source: Research/topics/link-budget.md — IEEE 802.3bs (HIGH)
FIBER_LENGTH_MAX_KM: float = 10.0

# Source: Research/topics/link-budget.md — ITU-T G.695, IEEE 802.3bs @1295 nm (HIGH)
FIBER_ATTENUATION_OS2_DBKM: float = 0.43

# Source: Research/topics/link-budget.md — OMA sensitivity −17.8 to −18.8 dBm
#         at KP4 BER = 2×10⁻⁴, PIN PD, Optica OE (HIGH). Use −18.3 dBm as nominal.
RX_SENSITIVITY_400G_LR8_DBM: float = -18.3

# Source: Research/topics/link-budget.md — IEEE 802.3bs / KP4 RS(544,514,10) (HIGH)
KP4_BER_THRESHOLD: float = 2.4e-4

# Source: Research/topics/link-budget.md — IEEE 802.3bs passive channel budget (HIGH)
PASSIVE_BUDGET_400G_LR8_DB: float = 6.3

# Source: Research/topics/link-budget.md — IEEE 802.3bs total budget incl. MPI+TDP+CD (HIGH)
TOTAL_BUDGET_400G_LR8_DB: float = 8.4


# ── Conversion functions ──────────────────────────────────────────────────────

def dBm_to_mW(power_dBm: float) -> float:
    """
    Convert optical power from dBm to milliwatts.

    Args:
        power_dBm: Optical power (dBm). Must be finite.

    Returns:
        Power in milliwatts (mW). Always positive.

    Raises:
        ValueError: if power_dBm is NaN or infinite.

    Example:
        >>> dBm_to_mW(0.0)
        1.0
        >>> dBm_to_mW(-3.0)  # approx 0.5 mW
        0.5011...
    """
    if not math.isfinite(power_dBm):
        raise ValueError(f"power_dBm must be finite, got {power_dBm}")
    return 10.0 ** (power_dBm / 10.0)


def mW_to_dBm(power_mW: float) -> float:
    """
    Convert optical power from milliwatts to dBm.

    Args:
        power_mW: Optical power (mW). Must be > 0.

    Returns:
        Power in dBm.

    Raises:
        ValueError: if power_mW <= 0 (log of non-positive is undefined).

    Example:
        >>> mW_to_dBm(1.0)
        0.0
    """
    if not math.isfinite(power_mW):
        raise ValueError(
            f"power_mW must be finite, got {power_mW}."
        )
    if power_mW <= 0:
        raise ValueError(
            f"power_mW must be > 0, got {power_mW}. "
            "Optical power cannot be zero or negative."
        )
    return 10.0 * math.log10(power_mW)


# ── Loss calculation functions ────────────────────────────────────────────────

def compute_on_chip_loss(
    coupler_loss_dB: float,
    waveguide_loss_dBcm: float,
    waveguide_length_cm: float,
    modulator_il_dB: float,
    mux_demux_il_dB: float,
) -> float:
    """
    Compute total on-chip insertion loss for one optical path (TX to PD).

    Assumes TWO coupling events: one at the fiber input facet (TX side)
    and one at the fiber output facet (RX side). This is the standard
    assumption for a CPO link with fiber array unit (FAU) at both ends.
    Revise to 1× if modeling a remote-laser architecture with a single FAU.

    Loss is additive in dB:
        total = 2 × coupler + waveguide_loss_per_cm × length + modulator + mux/demux

    Args:
        coupler_loss_dB: Per-facet fiber-to-chip coupling loss (dB).
            Valid range: 0.08–4.5 dB.
            Source: Research/topics/optical-coupling.md
            HIGH confidence (IMEC 1.5 dB/facet); MEDIUM (TSMC 0.08 dB).
        waveguide_loss_dBcm: Waveguide propagation loss (dB/cm).
            Valid range: 0.1–4.0 dB/cm.
            Source: Research/topics/link-budget.md
            HIGH confidence: rib 0.27 dB/cm; strip 3.6 dB/cm.
        waveguide_length_cm: On-chip routing length (cm).
            Valid range: 0.1–5.0 cm.
        modulator_il_dB: Modulator insertion loss (dB).
            Valid range: 0.1–4.0 dB.
            Source: Research/topics/link-budget.md
            HIGH: ring off-resonance <0.2 dB. MEDIUM: MZM 1.29–1.8 dB.
        mux_demux_il_dB: WDM MUX/DEMUX insertion loss per port (dB).
            Valid range: 0.5–3.0 dB.
            PROVISIONAL: wdm-grid topic PLANNED. Default 1.0 dB is an estimate.

    Returns:
        Total on-chip loss (dB). Always >= 0.

    Raises:
        ValueError: if any parameter is outside its valid range.
    """
    # Validate inputs against corpus-sourced ranges
    if not (COUPLER_LOSS_MIN_DB <= coupler_loss_dB <= COUPLER_LOSS_MAX_DB):
        raise ValueError(
            f"coupler_loss_dB={coupler_loss_dB} out of range "
            f"[{COUPLER_LOSS_MIN_DB}, {COUPLER_LOSS_MAX_DB}] dB. "
            "Source: Research/topics/optical-coupling.md"
        )
    if not (WAVEGUIDE_LOSS_MIN_DBCM <= waveguide_loss_dBcm <= WAVEGUIDE_LOSS_MAX_DBCM):
        raise ValueError(
            f"waveguide_loss_dBcm={waveguide_loss_dBcm} out of range "
            f"[{WAVEGUIDE_LOSS_MIN_DBCM}, {WAVEGUIDE_LOSS_MAX_DBCM}] dB/cm. "
            "Source: Research/topics/link-budget.md"
        )
    if not (0.1 <= waveguide_length_cm <= 5.0):
        raise ValueError(
            f"waveguide_length_cm={waveguide_length_cm} out of range [0.1, 5.0] cm."
        )
    if not (MODULATOR_IL_MIN_DB <= modulator_il_dB <= MODULATOR_IL_MAX_DB):
        raise ValueError(
            f"modulator_il_dB={modulator_il_dB} out of range "
            f"[{MODULATOR_IL_MIN_DB}, {MODULATOR_IL_MAX_DB}] dB. "
            "Source: Research/topics/link-budget.md"
        )
    if not (MUX_DEMUX_IL_MIN_DB <= mux_demux_il_dB <= MUX_DEMUX_IL_MAX_DB):
        raise ValueError(
            f"mux_demux_il_dB={mux_demux_il_dB} out of range "
            f"[{MUX_DEMUX_IL_MIN_DB}, {MUX_DEMUX_IL_MAX_DB}] dB."
        )

    # Two coupling facets: TX fiber-in and RX fiber-out
    coupling_total_dB = 2.0 * coupler_loss_dB

    # Waveguide propagation loss is linear in dB over length
    waveguide_total_dB = waveguide_loss_dBcm * waveguide_length_cm

    # All losses additive in dB domain — never mix with linear mW
    total = coupling_total_dB + waveguide_total_dB + modulator_il_dB + mux_demux_il_dB

    assert total >= 0, "On-chip loss must be non-negative — check for unit mixing."
    return total


def compute_fiber_loss(
    fiber_attenuation_dBkm: float,
    fiber_length_km: float,
) -> float:
    """
    Compute optical loss over a single-mode fiber span.

    Uses bulk attenuation only. Does not include connector loss,
    splice loss, or bend loss — add those as separate terms if needed.

    Args:
        fiber_attenuation_dBkm: Fiber bulk attenuation (dB/km).
            Corpus value: 0.43 dB/km at 1295 nm on OS2 SMF (G.652).
            Source: Research/topics/link-budget.md — IEEE 802.3bs, ITU-T G.695 (HIGH).
        fiber_length_km: Fiber span length (km).
            Valid range: 0.0–10.0 km (400G-LR8 max reach).

    Returns:
        Total fiber span loss (dB). Always >= 0.

    Raises:
        ValueError: if fiber_length_km < 0 or fiber_attenuation_dBkm <= 0.
    """
    if fiber_attenuation_dBkm <= 0:
        raise ValueError(
            f"fiber_attenuation_dBkm must be > 0, got {fiber_attenuation_dBkm}."
        )
    if fiber_length_km < 0:
        raise ValueError(
            f"fiber_length_km must be >= 0, got {fiber_length_km}."
        )
    if fiber_length_km > FIBER_LENGTH_MAX_KM:
        raise ValueError(
            f"fiber_length_km={fiber_length_km} exceeds 400G-LR8 max reach "
            f"of {FIBER_LENGTH_MAX_KM} km. "
            "Source: Research/topics/link-budget.md — IEEE 802.3bs (HIGH)."
        )

    return fiber_attenuation_dBkm * fiber_length_km


def compute_link_budget(
    tx_power_dBm: float,
    on_chip_loss_dB: float,
    fiber_loss_dB: float,
    rx_sensitivity_dBm: float,
    system_margin_dB: float,
) -> Dict[str, object]:
    """
    Compute full optical link budget and determine if the link closes.

    Link equation (all in dB domain):
        rx_power_dBm = tx_power_dBm - total_loss_dB
        link_margin_dB = rx_power_dBm - rx_sensitivity_dBm - system_margin_dB
        link closes if link_margin_dB >= 0

    Args:
        tx_power_dBm: Transmitter launch power into fiber (dBm).
        on_chip_loss_dB: Total on-chip loss from compute_on_chip_loss() (dB).
            Must be >= 0.
        fiber_loss_dB: Fiber span loss from compute_fiber_loss() (dB).
            Must be >= 0.
        rx_sensitivity_dBm: Receiver sensitivity floor (dBm).
            Corpus nominal: −18.3 dBm (400G-LR8, KP4 BER = 2×10⁻⁴, PIN PD).
            Source: Research/topics/link-budget.md — Optica OE 2016 (HIGH).
        system_margin_dB: Required engineering margin above sensitivity floor (dB).
            Must be >= 0. Typical: 1.0–2.0 dB.

    Returns:
        dict with keys:
            'total_loss_dB'   (float): Sum of on-chip and fiber losses (dB).
            'rx_power_dBm'    (float): Power at receiver input (dBm).
            'link_margin_dB'  (float): Margin above (sensitivity + margin floor) (dB).
            'link_status'     (str):   'PASS' if link_margin_dB >= 0, else 'FAIL'.

    Raises:
        ValueError: if on_chip_loss_dB < 0, fiber_loss_dB < 0, or system_margin_dB < 0.
    """
    if on_chip_loss_dB < 0:
        raise ValueError(f"on_chip_loss_dB must be >= 0, got {on_chip_loss_dB}.")
    if fiber_loss_dB < 0:
        raise ValueError(f"fiber_loss_dB must be >= 0, got {fiber_loss_dB}.")
    if system_margin_dB < 0:
        raise ValueError(f"system_margin_dB must be >= 0, got {system_margin_dB}.")

    # All arithmetic in dB domain — never convert to mW here
    total_loss_dB = on_chip_loss_dB + fiber_loss_dB

    # Sanity check: total loss must not be negative (would imply gain without amplifier)
    assert total_loss_dB >= 0, (
        "Total loss must be non-negative — check for unit mixing (dB vs dBm)."
    )

    rx_power_dBm = tx_power_dBm - total_loss_dB
    link_margin_dB = rx_power_dBm - rx_sensitivity_dBm - system_margin_dB
    link_status = "PASS" if link_margin_dB >= 0.0 else "FAIL"

    return {
        "total_loss_dB":  total_loss_dB,
        "rx_power_dBm":   rx_power_dBm,
        "link_margin_dB": link_margin_dB,
        "link_status":    link_status,
    }


def compute_budget_breakdown(
    coupler_loss_dB: float,
    waveguide_loss_dBcm: float,
    waveguide_length_cm: float,
    modulator_il_dB: float,
    mux_demux_il_dB: float,
    fiber_attenuation_dBkm: float,
    fiber_length_km: float,
) -> Dict[str, float]:
    """
    Return per-component optical loss breakdown for waterfall chart rendering.

    All values in dB. Calls compute_on_chip_loss and compute_fiber_loss
    internally — their validation applies.

    Returns:
        Ordered dict mapping component label to loss (dB):
        {
            'Fiber-chip coupling (×2)': float,  # 2 × coupler_loss_dB
            'Waveguide routing':        float,  # loss_dBcm × length_cm
            'Modulator IL':             float,
            'MUX/DEMUX':               float,  # PROVISIONAL — wdm-grid topic PLANNED
            'Fiber span':               float,  # attenuation × length
            'Total':                    float,
        }

    Source: Component values from Research/topics/link-budget.md and
            Research/topics/optical-coupling.md.
    """
    # Validate on-chip parameters via compute_on_chip_loss (reuses its range checks)
    on_chip_total = compute_on_chip_loss(
        coupler_loss_dB, waveguide_loss_dBcm, waveguide_length_cm,
        modulator_il_dB, mux_demux_il_dB,
    )
    fiber_total = compute_fiber_loss(fiber_attenuation_dBkm, fiber_length_km)

    # Decompose for waterfall chart — individual components from validated inputs
    coupling_total = 2.0 * coupler_loss_dB
    waveguide_total = waveguide_loss_dBcm * waveguide_length_cm
    total = on_chip_total + fiber_total

    return {
        "Fiber-chip coupling (×2)": round(coupling_total, 4),
        "Waveguide routing":        round(waveguide_total, 4),
        "Modulator IL":             round(modulator_il_dB, 4),
        "MUX/DEMUX":               round(mux_demux_il_dB, 4),  # PROVISIONAL
        "Fiber span":               round(fiber_total, 4),
        "Total":                    round(total, 4),
    }
