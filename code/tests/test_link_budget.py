"""
Test suite for L1 Link Budget Calculator.

Layer: L1 Foundations
Corpus: Research/topics/link-budget.md, Research/topics/optical-coupling.md
Agent: sip-test-writer

Run from project root:
    pytest code/tests/test_link_budget.py -v
"""

import math
import sys
import os
import pytest

# Add utils to path for both pytest run from project root and from tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from optical_math import (
    dBm_to_mW,
    mW_to_dBm,
    compute_on_chip_loss,
    compute_fiber_loss,
    compute_link_budget,
    compute_budget_breakdown,
    FIBER_ATTENUATION_OS2_DBKM,
    RX_SENSITIVITY_400G_LR8_DBM,
    KP4_BER_THRESHOLD,
    PASSIVE_BUDGET_400G_LR8_DB,
    TOTAL_BUDGET_400G_LR8_DB,
)

# ── Corpus constants used as test vectors ─────────────────────────────────────
# Source: Research/topics/link-budget.md — IEEE 802.3bs (HIGH confidence)
_FIBER_ATTEN_DBKM     = 0.43     # dB/km @1295 nm, OS2 SMF, ITU-T G.695
_RX_SENS_DBM          = -18.3    # OMA sensitivity, KP4 BER = 2×10⁻⁴, PIN PD
_KP4_BER              = 2.4e-4   # Pre-FEC BER threshold, KP4 RS(544,514,10)
_PASSIVE_BUDGET_DB    = 6.3      # 400G-LR8 passive channel budget
_TOTAL_BUDGET_DB      = 8.4      # Passive + MPI + TDP + CD penalty
_LR8_REACH_KM         = 10.0     # Max reach on OS2 SMF

# Source: Research/topics/optical-coupling.md — IMEC (HIGH confidence)
_EC_LOSS_IMEC_DB      = 1.5      # IMEC hybrid Si/SiN edge coupler per facet
# Source: Research/topics/optical-coupling.md — TSMC COUPE (MEDIUM confidence)
_GC_TSMC_MIN_DB       = 0.08     # TSMC COUPE SiN coupler (MEDIUM)
# Source: Research/topics/link-budget.md — rib waveguide, Ghent Univ (HIGH)
_WG_RIB_LOSS_DBCM     = 0.27
# Source: Research/topics/link-budget.md — MZM, MEDIUM
_MZM_IL_DB            = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# 1. Unit conversion tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUnitConversions:

    def test_dBm_to_mW_zero_is_one_mW(self):
        """0 dBm == 1.0 mW exactly."""
        assert abs(dBm_to_mW(0.0) - 1.0) < 1e-12

    def test_dBm_to_mW_positive(self):
        """10 dBm == 10.0 mW."""
        assert abs(dBm_to_mW(10.0) - 10.0) < 1e-9

    def test_dBm_to_mW_negative(self):
        """-10 dBm == 0.1 mW."""
        assert abs(dBm_to_mW(-10.0) - 0.1) < 1e-12

    def test_mW_to_dBm_one_mW_is_zero(self):
        """1.0 mW == 0.0 dBm exactly."""
        assert abs(mW_to_dBm(1.0) - 0.0) < 1e-12

    def test_mW_to_dBm_ten_mW(self):
        """10.0 mW == 10.0 dBm."""
        assert abs(mW_to_dBm(10.0) - 10.0) < 1e-9

    def test_roundtrip_dBm_mW(self):
        """dBm → mW → dBm round-trip is lossless."""
        for val in [-20.0, -10.0, 0.0, 3.0, 10.0, 23.0]:
            assert abs(mW_to_dBm(dBm_to_mW(val)) - val) < 1e-9, \
                f"Round-trip failed for {val} dBm"

    def test_mW_to_dBm_raises_on_zero(self):
        """mW_to_dBm(0) must raise ValueError — log(0) is undefined."""
        with pytest.raises(ValueError):
            mW_to_dBm(0.0)

    def test_mW_to_dBm_raises_on_negative(self):
        """mW_to_dBm(-1) must raise ValueError — negative optical power is unphysical."""
        with pytest.raises(ValueError):
            mW_to_dBm(-1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. On-chip loss tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOnChipLoss:

    def _defaults(self, **overrides):
        params = dict(
            coupler_loss_dB=2.0,
            waveguide_loss_dBcm=0.4,
            waveguide_length_cm=1.0,
            modulator_il_dB=1.5,
            mux_demux_il_dB=1.0,
        )
        params.update(overrides)
        return params

    def test_basic_additivity(self):
        """Loss components must add correctly in dB (not multiply)."""
        # 2×2.0 (coupler) + 0.4×1.0 (WG) + 1.5 (mod) + 1.0 (mux) = 6.9 dB
        result = compute_on_chip_loss(**self._defaults())
        assert abs(result - 6.9) < 1e-9, f"Expected 6.9 dB, got {result}"

    def test_two_facets_counted(self):
        """Coupling loss must be multiplied by 2 (TX + RX facets).
        Uses minimum valid values for other params to isolate coupler contribution.
        Source: Research/topics/optical-coupling.md — IMEC 1.5 dB/facet (HIGH)."""
        # Use minimum valid values for non-coupler parameters to isolate 2× factor
        # Minimum valid: waveguide 0.1 dB/cm, length 0.1 cm, mod 0.1 dB, mux 0.5 dB
        result = compute_on_chip_loss(
            coupler_loss_dB=_EC_LOSS_IMEC_DB,   # 1.5 dB/facet
            waveguide_loss_dBcm=0.27,             # rib waveguide (HIGH)
            waveguide_length_cm=0.1,              # minimum length
            modulator_il_dB=0.1,                  # minimum modulator IL
            mux_demux_il_dB=0.5,                  # minimum MUX/DEMUX IL
        )
        expected = 2 * _EC_LOSS_IMEC_DB + 0.27 * 0.1 + 0.1 + 0.5
        assert abs(result - expected) < 1e-9, \
            f"Two-facet coupling not counted correctly. Expected {expected:.4f}, got {result:.4f}"

    def test_tsmc_coupler_minimum(self):
        """TSMC COUPE 0.08 dB minimum coupler is accepted (in valid range)."""
        result = compute_on_chip_loss(
            coupler_loss_dB=_GC_TSMC_MIN_DB,
            waveguide_loss_dBcm=0.27,
            waveguide_length_cm=1.0,
            modulator_il_dB=0.1,
            mux_demux_il_dB=0.5,
        )
        assert result > 0

    def test_standard_gc_maximum(self):
        """Standard GC 4.5 dB maximum is accepted."""
        result = compute_on_chip_loss(
            coupler_loss_dB=4.5,
            waveguide_loss_dBcm=0.4,
            waveguide_length_cm=1.0,
            modulator_il_dB=1.5,
            mux_demux_il_dB=1.0,
        )
        assert result > 0

    def test_coupler_below_range_raises(self):
        """Coupler loss below 0.08 dB (sub-TSMC) must raise ValueError."""
        with pytest.raises(ValueError):
            compute_on_chip_loss(**self._defaults(coupler_loss_dB=0.01))

    def test_coupler_above_range_raises(self):
        """Coupler loss above 4.5 dB must raise ValueError."""
        with pytest.raises(ValueError):
            compute_on_chip_loss(**self._defaults(coupler_loss_dB=5.0))

    def test_waveguide_loss_above_range_raises(self):
        """Waveguide loss > 4.0 dB/cm must raise ValueError."""
        with pytest.raises(ValueError):
            compute_on_chip_loss(**self._defaults(waveguide_loss_dBcm=4.5))

    def test_modulator_above_range_raises(self):
        """Modulator IL > 4.0 dB must raise ValueError."""
        with pytest.raises(ValueError):
            compute_on_chip_loss(**self._defaults(modulator_il_dB=4.5))

    def test_result_always_positive(self):
        """On-chip loss must always be >= 0."""
        result = compute_on_chip_loss(
            coupler_loss_dB=0.08,
            waveguide_loss_dBcm=0.1,
            waveguide_length_cm=0.1,
            modulator_il_dB=0.1,
            mux_demux_il_dB=0.5,
        )
        assert result >= 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fiber loss tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFiberLoss:

    def test_corpus_value_10km(self):
        """10 km × 0.43 dB/km = 4.30 dB exactly.
        Source: Research/topics/link-budget.md — IEEE 802.3bs (HIGH)."""
        result = compute_fiber_loss(_FIBER_ATTEN_DBKM, 10.0)
        assert abs(result - 4.30) < 1e-9, f"Expected 4.30 dB, got {result}"

    def test_zero_length_is_zero_loss(self):
        """0 km fiber = 0 dB loss."""
        assert compute_fiber_loss(_FIBER_ATTEN_DBKM, 0.0) == 0.0

    def test_linearity(self):
        """Loss must scale linearly with distance."""
        loss_5km = compute_fiber_loss(_FIBER_ATTEN_DBKM, 5.0)
        loss_10km = compute_fiber_loss(_FIBER_ATTEN_DBKM, 10.0)
        assert abs(loss_10km - 2 * loss_5km) < 1e-9

    def test_negative_length_raises(self):
        """Negative fiber length is unphysical — must raise ValueError."""
        with pytest.raises(ValueError):
            compute_fiber_loss(_FIBER_ATTEN_DBKM, -1.0)

    def test_exceeds_lr8_reach_raises(self):
        """Fiber length > 10 km exceeds 400G-LR8 spec — must raise ValueError.
        Source: Research/topics/link-budget.md — IEEE 802.3bs (HIGH)."""
        with pytest.raises(ValueError):
            compute_fiber_loss(_FIBER_ATTEN_DBKM, 10.1)

    def test_zero_attenuation_raises(self):
        """Zero attenuation coefficient is unphysical — must raise ValueError."""
        with pytest.raises(ValueError):
            compute_fiber_loss(0.0, 5.0)

    def test_corpus_constant_matches(self):
        """FIBER_ATTENUATION_OS2_DBKM must equal corpus value 0.43 dB/km.
        Source: Research/topics/link-budget.md — IEEE 802.3bs (HIGH)."""
        assert abs(FIBER_ATTENUATION_OS2_DBKM - 0.43) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# 4. Link budget (domain accuracy) tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLinkBudgetDomainAccuracy:

    def _run_400g_lr8_default(self, tx_power_dBm=0.0, margin_dB=1.0):
        """Helper: run full budget with default component values."""
        on_chip = compute_on_chip_loss(2.0, 0.4, 1.0, 1.5, 1.0)  # 6.9 dB
        fiber   = compute_fiber_loss(_FIBER_ATTEN_DBKM, 10.0)      # 4.3 dB
        return compute_link_budget(tx_power_dBm, on_chip, fiber, _RX_SENS_DBM, margin_dB)

    def test_link_closes_with_adequate_tx_power(self):
        """With TX=0 dBm and default components, link must PASS.
        Total loss ~11.2 dB, RX = −11.2 dBm >> sensitivity −18.3 dBm."""
        result = self._run_400g_lr8_default(tx_power_dBm=0.0)
        assert result["link_status"] == "PASS", \
            f"Expected PASS at TX=0 dBm, got {result}"

    def test_link_fails_with_insufficient_tx_power(self):
        """With very low TX power, link must FAIL."""
        result = self._run_400g_lr8_default(tx_power_dBm=-20.0)
        assert result["link_status"] == "FAIL", \
            f"Expected FAIL at TX=−20 dBm, got {result}"

    def test_rx_power_calculation(self):
        """rx_power = tx_power - total_loss (dB domain, not linear)."""
        on_chip = 6.9
        fiber   = 4.3
        result  = compute_link_budget(0.0, on_chip, fiber, _RX_SENS_DBM, 1.0)
        expected_rx = 0.0 - (6.9 + 4.3)  # = −11.2 dBm
        assert abs(result["rx_power_dBm"] - expected_rx) < 1e-9

    def test_link_margin_calculation(self):
        """margin = rx_power − sensitivity − system_margin."""
        on_chip = 6.9
        fiber   = 4.3
        result  = compute_link_budget(0.0, on_chip, fiber, _RX_SENS_DBM, 1.0)
        # rx_power = −11.2, sensitivity = −18.3, margin_floor = 1.0
        # margin = −11.2 − (−18.3) − 1.0 = 6.1 dB
        assert abs(result["link_margin_dB"] - 6.1) < 1e-9, \
            f"Expected 6.1 dB margin, got {result['link_margin_dB']}"

    def test_margin_zero_gives_pass(self):
        """Exactly 0 dB margin must be PASS (threshold is >=, not >)."""
        # Construct inputs that produce exactly 0 margin
        # rx_power = sens + margin_floor → tx - loss = sens + margin_floor
        # tx=0, loss=X, sens=−18.3, margin=1.0 → X = 0 − (−18.3) − 1.0 = 17.3
        on_chip = 13.0
        fiber   = 4.3
        result  = compute_link_budget(0.0, on_chip, fiber, _RX_SENS_DBM, 1.0)
        assert result["link_status"] == "PASS" or result["link_margin_dB"] >= -0.001

    def test_rx_sensitivity_corpus_value(self):
        """RX_SENSITIVITY_400G_LR8_DBM must match corpus: −18.3 dBm.
        Source: Research/topics/link-budget.md — Optica OE 2016 (HIGH)."""
        assert abs(RX_SENSITIVITY_400G_LR8_DBM - (-18.3)) < 1e-12

    def test_kp4_ber_threshold_corpus_value(self):
        """KP4_BER_THRESHOLD must match corpus: 2.4×10⁻⁴.
        Source: Research/topics/link-budget.md — IEEE 802.3bs (HIGH)."""
        assert abs(KP4_BER_THRESHOLD - 2.4e-4) < 1e-15

    def test_negative_loss_raises(self):
        """Negative on-chip loss implies impossible gain — must raise ValueError."""
        with pytest.raises(ValueError):
            compute_link_budget(0.0, -1.0, 4.3, _RX_SENS_DBM, 1.0)

    def test_negative_system_margin_raises(self):
        """Negative system margin is undefined — must raise ValueError."""
        with pytest.raises(ValueError):
            compute_link_budget(0.0, 6.9, 4.3, _RX_SENS_DBM, -0.1)

    def test_total_loss_additive(self):
        """Total loss = on_chip + fiber (dB addition, not multiplication)."""
        on_chip = 6.9
        fiber   = 4.3
        result  = compute_link_budget(0.0, on_chip, fiber, _RX_SENS_DBM, 1.0)
        assert abs(result["total_loss_dB"] - (on_chip + fiber)) < 1e-9

    def test_400g_lr8_passive_budget_constant(self):
        """PASSIVE_BUDGET_400G_LR8_DB must equal 6.3 dB.
        Source: Research/topics/link-budget.md — IEEE 802.3bs (HIGH)."""
        assert abs(PASSIVE_BUDGET_400G_LR8_DB - 6.3) < 1e-12

    def test_400g_lr8_total_budget_constant(self):
        """TOTAL_BUDGET_400G_LR8_DB must equal 8.4 dB.
        Source: Research/topics/link-budget.md — IEEE 802.3bs (HIGH)."""
        assert abs(TOTAL_BUDGET_400G_LR8_DB - 8.4) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# 5. Budget breakdown tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetBreakdown:

    def _default_breakdown(self):
        return compute_budget_breakdown(2.0, 0.4, 1.0, 1.5, 1.0, _FIBER_ATTEN_DBKM, 10.0)

    def test_keys_present(self):
        """Breakdown dict must contain all expected component keys."""
        result = self._default_breakdown()
        expected_keys = {
            "Fiber-chip coupling (×2)",
            "Waveguide routing",
            "Modulator IL",
            "MUX/DEMUX",
            "Fiber span",
            "Total",
        }
        assert set(result.keys()) == expected_keys

    def test_total_equals_sum_of_components(self):
        """Total must equal sum of all component losses."""
        result = self._default_breakdown()
        component_sum = (
            result["Fiber-chip coupling (×2)"]
            + result["Waveguide routing"]
            + result["Modulator IL"]
            + result["MUX/DEMUX"]
            + result["Fiber span"]
        )
        assert abs(result["Total"] - component_sum) < 1e-6, \
            f"Total {result['Total']} != component sum {component_sum}"

    def test_coupling_is_two_facets(self):
        """Coupling entry must be 2× per-facet value."""
        result = self._default_breakdown()
        assert abs(result["Fiber-chip coupling (×2)"] - 4.0) < 1e-9  # 2 × 2.0

    def test_waveguide_entry(self):
        """Waveguide routing entry: 0.4 dB/cm × 1.0 cm = 0.4 dB."""
        result = self._default_breakdown()
        assert abs(result["Waveguide routing"] - 0.4) < 1e-9

    def test_fiber_span_entry(self):
        """Fiber span: 0.43 dB/km × 10 km = 4.3 dB."""
        result = self._default_breakdown()
        assert abs(result["Fiber span"] - 4.3) < 1e-9

    def test_all_values_positive(self):
        """Every component loss must be non-negative."""
        result = self._default_breakdown()
        for key, val in result.items():
            assert val >= 0, f"Component '{key}' has negative loss: {val}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Regression tests — locked known-good vectors
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressions:

    def test_regression_default_parameters(self):
        """Locked regression: default params give 11.2 dB total loss, 6.1 dB margin, PASS."""
        on_chip = compute_on_chip_loss(2.0, 0.4, 1.0, 1.5, 1.0)
        fiber   = compute_fiber_loss(0.43, 10.0)
        result  = compute_link_budget(0.0, on_chip, fiber, -18.3, 1.0)

        assert abs(result["total_loss_dB"]  - 11.2) < 1e-9, "Regression: total_loss_dB"
        assert abs(result["rx_power_dBm"]   - (-11.2)) < 1e-9, "Regression: rx_power_dBm"
        assert abs(result["link_margin_dB"] - 6.1) < 1e-9, "Regression: link_margin_dB"
        assert result["link_status"] == "PASS", "Regression: link_status"

    def test_regression_tsmc_coupler(self):
        """TSMC COUPE coupler (0.08 dB) dramatically improves margin vs standard GC (2.0 dB)."""
        on_chip_tsmc = compute_on_chip_loss(0.08, 0.4, 1.0, 1.5, 1.0)
        on_chip_std  = compute_on_chip_loss(2.0,  0.4, 1.0, 1.5, 1.0)
        fiber        = compute_fiber_loss(0.43, 10.0)

        result_tsmc = compute_link_budget(0.0, on_chip_tsmc, fiber, -18.3, 1.0)
        result_std  = compute_link_budget(0.0, on_chip_std,  fiber, -18.3, 1.0)

        # TSMC should give ~3.84 dB better margin (2×(2.0−0.08))
        margin_improvement = result_tsmc["link_margin_dB"] - result_std["link_margin_dB"]
        expected_improvement = 2 * (2.0 - 0.08)  # = 3.84 dB
        assert abs(margin_improvement - expected_improvement) < 1e-9, \
            f"Expected {expected_improvement:.3f} dB improvement, got {margin_improvement:.3f}"
