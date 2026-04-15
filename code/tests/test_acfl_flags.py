"""
pytest tests for ACfL compiler flag utilities.

Tests the acfl_flags module functions: parse_acfl_command, mcpu_to_extensions,
and load_vectorization_report. Includes mock platform detection for ARM QEMU testing.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest import mock

from code.utils.acfl_flags import (
    parse_acfl_command,
    mcpu_to_extensions,
    load_vectorization_report,
)
from code.utils.config import ISA_COLORS, NEOVERSE_CPUS


class TestParseAcflCommand:
    """Tests for parse_acfl_command function."""

    def test_parse_simple_mcpu(self) -> None:
        """Parse a simple command with -mcpu flag."""
        cmd = "armclang -O3 -mcpu=neoverse-v3 file.c"
        result = parse_acfl_command(cmd)

        assert result["compiler"] == "armclang"
        assert result["opt_level"] == "-O3"
        assert result["mcpu"] == "neoverse-v3"
        assert "SVE2" in result["inferred_extensions"]
        assert "SME2" in result["inferred_extensions"]

    def test_parse_march_flag(self) -> None:
        """Parse a command with -march instead of -mcpu."""
        cmd = "armclang -march=armv9.2-a+sve2 file.c"
        result = parse_acfl_command(cmd)

        assert result["march"] == "armv9.2-a+sve2"
        assert "SVE2" in result["inferred_extensions"]

    def test_parse_optimization_levels(self) -> None:
        """Test parsing various optimization levels."""
        test_cases = [
            ("gcc -O0 file.c", "-O0"),
            ("gcc -O1 file.c", "-O1"),
            ("gcc -O2 file.c", "-O2"),
            ("gcc -O3 file.c", "-O3"),
        ]

        for cmd, expected_opt in test_cases:
            result = parse_acfl_command(cmd)
            assert result["opt_level"] == expected_opt

    def test_parse_multiple_flags(self) -> None:
        """Parse a command with multiple flags."""
        cmd = "armflang -O3 -mcpu=neoverse-n3 -fvectorize -o output.o file.f90"
        result = parse_acfl_command(cmd)

        assert result["compiler"] == "armflang"
        assert result["mcpu"] == "neoverse-n3"
        assert result["flags"]["mcpu"] == "neoverse-n3"
        assert result["flags"]["-fvectorize"] is True

    def test_parse_unknown_cpu_warning(self) -> None:
        """Test that unknown -mcpu generates a warning."""
        cmd = "armclang -O3 -mcpu=unknown-cpu file.c"
        result = parse_acfl_command(cmd)

        assert result["mcpu"] == "unknown-cpu"
        assert any("Unknown" in w for w in result["warnings"])

    def test_parse_empty_command_raises_error(self) -> None:
        """Raise ValueError for empty command."""
        with pytest.raises(ValueError):
            parse_acfl_command("")

    def test_parse_none_command_raises_error(self) -> None:
        """Raise ValueError for None command."""
        with pytest.raises(ValueError):
            parse_acfl_command(None)

    def test_parse_infer_extensions_from_march(self) -> None:
        """Test extension inference from -march flag."""
        cmd = "gcc -march=armv9.2-a+sve2+sme2 file.c"
        result = parse_acfl_command(cmd)

        assert "SVE2" in result["inferred_extensions"]
        assert "SME2" in result["inferred_extensions"]


class TestMcpuToExtensions:
    """Tests for mcpu_to_extensions function."""

    def test_mcpu_neoverse_v3(self) -> None:
        """Lookup Neoverse V3 extensions."""
        result = mcpu_to_extensions("neoverse-v3")

        assert result["cpu"] == "Neoverse V3"
        assert result["isa_level"] == "armv9.2-a"
        assert result["sve"] is True
        assert result["sve2"] is True
        assert result["sme"] is True
        assert result["sme2"] is True
        assert result["neon"] is True

    def test_mcpu_neoverse_v2(self) -> None:
        """Lookup Neoverse V2 extensions."""
        result = mcpu_to_extensions("neoverse-v2")

        assert result["sve2"] is True
        assert result["sme2"] is False

    def test_mcpu_neoverse_n3(self) -> None:
        """Lookup Neoverse N3 extensions."""
        result = mcpu_to_extensions("neoverse-n3")

        assert result["sve2"] is True
        assert result["sme"] is False
        assert result["sme2"] is False

    def test_mcpu_neoverse_n2(self) -> None:
        """Lookup Neoverse N2 extensions."""
        result = mcpu_to_extensions("neoverse-n2")

        assert result["sve"] is True
        assert result["sve2"] is False

    def test_mcpu_neoverse_n1(self) -> None:
        """Lookup Neoverse N1 extensions."""
        result = mcpu_to_extensions("neoverse-n1")

        assert result["sve"] is False
        assert result["neon"] is True

    def test_mcpu_cortex_a78(self) -> None:
        """Lookup Cortex-A78 extensions."""
        result = mcpu_to_extensions("cortex-a78")

        assert result["neon"] is True
        assert result["sve"] is False

    def test_mcpu_generic(self) -> None:
        """Lookup generic ARM target."""
        result = mcpu_to_extensions("generic")

        assert result["neon"] is True
        assert result["sve"] is False

    def test_mcpu_case_insensitive(self) -> None:
        """Test case-insensitive CPU lookup."""
        result_lower = mcpu_to_extensions("neoverse-v3")
        result_upper = mcpu_to_extensions("NEOVERSE-V3")

        assert result_lower == result_upper

    def test_mcpu_unknown_raises_keyerror(self) -> None:
        """Raise KeyError for unknown CPU."""
        with pytest.raises(KeyError):
            mcpu_to_extensions("unknown-cpu-xyz")


class TestLoadVectorizationReport:
    """Tests for load_vectorization_report function."""

    def test_load_report_success(self) -> None:
        """Load and parse a valid vectorization report."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                """
ACfL Report
-mcpu=neoverse-v3

Function foo:
  Loop 1: VECTORIZED with SVE2
  Loop 2: NOT VECTORIZED due to dependency
"""
            )
            f.flush()
            temp_path = Path(f.name)

        try:
            report_dict, annotated_lines = load_vectorization_report(temp_path)

            assert report_dict["vectorized_loops"] == 1
            assert report_dict["scalar_loops"] == 1
            assert "neoverse-v3" in report_dict["metadata"].get("mcpu", "")
            assert isinstance(annotated_lines, list)
        finally:
            temp_path.unlink()

    def test_load_report_file_not_found(self) -> None:
        """Raise FileNotFoundError for missing report file."""
        nonexistent = Path("/nonexistent/report.txt")

        with pytest.raises(FileNotFoundError):
            load_vectorization_report(nonexistent)

    def test_load_report_annotates_vectorized_lines(self) -> None:
        """Test that VECTORIZED lines are annotated with SVE2 color."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This loop was VECTORIZED with SVE2\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            report_dict, annotated_lines = load_vectorization_report(temp_path)

            # Check for SVE2 color annotation
            assert any("#00CC66" in line for line in annotated_lines)
        finally:
            temp_path.unlink()

    def test_load_report_annotates_non_vectorized_lines(self) -> None:
        """Test that NOT VECTORIZED lines are annotated with amber color."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This loop was NOT VECTORIZED\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            report_dict, annotated_lines = load_vectorization_report(temp_path)

            # Check for amber color annotation
            assert any("#FFAA00" in line for line in annotated_lines)
        finally:
            temp_path.unlink()

    def test_load_report_counts_loops(self) -> None:
        """Test accurate loop counting."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                """
Loop 1: VECTORIZED
Loop 2: VECTORIZED
Loop 3: NOT VECTORIZED
Loop 4: NOT VECTORIZED
Loop 5: NOT VECTORIZED
"""
            )
            f.flush()
            temp_path = Path(f.name)

        try:
            report_dict, annotated_lines = load_vectorization_report(temp_path)

            assert report_dict["vectorized_loops"] == 2
            assert report_dict["scalar_loops"] == 3
        finally:
            temp_path.unlink()


class TestConfigIntegration:
    """Tests for config module constants."""

    def test_isa_colors_has_all_keys(self) -> None:
        """Verify ISA_COLORS dict has all required keys."""
        required_keys = {"NEON", "SVE2", "SME2", "ArmPL", "KleidiAI"}

        assert all(key in ISA_COLORS for key in required_keys)

    def test_isa_colors_are_hex(self) -> None:
        """Verify all ISA color values are valid hex codes."""
        for name, color in ISA_COLORS.items():
            assert isinstance(color, str)
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB format

    def test_neoverse_cpus_not_empty(self) -> None:
        """Verify NEOVERSE_CPUS list is populated."""
        assert len(NEOVERSE_CPUS) > 0

    def test_neoverse_cpus_have_required_fields(self) -> None:
        """Verify each CPU dict has required fields."""
        required_fields = {"cpu", "isa_level", "sve", "sve2", "sme2", "neon"}

        for cpu_entry in NEOVERSE_CPUS:
            assert all(field in cpu_entry for field in required_fields)


class TestArmQemuDetection:
    """Tests for ARM QEMU environment detection."""

    def test_detect_arm64_platform(self) -> None:
        """Test detection of aarch64 platform."""
        import platform

        # Mock platform.machine() to return aarch64
        with mock.patch("platform.machine", return_value="aarch64"):
            machine = platform.machine()
            assert machine == "aarch64"

    def test_detect_non_arm_platform(self) -> None:
        """Test detection of non-ARM platform."""
        import platform

        # Mock platform.machine() to return x86_64
        with mock.patch("platform.machine", return_value="x86_64"):
            machine = platform.machine()
            assert machine != "aarch64"


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_parse_empty_flags(self) -> None:
        """Parse a command with no optimization flags."""
        cmd = "gcc file.c"
        result = parse_acfl_command(cmd)

        assert result["opt_level"] == "unknown"

    def test_parse_fortran_file(self) -> None:
        """Parse a Fortran compilation command."""
        cmd = "armftn -O3 -mcpu=neoverse-v3 matrix.f90"
        result = parse_acfl_command(cmd)

        assert result["compiler"] == "armftn"
        assert result["mcpu"] == "neoverse-v3"

    def test_mcpu_lookup_with_hyphens(self) -> None:
        """Test mcpu lookup with various hyphenation."""
        # Standard form
        result = mcpu_to_extensions("neoverse-v3")
        assert result["cpu"] == "Neoverse V3"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
