"""
Tests for map_profiler utility module.

Verifies parse_map_file(), generate_flamechart_data(), and
analyze_profile_for_optimization() across valid and edge-case inputs.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from code.utils.map_profiler import (
    analyze_profile_for_optimization,
    generate_flamechart_data,
    parse_map_file,
)


@pytest.fixture
def valid_map_data() -> dict:
    """Return valid mock MAP profile data."""
    return {
        "metadata": {
            "tool": "Linaro Forge MAP 25.1.2",
            "generated_by": "synthetic",
            "app": "hpl_benchmark",
            "ranks": 8,
            "runtime_s": 47.3,
        },
        "time_breakdown": {
            "cpu_pct": 71.2,
            "mpi_pct": 22.4,
            "io_pct": 4.1,
            "thread_overhead_pct": 2.3,
        },
        "hotspots": [
            {
                "function": "dgemm_",
                "file": "hpl_dgemm.c",
                "line": 142,
                "cpu_pct": 34.5,
                "mpi_pct": 0.1,
            },
            {
                "function": "MPI_Allreduce",
                "file": "hpl_comm.c",
                "line": 87,
                "cpu_pct": 1.2,
                "mpi_pct": 18.3,
            },
            {
                "function": "dtrsm_",
                "file": "hpl_dtrsm.c",
                "line": 201,
                "cpu_pct": 18.7,
                "mpi_pct": 0.3,
            },
        ],
    }


class TestParseMapFile:
    """Tests for parse_map_file()."""

    def test_parse_map_file_structure(self, valid_map_data: dict) -> None:
        """Verify parse_map_file returns dict with required keys."""
        result = parse_map_file(valid_map_data)

        required_keys = {
            "total_runtime_s",
            "ranks",
            "samples",
            "cpu_time_pct",
            "mpi_time_pct",
            "io_time_pct",
            "thread_overhead_pct",
            "hotspots",
        }
        assert required_keys.issubset(result.keys())

    def test_parse_map_file_values(self, valid_map_data: dict) -> None:
        """Verify parse_map_file extracts correct values."""
        result = parse_map_file(valid_map_data)

        assert result["total_runtime_s"] == 47.3
        assert result["ranks"] == 8
        assert result["samples"] == 3
        assert result["cpu_time_pct"] == 71.2
        assert result["mpi_time_pct"] == 22.4
        assert result["io_time_pct"] == 4.1
        assert result["thread_overhead_pct"] == 2.3

    def test_parse_map_file_hotspot_normalization(
        self, valid_map_data: dict
    ) -> None:
        """Verify hotspots are normalized to expected keys."""
        result = parse_map_file(valid_map_data)

        for hs in result["hotspots"]:
            required_hs_keys = {
                "function_name",
                "file",
                "line",
                "pct_cpu_time",
                "pct_mpi_time",
            }
            assert required_hs_keys.issubset(hs.keys())

    def test_parse_map_file_missing_metadata(self) -> None:
        """Verify KeyError raised when 'metadata' key missing."""
        invalid_data = {
            "time_breakdown": {"cpu_pct": 100.0},
            "hotspots": [],
        }
        with pytest.raises(KeyError):
            parse_map_file(invalid_data)

    def test_parse_map_file_invalid_runtime(self, valid_map_data: dict) -> None:
        """Verify ValueError raised for invalid runtime_s."""
        valid_map_data["metadata"]["runtime_s"] = 0
        with pytest.raises(ValueError):
            parse_map_file(valid_map_data)

    def test_parse_map_file_invalid_ranks(self, valid_map_data: dict) -> None:
        """Verify ValueError raised for invalid ranks."""
        valid_map_data["metadata"]["ranks"] = 0
        with pytest.raises(ValueError):
            parse_map_file(valid_map_data)

    def test_parse_map_file_percentage_mismatch(
        self, valid_map_data: dict
    ) -> None:
        """Verify ValueError raised when percentages do not sum to ~100%."""
        valid_map_data["time_breakdown"]["cpu_pct"] = 50.0
        valid_map_data["time_breakdown"]["mpi_pct"] = 20.0
        valid_map_data["time_breakdown"]["io_pct"] = 20.0
        valid_map_data["time_breakdown"]["thread_overhead_pct"] = 1.0

        with pytest.raises(ValueError):
            parse_map_file(valid_map_data)


class TestGenerateFlamechartData:
    """Tests for generate_flamechart_data()."""

    def test_generate_flamechart_data_entries(self, valid_map_data: dict) -> None:
        """Verify generate_flamechart_data returns list with required keys."""
        profile = parse_map_file(valid_map_data)
        result = generate_flamechart_data(profile)

        assert isinstance(result, list)
        for entry in result:
            required_keys = {"x_start", "x_end", "y_level", "label", "color", "pct"}
            assert required_keys.issubset(entry.keys())

    def test_generate_flamechart_data_color_compute(
        self, valid_map_data: dict
    ) -> None:
        """Verify compute-heavy hotspots colored SVE2 green."""
        profile = parse_map_file(valid_map_data)
        result = generate_flamechart_data(profile)

        dgemm_entry = [e for e in result if "dgemm" in e["label"].lower()]
        assert len(dgemm_entry) > 0
        assert dgemm_entry[0]["color"] == "#00CC66"

    def test_generate_flamechart_data_color_mpi(self, valid_map_data: dict) -> None:
        """Verify MPI-heavy hotspots colored NEON blue."""
        profile = parse_map_file(valid_map_data)
        result = generate_flamechart_data(profile)

        mpi_entry = [e for e in result if "MPI" in e["label"]]
        assert len(mpi_entry) > 0
        assert mpi_entry[0]["color"] == "#0066CC"

    def test_generate_flamechart_data_empty_hotspots(self) -> None:
        """Verify empty list returned for profile with no hotspots."""
        profile = {
            "total_runtime_s": 10.0,
            "ranks": 1,
            "samples": 0,
            "cpu_time_pct": 100.0,
            "mpi_time_pct": 0.0,
            "io_time_pct": 0.0,
            "thread_overhead_pct": 0.0,
            "hotspots": [],
        }
        result = generate_flamechart_data(profile)
        assert result == []


class TestAnalyzeProfileForOptimization:
    """Tests for analyze_profile_for_optimization()."""

    def test_analyze_mpi_heavy_profile(self, valid_map_data: dict) -> None:
        """Verify MPI > 20% triggers HIGH priority MPI suggestion."""
        profile = parse_map_file(valid_map_data)
        result = analyze_profile_for_optimization(profile)

        mpi_recs = [r for r in result if r["category"] == "MPI"]
        assert len(mpi_recs) > 0
        assert mpi_recs[0]["priority"] == "HIGH"
        assert "MPI" in mpi_recs[0]["suggestion"]

    def test_analyze_compute_heavy_profile(self, valid_map_data: dict) -> None:
        """Verify high CPU + concentrated hotspot triggers compute suggestion."""
        profile = parse_map_file(valid_map_data)
        result = analyze_profile_for_optimization(profile)

        compute_recs = [r for r in result if r["category"] == "Compute"]
        assert len(compute_recs) > 0
        assert "dgemm_" in compute_recs[0]["suggestion"]

    def test_analyze_io_heavy_profile(self) -> None:
        """Verify I/O > 10% triggers I/O suggestion."""
        io_heavy_data = {
            "metadata": {
                "tool": "Linaro Forge MAP 25.1.2",
                "generated_by": "synthetic",
                "app": "io_benchmark",
                "ranks": 4,
                "runtime_s": 30.0,
            },
            "time_breakdown": {
                "cpu_pct": 60.0,
                "mpi_pct": 10.0,
                "io_pct": 25.0,
                "thread_overhead_pct": 5.0,
            },
            "hotspots": [
                {
                    "function": "write_checkpoint",
                    "file": "io.c",
                    "line": 100,
                    "cpu_pct": 20.0,
                    "mpi_pct": 0.0,
                },
            ],
        }
        profile = parse_map_file(io_heavy_data)
        result = analyze_profile_for_optimization(profile)

        io_recs = [r for r in result if r["category"] == "I/O"]
        assert len(io_recs) > 0
        assert io_recs[0]["priority"] == "HIGH"

    def test_analyze_profile_no_hotspots(self) -> None:
        """Verify analysis handles profile with no hotspots."""
        profile = {
            "total_runtime_s": 10.0,
            "ranks": 1,
            "samples": 0,
            "cpu_time_pct": 100.0,
            "mpi_time_pct": 0.0,
            "io_time_pct": 0.0,
            "thread_overhead_pct": 0.0,
            "hotspots": [],
        }
        result = analyze_profile_for_optimization(profile)
        assert isinstance(result, list)

    def test_analyze_profile_priority_order(self, valid_map_data: dict) -> None:
        """Verify recommendations sorted by priority (HIGH > MEDIUM > LOW)."""
        profile = parse_map_file(valid_map_data)
        result = analyze_profile_for_optimization(profile)

        if len(result) > 1:
            priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            for i in range(len(result) - 1):
                assert priority_order[result[i]["priority"]] <= priority_order[
                    result[i + 1]["priority"]
                ]


class TestIntegration:
    """Integration tests: parse → flamechart → analyze."""

    def test_load_mock_profile_from_json(self) -> None:
        """Verify mock MAP profile JSON loads and parses correctly."""
        mock_path = Path(__file__).parent.parent / "data" / "mock_map_profile.json"
        assert mock_path.exists(), f"Mock profile not found at {mock_path}"

        with open(mock_path) as f:
            data = json.load(f)

        profile = parse_map_file(data)
        assert profile["ranks"] == 8
        assert profile["total_runtime_s"] == 47.3
        assert len(profile["hotspots"]) >= 8

    def test_full_pipeline(self, valid_map_data: dict) -> None:
        """Verify full pipeline: parse → flamechart → analyze."""
        profile = parse_map_file(valid_map_data)
        flamechart = generate_flamechart_data(profile)
        recommendations = analyze_profile_for_optimization(profile)

        assert len(flamechart) > 0
        assert len(recommendations) > 0
