"""
Test suite for code.utils.armpl_benchmark module.

Tests verify synthetic data generation, shape/structure contracts,
determinism, and roofline bounds.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from code.utils.armpl_benchmark import (
    compute_roofline_model,
    generate_dgemm_throughput,
    load_fft_scaling,
    load_library_comparison,
)


class TestGenerateDGEMMThroughput:
    """Tests for generate_dgemm_throughput()."""

    def test_generate_dgemm_throughput_shape_numpy(self):
        """Verify output DataFrame has correct shape and columns for numpy."""
        sizes = [64, 128, 256, 512]
        df = generate_dgemm_throughput(sizes, library="numpy")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sizes)
        assert list(df.columns) == ["matrix_size", "throughput_gflops", "library"]
        assert all(df["library"] == "numpy")

    def test_generate_dgemm_throughput_shape_armpl(self):
        """Verify output DataFrame has correct shape and columns for armpl."""
        sizes = [64, 128, 256, 512]
        df = generate_dgemm_throughput(sizes, library="armpl")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sizes)
        assert list(df.columns) == ["matrix_size", "throughput_gflops", "library"]
        assert all(df["library"] == "armpl")

    def test_generate_dgemm_throughput_deterministic(self):
        """Verify same input produces same output (reproducibility)."""
        sizes = [64, 128, 256, 512]
        df1 = generate_dgemm_throughput(sizes, library="numpy")
        df2 = generate_dgemm_throughput(sizes, library="numpy")

        # DataFrames should be equal (within floating-point tolerance)
        pd.testing.assert_frame_equal(df1, df2)

    def test_generate_dgemm_throughput_armpl_higher_than_numpy(self):
        """Verify ArmPL curves are >= NumPy on average (SVE2 benefit)."""
        sizes = [256, 512, 1024, 2048]
        df_numpy = generate_dgemm_throughput(sizes, library="numpy")
        df_armpl = generate_dgemm_throughput(sizes, library="armpl")

        # On large matrices, ArmPL should be >= NumPy
        for idx in range(len(sizes)):
            armpl_val = df_armpl.loc[df_armpl["matrix_size"] == sizes[idx], "throughput_gflops"].values[0]
            numpy_val = df_numpy.loc[df_numpy["matrix_size"] == sizes[idx], "throughput_gflops"].values[0]
            assert armpl_val >= numpy_val * 0.95, f"ArmPL not competitive at size {sizes[idx]}"

    def test_generate_dgemm_throughput_invalid_library(self):
        """Verify ValueError on invalid library name."""
        with pytest.raises(ValueError, match="library must be"):
            generate_dgemm_throughput([64, 128], library="invalid")

    def test_generate_dgemm_throughput_throughputs_positive(self):
        """Verify all throughput values are positive."""
        sizes = [64, 128, 256, 512, 1024]
        df = generate_dgemm_throughput(sizes, library="numpy")

        assert all(df["throughput_gflops"] > 0), "All throughputs must be positive"

    def test_generate_dgemm_throughput_empty_sizes(self):
        """Verify handling of empty size list."""
        df = generate_dgemm_throughput([], library="numpy")
        assert len(df) == 0
        assert list(df.columns) == ["matrix_size", "throughput_gflops", "library"]


class TestLoadLibraryComparison:
    """Tests for load_library_comparison()."""

    def test_load_library_comparison_structure(self):
        """Verify loaded data has required columns and structure."""
        data_path = Path("code/data/dgemm_comparison.json")

        if not data_path.exists():
            pytest.skip(f"Data file not found: {data_path}")

        df = load_library_comparison(data_path)

        # Verify shape and columns
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "matrix_size" in df.columns
        assert "numpy_gflops" in df.columns
        assert "armpl_gflops" in df.columns

    def test_load_library_comparison_file_not_found(self):
        """Verify FileNotFoundError on missing file."""
        with pytest.raises(FileNotFoundError):
            load_library_comparison(Path("nonexistent.json"))

    def test_load_library_comparison_data_types(self):
        """Verify loaded data types are correct."""
        data_path = Path("code/data/dgemm_comparison.json")

        if not data_path.exists():
            pytest.skip(f"Data file not found: {data_path}")

        df = load_library_comparison(data_path)

        assert df["matrix_size"].dtype in (int, "int64", "int32")
        assert pd.api.types.is_numeric_dtype(df["numpy_gflops"])
        assert pd.api.types.is_numeric_dtype(df["armpl_gflops"])


class TestComputeRooflineModel:
    """Tests for compute_roofline_model()."""

    def test_roofline_model_bound(self):
        """Verify roofline is always <= peak_gflops."""
        peak_gflops = 2560.0
        peak_bandwidth_gbs = 200.0
        sizes = [64, 128, 256, 512, 1024]

        df = compute_roofline_model(peak_gflops, peak_bandwidth_gbs, sizes)

        assert all(df["roofline_gflops"] <= peak_gflops * 1.01), \
            "Roofline must not exceed peak GFLOP/s (with 1% tolerance)"

    def test_roofline_model_shape(self):
        """Verify output DataFrame shape and columns."""
        peak_gflops = 2560.0
        peak_bandwidth_gbs = 200.0
        sizes = [64, 128, 256, 512]

        df = compute_roofline_model(peak_gflops, peak_bandwidth_gbs, sizes)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sizes)
        assert list(df.columns) == ["matrix_size", "roofline_gflops"]

    def test_roofline_model_increases_with_intensity(self):
        """Verify roofline generally increases with matrix size (higher AI)."""
        peak_gflops = 2560.0
        peak_bandwidth_gbs = 200.0
        sizes = [64, 256, 1024, 4096]

        df = compute_roofline_model(peak_gflops, peak_bandwidth_gbs, sizes)
        rooflines = df["roofline_gflops"].values

        # Higher arithmetic intensity → higher roofline (up to peak)
        for i in range(len(rooflines) - 1):
            assert rooflines[i+1] >= rooflines[i] * 0.99, \
                "Roofline should not decrease with increasing matrix size"

    def test_roofline_model_positive(self):
        """Verify all roofline values are positive."""
        peak_gflops = 2560.0
        peak_bandwidth_gbs = 200.0
        sizes = [64, 128, 256, 512]

        df = compute_roofline_model(peak_gflops, peak_bandwidth_gbs, sizes)

        assert all(df["roofline_gflops"] > 0), "All roofline values must be positive"


class TestLoadFFTScaling:
    """Tests for load_fft_scaling()."""

    def test_load_fft_scaling_structure(self):
        """Verify loaded FFT data has required columns and structure."""
        data_path = Path("code/data/fft_scaling.json")

        if not data_path.exists():
            pytest.skip(f"Data file not found: {data_path}")

        df = load_fft_scaling(data_path)

        # Verify shape and columns
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "fft_size" in df.columns
        assert "armpl_gflops" in df.columns
        assert "numpy_fft_gflops" in df.columns

    def test_load_fft_scaling_file_not_found(self):
        """Verify FileNotFoundError on missing file."""
        with pytest.raises(FileNotFoundError):
            load_fft_scaling(Path("nonexistent_fft.json"))

    def test_load_fft_scaling_decreases_with_size(self):
        """Verify FFT throughput decreases with size (cache effects)."""
        data_path = Path("code/data/fft_scaling.json")

        if not data_path.exists():
            pytest.skip(f"Data file not found: {data_path}")

        df = load_fft_scaling(data_path)

        # Larger FFTs → lower throughput due to cache misses
        armpl_vals = df["armpl_gflops"].values
        for i in range(len(armpl_vals) - 1):
            assert armpl_vals[i+1] <= armpl_vals[i] * 1.01, \
                "FFT throughput should decrease with size"

    def test_load_fft_scaling_armpl_advantage(self):
        """Verify ArmPL FFT is >= NumPy FFT on average."""
        data_path = Path("code/data/fft_scaling.json")

        if not data_path.exists():
            pytest.skip(f"Data file not found: {data_path}")

        df = load_fft_scaling(data_path)

        # ArmPL should be competitive with NumPy FFT
        assert (df["armpl_gflops"] >= df["numpy_fft_gflops"] * 0.95).all(), \
            "ArmPL FFT should be >= 95% of NumPy FFT throughput"

    def test_load_fft_scaling_data_types(self):
        """Verify loaded data types are correct."""
        data_path = Path("code/data/fft_scaling.json")

        if not data_path.exists():
            pytest.skip(f"Data file not found: {data_path}")

        df = load_fft_scaling(data_path)

        assert df["fft_size"].dtype in (int, "int64", "int32")
        assert pd.api.types.is_numeric_dtype(df["armpl_gflops"])
        assert pd.api.types.is_numeric_dtype(df["numpy_fft_gflops"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
