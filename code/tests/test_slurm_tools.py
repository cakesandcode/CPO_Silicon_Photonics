"""
Tests for slurm_tools module.

Validates parsing, topology generation, binding validation, and strategy suggestion.
"""

import pytest
from code.utils.slurm_tools import (
    parse_sbatch_script,
    generate_numa_topology,
    validate_binding_flags,
    suggest_binding_strategy,
)


class TestParseSbatchScript:
    """Test sbatch script parsing."""

    def test_parse_sbatch_basic(self):
        """Parse a minimal sbatch script and verify known directives extracted."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

echo "Hello"
"""
        result = parse_sbatch_script(script)

        assert result['job_name'] == 'test_job'
        assert result['nodes'] == '2'
        assert result['ntasks_per_node'] == '64'
        assert result['cpus_per_task'] == '1'
        assert result['time'] == '00:30:00'
        assert result['partition'] is None

    def test_parse_sbatch_unknown_directive(self):
        """Unknown directive goes to 'other' list."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --unknown-flag=value
#SBATCH --another-unknown

echo "Test"
"""
        result = parse_sbatch_script(script)

        assert result['job_name'] == 'test_job'
        assert len(result['other']) == 2
        assert '--unknown-flag=value' in result['other']
        assert '--another-unknown' in result['other']

    def test_parse_sbatch_with_equals(self):
        """Parse directives with --key=value syntax."""
        script = """#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=hpc
#SBATCH --account=research
"""
        result = parse_sbatch_script(script)

        assert result['job_name'] == 'benchmark'
        assert result['partition'] == 'hpc'
        assert result['account'] == 'research'

    def test_parse_sbatch_binding_flags(self):
        """Parse CPU and memory binding directives."""
        script = """#!/bin/bash
#SBATCH --cpu-bind=mask_cpu
#SBATCH --mem-bind=local
"""
        result = parse_sbatch_script(script)

        assert result['cpu_bind'] == 'mask_cpu'
        assert result['mem_bind'] == 'local'


class TestGenerateNumaTopology:
    """Test NUMA topology generation."""

    def test_generate_numa_topology_v3(self):
        """Verify V3 topology returns expected structure."""
        result = generate_numa_topology('neoverse-v3')

        assert result['cores_per_socket'] == 128
        assert result['numa_nodes_per_socket'] == 2
        assert result['cores_per_numa'] == 64
        assert result['confidence'] == 'estimated'
        assert 'note' in result

    def test_generate_numa_topology_n3(self):
        """Verify N3 topology returns expected structure."""
        result = generate_numa_topology('neoverse-n3')

        assert result['cores_per_socket'] == 64
        assert result['numa_nodes_per_socket'] == 1
        assert result['cores_per_numa'] == 64
        assert result['confidence'] == 'verified'
        assert 'note' in result

    def test_generate_numa_topology_invalid(self):
        """ValueError for unknown CPU model."""
        with pytest.raises(ValueError, match="Unknown CPU model"):
            generate_numa_topology('unknown-cpu')


class TestValidateBindingFlags:
    """Test binding flag validation."""

    def test_validate_binding_valid(self):
        """No warnings for known-good flags."""
        flags = {
            'cpu_bind': 'mask_cpu',
            'mem_bind': 'local',
        }
        warnings = validate_binding_flags(flags)

        assert len(warnings) == 0

    def test_validate_binding_rank(self):
        """Rank-based binding is valid."""
        flags = {
            'cpu_bind': 'rank',
            'mem_bind': 'local',
        }
        warnings = validate_binding_flags(flags)

        assert len(warnings) == 0

    def test_validate_binding_none(self):
        """'none' is valid for both cpu_bind and mem_bind."""
        flags = {
            'cpu_bind': 'none',
            'mem_bind': 'none',
        }
        warnings = validate_binding_flags(flags)

        assert len(warnings) == 0

    def test_validate_binding_invalid_cpu_bind(self):
        """Warning produced for unknown cpu_bind value."""
        flags = {
            'cpu_bind': 'invalid_value',
        }
        warnings = validate_binding_flags(flags)

        assert len(warnings) == 1
        assert 'cpu_bind' in warnings[0]

    def test_validate_binding_invalid_mem_bind(self):
        """Warning produced for unknown mem_bind value."""
        flags = {
            'mem_bind': 'global',
        }
        warnings = validate_binding_flags(flags)

        assert len(warnings) == 1
        assert 'mem_bind' in warnings[0]

    def test_validate_binding_both_invalid(self):
        """Multiple warnings if both are invalid."""
        flags = {
            'cpu_bind': 'bad_cpu',
            'mem_bind': 'bad_mem',
        }
        warnings = validate_binding_flags(flags)

        assert len(warnings) == 2


class TestSuggestBindingStrategy:
    """Test binding strategy suggestion."""

    def test_suggest_binding_strategy_v3(self):
        """Returns valid binding dict for V3 + 128 ranks."""
        topology = generate_numa_topology('neoverse-v3')
        result = suggest_binding_strategy(topology, mpi_ranks=128)

        assert result['cpu_bind_flag'] == 'mask_cpu'
        assert result['mem_bind_flag'] == 'local'
        assert result['ntasks_per_node'] == 128
        assert result['cpus_per_task'] == 1
        assert 'rationale' in result
        assert len(result['rationale']) > 0

    def test_suggest_binding_strategy_n3(self):
        """Returns valid binding dict for N3."""
        topology = generate_numa_topology('neoverse-n3')
        result = suggest_binding_strategy(topology, mpi_ranks=32)

        assert result['cpu_bind_flag'] == 'mask_cpu'
        assert result['mem_bind_flag'] == 'local'
        assert result['ntasks_per_node'] == 32
        assert result['cpus_per_task'] == 1
        assert 'rationale' in result

    def test_suggest_binding_strategy_oversubscribed(self):
        """Handles oversubscribed case (more ranks than cores)."""
        topology = generate_numa_topology('neoverse-v3')
        result = suggest_binding_strategy(topology, mpi_ranks=256)

        assert result['cpu_bind_flag'] == 'mask_cpu'
        assert result['mem_bind_flag'] == 'local'
        # 256 ranks on 128 cores: cpus_per_task should be >= 1
        assert result['cpus_per_task'] >= 1
        assert 'rationale' in result
