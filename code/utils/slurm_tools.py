"""
Slurm job script tools for Neoverse N3/V3 HPC clusters.

Educational module teaching Slurm job script anatomy: NUMA topology awareness,
CPU binding, MPI integration on modern ARM processors.
"""

from typing import Dict, List
import re

from code.utils.config import SLURM_BINDING_CONFIG


def parse_sbatch_script(script: str) -> Dict:
    """Parse an sbatch script string and extract all #SBATCH directives.

    Extracts both recognized directives (nodes, ntasks, ntasks_per_node,
    cpus_per_task, mem_bind, cpu_bind, partition, time, job_name, account)
    and any unrecognized directives placed in an 'other' list.

    Args:
        script: A string containing an sbatch script with #SBATCH lines.

    Returns:
        A dict with keys: nodes, ntasks, ntasks_per_node, cpus_per_task,
        mem_bind, cpu_bind, partition, time, job_name, account, other.
        Known keys are strings or None if not found. 'other' is a list of
        unrecognized #SBATCH directives.

    Raises:
        None — unrecognized directives are collected, not errored.
    """
    result = {
        "nodes": None,
        "ntasks": None,
        "ntasks_per_node": None,
        "cpus_per_task": None,
        "mem_bind": None,
        "cpu_bind": None,
        "partition": None,
        "time": None,
        "job_name": None,
        "account": None,
        "other": [],
    }

    # Regex to match #SBATCH --key=value or --key value patterns
    lines = script.split('\n')
    for line in lines:
        if not line.strip().startswith('#SBATCH'):
            continue

        # Remove #SBATCH prefix and strip
        sbatch_content = line.replace('#SBATCH', '').strip()

        # Parse --key=value or --key value (allow hyphens in key names)
        match = re.match(r'--([a-z0-9\-]+)(?:=(.+?))?(?:\s+(.+))?$', sbatch_content)
        if not match:
            continue

        key = match.group(1)
        value = match.group(2) or match.group(3)

        if value:
            value = value.strip()

        # Map known directives (normalize key with underscores for consistency)
        if key == 'nodes':
            result['nodes'] = value
        elif key == 'ntasks':
            result['ntasks'] = value
        elif key == 'ntasks-per-node':
            result['ntasks_per_node'] = value
        elif key == 'cpus-per-task':
            result['cpus_per_task'] = value
        elif key == 'mem-bind':
            result['mem_bind'] = value
        elif key == 'cpu-bind':
            result['cpu_bind'] = value
        elif key == 'partition':
            result['partition'] = value
        elif key == 'time':
            result['time'] = value
        elif key == 'job-name':
            result['job_name'] = value
        elif key == 'account':
            result['account'] = value
        else:
            # Collect unrecognized directives
            result['other'].append(f"--{key}={value}" if value else f"--{key}")

    return result


def generate_numa_topology(cpu_model: str) -> Dict:
    """Generate NUMA topology for known Neoverse CPU models.

    Provides NUMA configuration for job binding guidance on Neoverse N3/V3.
    Topology is derived from public documentation; V3 estimates are reasoned
    based on core count and typical ARM platform patterns.

    Args:
        cpu_model: CPU model string. Supported: 'neoverse-v3', 'neoverse-n3'.

    Returns:
        A dict with keys:
        - cores_per_socket: Total cores per socket.
        - numa_nodes_per_socket: Number of NUMA nodes per socket.
        - cores_per_numa: Cores per NUMA node.
        - note: Description and confidence level.
        - confidence: 'verified' or 'estimated'.

    Raises:
        ValueError: If cpu_model is not recognized.
    """
    topologies = {
        'neoverse-v3': {
            'cores_per_socket': 128,
            'numa_nodes_per_socket': 2,
            'cores_per_numa': 64,
            'note': 'V3 NUMA topology is not confirmed in public docs — this is a reasoned estimate based on 128 cores/socket',
            'confidence': 'estimated',
        },
        'neoverse-n3': {
            'cores_per_socket': 64,
            'numa_nodes_per_socket': 1,
            'cores_per_numa': 64,
            'note': 'N3 production configs (Graviton4, Axion) have 64+ cores per socket',
            'confidence': 'verified',
        },
    }

    if cpu_model not in topologies:
        raise ValueError(
            f"Unknown CPU model: {cpu_model}. Supported: {list(topologies.keys())}"
        )

    return topologies[cpu_model]


def validate_binding_flags(flags: Dict) -> List[str]:
    """Validate CPU and memory binding flags against known good patterns.

    Args:
        flags: A dict with optional keys 'cpu_bind' and 'mem_bind' containing
               the flag values to validate.

    Returns:
        A list of warning strings (empty list means valid). Warnings issued for:
        - cpu_bind not in valid_cpu_binds from config
        - mem_bind not in valid_mem_binds from config
    """
    warnings = []

    valid_cpu_binds = SLURM_BINDING_CONFIG["valid_cpu_binds"]
    valid_mem_binds = SLURM_BINDING_CONFIG["valid_mem_binds"]

    if 'cpu_bind' in flags and flags['cpu_bind']:
        if flags['cpu_bind'] not in valid_cpu_binds:
            warnings.append(
                f"cpu_bind '{flags['cpu_bind']}' not in known patterns {valid_cpu_binds}"
            )

    if 'mem_bind' in flags and flags['mem_bind']:
        if flags['mem_bind'] not in valid_mem_binds:
            warnings.append(
                f"mem_bind '{flags['mem_bind']}' not in known patterns {valid_mem_binds}"
            )

    return warnings


def suggest_binding_strategy(topology: Dict, mpi_ranks: int) -> Dict:
    """Suggest optimal CPU/memory binding strategy for MPI on Neoverse.

    Given a NUMA topology and number of MPI ranks per node, recommends
    binding flags, task distribution, and rationale.

    Args:
        topology: A NUMA topology dict (as returned by generate_numa_topology).
        mpi_ranks: Number of MPI ranks to place on one node. Must be > 0.

    Returns:
        A dict with keys:
        - cpu_bind_flag: Recommended --cpu-bind value.
        - mem_bind_flag: Recommended --mem-bind value.
        - ntasks_per_node: Recommended --ntasks-per-node.
        - cpus_per_task: Recommended --cpus-per-task.
        - rationale: Explanation of the recommendation.
    """
    if mpi_ranks <= 0:
        raise ValueError(f"mpi_ranks must be > 0, got {mpi_ranks}")

    cores_per_socket = topology['cores_per_socket']
    numa_nodes_per_socket = topology['numa_nodes_per_socket']
    cores_per_numa = topology['cores_per_numa']

    # Assume single socket (most common educational case)
    total_cores = cores_per_socket

    # If MPI ranks fit within cores, 1 rank per core with cpus_per_task=1
    if mpi_ranks <= total_cores:
        cpus_per_task = 1
        ntasks_per_node = mpi_ranks
        rationale = (
            f"One MPI rank per core ({mpi_ranks} cores available). "
            f"mask_cpu ensures NUMA-aware binding; local memory binding keeps "
            f"data close to computation."
        )
    else:
        # Oversubscribe: multiple ranks per core
        cpus_per_task = max(1, mpi_ranks // total_cores)
        ntasks_per_node = mpi_ranks
        rationale = (
            f"Over-subscribed: {mpi_ranks} ranks on {total_cores} cores. "
            f"Each rank gets {cpus_per_task} CPU(s). Binding helps scheduler "
            f"distribute work fairly across NUMA nodes."
        )

    return {
        'cpu_bind_flag': 'mask_cpu',
        'mem_bind_flag': 'local',
        'ntasks_per_node': ntasks_per_node,
        'cpus_per_task': cpus_per_task,
        'rationale': rationale,
    }
