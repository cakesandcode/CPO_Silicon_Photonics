#!/bin/bash
# Annotated Slurm job script for HPC MPI benchmark on Neoverse V3
# Demonstrates NUMA-aware resource allocation and binding
# Slurm 25.11 (current stable) + OpenMPI 5.x + TaskPlugin: affinity,cgroup_v2

#SBATCH --job-name=hpc_benchmark
# Job name for identification in queue and logs. Use descriptive names for tracking.

#SBATCH --nodes=4
# Request 4 compute nodes. Neoverse V3 clusters typically offer multi-socket nodes;
# adjust based on cluster size and queue limits.

#SBATCH --ntasks-per-node=128
# One MPI rank per core on Neoverse V3 (128 cores per socket, single-socket assumption).
# This maximizes parallelism for embarrassingly parallel HPC codes. Adjust for
# hybrid MPI+OpenMP models (e.g., 64 ranks × 2 threads = 128 logical tasks).

#SBATCH --cpus-per-task=1
# Each MPI rank gets 1 CPU. Combined with ntasks-per-node=128, this fully occupies
# the socket. For multithreaded ranks, increase to 2+ (requires proportional ntasks reduction).

#SBATCH --mem-bind=local
# Memory is allocated from NUMA nodes local to the CPU running the task.
# Critical on Neoverse V3 (2 NUMA domains per socket): ensures minimal remote memory latency.
# Use 'local' for latency-sensitive codes; use 'none' for memory-limited workloads
# that benefit from global visibility.

#SBATCH --cpu-bind=mask_cpu
# Hardware-aware CPU binding via Linux affinity masks. Slurm TaskPlugin (affinity)
# applies this at task launch. 'mask_cpu' lets Slurm choose best available cores
# based on node topology. Alternative: 'rank' (rank 0 → core 0, etc.) or 'none' (no binding).
# mask_cpu is preferred for heterogeneous topologies.

#SBATCH --exclusive
# Exclusive node access: no other jobs share this compute node. Eliminates OS noise
# from sibling tasks. Important for reproducible HPC benchmarking. May have longer
# queue wait; acceptable for research/development.

#SBATCH --time=01:00:00
# Wall-clock time limit: 1 hour. Adjust based on application complexity.
# Slurm kills the job if it exceeds this wall time. Check historical job logs
# to set reasonable limits (avoid over-requesting, which increases queue wait).

#SBATCH --partition=hpc
# Submit to 'hpc' partition (queue). Cluster-specific: check available partitions
# with 'sinfo'. Some clusters name it 'standard', 'gpu', 'long', etc.
# HPC partitions typically have higher core counts and better interconnect.

#SBATCH --account=research
# Billing account for resource allocation (SLURM AccountingStorage plugin).
# Required on shared HPC systems; may be optional in dev clusters.
# Contact cluster admin if unsure.

# ============================================================================
# Module loading: Set up compiler, MPI, and math libraries
# ============================================================================

# Load ARM Compiler for Linux (ACfL) 25.04 — provides armclang, armflang, and OpenMPI
module load acfl/25.04
# Verify load: `which armclang` should return /opt/acfl/25.04/bin/armclang

# Load ARM Performance Libraries (ArmPL) 25.04 — BLAS, LAPACK, FFT, sparse solver
module load armpl/25.04
# ArmPL is auto-tuned for Neoverse V3: excellent DGEMM, FFT performance without
# manual optimization. Link via: -L$ARMPL_DIR/lib -larmpl_lp64

# ============================================================================
# Launch MPI application
# ============================================================================

# mpirun (OpenMPI 5.x launcher) — recommended for Slurm 22.05+ with PMIx v2–v5
# mpirun respects Slurm environment (SLURM_NTASKS, SLURM_NTASKS_PER_NODE) and
# negotiates process placement with Slurm's TaskPlugin.
# Alternative: srun --mpi=pmix ./hpc_benchmark (if site prefers srun)
#
# Note: PMI-2 is deprecated in OpenMPI 5.x; PMIx (Process Management Interface for
# Exascale) is the standard. Slurm 25.11 includes pmix plugin out-of-box.

mpirun -np 512 ./hpc_benchmark
# Launch 512 MPI processes: 4 nodes × 128 ranks/node = 512 processes.
# -np must match (nodes × ntasks-per-node).
# mpirun auto-distributes ranks across Slurm-allocated cores respecting
# the --cpu-bind and --mem-bind directives above.

# ============================================================================
# Common Debugging and Tuning
# ============================================================================

# To see what binding was applied:
#   srun --cpu-bind=verbose echo "Core affinity: $(taskset -cp $$)"
#
# To measure NUMA traffic:
#   numastat -n -c ./hpc_benchmark_on_slurm (post-job, if available)
#
# To inspect job output and errors:
#   sacct -j ${SLURM_JOB_ID} -o JobID,State,ExitCode,Comment
#
# To profile across nodes:
#   srun pfmon -e CYCLES_CNT,INST_RETIRED ./hpc_benchmark
