#!/bin/bash
#
# SLURM job script — 8-node DPO training on LUMI
#
# Submit with:  sbatch launch.sh
# Or use:       ./submit_and_tail.sh launch.sh

#SBATCH --account=project_465002038
#SBATCH --partition=standard-g
#SBATCH --exclusive=user
#SBATCH --nodes=8
#SBATCH --gpus-per-node=mi250:8
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --hint=nomultithread
#SBATCH --job-name=dpo-8ki-8node

set -euo pipefail

# ---- Modules ----
module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems

# ---- Compiler (something inside the container occasionally needs to compile) ----
export CC=gcc-12
export CXX=g++-12

# ---- Networking ----
export HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export OMP_NUM_THREADS=1
export NCCL_NET_GDR_LEVEL=PHB

# ---- Container ----
CONTAINER=/scratch/project_465002038/environment/containers/rocm624_torch271.sif

srun singularity exec \
  -B /usr/x86_64-suse-linux:/usr/x86_64-suse-linux \
  --env CXX="$CXX" \
  --env CC="$CC" \
  "$CONTAINER" \
  bash container_entry.sh
