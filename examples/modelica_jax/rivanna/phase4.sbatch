#!/bin/bash
#SBATCH --job-name=mjax-phase4
#SBATCH --account=bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/meshswarm/modelica_jax/logs/%x-%j.out

REPO=/scratch/$USER/meshswarm/repo
source "$REPO/examples/modelica_jax/rivanna/env.sh"
set -euo pipefail
trap 'rm -rf "$MESH_LOCAL"' EXIT

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

hostname
lscpu | grep "Model name" || true
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
cd "$REPO"
python examples/modelica_jax/phase4/run_phase4.py \
  --out /scratch/$USER/meshswarm/modelica_jax/phase4_metrics.json
