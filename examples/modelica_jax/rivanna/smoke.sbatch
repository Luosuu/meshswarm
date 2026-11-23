#!/bin/bash
#SBATCH --job-name=mjax-smoke
#SBATCH --account=bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/meshswarm/modelica_jax/logs/%x-%j.out

REPO=/scratch/$USER/meshswarm/repo
source "$REPO/examples/modelica_jax/rivanna/env.sh"
set -euo pipefail

echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python --version

WORK_TMP="${TMPDIR:-/tmp}/$USER-mjax-${SLURM_JOB_ID:-$$}"
mkdir -p "$WORK_TMP"
trap 'rm -rf "$WORK_TMP"' EXIT

cd "$REPO"
python examples/modelica_jax/smoke_pipeline.py --workdir "$WORK_TMP" --n-ensemble 1000
