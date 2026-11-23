#!/bin/bash
#SBATCH --job-name=mjax-regrade
#SBATCH --account=bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/meshswarm/modelica_jax/logs/%x-%j.out
# Usage: sbatch regrade.sbatch TIER WORKSPACE [WORKSPACE ...]

REPO=/scratch/$USER/meshswarm/repo
source "$REPO/examples/modelica_jax/rivanna/env.sh"
set -euo pipefail
trap 'rm -rf "$MESH_LOCAL"' EXIT

hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cd "$REPO"
TIER=$1; shift
python examples/modelica_jax/regrade.py --tier "$TIER" "$@"
