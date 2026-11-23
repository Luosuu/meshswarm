#!/bin/bash
#SBATCH --job-name=meshmjax
#SBATCH --account=bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/meshswarm/modelica_jax/logs/%x-%j.out
# Usage:
#   sbatch job.sbatch --selftest
#   sbatch job.sbatch --tier tier1_blanket_dae --generations 3 --population 2
#   sbatch --gres=gpu:2 job.sbatch --tier tier2_transform_hx --generations 3 --population 2 --parallel

REPO=/scratch/$USER/meshswarm/repo
source "$REPO/examples/modelica_jax/rivanna/env.sh"
set -euo pipefail
trap 'rm -rf "$MESH_LOCAL"' EXIT

echo "Host: $(hostname) | command network: $MESH_RUN_NETWORK"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cd "$REPO"
python examples/modelica_jax/run.py --require-gpu --api-key-file /scratch/$USER/meshswarm/.tinker_key "$@"
