#!/bin/bash
#SBATCH --job-name=arc-gen45
#SBATCH --account=bii_dsc_community
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/meshswarm/tmap8/logs/%x-%j.out

REPO=/scratch/$USER/meshswarm/repo
source "$REPO/examples/tmap8_arc_plant/rivanna/env.sh"
set -euo pipefail
trap 'rm -rf "$MESH_LOCAL"' EXIT

echo "Host: $(hostname) | CPUs: $(nproc) | command network: $MESH_RUN_NETWORK"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU attached (CPU JAX + multi-core MOOSE mode)"
KEY_FILE="${TINKER_KEY_FILE:-/scratch/$USER/meshswarm/.tinker_key}"
cd "$REPO"
python examples/tmap8_arc_plant/run.py \
  --gpus "" \
  --api-key-file "$KEY_FILE" \
  --run-dir "/scratch/$USER/meshswarm/tmap8/runs" \
  "$@"
