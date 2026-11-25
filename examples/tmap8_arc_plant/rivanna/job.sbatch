#!/bin/bash
#SBATCH --job-name=meshtmap8
#SBATCH --account=bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/meshswarm/tmap8/logs/%x-%j.out
# Usage:
#   sbatch job.sbatch --selftest
#   sbatch --gres=gpu:a100:2 job.sbatch --generations 3 --population 2 --controls 1

REPO=/scratch/$USER/meshswarm/repo
source "$REPO/examples/tmap8_arc_plant/rivanna/env.sh"
set -euo pipefail
trap 'rm -rf "$MESH_LOCAL"' EXIT

echo "Host: $(hostname) | command network: $MESH_RUN_NETWORK"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
KEY_FILE="${TINKER_KEY_FILE:-/scratch/$USER/meshswarm/.tinker_key}"
cd "$REPO"
python examples/tmap8_arc_plant/run.py \
  --require-gpu \
  --api-key-file "$KEY_FILE" \
  --run-dir "/scratch/$USER/meshswarm/tmap8/runs" \
  "$@"
