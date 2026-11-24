#!/bin/bash
#SBATCH --job-name=mjax-regrade-all
#SBATCH --account=bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/meshswarm/modelica_jax/logs/%x-%j.out
# Re-grades an entire multi-tier Phase 3 campaign sequentially on one NVIDIA A100 GPU.
# Usage: sbatch regrade_campaign.sbatch [TAG]

TAG="${1:-r1}"
REPO="/scratch/$USER/meshswarm/repo"
RUNS="/scratch/$USER/meshswarm/modelica_jax/runs"

source "$REPO/examples/modelica_jax/rivanna/env.sh"
set -euo pipefail
trap 'rm -rf "$MESH_LOCAL"' EXIT

hostname
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
cd "$REPO"

echo "=== REGRADE tier1_blanket_dae ==="
python examples/modelica_jax/regrade.py --tier tier1_blanket_dae \
  "$RUNS/${TAG}-tier1-g1"/gen-01/*/ \
  "$RUNS/${TAG}-tier1-g2"/gen-01/*/ \
  "$RUNS/${TAG}-tier1-g3"/gen-01/*/

echo "=== REGRADE tier2_transform_hx ==="
python examples/modelica_jax/regrade.py --tier tier2_transform_hx \
  "$RUNS/${TAG}-tier2-g1"/gen-01/*/ \
  "$RUNS/${TAG}-tier2-g2"/gen-01/*/

echo "=== REGRADE tier3_hdt_column ==="
python examples/modelica_jax/regrade.py --tier tier3_hdt_column \
  "$RUNS/${TAG}-tier3-g1"/gen-01/*/ \
  "$RUNS/${TAG}-tier3-g2"/gen-01/*/

echo "REGRADE_CAMPAIGN_DONE"
