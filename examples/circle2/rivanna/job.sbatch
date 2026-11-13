#!/bin/bash
#SBATCH --job-name=circle2
#SBATCH --account=bii_dsc_community
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/%u/meshswarm/logs/%x-%j.out
# Usage: sbatch job.sbatch [arguments for run.py], e.g. --selftest or --generations 3 --population 2 --control
set -euo pipefail
ROOT=/scratch/$USER/meshswarm
source $ROOT/repo/examples/circle2/rivanna/env.sh
trap 'rm -rf "$MESH_LOCAL"' EXIT
echo "lean network: $MESH_LEAN_NETWORK"
cd $ROOT/repo
python3 --version
python3 examples/circle2/run.py --api-key-file $ROOT/.tinker_key "$@"
