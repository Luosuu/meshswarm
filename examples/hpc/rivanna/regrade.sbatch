#!/bin/bash
#SBATCH --job-name=regrade
#SBATCH --account=bii_dsc_community
#SBATCH --partition=standard
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/meshswarm/logs/%x-%j.out
# Grade saved workspaces again on a node reserved for this job, so that nothing else disturbs the
# timings and every workspace is measured on the same cores against one shared baseline.
# Usage: sbatch [--exclusive] regrade.sbatch KERNEL WORKSPACE [WORKSPACE ...]
# With --exclusive the whole node is reserved, which gives the cleanest timings but can wait many hours in the queue.
set -euo pipefail
ROOT=/scratch/$USER/meshswarm
export MESH_CORES=8
source $ROOT/repo/examples/hpc/rivanna/env.sh
trap 'rm -rf "$MESH_LOCAL"' EXIT
hostname; lscpu | grep -E "Model name|Thread|Core|Socket"
cd $ROOT/repo
KERNEL=$1; shift
python3 examples/hpc/regrade.py --kernel $KERNEL --strip-fast-math "$@"
