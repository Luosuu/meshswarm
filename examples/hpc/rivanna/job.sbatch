#!/bin/bash
#SBATCH --job-name=meshhpc
#SBATCH --account=bii_dsc_community
#SBATCH --partition=standard
#SBATCH --cpus-per-task=18
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/%u/meshswarm/logs/%x-%j.out
# Usage: sbatch job.sbatch [arguments for run.py], e.g. --kernel amr --selftest
# Each swarm gets its own 8 cores (two swarms = 16); the last two are left for the driver.
set -euo pipefail
ROOT=/scratch/$USER/meshswarm
export MESH_CORES=8
source $ROOT/repo/examples/hpc/rivanna/env.sh
trap 'rm -rf "$MESH_LOCAL"' EXIT
echo "command network: $MESH_RUN_NETWORK"
cd $ROOT/repo
python3 --version
python3 examples/hpc/run.py --cores $MESH_CORES --api-key-file $ROOT/.tinker_key "$@"
