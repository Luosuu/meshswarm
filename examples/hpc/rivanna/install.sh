#!/bin/bash
# One-time setup on Rivanna for the performance tasks: a container image with GCC, OpenMP, make and
# binutils (gprof). Downloads docker://gcc:14 from Docker Hub (about 500 MB) into
# /scratch/$USER/meshswarm. Nothing is written to the home directory.
set -euo pipefail
ROOT=/scratch/$USER/meshswarm
mkdir -p $ROOT/apptainer-cache $ROOT/logs $ROOT/runs
module load apptainer
export APPTAINER_CACHEDIR=$ROOT/apptainer-cache
cd $ROOT
if [ ! -f gcc.sif ]; then
  apptainer pull gcc.sif docker://gcc:14
fi
apptainer exec --containall --no-home gcc.sif gcc --version | head -1
echo INSTALL_OK
