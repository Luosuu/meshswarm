# Source this inside a Slurm job on Rivanna after install.sh. Sets MESH_RUN_WRAP (a JSON list) so that
# every command an agent runs happens inside a container that sees only that swarm's workspace:
# no home directory, no other files, and no network where the cluster allows cutting it.
ROOT=/scratch/$USER/meshswarm
module load apptainer miniforge
export MESH_LOCAL=${MESH_LOCAL:-${TMPDIR:-/tmp}/$USER-meshhpc-${SLURM_JOB_ID:-$$}}
mkdir -p $MESH_LOCAL
cp $ROOT/gcc.sif $MESH_LOCAL/gcc.sif
export MESH_RUN_SCRATCH=$MESH_LOCAL
BASE="apptainer exec --containall --no-home --pwd {cwd} --bind {cwd}:{cwd} --env OMP_NUM_THREADS=${MESH_CORES:-8} --env OMP_PROC_BIND=true"
if $BASE --net --network none $MESH_LOCAL/gcc.sif true 2>/dev/null; then
  BASE="$BASE --net --network none"
  export MESH_RUN_NETWORK=none
else
  export MESH_RUN_NETWORK=host
fi
export MESH_RUN_WRAP=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1].split() + sys.argv[2:]))' "$BASE" $MESH_LOCAL/gcc.sif bash -c)
