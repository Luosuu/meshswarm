# Source this inside a Slurm job on Rivanna after install.sh. It copies Lean and Mathlib's compiled
# files to the node's local disk (the shared scratch filesystem can make a Mathlib import take from
# one to more than twenty-five minutes), then sets MESH_LEAN_CMD (a JSON list) so that Lean runs in a
# container that sees only those files (read-only) and a scratch folder.
ROOT=/scratch/$USER/meshswarm
TC=leanprover--lean4---v4.33.1
module load apptainer miniforge
export MESH_LOCAL=${MESH_LOCAL:-${TMPDIR:-/tmp}/$USER-meshlean-${SLURM_JOB_ID:-$$}}
mkdir -p $MESH_LOCAL/tc/bin $MESH_LOCAL/scratch $MESH_LOCAL/packages
df -h $MESH_LOCAL | tail -1
SECONDS=0
cp $ROOT/elan/toolchains/$TC/bin/lean $MESH_LOCAL/tc/bin/
rsync -a $ROOT/elan/toolchains/$TC/lib $MESH_LOCAL/tc/
for lib in $ROOT/leanproj/.lake/packages/*/.lake/build/lib; do
  pkg=$(basename $(dirname $(dirname $(dirname $lib))))
  mkdir -p $MESH_LOCAL/packages/$pkg/.lake/build
  rsync -a $lib $MESH_LOCAL/packages/$pkg/.lake/build/
done
cp $ROOT/lean.sif $MESH_LOCAL/lean.sif
echo "copied Lean and Mathlib to $MESH_LOCAL in ${SECONDS}s ($(du -sh $MESH_LOCAL | cut -f1))"
LEAN_PATH_VALUE=$(ls -d $MESH_LOCAL/packages/*/.lake/build/lib/lean | paste -sd:)
export MESH_LEAN_SCRATCH=$MESH_LOCAL/scratch
BASE="apptainer exec --containall --no-home --pwd $MESH_LOCAL/scratch --bind $MESH_LOCAL/tc:$MESH_LOCAL/tc:ro --bind $MESH_LOCAL/packages:$MESH_LOCAL/packages:ro --bind $MESH_LOCAL/scratch:$MESH_LOCAL/scratch --env LEAN_PATH=$LEAN_PATH_VALUE"
# Cut the network too when this cluster lets an unprivileged user do so.
if $BASE --net --network none $MESH_LOCAL/lean.sif true 2>/dev/null; then
  BASE="$BASE --net --network none"
  export MESH_LEAN_NETWORK=none
else
  export MESH_LEAN_NETWORK=host
fi
export MESH_LEAN_CMD=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1].split() + sys.argv[2:]))' "$BASE" $MESH_LOCAL/lean.sif $MESH_LOCAL/tc/bin/lean)
