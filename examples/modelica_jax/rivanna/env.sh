# Source this inside a Slurm GPU job on Rivanna after install.sh.
# Configures MESH_RUN_WRAP so every agent command runs inside an isolated `--nv --containall --no-home`
# Apptainer container seeing only its workspace `{cwd}` (read-write) plus read-only Python/.venv & Modelica repos.
source /etc/profile
ROOT=/scratch/$USER/meshswarm/modelica_jax

module load apptainer/1.4.5 cuda/12.8.0 miniforge/26.9.4

export MODELICA_JAX_ROOT="$ROOT"
export MESH_LOCAL="${MESH_LOCAL:-${TMPDIR:-/tmp}/$USER-mjax-${SLURM_JOB_ID:-$$}}"
mkdir -p "$MESH_LOCAL"
if [ ! -f "$MESH_LOCAL/openmodelica.sif" ]; then
  cp "$ROOT/openmodelica.sif" "$MESH_LOCAL/openmodelica.sif"
fi

export OMC_SIF="$MESH_LOCAL/openmodelica.sif"
export MESH_RUN_SCRATCH="$MESH_LOCAL"
export MSL_DIR="$ROOT/repos/ModelicaStandardLibrary"
export TRANSFORM_LIB="$ROOT/repos/TRANSFORM-Library/TRANSFORM/package.mo"
export TRANSFORM_TESTS="$ROOT/repos/TRANSFORM-UnitTests"
export MODELICA_REPOS="$ROOT/repos"
export OPENMODELICALIBRARY="$ROOT/repos/ModelicaStandardLibrary:$ROOT/repos/TRANSFORM-Library:$ROOT/repos/TRANSFORM-UnitTests"
export JAX_ENABLE_X64=1
export GRADSOLVE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH="$ROOT/.venv/bin:$PATH"

CONTAINER_PATH="$ROOT/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
BASE=(
  apptainer exec --nv --containall --no-home
  --pwd "{cwd}"
  --bind "{cwd}:{cwd}"
  --bind "$ROOT/.venv:$ROOT/.venv:ro"
  --bind "$ROOT/uv-python:$ROOT/uv-python:ro"
  --bind "$ROOT/repos:$ROOT/repos:ro"
  --env "PATH=$CONTAINER_PATH"
  --env "OPENMODELICALIBRARY=$OPENMODELICALIBRARY"
  --env "TRANSFORM_LIB=$TRANSFORM_LIB"
  --env "JAX_ENABLE_X64=1"
  --env "GRADSOLVE_X64=1"
  --env "XLA_PYTHON_CLIENT_PREALLOCATE=false"
)
if apptainer exec --containall --no-home --net --network none "$OMC_SIF" true 2>/dev/null; then
  BASE+=(--net --network none)
  export MESH_RUN_NETWORK=none
else
  export MESH_RUN_NETWORK=host
fi
BASE+=("$OMC_SIF" bash -c)
export MESH_RUN_WRAP=$("$ROOT/.venv/bin/python" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${BASE[@]}")
