# Source this inside a Slurm GPU job on Rivanna after install.sh.
# Configures MESH_RUN_WRAP so every agent command runs inside an isolated `--nv --containall --no-home`
# Apptainer container (`idaholab/tmap8`) seeing only its workspace `{cwd}` (read-write) plus
# read-only Python/.venv (CUDA12 JAX) and read-only official INL TMAP8 inputs.
source /etc/profile
ROOT=/scratch/$USER/meshswarm/tmap8
MJAX_ROOT=/scratch/$USER/meshswarm/modelica_jax

module load apptainer/1.4.5 cuda/12.8.0 miniforge/26.9.4

export TMAP8_ROOT="$ROOT"
export MESH_LOCAL="${MESH_LOCAL:-${TMPDIR:-/tmp}/$USER-tmap8-${SLURM_JOB_ID:-$$}}"
mkdir -p "$MESH_LOCAL"
if [ ! -f "$MESH_LOCAL/tmap8.sif" ]; then
  cp "$ROOT/tmap8.sif" "$MESH_LOCAL/tmap8.sif"
fi

export TMAP8_SIF="$MESH_LOCAL/tmap8.sif"
export TMAP8_INPUTS="$ROOT/inputs"
export TMAP8_BIN_DIR=$(cat "$ROOT/tmap8_bin_dir.txt")
export MESH_RUN_SCRATCH="$MESH_LOCAL"
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ -f /etc/pki/tls/cert.pem ]; then
  export SSL_CERT_FILE=/etc/pki/tls/cert.pem
fi
export PATH="$MJAX_ROOT/.venv/bin:$PATH"

CONTAINER_PATH="$MJAX_ROOT/.venv/bin:$TMAP8_BIN_DIR:/opt/openmpi/bin:/opt/rh/gcc-toolset-13/root/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
CONTAINER_LD="/opt/rh/gcc-toolset-13/root/usr/lib64:/opt/openmpi/lib:/opt/petsc/lib:/opt/libmesh/lib:/opt/wasp/lib:/.singularity.d/libs"
BASE=(
  apptainer exec --nv --containall --no-home
  --pwd "{cwd}"
  --bind "{cwd}:{cwd}"
  --bind "$MJAX_ROOT/.venv:$MJAX_ROOT/.venv:ro"
  --bind "$MJAX_ROOT/uv-python:$MJAX_ROOT/uv-python:ro"
  --bind "$ROOT/inputs:$ROOT/inputs:ro"
  --env "PATH=$CONTAINER_PATH"
  --env "LD_LIBRARY_PATH=$CONTAINER_LD"
  --env "TMAP8_INPUTS=$ROOT/inputs"
  --env "JAX_ENABLE_X64=1"
  --env "XLA_PYTHON_CLIENT_PREALLOCATE=false"
)
if apptainer exec --containall --no-home --net --network none "$TMAP8_SIF" true 2>/dev/null; then
  BASE+=(--net --network none)
  export MESH_RUN_NETWORK=none
else
  export MESH_RUN_NETWORK=host
fi
BASE+=("$TMAP8_SIF" bash -c)
export MESH_RUN_WRAP=$("$MJAX_ROOT/.venv/bin/python" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${BASE[@]}")
