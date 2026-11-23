# Source this inside a Slurm GPU job on Rivanna after install.sh.
source /etc/profile
ROOT=/scratch/$USER/meshswarm/modelica_jax

module load apptainer/1.4.5 cuda/12.8.0 miniforge/26.9.4

export MODELICA_JAX_ROOT="$ROOT"
export OMC_SIF="$ROOT/openmodelica.sif"
export MSL_DIR="$ROOT/repos/ModelicaStandardLibrary"
export TRANSFORM_LIB="$ROOT/repos/TRANSFORM-Library/TRANSFORM/package.mo"
export TRANSFORM_TESTS="$ROOT/repos/TRANSFORM-UnitTests"
export MODELICA_REPOS="$ROOT/repos"
export OPENMODELICALIBRARY="$ROOT/repos/ModelicaStandardLibrary:$ROOT/repos/TRANSFORM-Library:$ROOT/repos/TRANSFORM-UnitTests"
export JAX_ENABLE_X64=1
export GRADSOLVE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH="$ROOT/.venv/bin:$PATH"
