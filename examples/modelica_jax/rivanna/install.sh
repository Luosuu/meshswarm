#!/bin/bash
# One-time setup on UVA Rivanna for the Modelica-to-JAX differentiable simulation & control pipeline.
# Everything is installed under /scratch/$USER/meshswarm/modelica_jax — zero files written to $HOME.
source /etc/profile
set -euo pipefail

ROOT=/scratch/$USER/meshswarm/modelica_jax
mkdir -p "$ROOT"/{apptainer-cache,apptainer-tmp,uv-cache,uv-python,repos,logs,runs}

module load apptainer/1.4.5 cuda/12.8.0 miniforge/26.9.4
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

export APPTAINER_CACHEDIR="$ROOT/apptainer-cache"
export APPTAINER_TMPDIR="$ROOT/apptainer-tmp"
export UV_CACHE_DIR="$ROOT/uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/uv-python"
export PIP_CACHE_DIR="$ROOT/uv-cache"

echo "=== [1/4] Installing relocatable Python 3.12 & JAX CUDA12 + Rumoca + GRADSOLVE ==="
uv python install 3.12
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  uv venv --python-preference only-managed --python 3.12 "$ROOT/.venv"
fi
uv pip install --python "$ROOT/.venv/bin/python" \
  "jax[cuda12]" \
  "rumoca>=0.10.4" \
  "gradsolve[diffrax]>=0.2.1" \
  "optax" \
  "equinox" \
  "scipy" \
  "numpy"

echo "=== [2/4] Pulling OpenModelica v1.27.1-minimal Apptainer image ==="
if [ ! -f "$ROOT/openmodelica.sif" ]; then
  apptainer pull "$ROOT/openmodelica.sif" docker://openmodelica/openmodelica:v1.27.1-minimal
fi

echo "=== [3/4] Cloning Modelica Standard Library v4.0.0, ORNL TRANSFORM-Library & TRANSFORM-UnitTests ==="
if [ ! -d "$ROOT/repos/ModelicaStandardLibrary/.git" ]; then
  git clone --depth 1 --branch v4.0.0 https://github.com/modelica/ModelicaStandardLibrary.git "$ROOT/repos/ModelicaStandardLibrary"
fi
if [ ! -d "$ROOT/repos/TRANSFORM-Library/.git" ]; then
  git clone --depth 1 https://github.com/WyrdLab/TRANSFORM-Library.git "$ROOT/repos/TRANSFORM-Library"
fi
if [ ! -d "$ROOT/repos/TRANSFORM-UnitTests/.git" ]; then
  git clone --depth 1 https://github.com/WyrdLab/TRANSFORM-UnitTests.git "$ROOT/repos/TRANSFORM-UnitTests"
fi

echo "=== [4/4] Verifying unified isolated container (omc + Python JAX/Rumoca/gradsolve) ==="
apptainer exec --containall --no-home \
  --bind "$ROOT/.venv:$ROOT/.venv:ro" \
  --bind "$ROOT/uv-python:$ROOT/uv-python:ro" \
  --bind "$ROOT/repos:$ROOT/repos:ro" \
  --env PATH="$ROOT/.venv/bin:/usr/bin:/bin" \
  --env JAX_PLATFORMS=cpu \
  "$ROOT/openmodelica.sif" \
  bash -c 'omc --version && python -c "import jax, rumoca, gradsolve; print(\"Unified container OK:\", jax.__version__, rumoca.version())"'
echo "INSTALL_MODELICA_JAX_OK"
