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

echo "=== [1/4] Creating Python 3.12 virtualenv & installing JAX CUDA12 + Rumoca + GRADSOLVE ==="
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "$ROOT/.venv"
  else
    CONDA_PKGS_DIRS="$ROOT/conda-pkgs" conda create -y -p "$ROOT/.venv" python=3.12 pip
  fi
fi
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$ROOT/.venv/bin/python" \
    "jax[cuda12]" \
    "rumoca>=0.10.4" \
    "gradsolve[diffrax]>=0.2.1" \
    "optax" \
    "equinox" \
    "scipy" \
    "numpy"
else
  "$ROOT/.venv/bin/pip" install \
    "jax[cuda12]" \
    "rumoca>=0.10.4" \
    "gradsolve[diffrax]>=0.2.1" \
    "optax" \
    "equinox" \
    "scipy" \
    "numpy"
fi

"$ROOT/.venv/bin/python" -c "import jax, rumoca, gradsolve, optax; print('Python stack OK:', 'jax', jax.__version__, 'rumoca', rumoca.version())"

echo "=== [2/4] Pulling OpenModelica v1.27.1-minimal Apptainer image ==="
if [ ! -f "$ROOT/openmodelica.sif" ]; then
  apptainer pull "$ROOT/openmodelica.sif" docker://openmodelica/openmodelica:v1.27.1-minimal
fi
apptainer exec --containall --no-home "$ROOT/openmodelica.sif" omc --version

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

echo "=== [4/4] Verifying MSL + TRANSFORM packages & OpenModelica image ==="
ls -lh "$ROOT/repos/ModelicaStandardLibrary/Modelica/package.mo" "$ROOT/repos/TRANSFORM-Library/TRANSFORM/package.mo" "$ROOT/openmodelica.sif"
echo "INSTALL_MODELICA_JAX_OK"
