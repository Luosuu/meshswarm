#!/bin/bash
#SBATCH --job-name=tmap8-install
#SBATCH --account=bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/meshswarm/tmap8/logs/install-%j.out
#SBATCH --error=/scratch/%u/meshswarm/tmap8/logs/install-%j.err
# One-time setup on UVA Rivanna for the DOE INL TMAP8 ARC Fusion Whole-Plant benchmark.
# Pulls INL's official idaholab/tmap8 container (`tmap8-opt`), extracts official tests,
# and verifies unified GPU JAX + tmap8-opt execution inside `--nv --containall --no-home`.
source /etc/profile
set -euo pipefail

TMAP8_TAG="${TMAP8_TAG:-2026.05.10-0cdef71}"
ROOT=/scratch/$USER/meshswarm/tmap8
MJAX_ROOT=/scratch/$USER/meshswarm/modelica_jax
mkdir -p "$ROOT"/{apptainer-cache,apptainer-tmp,inputs,logs,runs}

module load apptainer/1.4.5 cuda/12.8.0 miniforge/26.9.4
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

export APPTAINER_CACHEDIR="$ROOT/apptainer-cache"
export APPTAINER_TMPDIR="${TMPDIR:-/tmp}/$USER-apptainer-tmp-$$"
mkdir -p "$APPTAINER_TMPDIR"

echo "=== [1/4] Ensuring Python 3.12 + JAX CUDA12 venv exists ==="
if [ ! -x "$MJAX_ROOT/.venv/bin/python" ]; then
  bash "$(dirname "$0")/../../modelica_jax/rivanna/install.sh"
fi

echo "=== [2/4] Pulling official DOE INL idaholab/tmap8:${TMAP8_TAG} Apptainer image ==="
if [ ! -f "$ROOT/tmap8.sif" ]; then
  LOCAL_SIF="${TMPDIR:-/tmp}/$USER-tmap8-$$.sif"
  apptainer pull "$LOCAL_SIF" "docker://idaholab/tmap8:${TMAP8_TAG}"
  mv -f "$LOCAL_SIF" "$ROOT/tmap8.sif"
fi
rm -rf "$APPTAINER_TMPDIR"

echo "=== [3/4] Extracting official INL TMAP8 fuel_cycle_Meschini & fuel_cycle_Abdou inputs ==="
WORK=$(mktemp -d "${TMPDIR:-/tmp}/$USER-tmap8-init-XXXXXX")
trap 'rm -rf "$WORK"' EXIT

apptainer exec --containall --no-home \
  --pwd "$WORK" \
  --bind "$WORK:$WORK" \
  "$ROOT/tmap8.sif" \
  bash -lc '
    TBIN=$(command -v tmap8-opt || find / -name tmap8-opt 2>/dev/null | head -n 1)
    echo "TMAP8_BIN=$TBIN"
    "$TBIN" --version || true
    "$TBIN" --copy-inputs tests
    ls -la tmap8/tests/fuel_cycle_Meschini
  ' | tee "$ROOT/tmap8_info.txt"

rm -rf "$ROOT/inputs/fuel_cycle_Meschini" "$ROOT/inputs/fuel_cycle_Abdou"
cp -r "$WORK/tmap8/tests/fuel_cycle_Meschini" "$ROOT/inputs/"
cp -r "$WORK/tmap8/tests/fuel_cycle_Abdou" "$ROOT/inputs/"
sha256sum "$ROOT/inputs/fuel_cycle_Meschini/fuel_cycle.i" | tee "$ROOT/inputs/fuel_cycle_Meschini.sha256"

echo "=== [4/4] Verifying unified GPU JAX + INL tmap8-opt container & timing pulsed runs ==="
TBIN_DIR=$(grep '^TMAP8_BIN=' "$ROOT/tmap8_info.txt" | head -n 1 | cut -d= -f2 | xargs dirname)
echo "$TBIN_DIR" > "$ROOT/tmap8_bin_dir.txt"

apptainer exec --nv --containall --no-home \
  --pwd "$WORK/tmap8/tests/fuel_cycle_Meschini" \
  --bind "$WORK:$WORK" \
  --bind "$MJAX_ROOT/.venv:$MJAX_ROOT/.venv:ro" \
  --bind "$MJAX_ROOT/uv-python:$MJAX_ROOT/uv-python:ro" \
  --env "PATH=$MJAX_ROOT/.venv/bin:$TBIN_DIR:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "JAX_ENABLE_X64=1" \
  "$ROOT/tmap8.sif" \
  bash -c '
    echo "--- Python JAX GPU check inside tmap8.sif ---"
    python -c "import jax; print(\"JAX_DEVS:\", jax.devices())"
    echo "--- Timing INL baseline (20-day pulse cutoff, 3-year total) ---"
    /usr/bin/time -p tmap8-opt -i fuel_cycle.i
    echo "--- Timing full 90-day pulsed run (no 20-day approximation) ---"
    /usr/bin/time -p tmap8-opt -i fuel_cycle.i \
      initial_refinement_time="\${units 7776000 s}" \
      simulation_time="\${units 7776000 s}" \
      Outputs/file_base=pulsed_90d_out
    echo "--- Timing full 1-year pulsed run (no 20-day approximation) ---"
    /usr/bin/time -p tmap8-opt -i fuel_cycle.i \
      initial_refinement_time="\${units 31557600 s}" \
      simulation_time="\${units 31557600 s}" \
      Outputs/file_base=pulsed_1y_out
    wc -l fuel_cycle_out.csv pulsed_90d_out.csv pulsed_1y_out.csv
  '

echo "INSTALL_TMAP8_OK"
