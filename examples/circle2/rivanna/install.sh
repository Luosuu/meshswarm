#!/bin/bash
# One-time setup on Rivanna: Lean 4.33.1 + Mathlib (prebuilt cache) and a container image,
# all under /scratch/$USER/meshswarm. Nothing is written to the home directory.
# Downloads: the elan installer (lean-lang.org), Lean and Mathlib (GitHub, Mathlib's cache server),
# and buildpack-deps:noble-scm (Docker Hub: Ubuntu 24.04 with curl and git). About 9 GB.
# Lean's bundled compiler needs a newer glibc than Rivanna's hosts have, so lake runs in the container.
set -euo pipefail
ROOT=/scratch/$USER/meshswarm
export ELAN_HOME=$ROOT/elan
mkdir -p $ROOT/leanproj $ROOT/scratch $ROOT/mathlib-cache $ROOT/apptainer-cache
cd $ROOT

module load apptainer
export APPTAINER_CACHEDIR=$ROOT/apptainer-cache
if [ ! -f lean.sif ]; then
  apptainer pull lean.sif docker://buildpack-deps:noble-scm
fi

if [ ! -x $ELAN_HOME/bin/elan ]; then
  curl -sSf https://elan.lean-lang.org/elan-init.sh | sh -s -- -y --no-modify-path --default-toolchain none
fi

cd $ROOT/leanproj
cat > lakefile.toml <<'L'
name = "leanproj"
defaultTargets = ["Leanproj"]

[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "v4.33.1"

[[lean_lib]]
name = "Leanproj"
L
echo 'leanprover/lean4:v4.33.1' > lean-toolchain
echo 'import Mathlib' > Leanproj.lean

IN="apptainer exec --containall --no-home --bind $ROOT:$ROOT --pwd $ROOT/leanproj --env ELAN_HOME=$ELAN_HOME --env MATHLIB_CACHE_DIR=$ROOT/mathlib-cache --env PATH=$ELAN_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin $ROOT/lean.sif"
$IN lake update
$IN lake exe cache get
echo 'example : (2 : ℝ) + 2 = 4 := by norm_num' | cat Leanproj.lean - > $ROOT/scratch/smoke.lean
$IN lake env lean $ROOT/scratch/smoke.lean
echo INSTALL_OK
