#!/bin/bash
# Submit the Phase 3 multi-generation & cross-task transfer DAG on UVA Rivanna:
#
#                   +--> tier1-g2 (+ctrl) --> tier1-g3 (+ctrl)
#   tier1-g1 (2) ---+--> tier2-g1 (+ctrl) --> tier2-g2 (+ctrl)   [transfer to TRANSFORM HX]
#                   +--> tier3-g1 (+ctrl) --> tier3-g2 (+ctrl)   [transfer to 16-stage HDT column]
#
# Usage on Rivanna: bash /scratch/$USER/meshswarm/repo/examples/modelica_jax/rivanna/submit_phase3.sh [TAG]
source /etc/profile
set -euo pipefail

TAG="${1:-p3}"
REPO="/scratch/$USER/meshswarm/repo"
RUNS="/scratch/$USER/meshswarm/modelica_jax/runs"
SBATCH_SCRIPT="$REPO/examples/modelica_jax/rivanna/job.sbatch"
COMMON=(--generations 1 --population 2 --parallel --max-cost 3.0 --timeout 3600)

mkdir -p "$RUNS" "/scratch/$USER/meshswarm/modelica_jax/logs"

J_T1G1=$(sbatch --parsable --job-name="mjax-${TAG}-t1g1" --time=02:30:00 "$SBATCH_SCRIPT" \
  --tier tier1_blanket_dae "${COMMON[@]}" \
  --out "$RUNS/${TAG}-tier1-g1")

J_T1G2=$(sbatch --parsable --dependency="afterok:${J_T1G1}" --job-name="mjax-${TAG}-t1g2" --time=02:30:00 "$SBATCH_SCRIPT" \
  --tier tier1_blanket_dae "${COMMON[@]}" --controls 1 \
  --lessons "$RUNS/${TAG}-tier1-g1/lessons.md" \
  --out "$RUNS/${TAG}-tier1-g2")

J_T1G3=$(sbatch --parsable --dependency="afterok:${J_T1G2}" --job-name="mjax-${TAG}-t1g3" --time=02:30:00 "$SBATCH_SCRIPT" \
  --tier tier1_blanket_dae "${COMMON[@]}" --controls 1 \
  --lessons "$RUNS/${TAG}-tier1-g2/lessons.md,$RUNS/${TAG}-tier1-g1/lessons.md" \
  --out "$RUNS/${TAG}-tier1-g3")

J_T2G1=$(sbatch --parsable --dependency="afterok:${J_T1G1}" --job-name="mjax-${TAG}-t2g1" --time=02:30:00 "$SBATCH_SCRIPT" \
  --tier tier2_transform_hx "${COMMON[@]}" --controls 1 \
  --lessons "$RUNS/${TAG}-tier1-g1/lessons.md" \
  --out "$RUNS/${TAG}-tier2-g1")

J_T2G2=$(sbatch --parsable --dependency="afterok:${J_T2G1}" --job-name="mjax-${TAG}-t2g2" --time=02:30:00 "$SBATCH_SCRIPT" \
  --tier tier2_transform_hx "${COMMON[@]}" --controls 1 \
  --lessons "$RUNS/${TAG}-tier2-g1/lessons.md,$RUNS/${TAG}-tier1-g1/lessons.md" \
  --out "$RUNS/${TAG}-tier2-g2")

J_T3G1=$(sbatch --parsable --dependency="afterok:${J_T1G1}" --job-name="mjax-${TAG}-t3g1" --time=02:30:00 "$SBATCH_SCRIPT" \
  --tier tier3_hdt_column "${COMMON[@]}" --controls 1 \
  --lessons "$RUNS/${TAG}-tier1-g1/lessons.md" \
  --out "$RUNS/${TAG}-tier3-g1")

J_T3G2=$(sbatch --parsable --dependency="afterok:${J_T3G1}" --job-name="mjax-${TAG}-t3g2" --time=02:30:00 "$SBATCH_SCRIPT" \
  --tier tier3_hdt_column "${COMMON[@]}" --controls 1 \
  --lessons "$RUNS/${TAG}-tier3-g1/lessons.md,$RUNS/${TAG}-tier1-g1/lessons.md" \
  --out "$RUNS/${TAG}-tier3-g2")

cat <<EOF
Submitted Phase 3 DAG (tag=${TAG}, 20 swarms across 7 jobs):
  Wave 1 (seed):
    tier1-g1: ${J_T1G1}
  Wave 2 (after ${J_T1G1}):
    tier1-g2: ${J_T1G2}
    tier2-g1: ${J_T2G1} (transfer -> TRANSFORM HX)
    tier3-g1: ${J_T3G1} (transfer -> 16-stage HDT column)
  Wave 3:
    tier1-g3: ${J_T1G3} (after ${J_T1G2})
    tier2-g2: ${J_T2G2} (after ${J_T2G1})
    tier3-g2: ${J_T3G2} (after ${J_T3G1})
EOF
