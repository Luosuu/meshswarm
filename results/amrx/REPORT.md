# Stricter AMR: three generations of swarms

Three generations of two swarms each were asked to make an adaptive-mesh-refinement stencil faster
on 8 cores. On a like-for-like re-grade the best swarm of every generation reached about 7.9×, with
or without lessons. What changed is the weaker swarm, which rose from 6.6× to 7.8× in generation 3,
and the cost, which fell from $4.90 to $1.39 per pair.

## Setup

- **Task:** the AMR kernel of the Parallel Research Kernels (`amr.c`, serial C, about 570 lines),
  patched so the benchmark's shortcuts no longer work (the `amrx` kernel in
  `examples/hpc/kernels.py`). The swarm gets the source and a sandboxed compile-and-run tool.
- **Why stricter:** in the earlier AMR run every swarm used the fact that the stencil weights sum
  to zero and the field is a plane, which lets the sweep be skipped. The patch adds a non-linear
  bump to the initial field and feeds a little of each sweep's output back into the input, so
  every sweep works on new data. The program's closed-form check is replaced by printing ten norms.
- **Reference:** the original serial program built with `-O3`.
- **Correctness:** the ten printed norms must match the original's to a relative 1e-9 on two
  held-out settings, `150 3000 600 3 12 6 4` and `200 4000 500 2 8 4 6`. Fast-math flags are not
  graded.
- **Swarm:** 3 agents, Inkling-2-Small with a 256K context, $15 cap and 2.5 hours per swarm.
- **Generations:** an outside reviewer reads both swarms of a generation and writes the lessons the
  next generation starts with. Generation 1 started with no lessons.
- **Machine:** UVA Rivanna, commands run in a container that sees only the swarm's workspace.

## Result of record

All six workspaces were timed one after another in a single job on one node (AMD EPYC 9454), five
runs each with the median counting, against one shared reference (24.90 s and 16.19 s). The raw
output is in `regrade-one-node.txt`.

| Generation | Swarm 1 | Swarm 2 | Average | Score (each) | Model cost (pair) | Time (each) |
|---|---|---|---|---|---|---|
| 1 | 7.90× | 6.66× | 7.28× | 3, 3 | $4.90 | 79, 64 min |
| 2 | 6.59× | 7.92× | 7.26× | 3, 3 | $3.83 | 81, 122 min |
| 3 | 7.81× | 7.91× | 7.86× | 3, 3 | $1.39 | 31, 59 min |

Total model cost was about $10.12. All six swarms marked the goal complete on their own.

## How to read the numbers

- **Ignore the scores printed during the runs** (`scores.json`: 6.31×, 7.55×, 4.98×, 9.47×, 6.81×,
  8.52×). The swarms of one generation were timed on different cores under different load, and
  generation 2's spread of 4.98× to 9.47× re-grades to 6.59× and 7.92×.
- **There is a ceiling near 7.9×.** Four of six swarms are within 0.1× of it. The swarms and the
  reviewer attribute it to memory bandwidth: the sweeps read and write whole grids.
- **The second reference varied.** It ranged from 15.23 s to 19.01 s over three runs, so a
  difference of a few percent between two swarms means nothing.
- **Two swarms per generation is a small sample.** One unlucky swarm in generation 3 would have
  erased the trend.

## What the swarms did

- **Found by all six:** OpenMP on the background sweep, the refinement sweeps, interpolation,
  initialisation and the final norms, with static scheduling.
- **Fewer passes over memory:** fusing the two update passes after each sweep into one, and in
  one swarm fusing them into the sweep itself with a second buffer.
- **Inner loop:** the star stencil unrolled with the sum kept in a local variable, same order of
  additions.
- **Flags:** `-O3 -fopenmp` with `-march=native` or `-march=x86-64-v3`.

Each swarm's own account is in `gen-N/run-M/FINAL.md` (where it wrote one), its source changes in
`changes.diff`, and the run statistics in `swarm-report.md`.

## How they collaborated

- **Roles:** one writer of the shared source, one verifier, one benchmarker, in every swarm. From
  generation 2 the lessons said to settle this at kickoff; generation 3 still re-negotiated the
  split three to five times before it held.
- **Channel:** direct messages, 46 to 88 per swarm, with one agent in 68–81% of them. The task
  board carried 1 to 5 tasks per swarm, and was used as the lock on the shared source file.
- **Sub-agents:** generations 1 and 2 delegated one or two bounded experiments each; generation 3
  delegated nothing.
- **What went wrong:** in generation 2 two agents both set out to rewrite the same file and a
  third had to arbitrate; one swarm timed through a wrapper script that added about 0.3 s per call
  and chased a start-up cost that did not exist; agents were idle 50–81% of the time.

## Caveats

- **The lessons did not raise the best result.** Generation 1 reached the ceiling with no
  lessons. The evidence here is for reliability and cost, not for a higher peak.
- **The lessons drifted towards the task.** They are mostly about organisation and timing, but by
  generation 3 they also name fused update passes and unrolled inner loops.
- **The tolerance is close to the noise for large grids.** One swarm measured a difference of
  about 1e-9 from parallel summation at 8000×8000. The held-out sizes stayed under 1e-10.

## Files

| Path | Contents |
|---|---|
| `regrade-one-node.txt` | Single-node re-grade of all six workspaces (job 23016325) |
| `gen-N/scores.json` | In-job scores and collaboration counts for both swarms |
| `gen-N/lessons_in.md`, `lessons_out.md` | Lessons the generation started with (empty for generation 1) and the reviewer's rewrite |
| `gen-N/summary.md` | The generation's summary table |
| `gen-N/run-M/` | `build.sh`, `changes.diff` against the patched original, the swarm's own notes, `swarm-report.md` |

The full workspaces and event logs are in the git-ignored `runs/amrx-g1` to `runs/amrx-g3`.
