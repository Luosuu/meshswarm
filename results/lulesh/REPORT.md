# LULESH: three generations of swarms

Three generations of two swarms each were asked to make LULESH 2.0 faster on 8 cores. On a
like-for-like re-grade the average speedup rose each generation, 1.60× then 1.69× then 1.88×, and
both the weaker and the stronger swarm of generation 3 beat every earlier swarm.

## Setup

- **Task:** LULESH 2.0 (LLNL, commit 3e01c40b3281, about 7,300 lines of C++), already parallel with
  OpenMP. The swarm gets the source and a sandboxed compile-and-run tool.
- **Reference:** the original source built with `-O3 -fopenmp -DUSE_MPI=0`, on the same 8 cores.
- **Correctness:** same iteration count and final origin energy as the reference (relative 2e-6) on
  two held-out settings, `-s 45` and `-s 40 -r 21 -b 2 -c 2`. Fast-math flags are not graded.
- **Swarm:** 3 agents, Inkling-2-Small with a 256K context, $15 cap and 2.5 hours per swarm.
- **Generations:** an outside reviewer reads both swarms of a generation and writes the lessons the
  next generation starts with. Generation 1 started from the lessons of an earlier PIC run.
- **Machine:** UVA Rivanna, commands run in a container that sees only the swarm's workspace.

## Result of record

All six workspaces were timed one after another in a single job on one node (Intel Xeon Gold 6248),
five runs each with the median counting, against one shared reference (29.54 s and 25.19 s).
The raw output is in `regrade-one-node.txt`.

| Generation | Swarm 1 | Swarm 2 | Average | Score (each) | Model cost (pair) |
|---|---|---|---|---|---|
| 1 | 1.56× | 1.64× | 1.60× | 4, 4 | $3.46 |
| 2 | 1.73× | 1.64× | 1.69× | 4, 4 | $3.27 |
| 3 | 1.80× | 1.96× | 1.88× | 5, 5 | $3.52 |

Total model cost was about $10.25. Five swarms ran to the time limit; generation 3's first swarm
marked the goal complete after 113 minutes.

## Each case on its own

The held-out columns are from the re-grade above (Intel Xeon Gold 6248). The shown case, `-s 30`,
was timed later in a separate job that landed on an AMD EPYC 9454 node (`regrade-shown-case.txt`,
reference 6.52 s), so compare that column across generations, not against the other two.

| Generation | Swarm | Shown `-s 30` | Held-out `-s 45` | Held-out `-s 40 -r 21 -b 2 -c 2` |
|---|---|---|---|---|
| 1 | 1 | 1.70× | 1.58× | 1.55× |
| 1 | 2 | 1.84× | 1.58× | 1.70× |
| 2 | 1 | 1.97× | 1.67× | 1.79× |
| 2 | 2 | 1.91× | 1.69× | 1.60× |
| 3 | 1 | 2.29× | 1.71× | 1.91× |
| 3 | 2 | 2.44× | 1.85× | 2.07× |

The ranking by generation is the same in all three columns.

## How to read the numbers

- **Ignore the scores printed during the runs** (`scores.json`: 3.01×, 1.80×, 2.38×, 2.33×, 2.48×,
  2.66×). Two swarms shared a node and LULESH is limited by memory speed, so the reference was timed
  while the other swarm was working and came out up to twice as slow. The grader now takes one
  workspace at a time, but only a re-grade of all workspaces in one job is comparable.
- **Speedup depends on the machine.** Generations 1 and 2 measured 1.83×, 1.92×, 1.97× and 1.99× on
  an AMD EPYC 9454 node. Compare generations only within one re-grade.
- **The re-grade's 8 cores spanned two memory regions** of a shared node. Every workspace and the
  reference had the same cores, and each measurement's spread was under 4%.
- **Two swarms per generation is a small sample.** The trend is consistent but not proven.

## What the swarms did

- **Largest win, found by all six:** stop allocating and freeing temporary arrays on every time
  step, by caching them.
- **Second win, from generation 2 on:** the lessons said to time each phase before editing. That
  pointed at the per-region equation-of-state update, which takes about half of a step.
- **Flags:** every swarm ended on `-O3 -march=native`, some with `-fno-math-errno`.

Each swarm's own account is in `gen-N/run-M/FINAL.md` (where it wrote one), its source changes in
`changes.diff`, and the run statistics in `swarm-report.md`.

## How they collaborated

- **Roles:** every swarm split into a benchmark owner, a kernel owner and a flags or verification
  owner within the first minutes, without being told to.
- **Channel:** direct messages, 51 to 116 per swarm, with one agent in 70–78% of them. The task
  board was used for 3 tasks in total and sub-agents once.
- **Safeguards they built:** a pristine reference build, snapshots before risky edits, and a
  compare-against-reference script.
- **What went wrong:** the shared build was broken by half-finished edits several times; one check
  script reported a pass on the empty output of a crashed binary; generation 3's second swarm ran
  out of time with a measured improvement never merged.

## Caveats

- **The lessons became specific to LULESH.** By generation 3 they name the equation-of-state
  phase, the repetition loop and an OpenMP schedule setting. Part of the gain is therefore a recipe
  handed down, not a better way of working. The reviewer was left unchanged by choice.
- **One optimisation removes work the benchmark adds on purpose.** The `-c` option repeats the
  equation-of-state update to model costlier materials. Generations 2 and 3 moved the unchanged
  part of that work out of the repetition loop. The answer is the same, but how much of the
  speedup on the second held-out setting comes from this was not measured.

## Files

| Path | Contents |
|---|---|
| `regrade-one-node.txt` | Single-node re-grade of all six workspaces (job 23002050) |
| `regrade-shown-case.txt` | The shown case timed for all six workspaces (job 23013162) |
| `gen-N/scores.json` | In-job scores and collaboration counts for both swarms |
| `gen-N/lessons_in.md`, `lessons_out.md` | Lessons the generation started with and the reviewer's rewrite |
| `gen-N/summary.md` | The generation's summary table |
| `gen-N/run-M/` | `build.sh`, `changes.diff` against the original source, `FINAL.md`, `swarm-report.md` |

The full workspaces and event logs (about 90 MB) are in the git-ignored `runs/lulesh-g1` to
`runs/lulesh-g3`.
