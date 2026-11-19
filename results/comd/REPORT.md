# CoMD: three generations of swarms

Three generations of two swarms each were asked to make CoMD, a molecular dynamics
mini-application, faster on 8 cores. On a like-for-like re-grade the average speedup was 1.70×,
1.76× and 1.96×. The best swarm barely moved (1.90×, 1.98×, 1.99×); the gain came from the weaker
swarm, which sat at about 1.5× in generations 1 and 2 and reached 1.93× in generation 3.

## Setup

- **Task:** CoMD (ECP-copa/CoMD, commit 3d48396, `src-openmp`, about 5,000 lines of C), already
  parallel with OpenMP. Atoms in a copper crystal move under a short-range force, Lennard-Jones or
  the embedded-atom method, found through a grid of link cells. The swarm gets the source and a
  sandboxed compile-and-run tool.
- **Reference:** the original source built with `-O3 -fopenmp`, on the same 8 cores.
- **Correctness:** the last row of the energy table must show the same step and atom count as the
  reference and the same total, potential and kinetic energy (relative 1e-7), with no atoms lost,
  on two held-out settings: `-x 60 -y 60 -z 60 -N 100` (Lennard-Jones) and
  `-e -d . -x 48 -y 40 -z 36 -N 150 -T 900` (embedded-atom). Fast-math flags are not graded.
- **Swarm:** 3 agents, Inkling-2-Small with a 256K context, $15 cap and 2.5 hours per swarm.
- **Generations:** an outside reviewer reads both swarms of a generation and writes the lessons the
  next generation starts with. Generation 1 started from the lessons of the LULESH run.
- **Machine:** UVA Rivanna, commands run in a container that sees only the swarm's workspace.

## Result of record

All six workspaces were timed one after another in a single job on one node (AMD EPYC 9454), five
runs each with the median counting, against one shared reference (19.91 s and 17.33 s). Every
measurement's spread was under 1%. The raw output is in `regrade-one-node.txt`.

| Generation | Swarm 1 | Swarm 2 | Average | Score (each) | Model cost (pair) | Time (each) |
|---|---|---|---|---|---|---|
| 1 | 1.90× | 1.49× | 1.70× | 5, 3 | $4.37 | 58, 102 min |
| 2 | 1.53× | 1.98× | 1.76× | 4, 5 | $6.49 | 93, 114 min |
| 3 | 1.99× | 1.93× | 1.96× | 5, 5 | $6.03 | 80, 56 min |

Total model cost was about $16.89. All six swarms marked the goal complete on their own.

## How to read the numbers

- **The in-job scores agree with the re-grade this time** (`scores.json`: 2.01×, 1.51×, 1.52×,
  1.95×, 1.93×, 1.87×). Generation 1 ran on an Intel node and the others on AMD, so only the
  re-grade compares generations on one machine.
- **No swarm crossed 2.0×.** Three swarms stopped within 0.1× of it.
- **The reference is a strong one.** The original already scales about 6.5× on 8 threads, so a
  2× gain here is on top of working OpenMP.
- **Two swarms per generation is a small sample.** One unlucky swarm in generation 3 would have
  erased the trend.

## What the swarms did

- **Largest win:** compute each pair of atoms once instead of twice in the force loops, with
  per-thread scratch arrays for the second atom's share, merged afterwards. Every swarm that
  reached 1.9× did this for both force models.
- **The same idea failed where it was done badly.** In generation 1 the second swarm's version for
  the embedded-atom force came out about 2.4× slower and was reverted.
- **Smaller wins:** scratch buffers for the boundary exchange kept between steps, zeroing only
  occupied slots, insertion sort in place of `qsort` for the atoms of a cell, parallel link-cell
  update.
- **Flags:** every swarm ended on `-O3 -march=native -fno-math-errno`; generation 3 added
  `-ffp-contract=fast`. Link-time optimisation, unrolling and profile-guided builds measured as
  noise.

Each swarm's own account is in `gen-N/run-M/FINAL.md` or `FINAL_NOTES.md`, its source changes in
`changes.diff`, and the run statistics in `swarm-report.md`.

## How they collaborated

- **Roles:** a force-kernel owner, an owner of link cells and boundary exchange, and a
  verification or benchmark owner, split by file.
- **Channel:** direct messages, 46 to 122 per swarm, with one agent in 74–80% of them. Five of six
  swarms posted 3 to 5 tasks on the board; no swarm used sub-agents.
- **Safeguards they built:** an untouched reference build, a check script over both force models,
  several box shapes and temperatures, and snapshots after each verified gain.
- **What went wrong:** generation 1's second swarm broke the shared build at least three times and
  its check script misread its own reference through a column off-by-one; in generation 2 one
  swarm's closing cleanup deleted the work tree while teammates still needed it, and the other
  ran out of budget with work unmerged; in generation 3 one swarm rebuilt its reference with
  other flags mid-run, read every benchmark as 0.92× and nearly rolled back good work. Agents
  were idle 44–75% of the time.

## Caveats

- **The lessons did not raise the best result.** Generation 1's first swarm was already at 1.90×.
  The evidence is for reliability, and it rests on generation 3 alone.
- **Lessons carried over from another task were not all dropped.** A lesson about LULESH's
  repetition loop, which CoMD does not have, survived two rewrites.
- **The lessons became specific to CoMD.** By generation 3 they point at pair symmetry in the hot
  loop and list the flags to use and the ones not worth trying.
- **Energy is a weak check on its own.** Agreement of the final energies to 1e-7 over 100 to 150
  steps would not catch every possible error, only wrong forces and skipped work.

## Files

| Path | Contents |
|---|---|
| `regrade-one-node.txt` | Single-node re-grade of all six workspaces (job 23019464) |
| `gen-N/scores.json` | In-job scores and collaboration counts for both swarms |
| `gen-N/lessons_in.md`, `lessons_out.md` | Lessons the generation started with and the reviewer's rewrite |
| `gen-N/summary.md` | The generation's summary table |
| `gen-N/run-M/` | `build.sh`, `changes.diff` against the original source, the swarm's own notes, `swarm-report.md` |

The full workspaces and event logs are in the git-ignored `runs/comd-g1` to `runs/comd-g3`.
