"""A performance task on a whole mini-application: LULESH 2.0, a shock-hydrodynamics proxy app from
Lawrence Livermore (about 5,000 lines of C++ in use, several dozen routines that depend on each other).

Unlike the two PRK kernels, the starting program is already parallel (OpenMP), so the reference is
the original built with `-O3 -fopenmp` and run on the same cores: the swarm is scored on how much
faster it makes a program that already uses every core.

Correctness is checked from outside: the grader runs the reference on problem settings the swarm
never saw and requires the swarm's program to print the same iteration count and the same final
origin energy. The program cannot know those answers without doing the simulation.
"""

from __future__ import annotations

import math
import re
import statistics
from pathlib import Path

from meshswarm import RunResult
from meshswarm.shell import ShellRunner, clip
from meshevolve import Grade

from kernels import _spread, _times

SRC = Path(__file__).with_name("lulesh_src")
SOURCES = ["lulesh.cc", "lulesh-comm.cc", "lulesh-viz.cc", "lulesh-util.cc", "lulesh-init.cc"]
HEADERS = ["lulesh.h", "lulesh_tuple.h"]
PROTECT = ["build.sh", "*.cc", "*.h", "*.cpp", "*.hpp", "*.c", "Makefile*"]
# Speedup over the original OpenMP build on the same cores. Published hand-tuning of LULESH gains
# roughly 1.5x to 3x, so the ladder is dense at the bottom and ends well past that.
THRESHOLDS = [1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
KIND = "making a scientific program faster on a multi-core machine, as a team with a compile-and-run tool"
DEV_ARGS = "-s 30"
TEST_ARGS = ["-s 45", "-s 40 -r 21 -b 2 -c 2"]
ENERGY_TOLERANCE = 2e-6  # the energy is printed with seven significant digits

ITERATIONS = re.compile(r"Iteration count\s*=\s*(\d+)")
ENERGY = re.compile(r"Final Origin Energy\s*=\s*([-+0-9.eE]+)")


def build_line(opt: str = "-O2", omp: str = "-fopenmp") -> str:
    return f"${{CXX:-g++}} {opt} {omp} -DUSE_MPI=0 -I. {' '.join(SOURCES)} -lm -o lulesh".replace("  ", " ")


def starting_files(omp: str = "-fopenmp") -> dict[str, str]:
    files = {f: (SRC / f).read_text() for f in SOURCES + HEADERS}
    files["build.sh"] = ("#!/bin/bash\n# Builds ./lulesh. Change this as you like; the grader runs it as `bash build.sh`.\n"
                         f"set -e\n{build_line(omp=omp)}\n")
    return files


def answer(output: str) -> tuple[int, float] | None:
    """(iteration count, final origin energy) printed by a run, or None if either is missing."""
    it, en = ITERATIONS.findall(output), ENERGY.findall(output)
    try:
        return (int(it[-1]), float(en[-1])) if it and en else None
    except ValueError:
        return None


def same_answer(got: tuple[int, float] | None, want: tuple[int, float]) -> bool:
    if got is None or got[0] != want[0] or not math.isfinite(got[1]):
        return False
    return abs(got[1] - want[1]) <= ENERGY_TOLERANCE * max(abs(want[1]), 1e-300)


def score_of(speedup: float) -> float:
    return round(10 * sum(1 for t in THRESHOLDS if speedup >= t) / len(THRESHOLDS), 2)


def task_text(cores: int) -> str:
    rungs = ", ".join(f"{t:g}x" for t in THRESHOLDS)
    return f"""Make LULESH, a shock-hydrodynamics mini-application, run as fast as possible on {cores} CPU cores, without
changing what it computes.

LULESH simulates a blast wave on an unstructured hexahedral mesh. Each time step computes forces on
the nodes (stress and hourglass terms), moves the nodes, then updates the elements (kinematics,
artificial viscosity, the equation of state region by region) and chooses the next time step. These
steps are several dozen routines that share the same mesh data, so a change in one place (for example
how the mesh data is laid out, or where temporary arrays are allocated) affects many of them.

Your workspace holds the program ({', '.join(SOURCES + HEADERS)}) and a `build.sh`
that compiles it to `./lulesh`. Nearly all the run time is in lulesh.cc; lulesh-init.cc sets the mesh
up, lulesh-util.cc reads the arguments and prints the result, and lulesh-comm.cc and lulesh-viz.cc are
only for MPI and plot files, which this build does not use. The program is already parallel with
OpenMP. Run it as:

    ./lulesh -s <mesh size> [-r <regions>] [-b <balance>] [-c <cost>] [-i <max iterations>]

for example `./lulesh {DEV_ARGS}`. It prints the iteration count and the final origin energy, then timings.

Deliverable: the workspace itself. `bash build.sh` must produce `./lulesh`, taking the same arguments
and printing its result in the same format. You may change the source and the build in any way that
keeps the computation the same: restructure loops and data, change how memory is allocated and reused,
change the threading, change compiler flags.

How it is graded: `build.sh` is run, then `./lulesh` is timed on {cores} cores with arguments you have not
seen: larger meshes than the example, and other region settings, each run five times with the median
counting. What is timed is the wall-clock time of the whole program from start to exit, including
setting the mesh up, not the time it prints: the time `run_command` reports for `./lulesh ...` is the same
measure. A run counts only if it prints the same iteration count as the original program and the same
final origin energy to six significant digits, so check both against the original after every change
(keep an untouched build of the original to compare with, and compare on more than one setting).
The score is the speedup over the original program built with `-O3 -fopenmp` and run on the same {cores}
cores, in rungs at {rungs}: reaching a rung earns its share, so a modest speedup is better than a
broken build. A build that fails, or a run with a different answer, scores nothing, so keep `build.sh`
working at all times.

Use the `run_command` tool to compile, run and time things. It returns the wall time of each command.
Measure before and after every change: timing is the only way to know whether a change helped, and
comparing the printed answer is the only way to know it is still correct. Timings vary by a few
percent from run to run, so repeat a measurement before trusting a small difference."""


def make_grader(runner: ShellRunner, repeats: int = 5, baseline_repeats: int = 3, omp: str = "-fopenmp",
                test_args: list[str] | None = None):
    cases = test_args or TEST_ARGS
    baselines: dict[tuple[str, str | None], tuple[list[float], tuple[int, float]]] = {}  # once per case and core set

    async def grade(result: RunResult) -> Grade:
        ws = Path(result.workspace)
        if not (ws / "build.sh").is_file():
            return Grade(0.0, "no build.sh in the workspace")
        # The reference is built from the original files in a folder agents cannot write, and timed
        # with this swarm's own cores so that the two times are comparable.
        base = ws / ".mesh" / "baseline"
        base.mkdir(parents=True, exist_ok=True)
        for f in SOURCES + HEADERS:
            (base / f).write_text((SRC / f).read_text())
        b = await runner.run(f"cd .mesh/baseline && {build_line('-O3', omp)}", ws, 600)
        if b.exit_code != 0:
            return Grade(0.0, f"reference build failed (setup problem): {clip(b.output, 300)}")
        built = await runner.run("rm -f ./lulesh && bash build.sh", ws, 900)
        if built.exit_code != 0 or not (ws / "lulesh").is_file():
            return Grade(0.0, f"build.sh failed: {clip(built.output.strip(), 300)}")
        speedups, notes = [], []
        for args in cases:
            key = (args, runner.cores_for(ws))
            if key not in baselines:
                t0, out0 = await _times(runner, f"./.mesh/baseline/lulesh {args}", ws, 1200, baseline_repeats,
                                        ok=lambda out: answer(out) is not None)
                if t0 is None:
                    return Grade(0.0, f"reference run failed (setup problem) on `{args}`: {clip(out0, 200)}")
                baselines[key] = (t0, answer(out0))
            t0, want = baselines[key]
            base_time = statistics.median(t0)
            t1, out1 = await _times(runner, f"./lulesh {args}", ws, max(120.0, 4 * base_time), repeats,
                                    ok=lambda out: same_answer(answer(out), want))
            if t1 is None:
                got = answer(out1)
                why = (f"printed {got[0]} iterations, energy {got[1]:.6e}; the original gives {want[0]}, {want[1]:.6e}"
                       if got else f"failed or timed out: {clip(out1.strip(), 200)}")
                return Grade(0.0, f"`./lulesh {args}` {why}")
            speedups.append(base_time / max(statistics.median(t1), 1e-6))
            notes.append(f"{_spread(t0)} -> {_spread(t1)}")
        speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        return Grade(score_of(speedup), f"speedup {speedup:.2f}x over the original OpenMP build on held-out settings, "
                                        f"median [min-max] of repeated runs ({'; '.join(notes)})")

    return grade
