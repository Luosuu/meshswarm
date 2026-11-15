"""A performance task on a second mini-application: CoMD, a classical molecular dynamics proxy app
from Los Alamos (about 5,000 lines of C). Atoms in a copper crystal move under a short-range force,
either Lennard-Jones or the embedded-atom method, found through a grid of link cells.

Like LULESH the starting program is already parallel with OpenMP, so the reference is the original
built with `-O3 -fopenmp` on the same cores. Unlike LULESH the cost is in particle pair loops and
neighbour search, not in sweeps over a mesh.

Correctness is checked from outside: on settings the swarm never saw, the last line of the energy
table must show the same step and atom count as the reference and the same total, potential and
kinetic energy per atom. Atom trajectories are chaotic, so the energies of two correct programs
drift apart slowly; the tolerance leaves room for that over the short runs used here, and is far
tighter than the change a wrong force or a skipped step makes.
"""

from __future__ import annotations

import asyncio
import math
import re
import statistics
from pathlib import Path

from meshswarm import RunResult
from meshswarm.shell import ShellRunner, clip
from meshevolve import Grade

from kernels import _spread, _times, unsafe_math_flag

SRC = Path(__file__).with_name("comd_src")
SOURCES = sorted(p.name for p in SRC.glob("*.c"))
HEADERS = sorted(p.name for p in SRC.glob("*.h"))
DATA = ["Cu_u6.eam"]
PROTECT = ["build.sh", "*.c", "*.h", "Makefile*", "*.eam"]
THRESHOLDS = [1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
KIND = "making a scientific program faster on a multi-core machine, as a team with a compile-and-run tool"
DEV_ARGS = "-x 30 -y 30 -z 30"
DEV_ARGS_EAM = "-e -d . -x 24 -y 24 -z 24"
TEST_ARGS = ["-x 60 -y 60 -z 60 -N 100", "-e -d . -x 48 -y 40 -z 36 -N 150 -T 900"]
ENERGY_TOLERANCE = 1e-7  # relative, on energies per atom printed with twelve decimals

# One row of the energy table: step, time, total, potential and kinetic energy, temperature, rate, atoms.
ROW = re.compile(r"^\s*(\d+)\s+[0-9.]+\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+(\d+)\s*$", re.M)

Answer = tuple[int, int, float, float, float]


def build_line(opt: str = "-O2", omp: str = "-fopenmp") -> str:
    return f"${{CC:-gcc}} -std=c99 {opt} {omp} -DDOUBLE -I. *.c -lm -o CoMD".replace("  ", " ")


def starting_files(omp: str = "-fopenmp") -> dict[str, str]:
    files = {f: (SRC / f).read_text() for f in SOURCES + HEADERS + DATA}
    files["build.sh"] = ("#!/bin/bash\n# Builds ./CoMD. Change this as you like; the grader runs it as `bash build.sh`.\n"
                         f"set -e\n{build_line(omp=omp)}\n")
    return files


def answer(output: str) -> Answer | None:
    """(step, atoms, total, potential, kinetic energy per atom) from the last row of the energy table."""
    rows = ROW.findall(output)
    if not rows or "no atoms lost" not in output:
        return None
    step, total, potential, kinetic, atoms = rows[-1]
    try:
        got = (int(step), int(atoms), float(total), float(potential), float(kinetic))
    except ValueError:
        return None
    return got if all(math.isfinite(v) for v in got[2:]) else None


def same_answer(got: Answer | None, want: Answer) -> bool:
    if got is None or got[:2] != want[:2]:
        return False
    return all(abs(g - w) <= ENERGY_TOLERANCE * max(abs(w), 1e-300) for g, w in zip(got[2:], want[2:]))


def show(a: Answer) -> str:
    return f"step {a[0]}, {a[1]} atoms, energies {a[2]:.12f} {a[3]:.12f} {a[4]:.12f}"


def score_of(speedup: float) -> float:
    return round(10 * sum(1 for t in THRESHOLDS if speedup >= t) / len(THRESHOLDS), 2)


def task_text(cores: int) -> str:
    rungs = ", ".join(f"{t:g}x" for t in THRESHOLDS)
    return f"""Make CoMD, a molecular dynamics mini-application, run as fast as possible on {cores} CPU cores, without
changing what it computes.

CoMD simulates atoms in a copper crystal. Each time step advances the velocities and positions,
moves atoms between link cells (a grid of boxes that makes neighbours cheap to find, with copies of
the atoms across the periodic boundary), and computes the force on every atom from its neighbours.
There are two force models: Lennard-Jones (ljForce.c, the default) and the embedded-atom method
(eam.c, chosen with `-e`, which reads the table in Cu_u6.eam and takes several passes over the
atoms). timestep.c drives a step, linkCells.c and haloExchange.c keep the cells and boundary copies
up to date, initAtoms.c builds the crystal, and CoMD.c holds `main` and prints the results. The
parallel.c layer is a stand-in for MPI, which this build does not use. These parts share the same
atom arrays, so a change to how atoms are stored or ordered affects all of them.

Your workspace holds the program (the .c and .h files, and the table Cu_u6.eam) and a `build.sh`
that compiles it to `./CoMD`. The program is already parallel with OpenMP. Run it as:

    ./CoMD [-x <cells> -y <cells> -z <cells>] [-N <steps>] [-T <temperature>] [-e -d .]

for example `./CoMD {DEV_ARGS}` for Lennard-Jones or `./CoMD {DEV_ARGS_EAM}` for the embedded-atom
method. It prints a table with one row every ten steps (step, time, total, potential and kinetic
energy per atom, temperature, rate, number of atoms), then a validation summary and timings. It
also writes a small .yaml file on every run; those are not graded and can be deleted.

Deliverable: the workspace itself. `bash build.sh` must produce `./CoMD`, taking the same arguments
and printing its results in the same format. You may change the source and the build in any way
that keeps the computation the same: restructure loops and data, change how neighbours are found,
change how memory is allocated and reused, change the threading, change compiler flags.

How it is graded: `build.sh` is run, then `./CoMD` is timed on {cores} cores with arguments you have not
seen: both force models, larger crystals than the examples, other box shapes, step counts and
temperatures, each run five times with the median counting. What is timed is the wall-clock time of
the whole program from start to exit, including building the crystal, not the rate it prints: the
time `run_command` reports for `./CoMD ...` is the same measure. A run counts only if the last row of
its table shows the same step and atom count as the original program, and the same total, potential
and kinetic energy to a relative difference of {ENERGY_TOLERANCE:g}, and no atoms are lost. That leaves
room for adding the same forces in another order, not for a different force or a skipped step, so
check against the original after every change (keep an untouched build of the original to compare
with, and compare on both force models and more than one size). Builds that use `-ffast-math`,
`-Ofast` or `-ffinite-math-only` are not graded at all. The score is the speedup over the original
program built with `-O3 -fopenmp` and run on the same {cores} cores, in rungs at {rungs}:
reaching a rung earns its share, so a modest speedup is better than a broken build. A build that
fails, or a run with a different answer, scores nothing, so keep `build.sh` working at all times.

Use the `run_command` tool to compile, run and time things. It returns the wall time of each command.
Measure before and after every change: timing is the only way to know whether a change helped, and
comparing the printed energies is the only way to know it is still correct. Timings vary by a few
percent from run to run, so repeat a measurement before trusting a small difference."""


def make_grader(runner: ShellRunner, repeats: int = 5, baseline_repeats: int = 3, omp: str = "-fopenmp",
                test_args: list[str] | None = None):
    cases = test_args or TEST_ARGS
    baselines: dict[tuple[str, str | None], tuple[list[float], Answer]] = {}  # once per case and core set

    turn = asyncio.Lock()  # one workspace is graded at a time: two programs timed at once slow each other down

    async def grade(result: RunResult) -> Grade:
        async with turn:
            return await grade_one(result)

    async def grade_one(result: RunResult) -> Grade:
        ws = Path(result.workspace)
        if not (ws / "build.sh").is_file():
            return Grade(0.0, "no build.sh in the workspace")
        unsafe = unsafe_math_flag(ws)
        if unsafe:
            return Grade(0.0, f"the build uses {unsafe}, which lets the compiler change the arithmetic")
        # The reference is built from the original files in a folder agents cannot write, and timed
        # with this swarm's own cores so that the two times are comparable.
        base = ws / ".mesh" / "baseline"
        base.mkdir(parents=True, exist_ok=True)
        for f in SOURCES + HEADERS + DATA:
            (base / f).write_text((SRC / f).read_text())
        b = await runner.run(f"cd .mesh/baseline && {build_line('-O3', omp)}", ws, 600)
        if b.exit_code != 0:
            return Grade(0.0, f"reference build failed (setup problem): {clip(b.output, 300)}")
        built = await runner.run("rm -f ./CoMD && bash build.sh", ws, 900)
        if built.exit_code != 0 or not (ws / "CoMD").is_file():
            return Grade(0.0, f"build.sh failed: {clip(built.output.strip(), 300)}")
        speedups, notes = [], []
        for args in cases:
            key = (args, runner.cores_for(ws))
            if key not in baselines:
                # The reference reads the potential table from its own folder, whatever `-d` says.
                t0, out0 = await _times(runner, f"(cd .mesh/baseline && ./CoMD {args})", ws, 1200, baseline_repeats,
                                        ok=lambda out: answer(out) is not None)
                if t0 is None:
                    return Grade(0.0, f"reference run failed (setup problem) on `{args}`: {clip(out0, 200)}")
                baselines[key] = (t0, answer(out0))
            t0, want = baselines[key]
            base_time = statistics.median(t0)
            t1, out1 = await _times(runner, f"./CoMD {args}", ws, max(120.0, 4 * base_time), repeats,
                                    ok=lambda out: same_answer(answer(out), want))
            if t1 is None:
                got = answer(out1)
                why = (f"ended at {show(got)}; the original gives {show(want)}" if got
                       else f"failed, timed out or lost atoms: {clip(out1.strip()[-200:], 200)}")
                return Grade(0.0, f"`./CoMD {args}` {why}")
            speedups.append(base_time / max(statistics.median(t1), 1e-6))
            notes.append(f"{_spread(t0)} -> {_spread(t1)}")
        speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        return Grade(score_of(speedup), f"speedup {speedup:.2f}x over the original OpenMP build on held-out settings, "
                                        f"median [min-max] of repeated runs ({'; '.join(notes)})")

    return grade
