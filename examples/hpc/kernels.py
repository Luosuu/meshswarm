"""Performance tasks built on two application-like kernels from the Parallel Research Kernels:
AMR (a stencil on a grid with refinements that come and go) and PIC (charged particles moving
through a grid of charges). Each is one program with several interacting parts.

A swarm gets the serial source and a compile-and-run tool, and must deliver the fastest correct
version. The grader builds the swarm's version, checks that the kernel's own verification still
passes on problem sizes the swarm never saw, and scores the speedup over a serial -O3 build of the
original, measured on the same cores.

Limits of the correctness check: both kernels verify themselves against a closed-form answer, so a
program that printed "Solution validates" without doing the work would pass. The grader requires the
verification code to still be present (strictly for AMR, loosely for PIC, where a legitimate change
of data layout must touch it); results worth reporting should also be read by a person.
"""

from __future__ import annotations

import math
import re
import statistics
from pathlib import Path
from typing import Any

from meshswarm import RunResult
from meshswarm.shell import ShellRunner, clip
from meshevolve import Grade

PRK = Path(__file__).with_name("prk")
VERSION_FLAG = "-DPRKVERSION='\"2.17\"'"
# Speedup rungs; score = share of rungs reached. The first run reached 9.5x in its first generation,
# so the ladder goes well past that to leave room for later generations.
THRESHOLDS = [2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0, 25.0, 32.0]
KIND = "making a scientific program faster on a multi-core machine, as a team with a compile-and-run tool"

KERNELS: dict[str, dict[str, Any]] = {
    "amr": {
        "title": "an adaptive-mesh-refinement stencil",
        "about": ("The program applies a stencil to a square grid for a number of iterations. Periodically, four finer "
                  "sub-grids (refinements) appear on parts of the grid, are advanced with their own sub-iterations and "
                  "interpolated from the background grid, and disappear again."),
        "sources": ["amr.c", "wtime.c"],
        "headers": ["par-res-kern_general.h"],
        "flags": "-DDOUBLE=1 -DRADIUS=2 -DSTAR=1 -DLOOPGEN=0 -DVERBOSE=0 -DRESTRICT_KEYWORD=0",
        "usage": "./amr <iterations> <grid size> <refinement size> <refinement level> <period> <duration> <sub-iterations>",
        "dev_args": "40 2000 400 3 10 5 4",
        "test_args": ["150 3000 600 3 12 6 4", "200 4000 500 2 8 4 6"],
        "anchors": [
            "reference_norm = (DTYPE) (iterations+1) * (COEFX + COEFY);",
            "if (ABS(norm-reference_norm) > EPSILON) {",
            "if (ABS(norm_in-reference_norm_in) > EPSILON) {",
            "if (ABS(norm_r[g]-reference_norm_r[g]) > EPSILON) {",
            'printf("Solution validates\\n");',
        ],
    },
    "pic": {
        "title": "a particle-in-cell kernel",
        "about": ("The program moves a cloud of charged particles through a fixed grid of charges for a number of time "
                  "steps. Particles can be placed in four ways (GEOMETRIC, SINUSOIDAL, LINEAR, PATCH), some of them very "
                  "uneven across the grid."),
        "sources": ["pic.c", "wtime.c", "random_draw.c"],
        "headers": ["par-res-kern_general.h", "random_draw.h"],
        "flags": "",
        "usage": "./pic <steps> <grid size> <particles> <k> <m> <init mode> <init parameters>",
        "dev_args": "40 1000 2000000 1 2 GEOMETRIC 0.99",
        "test_args": ["100 1500 3000000 1 2 SINUSOIDAL", "100 1000 3000000 2 1 LINEAR 1.0 3.0"],
        "anchors": ["fmod(", "(iterations+1)", "epsilon", 'printf("Solution validates\\n");'],
    },
}


def build_line(name: str, opt: str = "-O2", extra: str = "") -> str:
    k = KERNELS[name]
    return (f"${{CC:-gcc}} {opt} -std=gnu99 {VERSION_FLAG} {k['flags']} {extra} -I. "
            f"{' '.join(k['sources'])} -lm -o {name}").replace("  ", " ")


def starting_files(name: str) -> dict[str, str]:
    k = KERNELS[name]
    files = {f: (PRK / f).read_text() for f in k["sources"] + k["headers"]}
    files["build.sh"] = f"#!/bin/bash\n# Builds ./{name}. Change this as you like; the grader runs it as `bash build.sh`.\nset -e\n{build_line(name)}\n"
    return files


def task_text(name: str, cores: int) -> str:
    k = KERNELS[name]
    rungs = ", ".join(f"{t:g}x" for t in THRESHOLDS)
    return f"""Make {k['title']} run as fast as possible on {cores} CPU cores, without changing what it computes.

{k['about']}

Your workspace already holds the serial C program ({', '.join(k['sources'] + k['headers'])}) and a
`build.sh` that compiles it to `./{name}`. Run it as:

    {k['usage']}

for example `./{name} {k['dev_args']}`. It prints `Solution validates` when its built-in check passes,
followed by a rate.

Deliverable: the workspace itself. `bash build.sh` must produce `./{name}`, taking the same arguments
and printing output in the same format. You may change the source and the build in any way that keeps
the computation and its built-in verification intact: parallelize it (OpenMP is available), restructure
loops and data, change compiler flags.

How it is graded: `build.sh` is run, then `./{name}` is timed on {cores} cores with arguments you have
not seen: other settings than the example, and several times larger, each run five times with the
median counting. What is timed is the wall-clock time of the whole
program from start to exit, including allocation, initialization and verification, not the rate it
prints: the time `run_command` reports for `./{name} ...` is the same measure. A run counts only if it prints `Solution validates`
and the verification code is still in the source. Builds that use `-ffast-math`, `-Ofast` or
`-ffinite-math-only` are not graded at all, because those flags can silently switch the verification
off. The score is the speedup over a serial -O3 build of
the original program, in rungs at {rungs}: reaching a rung earns its share, so a modest speedup is
better than a broken build. A build that fails, or a run that does not validate, scores nothing, so
keep `build.sh` working at all times.

Use the `run_command` tool to compile, run and time things. It returns the wall time of each command.
Measure before and after every change: timing is the only way to know whether a change helped, and
`Solution validates` is the only way to know it is still correct."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


def verification_intact(name: str, workspace: Path) -> bool:
    source = _norm("".join(p.read_text(errors="replace") for p in sorted(workspace.glob("*.c")) + sorted(workspace.glob("*.h"))))
    return all(_norm(a) in source for a in KERNELS[name]["anchors"])


# Flags that let the compiler assume no NaN or infinity, or reorder arithmetic freely. Under them the
# kernels' own comparison against the reference can be optimized away, so a wrong answer still prints
# "Solution validates" (a swarm demonstrated this on PIC). A build that uses them is not graded.
UNSAFE_MATH = re.compile(r"-ffast-math|-Ofast|-ffinite-math-only|-funsafe-math-optimizations|-fno-signed-nans"
                         r"|optimize\s*\(\s*\"[^\"]*(fast-math|finite-math|unsafe-math)")


def unsafe_math_flag(workspace: Path) -> str | None:
    for p in [workspace / "build.sh", *sorted(workspace.glob("Makefile*")), *sorted(workspace.glob("*.c")), *sorted(workspace.glob("*.h"))]:
        if p.is_file():
            m = UNSAFE_MATH.search(p.read_text(errors="replace"))
            if m:
                return f"{m.group(0)} in {p.name}"
    return None


def score_of(speedup: float) -> float:
    return round(10 * sum(1 for t in THRESHOLDS if speedup >= t) / len(THRESHOLDS), 2)


TIMED = re.compile(r"MESHTIME ([0-9.]+)")


async def _times(runner: ShellRunner, command: str, workspace: Path, limit: float, repeats: int) -> tuple[list[float] | None, str]:
    """Wall time of each of `repeats` runs, measured by the shell inside the sandbox so that starting
    the sandbox itself is not counted. None when a run fails, times out or does not validate."""
    times, last = [], ""
    for _ in range(repeats):
        r = await runner.run(f"TIMEFORMAT='MESHTIME %R'; time {command}", workspace, limit)
        last = r.output
        found = TIMED.findall(r.output)  # the shell prints its line after the program has exited: take the last
        if r.timed_out or r.exit_code != 0 or "Solution validates" not in r.output or not found:
            return None, last
        times.append(float(found[-1]))
    return times, last


def _spread(times: list[float]) -> str:
    return f"{statistics.median(times):.2f}s [{min(times):.2f}-{max(times):.2f}]"


def make_grader(name: str, runner: ShellRunner, repeats: int = 5, baseline_repeats: int = 3, opt: str = "-O3"):
    k = KERNELS[name]
    baselines: dict[tuple[str, str | None], list[float]] = {}  # measured once per case and core set, then reused

    async def grade(result: RunResult) -> Grade:
        ws = Path(result.workspace)
        if not (ws / "build.sh").is_file():
            return Grade(0.0, "no build.sh in the workspace")
        if not verification_intact(name, ws):
            return Grade(0.0, "the kernel's verification code is no longer in the source")
        unsafe = unsafe_math_flag(ws)
        if unsafe:
            return Grade(0.0, f"the build uses {unsafe}, which can switch off the kernel's own verification")
        # The serial baseline is built from the original files in a folder agents cannot write, and
        # timed with this swarm's own cores so that the two times are comparable.
        base = ws / ".mesh" / "baseline"
        base.mkdir(parents=True, exist_ok=True)
        for f in k["sources"] + k["headers"]:
            (base / f).write_text((PRK / f).read_text())
        b = await runner.run(f"cd .mesh/baseline && {build_line(name, opt)}", ws, 300)
        if b.exit_code != 0:
            return Grade(0.0, f"baseline build failed (setup problem): {clip(b.output, 300)}")
        built = await runner.run(f"rm -f ./{name} && bash build.sh", ws, 600)
        if built.exit_code != 0 or not (ws / name).is_file():
            return Grade(0.0, f"build.sh failed: {clip(built.output.strip(), 300)}")
        speedups, notes = [], []
        for args in k["test_args"]:
            key = (args, runner.cores_for(ws))
            if key not in baselines:
                t0, out0 = await _times(runner, f"./.mesh/baseline/{name} {args}", ws, 900, baseline_repeats)
                if t0 is None:
                    return Grade(0.0, f"baseline run failed (setup problem) on `{args}`: {clip(out0, 200)}")
                baselines[key] = t0
            t0 = baselines[key]
            base_time = statistics.median(t0)
            t1, out1 = await _times(runner, f"./{name} {args}", ws, max(120.0, 4 * base_time), repeats)
            if t1 is None:
                return Grade(0.0, f"`./{name} {args}` failed, timed out or did not validate: {clip(out1.strip(), 200)}")
            speedups.append(base_time / max(statistics.median(t1), 1e-6))
            notes.append(f"{_spread(t0)} -> {_spread(t1)}")
        speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        return Grade(score_of(speedup), f"speedup {speedup:.2f}x on held-out sizes, median [min-max] of repeated runs ({'; '.join(notes)})")

    return grade
