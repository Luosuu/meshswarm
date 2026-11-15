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

AMR as published has a second weakness: its stencil weights sum to zero and its input field only ever
grows by a constant, so the stencil result is the same every iteration. Every swarm found this and
replaced the repeated sweeps with a cached result, which is correct but is not the work the benchmark
stands for. `amrx` is a variant without that property: the field starts uneven and each iteration feeds
a little of the output back into the input, so every sweep really has new data. It has no closed-form
answer, so it prints its norms and the grader compares them with the original program's.
"""

from __future__ import annotations

import asyncio
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

AMRX_TOLERANCE = 1e-9  # relative; a different summation order moves the norms by about 1e-13


def _amrx(text: str) -> str:
    """The AMR source with an uneven starting field and output fed back into the input (see above)."""
    def swap(old: str, new: str) -> None:
        nonlocal text
        if text.count(old) != 1:
            raise ValueError(f"amr.c is not the expected version: {old!r}")
        text = text.replace(old, new)

    swap("#define WEIGHT(ii,jj)   weight[ii+RADIUS][jj+RADIUS]",
         "#define WEIGHT(ii,jj)   weight[ii+RADIUS][jj+RADIUS]\n"
         "/* uneven part of the starting field, and the share of the output fed back into the input */\n"
         "#define BUMP(i,j)       ((DTYPE)(((i)*7+(j)*13)%11)*(DTYPE)0.25)\n"
         "#define FEEDBACK        ((DTYPE)1.0e-4)")
    swap("    IN(i,j)  = COEFX*i+COEFY*j;", "    IN(i,j)  = COEFX*i+COEFY*j+BUMP(i,j);")
    swap("        for (j=0; j<n_r_true; j++) for (i=0; i<n_r_true; i++) IN_R(g,i,j)+= (DTYPE)1.0;",
         "        for (j=0; j<n_r_true; j++) for (i=0; i<n_r_true; i++) IN_R(g,i,j)+= (DTYPE)1.0;\n"
         "        /* feed the solution back into the input field, so the next sweep sees new data */\n"
         "        for (j=RADIUS; j<n_r_true-RADIUS; j++) for (i=RADIUS; i<n_r_true-RADIUS; i++)\n"
         "          IN_R(g,i,j) += FEEDBACK*OUT_R(g,i,j);")
    swap("    for (j=0; j<n; j++) for (i=0; i<n; i++) IN(i,j)+= (DTYPE)1.0;",
         "    for (j=0; j<n; j++) for (i=0; i<n; i++) IN(i,j)+= (DTYPE)1.0;\n"
         "    /* feed the solution back into the input field, so the next sweep sees new data */\n"
         "    for (j=RADIUS; j<n-RADIUS; j++) for (i=RADIUS; i<n-RADIUS; i++)\n"
         "      IN(i,j) += FEEDBACK*OUT(i,j);")
    start = text.index("/* verify correctness of background grid solution and input field")
    end_mark = 'printf("Solution validates\\n");'
    end = text.index(end_mark) + len(end_mark)
    report = """/* this variant has no closed-form answer: print the norms, to be compared with the original program */
  full_cycles = ((iterations+1)/(period*4));
  leftover_iterations = (iterations+1)%(period*4);
  for (g=0; g<4; g++)
    iterations_r[g] = sub_iterations*(full_cycles*duration+
                      MIN(MAX(0,leftover_iterations-g*period),duration));
  printf("Norms: %.12e %.12e", (double)norm, (double)norm_in);
  for (g=0; g<4; g++) printf(" %.12e %.12e", (double)norm_r[g], (double)norm_in_r[g]);
  printf("\\n");"""
    return text[:start] + report + text[end:]


NORMS = re.compile(r"^Norms:((?: [-+0-9.eE]+){10})\s*$", re.M)


def norms(output: str) -> list[float] | None:
    m = NORMS.search(output)
    if not m:
        return None
    values = [float(x) for x in m.group(1).split()]
    return values if all(math.isfinite(v) for v in values) else None


def same_norms(got: list[float] | None, want: list[float]) -> bool:
    return got is not None and all(abs(g - w) <= AMRX_TOLERANCE * max(abs(w), 1.0) for g, w in zip(got, want))


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
    "amrx": {
        "title": "an adaptive-mesh-refinement stencil",
        "about": ("The program applies a stencil to a square grid for a number of iterations, and after every sweep "
                  "feeds a little of the result back into the input field, so each sweep works on new data. "
                  "Periodically, four finer sub-grids (refinements) appear on parts of the grid, are advanced with "
                  "their own sub-iterations and interpolated from the background grid, and disappear again."),
        "bin": "amr",
        "sources": ["amr.c", "wtime.c"],
        "headers": ["par-res-kern_general.h"],
        "patch": {"amr.c": _amrx},
        "flags": "-DDOUBLE=1 -DRADIUS=2 -DSTAR=1 -DLOOPGEN=0 -DVERBOSE=0 -DRESTRICT_KEYWORD=0",
        "usage": "./amr <iterations> <grid size> <refinement size> <refinement level> <period> <duration> <sub-iterations>",
        "dev_args": "40 2000 400 3 10 5 4",
        "test_args": ["150 3000 600 3 12 6 4", "200 4000 500 2 8 4 6"],
        "anchors": [],
        "norms": True,
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


def binary(name: str) -> str:
    return KERNELS[name].get("bin", name)


def build_line(name: str, opt: str = "-O2", extra: str = "") -> str:
    k = KERNELS[name]
    return (f"${{CC:-gcc}} {opt} -std=gnu99 {VERSION_FLAG} {k['flags']} {extra} -I. "
            f"{' '.join(k['sources'])} -lm -o {binary(name)}").replace("  ", " ")


def original_files(name: str) -> dict[str, str]:
    """The program a swarm starts from, which is also what its version is compared with."""
    k = KERNELS[name]
    patch = k.get("patch", {})
    return {f: patch.get(f, lambda t: t)((PRK / f).read_text()) for f in k["sources"] + k["headers"]}


def starting_files(name: str) -> dict[str, str]:
    files = original_files(name)
    files["build.sh"] = f"#!/bin/bash\n# Builds ./{binary(name)}. Change this as you like; the grader runs it as `bash build.sh`.\nset -e\n{build_line(name)}\n"
    return files


def task_text(name: str, cores: int) -> str:
    k = KERNELS[name]
    exe = binary(name)
    rungs = ", ".join(f"{t:g}x" for t in THRESHOLDS)
    if k.get("norms"):
        prints = ("It prints a line starting with `Norms:` that holds ten numbers describing the final state of the "
                  "grids, followed by a rate.")
        keep = ("the computation intact, meaning the same numbers come out")
        counts = (f"A run counts only if its `Norms:` line matches the one the original program prints for the same "
                  f"arguments, every number to a relative difference of {AMRX_TOLERANCE:g}: that leaves room for adding "
                  "the same values in another order, not for computing something else. There is no built-in check to "
                  "lean on, so keep a build of the original program and compare against it on several argument sets.")
        unsafe = "because they let the compiler change the arithmetic"
        know = "matching the original's `Norms:` line is the only way to know it is still correct"
        broken = "a run whose numbers differ"
    else:
        prints = "It prints `Solution validates` when its built-in check passes,\nfollowed by a rate."
        keep = "the computation and its built-in verification intact"
        counts = "A run counts only if it prints `Solution validates`\nand the verification code is still in the source."
        unsafe = "because those flags can silently switch the verification\noff"
        know = "`Solution validates` is the only way to know it is still correct"
        broken = "a run that does not validate"
    return f"""Make {k['title']} run as fast as possible on {cores} CPU cores, without changing what it computes.

{k['about']}

Your workspace already holds the serial C program ({', '.join(k['sources'] + k['headers'])}) and a
`build.sh` that compiles it to `./{exe}`. Run it as:

    {k['usage']}

for example `./{exe} {k['dev_args']}`. {prints}

Deliverable: the workspace itself. `bash build.sh` must produce `./{exe}`, taking the same arguments
and printing output in the same format. You may change the source and the build in any way that keeps
{keep}: parallelize it (OpenMP is available), restructure
loops and data, change compiler flags.

How it is graded: `build.sh` is run, then `./{exe}` is timed on {cores} cores with arguments you have
not seen: other settings than the example, and several times larger, each run five times with the
median counting. What is timed is the wall-clock time of the whole
program from start to exit, including allocation, initialization and verification, not the rate it
prints: the time `run_command` reports for `./{exe} ...` is the same measure. {counts} Builds that use `-ffast-math`, `-Ofast` or
`-ffinite-math-only` are not graded at all, {unsafe}. The score is the speedup over a serial -O3 build of
the original program, in rungs at {rungs}: reaching a rung earns its share, so a modest speedup is
better than a broken build. A build that fails, or {broken}, scores nothing, so
keep `build.sh` working at all times.

Use the `run_command` tool to compile, run and time things. It returns the wall time of each command.
Measure before and after every change: timing is the only way to know whether a change helped, and
{know}."""


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


def _validates(output: str) -> bool:
    return "Solution validates" in output


async def _times(runner: ShellRunner, command: str, workspace: Path, limit: float, repeats: int,
                 ok=_validates) -> tuple[list[float] | None, str]:
    """Wall time of each of `repeats` runs, measured by the shell inside the sandbox so that starting
    the sandbox itself is not counted. None when a run fails, times out or its output is not `ok`."""
    times, last = [], ""
    for _ in range(repeats):
        r = await runner.run(f"TIMEFORMAT='MESHTIME %R'; time {command}", workspace, limit)
        last = r.output
        found = TIMED.findall(r.output)  # the shell prints its line after the program has exited: take the last
        if r.timed_out or r.exit_code != 0 or not ok(r.output) or not found:
            return None, last
        times.append(float(found[-1]))
    return times, last


def _spread(times: list[float]) -> str:
    return f"{statistics.median(times):.2f}s [{min(times):.2f}-{max(times):.2f}]"


def make_grader(name: str, runner: ShellRunner, repeats: int = 5, baseline_repeats: int = 3, opt: str = "-O3"):
    k = KERNELS[name]
    exe = binary(name)
    by_norms = bool(k.get("norms"))
    baselines: dict[tuple[str, str | None], tuple[list[float], list[float] | None]] = {}  # measured once per case and core set, then reused

    turn = asyncio.Lock()  # one workspace is graded at a time: two programs timed at once slow each other down

    async def grade(result: RunResult) -> Grade:
        async with turn:
            return await grade_one(result)

    async def grade_one(result: RunResult) -> Grade:
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
        for f, text in original_files(name).items():
            (base / f).write_text(text)
        b = await runner.run(f"cd .mesh/baseline && {build_line(name, opt)}", ws, 300)
        if b.exit_code != 0:
            return Grade(0.0, f"baseline build failed (setup problem): {clip(b.output, 300)}")
        built = await runner.run(f"rm -f ./{exe} && bash build.sh", ws, 600)
        if built.exit_code != 0 or not (ws / exe).is_file():
            return Grade(0.0, f"build.sh failed: {clip(built.output.strip(), 300)}")
        speedups, notes = [], []
        for args in k["test_args"]:
            key = (args, runner.cores_for(ws))
            if key not in baselines:
                t0, out0 = await _times(runner, f"./.mesh/baseline/{exe} {args}", ws, 900, baseline_repeats,
                                        ok=(lambda o: norms(o) is not None) if by_norms else _validates)
                if t0 is None:
                    return Grade(0.0, f"baseline run failed (setup problem) on `{args}`: {clip(out0, 200)}")
                baselines[key] = (t0, norms(out0) if by_norms else None)
            t0, want = baselines[key]
            base_time = statistics.median(t0)
            t1, out1 = await _times(runner, f"./{exe} {args}", ws, max(120.0, 4 * base_time), repeats,
                                    ok=(lambda o: same_norms(norms(o), want)) if by_norms else _validates)
            if t1 is None:
                if by_norms and norms(out1) is not None:
                    return Grade(0.0, f"`./{exe} {args}` printed {NORMS.search(out1).group(0).strip()}; the original gives "
                                      f"Norms: {' '.join(f'{v:.12e}' for v in want)}")
                return Grade(0.0, f"`./{exe} {args}` failed, timed out or did not {'print its norms' if by_norms else 'validate'}: {clip(out1.strip(), 200)}")
            speedups.append(base_time / max(statistics.median(t1), 1e-6))
            notes.append(f"{_spread(t0)} -> {_spread(t1)}")
        speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        return Grade(score_of(speedup), f"speedup {speedup:.2f}x on held-out sizes, median [min-max] of repeated runs ({'; '.join(notes)})")

    return grade
