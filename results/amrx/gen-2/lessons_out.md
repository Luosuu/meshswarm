# Lessons for swarm-driven CPU optimization of a scientific compute program

**Name the single writer of the canonical source at kickoff.** Run-1 lost time to a double-write: one agent claimed the parallelization task and had edited; a second announced it would rewrite the same file; the verifier had to arbitrate. Run-2 (9.5x vs run-1's 5.0x) proposed the split in its first messages ("I take canonical, you verify, you benchmark") and merged cleanly. Make the split before anyone opens the file.

**Claim before code; release explicitly when handing off.** Treat the task board as the lock on the canonical file. Run-2's verifier claimed, then released its task in one clear message. Duplicating the read-only reference build is cheap insurance both runs accepted; duplicating canonical writes is not.

**Build the reference and arg matrix before optimizing.** Both runs agreed; gate every merge. Run-1's one real correctness bug was an uninitialized read on a rarely-taken path — exactly what never-taken-path cases catch — and small-set agreement was not enough: gate large grids too (worst diffs ~1e-10 against a 1e-9 tolerance).

**Sub-agents: one bounded experiment, absolute paths, keep/drop verdict.** Run-2's sub-agent returned clean verdicts without touching canonical code; run-1's wasted effort rediscovering the workspace from relative paths. The sub-agent never merges.

**Time A/B interleaved on a shared box.** Both runs saw load averages ~100 leaking in from outside the container; absolute times drift between sessions. Interleave variant and baseline, take repeated medians, and when flags are within noise prefer the portable one — run-1 adopted x86-64-v3 over -march=native at no measured cost; run-2 kept native after confirming the grader targets the same machine. Decide explicitly.

**Profile phases, spend where they dominate; time the whole program.** Run-2's phase breakdown focused its effort; the graded measure is start-to-exit wall including allocation and init — never the rate the program prints.

**Keep the rejected-ideas list; it compounds.** Tiling, double buffering, THP/madvise, collapse(2), SIMD-width hints, processor binding — measured and dropped in both runs. Never re-try; never stack speculative wins without a timing.

**Aim the idle verifier.** Both runs pointed the now-idle verifier at larger grids and rarer edge-case matrices; that de-risking caught issues hours before freeze.

**Freeze means no edits — even one pragma.** Run-2 froze, then found the canonical file had changed afterward and had to re-verify; an rm became a delete-then-correct false alarm. Re-run the full matrix after any late change; copy scratch in by reviewed diff, never bulk-cp; list and announce targets before any rm.

## Untested (experiments, not advice)
- **A merge-gatekeeper fourth agent** owning all canonical writes — does it remove conflicts or just bottleneck?
- **A verify-on-save watcher** running the arg matrix after each canonical edit — would it catch rare-path bugs before a human stops the line?
- **Load-aware benchmarking** that refuses timing claims on a noisy box — both runs attempted this ad hoc and wasted effort.