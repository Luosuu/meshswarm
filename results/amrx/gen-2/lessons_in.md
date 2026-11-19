# Lessons for swarm-driven CPU optimization of a scientific compute program

**Build the reference verifier first, before optimizing.** Both runs that scored well agreed on this: keep a pristine serial -O3 build, compare the candidate's output against it on a matrix of argument sets — including minimal-iteration and never-taken-path edge cases — and gate every merge on it. It caught real correctness bugs twice (a stale-buffer read on a rarely activated path; an over-eager removal of a zero-init whose buffer is read on paths the compute never writes). Never ship a change that has not passed this matrix, and extend the matrix whenever a bug slips through.

**Never bulk-copy scratch files over the canonical source.** Both runs lost or nearly lost work this way: one `cp` destroyed a teammate's patch; a mid-edit save was caught by a teammate before it spread. Work in scratch copies; merge into the canonical file by hand or by a reviewed diff; announce "I will not touch these loops" before stepping near them; if you must copy, diff-verify first and tell the team.

**Shared loop indices are a real hazard.** One run chased a phantom "compiler bug" for a long time; the actual cause was loop dummy variables not made private in parallel regions. Declare every loop variable private, and treat unexplained output divergence as your bug until proven otherwise — never re-tag it to the toolchain.

**Keep arithmetic operation order stable where the tolerance is tight.** Reassociation passes changed results, and one "speedup" turned out to be measured on a path that was not the one actually compiled. Confirm which path your flags select, re-time after any source change, and drop anything that breaks bitwise/tolerance matching.

**Measure on a quiet machine, median of repeated runs.** Both runs made claims while the box was loaded near saturating its 8 cores; numbers went stale and misled. Median-of-5 minimum; if load spikes, abandon the timing claim, not the review. The grader re-times, so real wins are what survive.

**Profile phases before micro-optimizing.** One run's phase breakdown showed a single block dominating runtime; effort aimed there paid off. Do a cheap breakdown in the first hour.

**Measure, then prune.** Double-buffering helped in one run; the same idea was a verified wall-clock wash in another. Huge pages, LTO, peeling, aggressive schedules — all tried, measured, dropped. Keep a "rejected ideas" list so the next generation does not re-try them, and never accumulate speculative wins without a timing.

**Delegate bounded experiments to sub-agents in scratch copies.** THP, ping-pong fusion, SIMD porting — each went to a sub-agent with "report keep/drop" and returned clean verdicts without touching canonical code. This pattern works; hand the sub-agent a self-contained copy and a verdict format.

**Assign explicit roles: one hot-loop owner, one verifier/flags owner.** Both runs converged on this split; announced boundaries prevented clobbering. Expect the verifier role to idle after early work — put it on edge-case matrices and freeze checks, not spin-waiting.

**Freeze discipline:** clean rebuild from scratch, full arg matrix, then stop touching the tree. Remove scratch binaries before declaring done.

## Untested (experiments, not advice)
- **A dedicated merge-gatekeeper** (a 4th agent owning all writes to the canonical source) — would this eliminate clobbering in larger swarms?
- **Automatic verify-on-save** (watcher running the arg matrix after every edit to the canonical file) instead of manual checks — would it have caught mid-edit states sooner?
- **Explicit load-aware benchmarking** (check machine load before accepting any timing claim, auto-invalidate noisy runs) — both runs handled this ad hoc and wasted effort.