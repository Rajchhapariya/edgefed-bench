# EdgeFed-Bench Audit Report

Audit date: 2026-11-19. Commit audited: 293d825 (main, working tree clean).
We ran everything in a scratch copy of the repository. During the audit no existing file was edited, nothing was committed or pushed, and nothing was sent to an external service. File and line references below point to commit 293d825.

## 1. Summary verdict

1. The current grid reproduces exactly. We reran all 320 runs (280 grid + 40 wrong-f, 50 rounds, seeds 0-4). Every run and all 16,000 per-round rows match `results/results.duckdb` bit for bit, and the README, `report/metrics.typ` and `report/tables.typ` regenerate unchanged.
2. The headline figures in the request (74.60% accuracy, 100% mitigation, 75% less uplink, 5 rounds) come from the README of the older commit 8e75282. That commit's code reproduces them exactly. The current code does not: for the same 5-round, seed-42 setting it gives 72.28% and 60% mitigation.
3. Only the 75% uplink cut still holds (101,774 vs 407,080 bytes per update). "100% mitigation" is false for sign-flip in the 50-round grid: only 16 of 30 Multi-Krum runs reject every attacker in every round. In `simulate.py`, median and trimmed mean always report 100% because of how the value is computed.
4. Results repeat only at a fixed torch thread count: the same run gives 85.23 at 10 threads and 85.21 at 2. Several statements in the report and README are wrong or stale: stochastic noise, "240 attacked configurations", "26 unit tests" and cross-platform reproducibility.
5. Verdict: a sound, reproducible engineering benchmark, but not yet a research paper. The 5-round numbers are taken before training has settled. There is one small model on one dataset, no validation split, no stated source for the hyperparameters and no significance testing.

## 2. Claims table

Verdicts: CONFIRMED = reproduced exactly, or the code does what the text says. MISMATCH = reproduced value differs, or the code contradicts the text. NOT REPRODUCIBLE = cannot be reproduced with the current code. UNVERIFIED = not measured or not testable here.

| Claim | Where stated (file:line) | Reproduced value | Verdict |
| :--- | :--- | :--- | :--- |
| Multi-Krum INT8 reaches 74.60% under 20% Byzantine (5 rounds, seed 42) | README.md:24 and :35 at commit 8e75282 (not in current README) | 8e75282 code: 74.60 (two identical runs). Current code: 72.28 | NOT REPRODUCIBLE with current code; CONFIRMED for 8e75282 |
| 74.39% is the "unquantized Defended baseline" | README.md:24 at 8e75282 | 74.39 is clean FedAvg (row 1). Defended Float32 Multi-Krum is 73.40 (row 3) | MISMATCH (mislabelled) |
| Poisoned FedAvg collapses to 19.43% | README.md:24, :33 at 8e75282 | 8e75282: 19.43. Current: 31.17 | NOT REPRODUCIBLE with current code |
| 100% Byzantine mitigation (Multi-Krum, Float32 and INT8) | README.md:23, :34-35 at 8e75282 | 8e75282: 100%. Current `simulate.py`: 60%. 50-round grid sign-flip: 90.8% to 99.8% mean rejection; only 16/30 runs perfect | NOT REPRODUCIBLE / MISMATCH |
| 75% less uplink with INT8 (397.5 KB to 99.4 KB) | README.md:23, :45, :49-52; report/metrics.typ:11 | 407,080 vs 101,774 bytes = 75.0% | CONFIRMED |
| Multi-Krum 20% sign-flip: INT8 84.90 +- 0.26, Float32 84.89 +- 0.24 | README.md:25; report/metrics.typ | Identical after full rerun | CONFIRMED |
| 30% sign-flip INT8: Multi-Krum 84.75, Trimmed mean 80.37, Median 80.04 | README.md:26 | Identical after full rerun | CONFIRMED |
| FedAvg 10.00% at 30% sign-flip | README.md:25 | 10.00 (diverged, loss 1e18-1e32) | CONFIRMED |
| Wrong f: Multi-Krum 51.65 / 84.25, Trimmed mean 17.61 / 80.08 | README.md:27 | Identical after full rerun | CONFIRMED |
| Rejection: Gaussian 100%, sign-flip 90.8% to 99.8% | README.md:24 | Identical; per-run minimum under sign-flip is 77.0% | CONFIRMED (as means only) |
| All tables in README section 3 (lines 34-83) | README.md:20-83 | Regenerated block identical (CRLF only differs) | CONFIRMED |
| INT8 vs Float32 difference smaller than one std for every aggregator | README.md:45 (text typed in experiments/make_report_assets.py:637) | Holds for current data (largest: median -0.23 vs std 0.62). Paired test for median at 30% sign-flip: -0.26 +- 0.10 (95% CI), significant | CONFIRMED as stated; hand-typed, not data-checked |
| "Stochastic truncation noise" | report/edgefed_bench_v2.typ:129 | Rounding is deterministic (engine/client.py:620-635) | MISMATCH |
| "Exact bit-level determinism on CPU" | report/edgefed_bench_v2.typ:163 | Holds only at a fixed thread count (10 vs 2 threads differ) | MISMATCH (conditional) |
| "Across the 240 attacked configurations" INT8 changed Multi-Krum accuracy... | report/edgefed_bench_v2.typ:175 | The comparison uses only the 20% sign-flip cell (10 runs) | MISMATCH |
| Table 4 "averaged across all 50 rounds and 5 seeds" | report/edgefed_bench_v2.typ:227 | Query pools INT8 and Float32 (10 runs per cell) (experiments/make_report_assets.py:379-391) | MISMATCH (incomplete description) |
| "26 unit tests across 5 test suites" | report/edgefed_bench_v2.typ:256 (also in the PDF) | 31 tests | MISMATCH (stale) |
| "Severe" non-IID label skew at alpha 0.5 | README.md:16; report/edgefed_bench_v2.typ:131 | Most clients hold 9 or 10 of 10 classes | MISMATCH (overstated) |
| Uplink "up to an order of magnitude slower" | README.md:11 | Cited to Konecny et al. 2016; not checked against the source | UNVERIFIED |
| "Sub-second access across 60,000 samples" | README.md:110 | Not measured anywhere in the repo | UNVERIFIED |
| Telemetry includes "mitigation status" | README.md:114 | Only `simulate.py` computes it; `global_eval` has no such column | MISMATCH |
| CPU-only "cross-platform numerical reproducibility" | results/ENVIRONMENT.md:11 | Not tested across platforms; thread count alone changes results | UNVERIFIED |
| Library versions match the environment | results/ENVIRONMENT.md | All listed versions match pip freeze; torchvision 0.29.1 missing | CONFIRMED with omission |
| Projected grid time | results/ENVIRONMENT.md:36-37 | Covers 280 runs only (13,619 s). Stored DB: 2.415 h grid + 0.439 h wrong-f | MISMATCH (incomplete) |
| One seed controls partition, init, Byzantine IDs, Gaussian noise, shuffling | report/edgefed_bench_v2.typ:163 | True in run_grid path. In `simulate.py` the `--seed` is ignored for data (bug B1) | CONFIRMED for grid; MISMATCH for simulate.py |
| Server is told f except in wrong-f experiment; no DP | README.md:152-153; report/edgefed_bench_v2.typ:242 | Matches code | CONFIRMED |

## 3. Test results

- Command: `python -m unittest discover tests -v` (Python 3.11.9, torch 2.14.0, Windows 11).
- Result: Ran 31 tests, OK. This matches `results/test_run.txt` (31 tests, OK).
- Specification tests present and passing: trimmed mean hand-computed example (tests/test_adversary_aggregator.py:133), k = 0 equals mean (:149), ignores outlier (:155), invalid k (:169), Multi-Krum invalid f raises (:177), seed determinism (tests/test_simulation.py:126), five seed factors (tests/test_simulation.py:187).
- Limits: the tests write into `data/telemetry.duckdb` and need prebuilt data or a download. The five-factor test checks each factor in isolation, not that one end-to-end seed controls all five together.
- Full grid rerun: 320/320 runs and 16,000/16,000 round rows identical to `results/results.duckdb` (default 10 threads, 4 parallel processes, 9.37 h total CPU wall time).
- `simulate.py` (5 rounds, seed 42): deterministic across two runs. Values given in section 2.

## 4. Bugs and risks

| ID | File:line | Issue | Severity |
| :--- | :--- | :--- | :--- |
| B1 | simulate.py:266-281 | `--seed` is ignored for data. The rebuild check looks at `data/clients`, but `EdgeStore` is built without a seed, so it always reads `data/clients`. With seed not equal to 42 on a clean clone, shards go to `clients_seed_{seed}` and are never read. | High |
| B2 | simulate.py:176-186 | Mitigation for median and trimmed mean only checks that the output is finite, so it is always 100%. FedAvg reports 0.0% because it is never evaluated. The metric is meaningless for three of four aggregators. | High |
| B3 | engine/storage.py:391-465 | `initialize_storage_pipeline` skips the rebuild when shards exist, regardless of alpha. Changing alpha silently reuses the old partition. | High |
| B4 | experiments/run_grid.py:39-56 | The config hash excludes lr, batch_size and momentum. Changing them reuses old results as if they were new. | Medium |
| B5 | experiments/run_grid.py:470-475 | `num_clients=20, alpha=0.5` are hard-coded instead of read from experiments/config.yaml. | Medium |
| B6 | engine/client.py:586; experiments/config.yaml | Momentum in the config is unused; 0.9 is hard-coded. The optimizer is recreated every round, so momentum resets each round. | Medium |
| B7 | engine/adversary.py (attack applied after quantization) | In INT8 mode Gaussian attackers send unquantized float noise, which is not a valid INT8 payload. Uplink bytes count attackers as honest INT8 payloads. | Medium |
| B8 | Determinism (torch threads) | Results depend on the torch thread count. Not recorded in ENVIRONMENT.md or the DB. | Medium |
| B9 | engine/storage.py:71-122 | Dirichlet partition has no minimum client size guarantee. | Low |
| B10 | engine/storage.py:184-200 | `data/test.parquet` is shared by all seeds; seed 42 uses `data/clients`, other seeds `clients_seed_{seed}`. Inconsistent layout. | Low |
| B11 | experiments/make_report_assets.py:350-355, :637 | Hand-typed interpretive claims ("optimal rejection envelope", "smaller than one standard deviation") are not computed from data and can become false silently. | Low |
| B12 | experiments/make_report_assets.py:379-391 | Table 6 query does not filter by precision. | Low |
| B13 | engine/aggregator.py:148-175 | FedAvg is unweighted (not by client sample count). This is stated in the report (line 157) but differs from standard FedAvg. | Low |
| B14 | report/edgefed_bench_v2.typ:129, :163, :175, :227, :256 | Wrong or stale prose (see section 2). The PDF still says "26 unit tests". | Low |

## 5. Gaps that would stop this from becoming a research paper (priority order)

1. Headline numbers. The 74.60% / 100% figures are from old code and 5 rounds, before convergence (clean accuracy at round 5 is about 77.6%, still rising at round 50). The paper must use the current 50-round grid, and convergence should be shown or more rounds run.
2. Statistics. 5 seeds with mean +- std only. No confidence intervals, no paired tests, no effect sizes. FedAvg at 20% sign-flip is bimodal (seeds 0 and 3 about 10%, others 52-68%), so mean +- std misrepresents it. Clean f = 0 differences of up to about 0.2 points come from summation order alone, which sets the noise floor.
3. Scope. One small MLP on one dataset (Fashion-MNIST), one alpha (0.5, which is mild: most clients hold 9-10 classes), two simple attacks. No CNN, no second dataset, no adaptive attacks (for example ALIE or IPM), no alpha sweep.
4. Methodology. No validation split; hyperparameter source: Not found in available documents. Momentum resets every round. Unweighted FedAvg. Gaussian attackers bypass INT8.
5. Related-work positioning. The claim that INT8 does not hurt Byzantine robustness needs comparison with prior work on quantized or compressed robust aggregation. Not found in available documents.
6. Theory. The bound argument in 8e75282's README (`||e||_inf <= S/2`) is not in the current paper as a proven statement or checked empirically against measured distances.
7. Reproducibility notes. The thread-count dependence, torchvision version and the B1/B3 data bugs must be fixed or documented.
8. Build specification `01_EdgeFed_Bench_v2_Build_Spec.md`: Not found in available documents, so compliance with it could not be checked.

## 6. Unfinished items from the handoff (handoffs/01_EdgeFed_Bench_handoff.md)

| Handoff task | Status found |
| :--- | :--- |
| Task 1: tests pass, required tests exist | Done. 31 tests pass. The five-factor test exists but checks factors in isolation. |
| Task 2: recompile report | Partly done. PDF is 4 pages with Segoe UI Variable; template decision is in docs/REPORT_NOTES.md. Not done: hand-typed or stale prose remains (typ:129, :163, :175, :227, :256). |
| Task 3: ENVIRONMENT.md matches pip freeze | Partly done. Versions match; torchvision missing; projected time covers 280 runs only; cross-platform claim untested. |
| Task 4: README check | Partly done. Generated block is fresh. Not done: hand-written claims outside the block (README.md:11, :16, :110, :114). Section numbering differs (reproduction is section 7, not 6). Clean-clone commands were not run in a fresh virtual environment in this audit: UNVERIFIED. |
| Task 5: publish | `.gitignore` is correct (keeps results/results.duckdb). Remote origin exists and main equals origin/main. Whether the repo is public and renders correctly was not checked (no network use allowed): UNVERIFIED. |

## 7. Status after fixes (2026-11-19, branch fix/audit-bugs, uncommitted)

Sections 1 to 6 describe commit 293d825. This section records what has changed since.

| ID | Status |
| :--- | :--- |
| B1 | Fixed. `simulate.py --seed` selects its own shard directory. |
| B2 | Fixed. Mitigation is reported for Multi-Krum only; other aggregators show n/a. |
| B3 | Fixed. Shards record alpha, seed and client count; a mismatch triggers a rebuild. |
| B4 | Fixed. lr, batch size and momentum enter the config hash when they differ from the stored values. |
| B5 | Fixed. Client count and alpha come from the config. |
| B6 | Fixed. Momentum comes from the config. Resetting it each round is intended and documented. |
| B7 | Fixed. In INT8 runs, attacker updates are now encoded as INT8, so byte counts are correct. The 160 INT8 attacked runs were rerun. |
| B8 | Mitigated. `run_grid.py --threads` sets the thread count; the stored runs used 10. |
| B9 | Fixed. A split that leaves a client empty raises an error. The current smallest client holds 409 samples. |
| B10 | Not changed. Seed 42 still uses `data/clients`; changing it would only move folders. |
| B11 | Fixed. The one-standard-deviation sentence is computed; the wrong-f notes describe the setting only. |
| B12 | Not changed. Table 6 still pools INT8 and Float32; the caption now says so. |
| B13 | Not changed. FedAvg stays unweighted; the README and report say so. |
| B14 | Fixed. Report prose corrected. |
| New | Fixed. On a fresh clone 5 tests failed because test setup skipped building the default shards. A clean copy now passes all 38 tests. |
| New | Fixed. `requirements.txt` now pins the exact versions used. |

Effect of the B7 rerun: 122 of the 160 rerun runs changed, while the other 160 runs are bit-identical. Most mean changes are under 0.6 points. The exception is the unstable wrong-f setting. There, trimmed mean at assumed f = 2 moved from 17.61% to 21.62%, and one seed moved by 23.6 points. Multi-Krum at 20% sign-flip with INT8 is now 84.87 +- 0.28%, against 84.89 +- 0.24% with Float32.

Removed: `verify_storage.py`, whose checks the storage tests cover, and `data/telemetry_time.duckdb`, an orphan file no code references.

Still open: every gap in section 5.
