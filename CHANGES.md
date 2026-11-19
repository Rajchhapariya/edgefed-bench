# Changes in the Working Tree

Branch `fix/audit-bugs`, based on commit 293d825. Nothing is committed.

The working tree holds two sets of changes:

- **Bug fixes** that you asked for after the audit. These change code logic.
- **Prose rewrite**, which is wording only.

The bug-fix diff is also saved on its own as a patch (text files only) at `C:\Users\Rajch\AppData\Local\Temp\claude\937fce3835b969ce\58d10613-5593-44b1-bbea-c3146da705c2\scratchpad\bugfix.patch`. That folder can be cleared by a reboot.

## How we checked them

- **Results unchanged.** Hashes, metrics, tables and the README numbers all match the stored results.
  - All 320 config hashes still match `results/results.duckdb`.
  - `report/metrics.typ` and `report/summary_metrics.json` regenerate byte for byte.
  - Every number in the README results block is unchanged.
  - One grid run (Multi-Krum, 20% sign-flip, INT8, seed 0) reproduces its stored accuracy exactly. `simulate.py` at seed 42 gives the same accuracies as before.
- **Rewrite is wording only.**
  - The code's syntax tree with docstrings removed is identical before and after the rewrite in every Python file.
  - The one exception is `experiments/make_report_assets.py`, which differs only in string literals.
- **Tests.** The suite passed before the rewrite (36 tests, OK) and after it (36 tests, OK).
- **Report.** `report/edgefed_bench_v2.pdf` is still 4 pages.

## Second round: fixes, cleanup and partial grid rerun

| File | Change |
| :--- | :--- |
| engine/adversary.py | New `wire_int8` option. In INT8 runs, attacker updates are quantized and reconstructed so they are valid INT8 payloads. |
| experiments/run_grid.py, simulate.py | Pass `wire_int8` for INT8 runs. `run_grid.py` gains `--threads`. |
| engine/storage.py | Raises an error if a Dirichlet split leaves a client with no samples. |
| requirements.txt | Pinned to the exact versions used for the stored results. |
| tests/test_grid_runner.py, test_simulation.py, test_storage.py, test_quantization_client.py | Test setup now always calls the storage pipeline, which skips work when shards exist. Before, 5 tests failed on a fresh clone. |
| tests/test_audit_fixes.py | Two more tests: attackers send valid INT8, and an empty client raises. |
| results/results.duckdb | The 160 INT8 attacked runs (120 grid, 40 wrong-f) were replaced with reruns under the INT8 fix. The file is compacted. The other 160 runs are unchanged. |
| README.md, report/*, results/figures/* | Regenerated from the updated database. The report says attacker updates are INT8 in INT8 runs, and the commands use `--threads 10`. |
| results/ENVIRONMENT.md | Wall-clock times recomputed, with a note on the rerun and the thread count. |
| results/test_run.txt | Fresh test output: 38 tests, OK. |
| AUDIT_REPORT.md | New section 7 gives the status of each finding after the fixes. |
| verify_storage.py | Removed. Its checks are covered by tests/test_storage.py. |
| data/telemetry_time.duckdb | Removed. It is an untracked orphan; a copy is in the scratchpad `removed` folder. |

A copy of the database from before the rerun is at `C:\Users\Rajch\AppData\Local\Temp\claude\937fce3835b969ce\58d10613-5593-44b1-bbea-c3146da705c2\scratchpad\removed\results_before_int8_fix.duckdb`.

## Bug fixes (code logic)

| File | Change |
| :--- | :--- |
| simulate.py | `--seed` now selects the matching shard directory. Mitigation rate is reported only for Multi-Krum and shows n/a for aggregators that do not select clients. |
| engine/storage.py | Shards record their alpha, seed and client count in `_partition_meta.json`. A mismatch triggers a rebuild, and stale shards are removed first. |
| engine/client.py | `EdgeClient` takes a `momentum` argument (default 0.9) instead of a hard-coded value. |
| experiments/run_grid.py | lr, batch size and momentum enter the config hash when they differ from the stored grid's values. Client count and alpha come from the config. Momentum is passed to clients. |
| experiments/make_report_assets.py | The "smaller than one standard deviation" sentence is computed from the data. The wrong-f notes describe the setting instead of an outcome. The Table 6 caption states that INT8 and Float32 are pooled. |
| results/ENVIRONMENT.md | Adds torchvision 0.29.1, the thread-count dependence of determinism and the measured grid and wrong-f wall-clock times. Removes the untested cross-platform claim. |
| tests/test_simulation.py | The median test now expects no mitigation rate, where it used to expect the old always-100% value. |
| tests/test_audit_fixes.py | New. Five regression tests for the hash, rebuild and momentum fixes. |
| AUDIT_REPORT.md | New. Audit findings for commit 293d825. |

## Prose rewrite (wording only)

| File | Change |
| :--- | :--- |
| README.md | Rewrote the intro, research question, architecture, methods, limitations and reproduction text in plain language. Limitations now state 5 seeds, no significance tests and that `simulate.py` defaults to 5 rounds. Removed an unsupported comparison with other simulators. The results block was regenerated from the script. |
| docs/REPORT_NOTES.md | Rewrote the three reasons for not importing the shared template in short sentences. |
| report/edgefed_bench_v2.typ | Rewrote sections 1 to 5 in first person plural. Each results section now opens with its number, and every number comes from `metrics.typ` or `tables.typ`. Adds that Gaussian attacks bypass INT8 and that FedAvg is unweighted. Removed "severe", "stochastic" and the stale test count. |
| report/edgefed_bench_v2.pdf | Recompiled from the edited `.typ`. Still 4 pages. |
| report/tables.typ | Regenerated. The header comment and wrong-f column title changed; values are the same. |
| experiments/make_report_assets.py | Plainer wording for the README key findings, section headings and captions, plus docstrings. Only string literals changed. |
| AUDIT_REPORT.md | Tightened the header and the summary verdict. Findings are unchanged. |
| engine/__init__.py | Plainer module docstring. |
| engine/adversary.py | Docstrings say what each attack does, without "realistic" or "variance explosion". |
| engine/aggregator.py | Module docstring now lists trimmed mean and says FedAvg is unweighted unless weights are given. Removed comments that restated code. |
| engine/client.py | Docstrings shortened. Comments now explain why the optimizer and loader are rebuilt each round. Removed comments that restated code. |
| engine/quantization.py | Removed "mathematically verified". Shortened the payload docstrings and the unused-`scale` comment. |
| engine/storage.py | Docstrings shortened, and "severe" removed from the alpha description. Removed comments that restated code. |
| experiments/run_grid.py | Plainer module and function docstrings. Removed comments that restated code. |
| simulate.py | Module docstring now lists the four regimes it actually runs and says the published results come from `run_grid.py`. Removed comments that restated code. |
| verify_storage.py | Plainer docstring. Removed one comment that restated code. |
| tests/test_grid_runner.py | Removed one comment that restated code. |
| tests/test_quantization_client.py | Plainer docstring. Removed two comments that restated code. |
| tests/test_simulation.py | Removed three comments that restated code. |
| tests/test_storage.py | Plainer docstring for the non-IID test. |

## Not changed

- Section 4 of the README keeps its heading, because `update_readme_results` falls back to it as a marker.
- The `+-` notation in generated tables is unchanged, because it is produced by the table code.
- The software design specification `01_EdgeFed_Bench_v2_Build_Spec.md` was not available: Not found in available documents.
