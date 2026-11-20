# Roadmap

Status of the items from the audit in `AUDIT_REPORT.md`. Every item below was worked on in version 2.1; items 2, 6, 8 and 12 were finished on 2026-11-20 and merged into `main`. Results are in `results/results.duckdb`, the README results block and `report/edgefed_bench_v2.pdf`.

## Needed before writing the paper

1. **More seeds and proper statistics.** Done.
   - The main grid and the wrong-f experiment now use 10 seeds (0 to 9).
   - `experiments/stats.py` gives paired 95% confidence intervals, exact permutation p-values and Holm correction within each aggregator. Results: README 3.6, report Section 3.3.
   - The equivalence margin of 0.5 points was chosen after seeing seeds 0 to 4. Seeds 5 to 9 are reported separately as a confirmation.
2. **Validation split and hyperparameter source.** Done.
   - 10,000 training images are held out as a validation split. 20 settings of learning rate, momentum and batch size were scored on it for clean FedAvg (README 3.14, report Section 4.7).
   - The fixed setting ranks 6 of 20. The best setting (lr 0.05, momentum 0.9, batch 64) was rerun for all four aggregators, clean, 20% sign-flip and 20% ALIE, Float32 and INT8, seeds 0 to 4: 120 runs (README 3.15, report Section 4.8). The largest INT8 minus Float32 difference for a robust aggregator is 0.38 points.
   - Where the original values came from: Not found in available documents.
3. **Measure the mechanism.** Done.
   - Every attacked run added in version 2.1 logs honest-to-honest distance, honest-to-attacker distance and the codec error per round (`round_mechanism` table; README 3.7, report Section 3.4).
4. **A stronger attack.** Done.
   - ALIE and IPM, both omniscient (README 3.8, report Section 4.1).

## Would strengthen the paper

5. **Alpha sweep.** Done for alpha 0.1 and 1.0 (README 3.9, report Section 4.2).
   - Correction to the earlier note: with a 1% threshold, clients at alpha 0.5 hold on average 7.9 classes (seeds 0 to 4), not 9 or 10. The `partition_stats` table holds the measured values.
6. **Second model or dataset.** Done.
   - A small CNN on Fashion-MNIST, with 3 seeds and 30 rounds (README 3.11, report Section 4.4).
   - CIFAR-10 with a 30,138-parameter CNN, 3 seeds and 50 rounds: 72 runs (README 3.12, report Section 4.5). Robust-aggregator INT8 differences reach 1.10 points but cannot be tested with 3 seeds.
7. **Lower bit widths.** Done: INT4 and 1-bit sign, without error feedback (README 3.10, report Section 4.3).
8. **Training to convergence.** Done as a check: 150 rounds for the three robust aggregators (README 3.13, report Section 4.6).
   - Convergence criterion: the gain over the last 50 rounds (round 100 to 150). It is +0.51 to +0.78 points, against +1.66 to +2.23 from round 50 to 150, so training is slowing but has not converged. The main grid stays at 50 rounds.
9. **Related work.** Done in `docs/RELATED_WORK.md` and report Section 5.

## Small cleanups

10. Seed 42 shards moved to `data/clients_seed_42`. Done.
11. Table 6 (Multi-Krum rejection) split by precision. Done.
12. CI workflow added in `.github/workflows/tests.yml`. Done. Its two steps (unit tests and the config dry run) pass locally and on GitHub Actions.
13. PyTorch thread count stored in the `torch_threads` column. Done.

## Open items

- More seeds on CIFAR-10, where the INT8 differences are larger but untested.
- Tuning per aggregator and attack.
- Error feedback for INT4 and Sign.
- Testing why coordinate median loses accuracy under INT8, and much more under INT4.
- More seeds for the extended studies. With 5 seeds the smallest possible p-value is 0.0625.
