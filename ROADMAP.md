# Roadmap

What remains before this project can support a research paper, most important first. The reasons come from the audit in `AUDIT_REPORT.md` and the current results in `results/results.duckdb`.

The full grid currently takes about 2.5 hours of CPU time. Items 1, 4, 5 and 7 each multiply that cost.

## Needed before writing the paper

1. **More seeds and proper statistics.**
   - Five seeds is too few for the unstable settings: FedAvg at 20% sign-flip is 41.67 +- 27.62%, and trimmed mean at wrong f = 2 is 21.62 +- 14.02%.
   - Move to 10 or more seeds.
   - Add paired tests and 95% confidence intervals for INT8 versus Float32 (same seed, same configuration), so the "smaller than one standard deviation" statement becomes a statistical claim.
2. **Validation split and hyperparameter source.**
   - The source of lr 0.05, momentum 0.9 and batch size 128: Not found in available documents.
   - Either hold out a validation set and tune on it, or cite where these values come from.
3. **Measure the mechanism.**
   - The report's hypothesis is that INT8 error is small next to the distance between honest and attacker updates. That distance was never measured.
   - Log, per round, the INT8 error norm, the mean honest-to-honest distance and the mean honest-to-attacker distance.
4. **A stronger attack.**
   - Sign-flip and Gaussian are easy for Multi-Krum to detect.
   - Add one attack designed to stay close to honest updates, such as A Little Is Enough (ALIE) or Inner Product Manipulation (IPM). This is where INT8 noise is most likely to change the outcome.

## Would strengthen the paper

5. **Alpha sweep.** Try alpha 0.1, 0.5 and 1.0. At alpha 0.5 most clients hold 9 or 10 of the 10 classes, so the skew is mild.
6. **Second model or dataset.** For example a small CNN on CIFAR-10.
7. **Lower bit widths.** 4-bit and sign-only updates, to find where quantization starts to hurt the defences.
8. **Training to convergence.** Accuracy is still rising at round 50. Run longer, or report a convergence criterion.
9. **Related work.** Position the result against prior work on Byzantine-robust aggregation with compressed or quantized updates. Not found in available documents.

## Small cleanups

10. Store seed 42 shards in `data/clients_seed_42` like the other seeds, instead of `data/clients`.
11. Split Table 6 (Multi-Krum rejection) by precision instead of pooling INT8 and Float32.
12. Add a CI workflow that runs the test suite on every push.
13. Record the PyTorch thread count in the `runs` table, not only in `results/ENVIRONMENT.md`.

## Suggested order

Item 3 first, then item 4. Both are cheap and are the first things a reviewer will ask about. Then item 1, which needs the most compute.
