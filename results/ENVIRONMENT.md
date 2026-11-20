# Experimental Environment and Hardware Specification

This document details the exact host environment, hardware profile, system software versions, and benchmark projection for EdgeFed-Bench v2.

---

## 1. Hardware Specification

- **Host Processor (CPU):** 13th Gen Intel(R) Core(TM) i7-13620H (10 physical cores, 16 logical threads, base clock 2.40 GHz, boost up to 4.90 GHz)
- **Host Memory (RAM):** 16.00 GB DDR5
- **Accelerator (GPU):** None used. All experiments execute on CPU. Runs are bit-for-bit repeatable on this machine at a fixed PyTorch thread count (default here: 10 threads, `torch.get_num_threads()`). A different thread count changes the floating-point summation order and the results slightly (for example 85.23% at 10 threads and 85.21% at 2 threads for Multi-Krum, 20% sign-flip, INT8, seed 0). Reproducibility on other CPUs or operating systems has not been tested.
- **Operating System:** Microsoft Windows 11 Home Single Language (Version 10.0.26300, Build 26300)
- **Architecture:** x86_64

---

## 2. Software Runtime and Library Versions

- **Python Runtime:** Python 3.11.9 (`C:\Users\Rajch\AppData\Local\Programs\Python\Python311\python.exe`)
- **PyTorch:** 2.14.0 (CPU build)
- **torchvision:** 0.29.1
- **DuckDB:** 1.5.6
- **PyArrow:** 26.0.0
- **NumPy:** 2.4.6
- **Matplotlib:** 3.11.2
- **PyYAML:** 6.0.3
- **Typst Compiler:** Typst 0.15.1 (`9dfd3a08`)

---

## 3. Benchmark Grid Profiling and Projection

A single-run benchmark profile consisting of 50 communication rounds across 20 edge clients (batch size 128, local SGD with learning rate 0.05 and momentum 0.9, INT8 quantization, Multi-Krum defense under 20% sign-flip Byzantine attackers) was executed on seed 0:

- **Measured Single Run Duration (50 rounds):** 48.64 seconds
- **Mean Duration per Communication Round:** 0.973 seconds/round
- **Total Configurations in Benchmark Grid:** 280 runs (40 clean runs + 240 attacked runs across 5 seeds)
- **Projected Total Grid Wall-Clock Time:** 13,619.2 seconds (approximately 3.78 hours)
- **Wrong-f Experiment:** 40 further runs (not included in the projection above).
- **Measured Wall-Clock Time (from `wall_clock_sec` in `results/results.duckdb`):** 4.594 hours summed over the 280-run grid and 1.183 hours over the 40 wrong-f runs.
  The 160 INT8 attacked runs (120 grid, 40 wrong-f) were rerun on 2026-11-19 after attacker updates were made valid INT8 payloads. They ran four processes at a time at 10 threads each, so their per-run times are inflated by CPU sharing. The other 160 runs were not rerun; they take 1.363 hours in total.
- **Thread Count:** the 320 runs above (seeds 0 to 4 of the grid and wrong-f experiment) used 10 PyTorch threads. `experiments/run_grid.py --threads 10` reproduces them.
- **Decision on Round Count:** Because the projected duration of 3.78 hours is substantially below the 24-hour threshold, the full protocol of **50 rounds** is retained without reduction.

---

## 4. Extended Study (2026-11-19 to 2026-11-20)

The 1,112 runs added in version 2.1 (grid and wrong-f seeds 5 to 9, and the six extended studies) ran as 8 parallel processes with 2 PyTorch threads each, 139 runs per process. Each run's thread count is stored in the `torch_threads` column of `runs`.

- **Elapsed time:** 15,148 seconds (about 4.2 hours) for the slowest process, from its log.
- **Summed run time (from `wall_clock_sec`):** 33.272 hours over the 1,112 runs. Because 8 processes shared the CPU, per-run times are longer than a run would take alone.
- **Mean seconds per round, by experiment:**

| Experiment | Runs | Summed hours | Seconds per round |
| :--- | :--- | :--- | :--- |
| grid (seeds 5 to 9) | 280 | 5.055 | 1.300 |
| wrong_f (seeds 5 to 9) | 40 | 0.743 | 1.338 |
| strong_attacks | 240 | 5.478 | 1.644 |
| alpha_sweep | 240 | 6.315 | 1.894 |
| bit_width | 120 | 3.235 | 1.941 |
| long_run | 60 | 4.519 | 1.807 |
| tuning | 60 | 1.703 | 2.043 |
| cnn | 72 | 6.225 | 10.374 |

- **CNN cost:** 10.374 seconds per round against 1.664 for the MLP at the same thread count and load.
- **Reproducing these runs:** use `--threads 2`. Results at 10 threads will differ slightly in the last digits.

### 4.1 CIFAR-10 and the rerun at the best setting (2026-11-20)

The 192 runs of the `cifar10` and `tuned` experiments ran as 8 parallel processes with 2 PyTorch threads each. 47 `tuned` runs finished in a first launch; the other 145 ran in a second launch, which took 7,566 seconds (about 2.1 hours) for the slowest process, from its log.

| Experiment | Runs | Summed hours | Seconds per round |
| :--- | :--- | :--- | :--- |
| tuned | 120 | 3.051 | 1.831 |
| cifar10 | 72 | 14.591 | 14.591 |

- **CIFAR-10 cost:** 1.4 times the seconds per round of the Fashion-MNIST CNN at the same thread count.
- **Download:** the first CIFAR-10 run downloads the dataset (about 170 MB) into `data/cifar10/`.

