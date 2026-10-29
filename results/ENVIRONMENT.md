# Experimental Environment and Hardware Specification

This document details the exact host environment, hardware profile, system software versions, and benchmark projection for EdgeFed-Bench v2.

---

## 1. Hardware Specification

- **Host Processor (CPU):** 13th Gen Intel(R) Core(TM) i7-13620H (10 physical cores, 16 logical threads, base clock 2.40 GHz, boost up to 4.90 GHz)
- **Host Memory (RAM):** 16.00 GB DDR5
- **Accelerator (GPU):** None used. All experiments execute strictly on CPU to guarantee bit-level determinism and cross-platform numerical reproducibility.
- **Operating System:** Microsoft Windows 11 Home Single Language (Version 10.0.26300, Build 26300)
- **Architecture:** x86_64

---

## 2. Software Runtime and Library Versions

- **Python Runtime:** Python 3.11.9 (`C:\Users\Rajch\AppData\Local\Programs\Python\Python311\python.exe`)
- **PyTorch:** 2.14.0 (CPU build)
- **DuckDB:** 1.5.6
- **PyArrow:** 26.0.0
- **NumPy:** 2.2.3
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
- **Decision on Round Count:** Because the projected duration of 3.78 hours is substantially below the 24-hour threshold, the full protocol of **50 rounds** is retained without reduction.
