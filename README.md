# EdgeFed-Bench: Byzantine-Robust Federated Learning with DuckDB-Partitioned Edge Stores and Quantized Gradients

A simulated federated learning benchmark. It measures whether INT8 compression of client updates weakens Byzantine-robust aggregation. Client data is stored as Parquet shards and read through DuckDB.

---

## 1. Problem and Research Question

Edge clients face two problems at once.

1. **Uplink is scarce.** Mobile and IoT clients often send over links that are up to an order of magnitude slower than the server's downlink (Konečný et al., 2016).
2. **Plain FedAvg trusts every client.** A few faulty or malicious clients can send updates that stop the model from converging (Blanchard et al., NeurIPS 2017).

Quantizing updates to INT8 helps with the first problem, but it adds rounding error to every update. Byzantine-robust aggregators work by spotting updates that look unlike the others, so that error could in principle hide attackers or make honest clients look like outliers.

> **Research question:** does INT8 uniform symmetric quantization of client updates reduce the accuracy or the attacker detection of Byzantine-robust aggregators (Multi-Krum, coordinate-wise median, trimmed mean) under moderate non-IID Dirichlet label skew ($\alpha = 0.5$)?

---

<!-- RESULTS:START -->
## 2. Key Findings (generated from the database)

- **Uplink:** INT8 cuts each client update from 397.5 KB to 99.4 KB per round, a 75.0% reduction. A 50-round run with 20 clients sends 97.1 MB instead of 388.2 MB.
- **Accuracy with INT8:** at 20% sign-flip, Multi-Krum ends at 84.87 +- 0.28% with INT8 and 84.89 +- 0.24% with Float32. Plain FedAvg ends at 10.00% at 30% sign-flip, which is chance level for 10 classes.
- **Defences at 30% sign-flip (INT8):** Multi-Krum 84.79 +- 0.45%, trimmed mean 80.21 +- 0.40%, coordinate median 79.99 +- 0.47%.
- **Multi-Krum attacker rejection (INT8 and Float32 pooled):** 100.0% in the worst Gaussian run. Under sign-flip the mean over seeds is between 90.8% and 99.8%, depending on attacker share, so some sign-flip updates are accepted.
- **Wrong assumed f (actual f = 6):** at assumed f = 2, Multi-Krum ends at 52.12% and trimmed mean at 21.62%. At assumed f = 8 they end at 84.21% and 80.13%.

## 3. Results by Configuration (50 Rounds, 20 Clients, $\alpha = 0.5$, 5 Seeds)

### 3.1. Final Accuracy by Attack and Aggregator (INT8)
Final test accuracy, mean $\pm$ standard deviation over seeds 0 to 4:

| Attack | Share | FedAvg | Multi-Krum | Coordinate Median | Trimmed Mean |
| :--- | :--- | :--- | :--- | :--- | :--- |
| None (Clean) | 0% | 85.30 +- 0.31% | 85.35 +- 0.25% | 84.65 +- 0.44% | 85.36 +- 0.27% |
| Gaussian | 10% | 48.22 +- 3.36% | 85.30 +- 0.39% | 84.54 +- 0.34% | 85.07 +- 0.40% |
| Gaussian | 20% | 41.74 +- 5.94% | 85.17 +- 0.32% | 84.57 +- 0.36% | 84.89 +- 0.23% |
| Gaussian | 30% | 43.00 +- 3.88% | 85.13 +- 0.29% | 84.67 +- 0.53% | 84.85 +- 0.40% |
| sign-flip | 10% | 78.44 +- 4.01% | 85.28 +- 0.38% | 83.97 +- 0.71% | 84.07 +- 0.63% |
| sign-flip | 20% | 41.67 +- 27.62% | 84.87 +- 0.28% | 83.03 +- 0.67% | 83.01 +- 0.44% |
| sign-flip | 30% | 10.00 +- 0.00% | 84.79 +- 0.45% | 79.99 +- 0.47% | 80.21 +- 0.40% |

### 3.2. Float32 versus INT8 at 20% Sign-Flip
INT8 cuts uplink by 75.0%. The Float32 to INT8 accuracy difference is smaller than one standard deviation for every aggregator in this table:

| Aggregator | Float32 Acc | INT8 Acc | Acc Delta | Float32 Uplink (KB) | INT8 Uplink (KB) | Bandwidth Cut |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FedAvg | 40.51 +- 28.25% | 41.67 +- 27.62% | +1.15% | 397.5 KB | 99.4 KB | -75.0% |
| Multi-Krum | 84.89 +- 0.24% | 84.87 +- 0.28% | -0.02% | 397.5 KB | 99.4 KB | -75.0% |
| Coordinate median | 83.21 +- 0.62% | 83.03 +- 0.67% | -0.18% | 397.5 KB | 99.4 KB | -75.0% |
| Trimmed mean | 82.92 +- 0.55% | 83.01 +- 0.44% | +0.09% | 397.5 KB | 99.4 KB | -75.0% |

### 3.3. Convergence and Attacker Share

<p align="center">
  <img src="results/figures/fig1_convergence_20pct_signflip.png" alt="Convergence Trajectory" width="48%" />
  <img src="results/figures/fig2_accuracy_vs_attacker_share.png" alt="Accuracy vs Attacker Share" width="48%" />
</p>

### 3.4. Wrong Assumed $f$ (Actual $f=6$, 30% Attackers, Sign-Flip, INT8)
The server is told a number of attackers $f$ that differs from the actual 6:

| Assumed f | Assumed Ratio (f/N) | Multi-Krum Final Acc | Trimmed Mean Final Acc | Setting |
| :--- | :--- | :--- | :--- | :--- |
| 2 | 2/20 (10%) | 52.12 +- 8.51% | 21.62 +- 14.02% | Underestimates attackers (assumed f < actual f = 6) |
| 4 | 4/20 (20%) | 81.85 +- 2.21% | 77.09 +- 2.96% | Underestimates attackers (assumed f < actual f = 6) |
| 6 | 6/20 (30%) | 84.79 +- 0.45% | 80.21 +- 0.40% | Assumed f equals actual f = 6 |
| 8 | 8/20 (40%) | 84.21 +- 0.92% | 80.13 +- 0.07% | Overestimates attackers (assumed f > actual f = 6) |

### 3.5. Multi-Krum Rejection Rates
Attacker rejection and honest false-rejection rates, averaged over all 50 rounds, pooled over INT8 and Float32 (10 runs per cell):

| Attack Vector | Attacker Share | Attacker Rejection Rate (%) | Honest False-Rejection Rate (%) |
| :--- | :--- | :--- | :--- |
| Gaussian | 10% | 100.00 +- 0.00% | 0.00 +- 0.00% |
| Gaussian | 20% | 100.00 +- 0.00% | 0.00 +- 0.00% |
| Gaussian | 30% | 100.00 +- 0.00% | 0.00 +- 0.00% |
| sign-flip | 10% | 99.80 +- 0.42% | 0.02 +- 0.05% |
| sign-flip | 20% | 90.80 +- 11.66% | 2.30 +- 2.92% |
| sign-flip | 30% | 95.50 +- 4.74% | 1.93 +- 2.03% |

<!-- RESULTS:END -->

## 4. Systems & Data Engineering Architecture

```
edgefed-bench/
├── requirements.txt
├── simulate.py                   # Quick 4-regime check with telemetry logging
├── data/                         # Parquet client shards, test split, telemetry (generated)
├── engine/
│   ├── storage.py                # Dirichlet partitioner, Snappy Parquet I/O, DuckDB EdgeStore
│   ├── quantization.py           # Uniform symmetric INT8 quantizer and SQNR analyzer
│   ├── client.py                 # EdgeClassifier model and local SGD
│   ├── adversary.py              # Sign-flip and Gaussian attacks
│   └── aggregator.py             # FedAvg, Coordinate Median, Trimmed Mean, Multi-Krum
├── experiments/
│   ├── config.yaml               # Grid and wrong-f definitions
│   ├── run_grid.py               # Resumable grid runner, writes results/results.duckdb
│   └── make_report_assets.py     # Figures, README tables, report data from the database
├── results/                      # results.duckdb, figures, ENVIRONMENT.md
├── report/                       # Typst report, generated tables and summary_metrics.json
└── tests/                        # Storage, quantization, attacks and aggregators, simulation, grid runner
```

### Why Parquet and DuckDB

Each client's partition is a separate Snappy-compressed Parquet file (`client_00.parquet` to `client_19.parquet`), one directory per seed. This keeps a partition fixed on disk once it is built, so every run with the same seed trains on exactly the same data. `EdgeStore` reads a shard with DuckDB's `.fetchnumpy()`, which returns NumPy arrays without converting row by row in Python. We have not benchmarked read time.

### Telemetry

`simulate.py` writes per-client telemetry (update norm, payload size, local loss) to the `round_telemetry` table and per-round evaluation (accuracy, loss, aggregator, attacker fraction, duration) to the `global_eval` table, both in `data/telemetry.duckdb`. Its summary table also prints a Multi-Krum mitigation rate: the share of rounds in which no attacker was selected. That rate is not stored. It shows as n/a for aggregators that do not select clients. Grid results are stored separately in `results/results.duckdb`.

---

## 5. Methods

### 5.1. INT8 Uniform Symmetric Quantization

Each client update $\mathbf{x} \in \mathbb{R}^D$ uses one scale for the whole vector. Rounding is to the nearest integer, so it is deterministic:

$$\alpha = \max_{i} |x_i|, \quad S = \frac{\alpha}{127}$$

$$q_i = \text{clip}\left(\left\lfloor \frac{x_i}{S} \right\rceil, -127, 127\right) \in \mathbb{Z}_{\text{int8}}$$

$$\hat{x}_i = q_i \cdot S \in \mathbb{R}_{\text{float32}}$$

The error on each element is at most half a step:
$$|x_i - \hat{x}_i| \le \frac{S}{2} = \frac{\alpha}{254}$$

### 5.2. Multi-Krum

For $N$ client updates $\{\mathbf{v}_1, \dots, \mathbf{v}_N\}$ and an assumed number of attackers $f$ with $2f + 2 < N$:

1. Compute all pairwise squared Euclidean distances in 64-bit floats:
   $$d_{ij}^2 = \|\mathbf{v}_i - \mathbf{v}_j\|_2^2 = \|\mathbf{v}_i\|_2^2 + \|\mathbf{v}_j\|_2^2 - 2 \langle \mathbf{v}_i, \mathbf{v}_j \rangle$$
2. Score each client $i$ by the sum of distances to its $N - f - 2$ nearest neighbours:
   $$s(i) = \sum_{k=1}^{N - f - 2} d_{i, (k)}^2$$
3. Average the $m = N - f$ updates with the lowest scores:
   $$\mathbf{v}_{\text{agg}} = \frac{1}{m} \sum_{i \in \text{Selected}} \mathbf{v}_i$$

---

## 6. Limitations

1. **One machine.** All clients run as a simulation on one computer, not on edge hardware such as a Raspberry Pi or Jetson. Network delay, packet loss and thermal throttling are not modelled.
2. **One small model on one dataset.** We train a multi-layer perceptron with 101,770 parameters on Fashion-MNIST with Dirichlet label skew ($\alpha = 0.5$) only. Convolutional networks or transformers may produce updates with different sparsity and noise.
3. **Five seeds.** Each result is the mean and standard deviation over seeds 0 to 4. We ran no significance tests. The grid runs 50 rounds; `simulate.py` defaults to 5 rounds and is meant only as a quick check.
4. **The server knows f.** In the 280-run grid the aggregator is given the true number of attackers. Section 3.4 covers what happens when it is given the wrong number.
5. **No differential privacy.** We did not add clipping or noise for differential privacy. That noise may interact with the INT8 rounding error.

---

## 7. Reproducing the Results

### 7.1. Setup

```bash
git clone https://github.com/Rajchhapariya/edgefed-bench.git
cd edgefed-bench
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used for the stored results.

### 7.2. Tests

```bash
python -m unittest discover tests -v
```

The tests cover Parquet storage and partitioning, INT8 error bounds, both attacks, all four aggregators, the simulation loop and the grid runner. Some tests build shards under `data/` and download Fashion-MNIST the first time.

### 7.3. Running the Grid

The full grid is 280 runs plus 40 wrong-f runs. Each finished run is stored under a config hash in `results/results.duckdb`, so a stopped grid resumes where it left off. Runs are bit-for-bit repeatable on the same machine at the same PyTorch thread count (see `results/ENVIRONMENT.md`). The stored results used 10 threads; pass `--threads 10` to match them.

```bash
python experiments/run_grid.py --dry_run          # list configs without running
python experiments/run_grid.py --experiment all --threads 10
python experiments/make_report_assets.py          # rewrites the results block above, figures and report data
```

`simulate.py` runs four short regimes (clean FedAvg, attacked FedAvg, Multi-Krum in Float32 and in INT8) as a quick check. Its numbers are not the grid results:

```bash
python simulate.py --rounds 5 --clients 20 --alpha 0.5
```

### 7.4. Querying the Results

```python
import duckdb

con = duckdb.connect("results/results.duckdb", read_only=True)
print(con.execute("""
    SELECT aggregator, attack, attacker_share, wire_precision,
           COUNT(*) AS runs, ROUND(AVG(final_accuracy), 2) AS mean_acc
    FROM runs
    WHERE experiment_type = 'grid'
    GROUP BY ALL
    ORDER BY 1, 2, 3, 4
""").fetchdf())
con.close()
```

---

## 8. Selected References

1. **Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017).** Machine learning with adversaries: Byzantine tolerant gradient descent. _Advances in Neural Information Processing Systems (NeurIPS 2017)_, 30.
2. **McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).** Communication-efficient learning of deep networks from decentralized data. _Artificial Intelligence and Statistics (AISTATS 2017)_, 1273-1282.
3. **Konečný, J., McMahan, H. B., Yu, F. X., Richtárik, P., Suresh, A. T., & Bacon, D. (2016).** Federated learning: Strategies for improving communication efficiency. _arXiv preprint arXiv:1610.05492_.
4. **Bernstein, J., Wang, Y. X., Azizzadenesheli, K., & Anandkumar, A. (2018).** signSGD: Compressed optimisation for non-convex problems. _International Conference on Machine Learning (ICML 2018)_, 560-569.
5. **Raasveldt, M., & Mühleisen, H. (2019).** DuckDB: an embeddable analytical database. _Proceedings of the 2019 International Conference on Management of Data (SIGMOD 2019)_, 1981-1984.

License: MIT
