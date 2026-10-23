# EdgeFed-Bench: Byzantine-Robust Federated Learning with DuckDB-Partitioned Edge Stores and Quantized Gradients

A reproducible research benchmark evaluating gradient compression trade-offs in Byzantine-tolerant federated learning over columnar edge storage engines.

---

## 1. Problem Formulation & Research Question

Federated learning systems deployed on heterogeneous edge devices face two simultaneous bottlenecks:

1. **Uplink Bandwidth Bottleneck:** Edge and mobile clients operate over constrained cellular or IoT networks where uplink bandwidth is up to an order of magnitude slower and more metered than central server broadcast downlink (Konečný et al., 2016).
2. **Adversarial Vulnerability:** Central aggregators using standard Federated Averaging (FedAvg) are vulnerable to Byzantine manipulation, where rogue or malfunctioning nodes transmit arbitrarily poisoned updates that invert or destabilize model convergence (Blanchard et al., NeurIPS 2017).

While uniform gradient quantization reduces transmission volume, it introduces quantization noise into local parameter deltas. This project evaluates:

> **Does INT8 uniform symmetric gradient quantization degrade the mathematical outlier-detection guarantees and convergence stability of Byzantine-resilient aggregators (Multi-Krum, Coordinate-wise Median) under severe non-IID Dirichlet label skew ($\alpha = 0.5$)?**

---

## 2. Key Findings

- **75.0% Uplink Bandwidth Reduction:** Symmetric INT8 quantization compresses local update vectors from 397.5 KB to 99.4 KB per client round ($101{,}770$ INT8 weights $+ 4$ bytes float32 scale scalar), reducing total communication volume from 38.82 MB to 9.71 MB across 5 rounds.
- **Outlier Rejection Invariance:** Multi-Krum achieves a **100% Byzantine mitigation rate** under both Float32 and INT8 quantization. Because INT8 quantization noise is strictly bounded ($\|e\|_\infty \le S / 2$), the maximum Euclidean perturbation across honest updates is negligible compared to the separation distance induced by adversarial attacks ($\|\mathbf{v}_{\text{honest}} - \mathbf{v}_{\text{byz}}\|_2 \gg S\sqrt{D}$).
- **Convergence Parity:** Under 20% Byzantine attackers and non-IID Dirichlet distribution ($\alpha = 0.5$), Quantized Defended Multi-Krum converges to **74.60% test accuracy**, matching the unquantized Defended baseline (**74.39%**) while unmitigated FedAvg collapses to **19.43%**.

---

## 3. Empirical Benchmark Matrix (5 Communication Rounds, 20 Clients, $\alpha = 0.5$)

| Regime                               | Aggregator | Wire Precision | Uplink / Client Round | Total Wire Bandwidth | Peak Accuracy | Final Accuracy | Byzantine Mitigation Rate |
| :----------------------------------- | :--------- | :------------- | :-------------------- | :------------------- | :------------ | :------------- | :------------------------ |
| **1. Baseline FedAvg**               | FedAvg     | Float32        | 397.5 KB              | 38.82 MB             | 76.22%        | 74.39%         | 100.0% (Clean)            |
| **2. Poisoned FedAvg**               | FedAvg     | Float32        | 397.5 KB              | 38.82 MB             | 35.31%        | 19.43%         | **0.0% (Collapsed)**      |
| **3. Defended Multi-Krum**           | Multi-Krum | Float32        | 397.5 KB              | 38.82 MB             | 75.92%        | 73.40%         | **100.0% (Mitigated)**    |
| **4. Quantized Defended Multi-Krum** | Multi-Krum | INT8           | **99.4 KB (-75%)**    | **9.71 MB (-75%)**   | 74.60%        | **74.60%**     | **100.0% (Mitigated)**    |

---

## 4. Systems & Data Engineering Architecture

```
edgefed-bench/
├── .gitignore
├── requirements.txt              # Core runtime dependencies
├── simulate.py                   # Multi-regime benchmark orchestrator & telemetry logger
├── verify_storage.py             # Phase 1 standalone storage validation script
├── data/
│   ├── cache/                    # Download cache for raw Fashion-MNIST
│   ├── clients/                  # 20 client Apache Parquet shards (client_00 to client_19)
│   ├── test.parquet              # Global evaluation partition
│   └── telemetry.duckdb          # Analytical SQL telemetry database
├── engine/
│   ├── __init__.py
│   ├── storage.py                # Dirichlet partitioner, Snappy Parquet I/O, DuckDB EdgeStore
│   ├── quantization.py           # Uniform symmetric INT8 quantizer & SQNR analyzer
│   ├── client.py                 # EdgeClassifier model & local stateless SGD optimization
│   ├── adversary.py              # Byzantine attack vectors (sign-flip and Gaussian injection)
│   └── aggregator.py             # Server aggregation: FedAvg, Coordinate Median, Multi-Krum
└── tests/
    ├── __init__.py
    ├── test_storage.py           # 6 unit tests for Parquet sharding, DuckDB CRUD & idempotency
    ├── test_quantization_client.py # 6 unit tests for INT8 bounds, noise metrics & edge SGD
    ├── test_adversary_aggregator.py # 5 unit tests for attacks, FedAvg failure & Multi-Krum
    └── test_simulation.py        # 4 unit tests for orchestrator, eval, and defense smoke tests
```

### Why DuckDB + Apache Parquet for Edge Simulation?

Traditional federated learning simulators store client partitions in a monolithic Python dictionary or raw PyTorch tensors, leading to high memory pressure and slow sequential iterations. EdgeFed-Bench shards non-IID partitions into individual compressed Apache Parquet files (`client_00.parquet` to `client_19.parquet`) and queries them through an in-memory DuckDB engine. By using DuckDB's `.fetchnumpy()`, label vectors and multidimensional feature arrays are extracted directly from columnar memory into NumPy buffers without Python-level row deserialization, achieving sub-second access across 60,000 samples.

### Analytical Telemetry

Client-side training telemetry (gradient norms, transmission payloads, local losses) and server-side evaluation metrics (global accuracy, test loss, mitigation status, execution time) are committed into a local DuckDB analytical database (`data/telemetry.duckdb`) using structured SQL tables (`round_telemetry` and `global_eval`).

---

## 5. Mathematical Formulations

### 5.1. INT8 Uniform Symmetric Quantization

Given a continuous parameter delta vector $\mathbf{x} \in \mathbb{R}^D$:

$$\alpha = \max_{i} |x_i|, \quad S = \frac{\alpha}{127}$$

$$q_i = \text{clip}\left(\left\lfloor \frac{x_i}{S} \right\rceil, -127, 127\right) \in \mathbb{Z}_{\text{int8}}$$

$$\hat{x}_i = q_i \cdot S \in \mathbb{R}_{\text{float32}}$$

Element-wise reconstruction error is bounded by:
$$|x_i - \hat{x}_i| \le \frac{S}{2} = \frac{\alpha}{254}$$

### 5.2. Multi-Krum Robust Aggregation

For $N$ client updates $\{\mathbf{v}_1, \dots, \mathbf{v}_N\}$ with up to $f$ Byzantine adversaries ($2f + 2 < N$):

1. Compute pairwise squared Euclidean distance matrix in 64-bit precision:
   $$d_{ij}^2 = \|\mathbf{v}_i - \mathbf{v}_j\|_2^2 = \|\mathbf{v}_i\|_2^2 + \|\mathbf{v}_j\|_2^2 - 2 \langle \mathbf{v}_i, \mathbf{v}_j \rangle$$
2. For each client $i$, compute its Krum score by summing distances to its $N - f - 2$ closest neighbors:
   $$s(i) = \sum_{k=1}^{N - f - 2} d_{i, (k)}^2$$
3. Select the $m = N - f$ candidates with the smallest Krum scores and compute their average:
   $$\mathbf{v}_{\text{agg}} = \frac{1}{m} \sum_{i \in \text{Selected}} \mathbf{v}_i$$

---

## 6. Reproduction & Execution

### 6.1. Environment Setup

```bash
git clone https://github.com/Rajchhapariya/edgefed-bench.git
cd edgefed-bench
pip install -r requirements.txt
```

### 6.2. Run Automated Test Suite (21 Tests)

```bash
python -m unittest discover tests -v
```

All 21 tests covering Parquet storage, Dirichlet partitioning, INT8 mathematical bounds, Byzantine corruption, Multi-Krum isolation, and end-to-end simulations run in approximately 3 seconds.

### 6.3. Execute Benchmark Simulation

Run standard benchmark across all 4 experimental regimes:

```bash
# Default: 5 communication rounds, 20 clients, alpha=0.5, sign-flip attack
python simulate.py --rounds 5 --clients 20 --alpha 0.5

# Test Gaussian noise explosion attack vector
python simulate.py --rounds 3 --clients 20 --attack_type gaussian

# Force re-partitioning and Parquet shard generation
python simulate.py --rounds 5 --clients 20 --force_rebuild
```

### 6.4. Inspect Analytical SQL Telemetry

Query stored experiment metrics directly via DuckDB CLI or Python:

```python
import duckdb

con = duckdb.connect("data/telemetry.duckdb")

# Inspect regime comparison
print(con.execute("""
    SELECT
        regime_name,
        ROUND(MAX(global_accuracy), 2) AS peak_acc,
        ROUND(MIN(global_loss), 4) AS best_loss,
        ROUND(AVG(round_duration_sec), 2) AS avg_duration_sec
    FROM global_eval
    GROUP BY regime_name
    ORDER BY regime_name
""").fetchdf())

# Query uplink bandwidth savings
print(con.execute("""
    SELECT
        regime_name,
        quantized,
        COUNT(*) AS total_client_rounds,
        ROUND(SUM(payload_bytes) / (1024.0 * 1024.0), 2) AS total_transmitted_mb
    FROM round_telemetry
    GROUP BY regime_name, quantized
""").fetchdf())

con.close()
```

---

## 7. Selected References

1. **Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017).** Machine learning with adversaries: Byzantine tolerant gradient descent. _Advances in Neural Information Processing Systems (NeurIPS 2017)_, 30.
2. **McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).** Communication-efficient learning of deep networks from decentralized data. _Artificial Intelligence and Statistics (AISTATS 2017)_, 1273-1282.
3. **Konečný, J., McMahan, H. B., Yu, F. X., Richtárik, P., Suresh, A. T., & Bacon, D. (2016).** Federated learning: Strategies for improving communication efficiency. _arXiv preprint arXiv:1610.05492_.
4. **Bernstein, J., Wang, Y. X., Azizzadenesheli, K., & Anandkumar, A. (2018).** signSGD: Compressed optimisation for non-convex problems. _International Conference on Machine Learning (ICML 2018)_, 560-569.
5. **Raasveldt, M., & Mühleisen, H. (2019).** DuckDB: an embeddable analytical database. _Proceedings of the 2019 International Conference on Management of Data (SIGMOD 2019)_, 1981-1984.
