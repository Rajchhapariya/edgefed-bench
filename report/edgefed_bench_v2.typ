// ==============================================================================
// EdgeFed-Bench v2: Technical Research Report
// Robustness Study Across Seeds, Attacks, and Defences
// ==============================================================================

#import "metrics.typ": m
#import "tables.typ": table_1_robustness, table_2_precision, table_3_wrong_f, table_4_multikrum

#set page(
  paper: "a4",
  margin: (top: 1.4cm, bottom: 1.4cm, left: 1.6cm, right: 1.6cm),
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8pt, fill: rgb("#64748b"), font: "Segoe UI")
      #grid(
        columns: (1fr, 1fr),
        align(left)[*EdgeFed-Bench v2* | Research Technical Report],
        align(right)[Doc ID: RES-FL-2026-V2]
      )
      #line(length: 100%, stroke: 0.5pt + rgb("#cbd5e1"))
    ]
  },
  footer: context [
    #line(length: 100%, stroke: 0.5pt + rgb("#cbd5e1"))
    #set text(size: 8pt, fill: rgb("#64748b"), font: "Segoe UI")
    #grid(
      columns: (1fr, 1fr),
      align(left)[EdgeFed-Bench: Open Research Benchmark | Version 2.0.0],
      align(right)[Page #counter(page).display("1 of 1", both: true)]
    )
  ]
)

#set text(font: "Segoe UI", size: 8.8pt, fill: rgb("#0f172a"))
#set par(justify: true, leading: 0.50em)

// Reusable Components
#let stat-card(label, value, subtext, color: rgb("#0f172a")) = [
  #block(
    fill: rgb("#f8fafc"),
    stroke: 0.5pt + rgb("#e2e8f0"),
    radius: 4pt,
    inset: 5pt,
    width: 100%
  )[
    #set text(size: 7pt, weight: "bold", fill: rgb("#64748b"))
    #upper(label)\
    #v(1pt)
    #set text(size: 12pt, weight: "bold", fill: color)
    #value\
    #v(1pt)
    #set text(size: 6.8pt, fill: rgb("#94a3b8"), weight: "regular")
    #subtext
  ]
]

#let callout(title, body-text, kind: "tip") = {
  let bg = rgb("#f8fafc")
  let border-c = rgb("#3b82f6")
  let text-c = rgb("#1e3a8a")

  if kind == "tip" {
    bg = rgb("#f0fdf4")
    border-c = rgb("#10b981")
    text-c = rgb("#065f46")
  } else if kind == "warning" {
    bg = rgb("#fffbeb")
    border-c = rgb("#f59e0b")
    text-c = rgb("#92400e")
  } else if kind == "important" {
    bg = rgb("#fff1f2")
    border-c = rgb("#f43f5e")
    text-c = rgb("#9f1239")
  }

  block(
    fill: bg,
    stroke: (left: 3pt + border-c, rest: 0.5pt + rgb("#e2e8f0")),
    radius: (right: 4pt),
    inset: (x: 8pt, y: 5pt),
    width: 100%
  )[
    #text(weight: "bold", size: 8.2pt, fill: text-c)[#title]\
    #v(1pt)
    #text(size: 7.8pt, fill: rgb("#334155"))[#body-text]
  ]
}

// Header Block
#align(center)[
  #block(
    fill: rgb("#0f172a"),
    inset: (x: 14pt, y: 10pt),
    radius: 5pt,
    width: 100%
  )[
    #text(size: 14pt, weight: "bold", fill: white)[EdgeFed-Bench v2: Empirical Robustness and Precision Trade-offs in Byzantine-Tolerant Federated Learning]\
    #v(2pt)
    #text(size: 9pt, fill: rgb("#94a3b8"), weight: "medium")[A 280-Configuration Factorial Study Across Seeds, Attacks, Quantization and Robust Defences]\
    #v(3pt)
    #line(length: 35%, stroke: 0.5pt + rgb("#334155"))
    #v(2pt)
    #text(size: 7.5pt, fill: rgb("#cbd5e1"))[
      *Author:* Raj Chhapariya #h(10pt) | #h(10pt) *Repository:* github.com/Rajchhapariya/edgefed-bench #h(10pt) | #h(10pt) *Date:* October 2026
    ]
  ]
]

#v(0.1cm)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 7pt,
  stat-card("Grid Scope", m.total_runs + " Runs", m.grid_runs + " grid + " + m.wrong_f_runs + " wrong-f across 5 seeds", color: rgb("#0284c7")),
  stat-card("Uplink Savings", "-" + m.savings + "%", m.f32_kb + " KB to " + m.int8_kb + " KB / client / round", color: rgb("#10b981")),
  stat-card("Multi-Krum at 30%", m.mk30 + "%", "final accuracy, sign-flip, INT8", color: rgb("#8b5cf6")),
  stat-card("Assumed f = 2", m.mk_wf2 + "%", "Multi-Krum accuracy, actual f = 6", color: rgb("#ef4444"))
)

#v(0.1cm)

== 1. Problem Formulation & Research Question

Federated learning coordinates distributed machine learning training across decentralized edge devices while preserving raw data locality. In production edge deployments, two structural constraints collide:

1. *Uplink Bandwidth Asymmetry:* Edge devices communicate across bandwidth-constrained, metered cellular and IoT channels where upstream transmission is substantially slower and costlier than downstream reception.
2. *Byzantine Vulnerability:* Decentralized edge topologies allow rogue, compromised, or malfunctioning clients to transmit poisoned gradient updates. Standard Federated Averaging (FedAvg) aggregates updates through linear averaging, rendering it provably vulnerable to even a single arbitrary outlier.

Gradient quantization reduces uplink communication costs by mapping continuous float32 parameter deltas into low-bitwidth integers. However, lossy quantization injects stochastic truncation noise. This investigation addresses the central empirical research question:

*Does uniform symmetric INT8 gradient quantization degrade the mathematical outlier separation distance and convergence guarantees of Byzantine-tolerant aggregators (Multi-Krum, Coordinate-wise Median, Coordinate-wise Trimmed Mean) under severe non-IID Dirichlet label skew ($alpha = 0.5$)?*

#v(0.1cm)

== 2. Experimental Setup & System Architecture

=== 2.1. Dataset and Non-IID Dirichlet Partitioning
Experiments execute on Fashion-MNIST (60,000 training samples, 10,000 test samples, 10 classes). To simulate real-world heterogeneous edge deployments, training samples are partitioned across $N = 20$ edge clients using a Dirichlet distribution with concentration parameter $alpha = 0.5$. Each client receives a skewed subset of labels, creating substantial class imbalance across clients. To guarantee reproducibility, partitions are serialized into independent Apache Parquet shards per seed (`data/clients_seed_{seed}/client_{id}.parquet`) using Snappy compression and queried via an in-memory DuckDB columnar engine (`EdgeStore`).

=== 2.2. Edge Classifier and Local Optimization
Each edge client trains an `EdgeClassifier` Multi-Layer Perceptron (784-dimensional input, 128 hidden units with ReLU activation, 10 output logits; 101,770 total trainable parameters). Local optimization uses stateless Stochastic Gradient Descent (SGD) with learning rate $eta = 0.05$, momentum $0.9$, batch size $128$, and $E = 1$ local epoch per communication round. In each round $t$, client $i$ computes its local parameter update:
$ Delta bold(w)_i^(t) = bold(w)_(i, "local")^(t) - bold(w)_("global")^(t) $

=== 2.3. Uniform Symmetric INT8 Quantization
Under INT8 wire precision, each client compresses its continuous update vector $bold(x) in RR^D$ into signed 8-bit integers using symmetric scaling:
$ alpha = max_j |x_j|, quad S = alpha / 127, quad q_j = "clip"("round"(x_j / S), -127, 127) in ZZ_"int8" $
Reconstruction on the server yields $hat(x)_j = q_j dot S$. The element-wise quantization error is bounded by $|x_j - hat(x)_j| <= S / 2 = alpha / 254$. Payload per client per round drops from #m.f32_bytes bytes ($101,770 times 4$ bytes float32) to #m.int8_bytes bytes ($101,770 times 1$ byte int8 $+ 4$ bytes float32 scale), a #m.savings% uplink reduction.

#pagebreak()

=== 2.4. Byzantine Adversarial Attack Vectors
The benchmark evaluates two primary attack vectors deployed by a designated fraction $f / N in {10%, 20%, 30%}$ of clients:
- *Sign-Flip Attack:* Adversaries invert their true gradient direction and scale its magnitude: $bold(v)_"byz" = -2.5 dot bold(v)_"honest"$. This directly steers the server model away from optimal empirical minima toward higher loss regimes.
- *Gaussian Noise Injection:* Adversaries transmit uncorrelated Gaussian noise vectors: $bold(v)_"byz" tilde cal(N)(0, sigma^2 bold(I))$ with $sigma = 5.0$, inducing variance explosion in central updates.

=== 2.5. Robust Aggregation Algorithms
- *FedAvg (Baseline):* Computes standard coordinate-wise arithmetic mean: $bold(v)_"agg" = 1 / N sum_(i=1)^N bold(v)_i$.
- *Multi-Krum (Blanchard et al., NeurIPS 2017):* Computes pairwise squared Euclidean distances $d_(i j)^2 = ||bold(v)_i - bold(v)_j||_2^2$ in float64 precision. For each update, it sums distances to its $N - f - 2$ closest neighbors to establish a Krum score $s(i)$. It selects the $m = N - f$ updates with the lowest scores and averages them. Multi-Krum requires $2f + 2 < N$. For $N = 20$ and $f = 6$ (30% attackers), $2(6) + 2 = 14 < 20$, strictly satisfying the theoretical bound.
- *Coordinate-wise Median:* Computes the median independently along each parameter dimension: $[bold(v)_"agg"]_j = "median"({[bold(v)_1]_j, dots, [bold(v)_N]_j})$. Resilient up to a 50% breakdown point.
- *Coordinate-wise Trimmed Mean:* For each coordinate $j$, sorts candidate values, excludes the $k = ceil(beta N)$ largest and $k$ smallest values (where $beta = f / N$), and averages the remaining $N - 2k$ coordinates. Requires $2k < N$.

=== 2.6. Full Factorial Experiment Grid
The experimental grid sweeps #m.grid_runs configurations: 4 Aggregators $times$ 2 Attacks (sign-flip, Gaussian) $times$ 3 Attacker Shares (10%, 20%, 30%) $times$ 2 Wire Precisions (Float32, INT8) $times$ 5 Seeds (0, 1, 2, 3, 4) across 50 rounds, plus 40 clean baseline runs (0% attackers). A single seed controls Dirichlet partitioning, model weight initialization, Byzantine node selection, Gaussian attack noise, and client-level batch shuffling, guaranteeing exact bit-level determinism on CPU.

#v(0.15cm)

== 3. Empirical Results & Comparative Analysis

#table_1_robustness

#v(0.12cm)

#callout(
  "Core Result: Quantization Noise Invariance under Severe Non-IID Skew",
  "Across the " + m.attacked_runs + " attacked configurations, switching from Float32 to INT8 changed Multi-Krum final accuracy at 20% sign-flip from " + m.mk20_f32 + "% to " + m.mk20 + "%, a difference smaller than one standard deviation (" + m.mk20_sd + "). Trimmed Mean reaches " + m.tm20 + "% in the same setting. Per-element INT8 error is bounded by S/2; a likely reason the defences are unaffected is that this error is small next to the distance between honest and attacker updates, but that distance was not measured here, so this is a hypothesis.",
  kind: "tip"
)

#pagebreak()

=== 3.1. Precision Trade-off: Float32 vs. INT8 Quantization
Table 2 isolates the wire precision trade-off under a 20% sign-flip attack across 50 communication rounds:

#v(0.08cm)

#table_2_precision

#v(0.15cm)

=== 3.2. Learning Dynamics and Convergence Stability

#align(center)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    [
      #image("figures/fig1_convergence_20pct_signflip.png", width: 100%)
      #text(size: 7.2pt, fill: rgb("#64748b"), style: "italic")[Figure 1: Convergence trajectory under 20% sign-flip attack (INT8, 5 seeds, shaded band = +-1 std).]
    ],
    [
      #image("figures/fig2_accuracy_vs_attacker_share.png", width: 100%)
      #text(size: 7.2pt, fill: rgb("#64748b"), style: "italic")[Figure 2: Final test accuracy as a function of Byzantine attacker share (10%, 20%, 30%).]
    ]
  )
]

#v(0.15cm)

=== 3.3. Vulnerability to Misconfigured Server Assumptions (The Wrong-f Experiment)
Production federated learning servers do not possess oracle knowledge of the true number of compromised edge clients. To evaluate sensitivity to parameter misconfiguration, an extra factorial experiment was executed fixing actual attackers at $f_("act") = 6$ (30% sign-flip, INT8, 5 seeds) while varying the server's assumed $f_("assumed") in {2, 4, 6, 8}$:

#v(0.08cm)

#table_3_wrong_f

#v(0.12cm)

#callout(
  "Critical Architectural Vulnerability: Asymmetric Risk of Misconfigured f",
  "Underestimating f hurts most: with assumed f = 2 and actual f = 6, Multi-Krum falls to " + m.mk_wf2 + "% and Trimmed Mean to " + m.tm_wf2 + "%, because attacker vectors reach the aggregate. Overestimating (assumed f = 8) keeps Multi-Krum at " + m.mk_wf8 + "% and Trimmed Mean at " + m.tm_wf8 + "%, at the cost of discarding extra honest updates. Setting assumed f at or above the expected attacker count is the safer choice in this setup.",
  kind: "warning"
)

#pagebreak()

=== 3.4. Multi-Krum Isolation and False-Rejection Dynamics
To evaluate whether non-IID label skew causes honest client updates to be falsely rejected as adversaries, Table 4 reports the Attacker Rejection Rate ($|B inter R| / |B|$) and Honest False-Rejection Rate ($|H inter R| / |H|$) averaged across all 50 rounds and 5 seeds:

#v(0.08cm)

#table_4_multikrum

#v(0.15cm)

== 4. System Limitations

To ensure rigorous academic grounding, the scope and boundaries of this benchmark are explicitly stated:

1. *Simulation Environment:* All experiments execute on a single physical host (13th Gen Intel Core i7-13620H) utilizing virtualized client processes rather than physical embedded IoT or mobile hardware. Real-world network jitter, asymmetric packet drop rates, and edge device battery thermal throttling are not modeled.
2. *Model and Dataset Complexity:* Experiments utilize a compact Multi-Layer Perceptron (101,770 parameters) trained on Fashion-MNIST. Deep convolutional architectures (e.g., ResNet) or Vision Transformers over larger feature dimensions may exhibit different gradient sparsity characteristics.
3. *Server Oracle Knowledge:* In the main 280-run grid, the central coordinator is parameterized with the exact true attacker count $f$, representing an upper bound on defense capability. The sensitivity to unknown $f$ is explored strictly in the dedicated wrong-$f$ experiment (Section 3.3).
4. *Privacy Guarantees:* The benchmark evaluates robust aggregation and gradient compression in isolation. It does not integrate Differential Privacy (DP) mechanisms (e.g., Gaussian gradient clipping and noise addition), which may introduce compound noise interactions with INT8 quantization.

#v(0.15cm)

== 5. Reproduction Instructions

The benchmark is fully automated, self-contained, and reproducible from a single command:

```bash
# 1. Clone repository and install pinned dependencies
git clone https://github.com/Rajchhapariya/edgefed-bench.git
cd edgefed-bench
pip install -r requirements.txt

# 2. Run comprehensive unit test suite (26 unit tests across 5 test suites)
python -m unittest discover tests -v

# 3. Execute full 280-run factorial grid and wrong-f experiment into DuckDB
python experiments/run_grid.py --config experiments/config.yaml

# 4. Generate all empirical figures, Markdown tables, and Typst assets
python experiments/make_report_assets.py

# 5. Compile 4-page PDF technical report via Typst compiler
typst compile report/edgefed_bench_v2.typ report/edgefed_bench_v2.pdf
```

#v(0.15cm)

== 6. Selected References

1. *Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017).* Machine learning with adversaries: Byzantine tolerant gradient descent. _Advances in Neural Information Processing Systems (NeurIPS 2017)_, 30.
2. *McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).* Communication-efficient learning of deep networks from decentralized data. _AISTATS 2017_, 1273-1282.
3. *Konečný, J., McMahan, H. B., Yu, F. X., Richtárik, P., Suresh, A. T., & Bacon, D. (2016).* Federated learning: Strategies for improving communication efficiency. _arXiv preprint arXiv:1610.05492_.
4. *Bernstein, J., Wang, Y. X., Azizzadenesheli, K., & Anandkumar, A. (2018).* signSGD: Compressed optimisation for non-convex problems. _ICML 2018_, 560-569.
5. *Raasveldt, M., & Mühleisen, H. (2019).* DuckDB: an embeddable analytical database. _SIGMOD 2019_, 1981-1984.
