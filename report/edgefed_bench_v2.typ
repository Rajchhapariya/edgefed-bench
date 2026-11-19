// ==============================================================================
// EdgeFed-Bench v2 report. Numbers come from metrics.typ and tables.typ,
// which experiments/make_report_assets.py writes from results/results.duckdb.
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

#align(center)[
  #block(
    fill: rgb("#0f172a"),
    inset: (x: 14pt, y: 10pt),
    radius: 5pt,
    width: 100%
  )[
    #text(size: 14pt, weight: "bold", fill: white)[EdgeFed-Bench v2: INT8 Update Compression and Byzantine-Robust Aggregation in Federated Learning]\
    #v(2pt)
    #text(size: 9pt, fill: rgb("#94a3b8"), weight: "medium")[280 grid runs over attacks, defences, wire precision and 5 seeds, plus a wrong-f experiment]\
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

== 1. Problem and Research Question

In federated learning, clients train on their own data and send only model updates to a server. Edge deployments add two constraints.

1. *Uplink is the scarce direction.* Phones and IoT devices send over metered cellular or low-power links, where upload is slower and costs more than download.
2. *Plain averaging trusts every client.* FedAvg takes the arithmetic mean of all updates, so one arbitrary update can move the result anywhere (Blanchard et al., NeurIPS 2017).

Quantizing updates to INT8 cuts the upload to about a quarter, but it adds a deterministic rounding error to every coordinate. Byzantine-robust aggregators decide what to keep by comparing updates, so that error could in principle blur the line between honest and attacker updates. We test whether it does:

*Research question: does uniform symmetric INT8 quantization of client updates reduce the final accuracy or the attacker rejection of Multi-Krum, coordinate-wise median and coordinate-wise trimmed mean under moderate non-IID Dirichlet label skew ($alpha = 0.5$)?*

#v(0.1cm)

== 2. Setup

=== 2.1. Data and Partitioning
We use Fashion-MNIST: 60,000 training images, 10,000 test images, 10 classes. The training set is split across $N = 20$ clients with a Dirichlet distribution, $alpha = 0.5$, so each client holds a different mix of labels. Each seed's split is written once to Snappy-compressed Parquet shards (`data/clients_seed_{seed}/client_{id}.parquet`) and read back through DuckDB (`EdgeStore`), so every run with that seed sees the same data.

=== 2.2. Model and Local Training
Every client trains the same multi-layer perceptron, `EdgeClassifier`: 784 inputs, 128 ReLU hidden units, 10 outputs, 101,770 parameters. Local training is one epoch of SGD per round with learning rate $eta = 0.05$, momentum $0.9$ and batch size $128$. The optimizer is created fresh each round, so momentum does not carry across rounds. In round $t$, client $i$ sends its update:
$ Delta bold(w)_i^(t) = bold(w)_(i, "local")^(t) - bold(w)_("global")^(t) $

=== 2.3. INT8 Quantization
In INT8 mode a client maps its update $bold(x) in RR^D$ to signed 8-bit integers with one scale per vector:
$ alpha = max_j |x_j|, quad S = alpha / 127, quad q_j = "clip"("round"(x_j / S), -127, 127) in ZZ_"int8" $
The server reconstructs $hat(x)_j = q_j dot S$. The error per element is at most $|x_j - hat(x)_j| <= S / 2 = alpha / 254$. One update shrinks from #m.f32_bytes bytes ($101,770 times 4$ bytes) to #m.int8_bytes bytes ($101,770$ int8 values plus a 4-byte float32 scale), a #m.savings% cut.

#pagebreak()

=== 2.4. Attacks
A fixed share of clients, $f / N in {10%, 20%, 30%}$, is Byzantine. They use one of two attacks:
- *Sign-flip:* send $bold(v)_"byz" = -2.5 dot bold(v)_"honest"$, which pushes the model against the direction honest clients move it.
- *Gaussian:* send pure noise, $bold(v)_"byz" tilde cal(N)(0, sigma^2 bold(I))$ with $sigma = 5.0$.
In INT8 runs the attacker update is also encoded as INT8, since the server can only decode INT8 payloads.

=== 2.5. Aggregators
- *FedAvg (baseline):* unweighted mean, $bold(v)_"agg" = 1 / N sum_(i=1)^N bold(v)_i$. Clients are not weighted by sample count.
- *Multi-Krum (Blanchard et al., NeurIPS 2017):* computes pairwise squared distances $d_(i j)^2 = ||bold(v)_i - bold(v)_j||_2^2$ in float64. Each update's score $s(i)$ is the sum of distances to its $N - f - 2$ nearest neighbours. The $m = N - f$ lowest-scoring updates are averaged. This needs $2f + 2 < N$; for $N = 20$ and $f = 6$, $2(6) + 2 = 14 < 20$.
- *Coordinate-wise median:* the median of each coordinate across clients. Its breakdown point is 50%.
- *Coordinate-wise trimmed mean:* for each coordinate, drop the $k = ceil(beta N)$ largest and $k$ smallest values, with $beta = f / N$, and average the other $N - 2k$. This needs $2k < N$.

=== 2.6. Grid
The grid has #m.grid_runs runs: 4 aggregators $times$ 2 attacks $times$ 3 attacker shares (10%, 20%, 30%) $times$ 2 precisions (Float32, INT8) $times$ 5 seeds (0 to 4), plus 40 clean runs with no attackers. Every run lasts 50 rounds. One seed sets the Dirichlet split, the model initialization, which clients are Byzantine, the Gaussian noise and the batch order. Runs repeat bit for bit on CPU at a fixed PyTorch thread count. A different thread count changes the order of floating-point sums and moves results slightly.

#v(0.15cm)

== 3. Results

At 30% sign-flip with INT8, FedAvg ends at #m.fa30%, which is chance level for 10 classes. Multi-Krum ends at #m.mk30%, trimmed mean at #m.tm30% and coordinate median at #m.cm30%. Table 1 gives final accuracy for every attack and share.

#table_1_robustness

#v(0.12cm)

#callout(
  "INT8 did not change Multi-Krum accuracy at 20% sign-flip",
  "With 5 seeds per precision, Multi-Krum ends at " + m.mk20_f32 + "% with Float32 and " + m.mk20 + "% with INT8. The difference is smaller than one standard deviation (" + m.mk20_sd + "). Trimmed mean reaches " + m.tm20 + "% in the same setting. The INT8 error per element is at most S/2. Our guess is that this error is small next to the distance between honest and attacker updates. We did not measure that distance, so this remains a hypothesis.",
  kind: "tip"
)

#pagebreak()

=== 3.1. Float32 versus INT8
Table 2 compares the two precisions at 20% sign-flip over 50 rounds. Uplink drops by #m.savings% for every aggregator.

#v(0.08cm)

#table_2_precision

#v(0.15cm)

=== 3.2. Convergence and Attacker Share

#align(center)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    [
      #image("figures/fig1_convergence_20pct_signflip.png", width: 100%)
      #text(size: 7.2pt, fill: rgb("#64748b"), style: "italic")[Figure 1: Test accuracy per round at 20% sign-flip (INT8, 5 seeds, shaded band = +-1 std).]
    ],
    [
      #image("figures/fig2_accuracy_vs_attacker_share.png", width: 100%)
      #text(size: 7.2pt, fill: rgb("#64748b"), style: "italic")[Figure 2: Final test accuracy against attacker share (10%, 20%, 30%).]
    ]
  )
]

#v(0.15cm)

=== 3.3. Wrong Assumed f
A real server does not know how many clients are malicious. We fixed the true number of attackers at $f_("act") = 6$ (30% sign-flip, INT8, 5 seeds) and told the server $f_("assumed") in {2, 4, 6, 8}$:

#v(0.08cm)

#table_3_wrong_f

#v(0.12cm)

#callout(
  "Underestimating f is the costly mistake",
  "With assumed f = 2 and actual f = 6, Multi-Krum falls to " + m.mk_wf2 + "% and trimmed mean to " + m.tm_wf2 + "%, because attacker updates reach the average. With assumed f = 8, Multi-Krum ends at " + m.mk_wf8 + "% and trimmed mean at " + m.tm_wf8 + "%, at the cost of discarding some honest updates. In this setup it is safer to set assumed f at or above the expected number of attackers.",
  kind: "warning"
)

#pagebreak()

=== 3.4. Multi-Krum Rejection Rates
Under Gaussian noise, Multi-Krum rejected #m.rej_gauss_min% of attacker updates even in its worst run. Sign-flip is harder: averaged over seeds, rejection ranges from #m.rej_sf_min% to #m.rej_sf_max% depending on attacker share. Table 4 also shows how often honest updates were rejected, which tells us whether label skew makes honest clients look like attackers. Rates are averaged over all 50 rounds and pooled over 5 seeds and both precisions (10 runs per cell):

#v(0.08cm)

#table_4_multikrum

#v(0.15cm)

== 4. Limitations

1. *One machine.* All clients are simulated on one host (13th Gen Intel Core i7-13620H), not on phones or embedded boards. We do not model network jitter, packet loss or thermal throttling.
2. *One small model on one dataset.* We train a 101,770-parameter MLP on Fashion-MNIST only. Convolutional networks or vision transformers may produce updates with different sparsity.
3. *Five seeds, no significance tests.* Every number is a mean and standard deviation over 5 seeds. Accuracy is reported after 50 rounds.
4. *The server knows f.* In the 280-run grid the aggregator is given the true number of attackers, which is the best case for the defences. Only the wrong-f experiment (Section 3.3) relaxes this.
5. *No differential privacy.* We add no clipping or privacy noise. That noise could interact with the INT8 rounding error.

#v(0.15cm)

== 5. Reproduction

```bash
# 1. Clone and install pinned dependencies
git clone https://github.com/Rajchhapariya/edgefed-bench.git
cd edgefed-bench
pip install -r requirements.txt

# 2. Run the unit tests
python -m unittest discover tests -v

# 3. Run the 280-run grid and the wrong-f experiment into DuckDB
python experiments/run_grid.py --config experiments/config.yaml --threads 10

# 4. Regenerate figures, README tables and Typst data from the database
python experiments/make_report_assets.py

# 5. Compile this report
typst compile report/edgefed_bench_v2.typ report/edgefed_bench_v2.pdf
```

#v(0.15cm)

== 6. Selected References

1. *Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017).* Machine learning with adversaries: Byzantine tolerant gradient descent. _Advances in Neural Information Processing Systems (NeurIPS 2017)_, 30.
2. *McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).* Communication-efficient learning of deep networks from decentralized data. _AISTATS 2017_, 1273-1282.
3. *Konečný, J., McMahan, H. B., Yu, F. X., Richtárik, P., Suresh, A. T., & Bacon, D. (2016).* Federated learning: Strategies for improving communication efficiency. _arXiv preprint arXiv:1610.05492_.
4. *Bernstein, J., Wang, Y. X., Azizzadenesheli, K., & Anandkumar, A. (2018).* signSGD: Compressed optimisation for non-convex problems. _ICML 2018_, 560-569.
5. *Raasveldt, M., & Mühleisen, H. (2019).* DuckDB: an embeddable analytical database. _SIGMOD 2019_, 1981-1984.
