// Auto-generated Typst tables from results/results.duckdb
// Strictly zero hand-typed values.

#let table_1_robustness = table(
  columns: (1.4fr, 1fr, 2fr, 2fr, 2.2fr, 2fr),
  fill: (col, row) => if row == 0 { rgb("#f1f5f9") } else { none },
  stroke: 0.5pt + rgb("#cbd5e1"),
  inset: 4pt,
  [*Attack Vector*], [*Share*], [*FedAvg*], [*Multi-Krum*], [*Coord Median*], [*Trimmed Mean*],
  [None (Clean)], [0%], [85.30 +- 0.31%], [85.35 +- 0.25%], [84.65 +- 0.44%], [85.36 +- 0.27%],
  [Gaussian], [10%], [48.79 +- 3.51%], [85.30 +- 0.39%], [84.54 +- 0.35%], [85.11 +- 0.40%],
  [Gaussian], [20%], [41.79 +- 6.24%], [85.17 +- 0.32%], [84.64 +- 0.34%], [84.89 +- 0.22%],
  [Gaussian], [30%], [42.81 +- 4.09%], [85.13 +- 0.29%], [84.73 +- 0.50%], [84.87 +- 0.39%],
  [sign-flip], [10%], [78.79 +- 3.58%], [85.28 +- 0.38%], [83.96 +- 0.63%], [84.11 +- 0.65%],
  [sign-flip], [20%], [41.60 +- 29.28%], [84.90 +- 0.26%], [82.97 +- 0.62%], [83.00 +- 0.45%],
  [sign-flip], [30%], [10.00 +- 0.00%], [84.75 +- 0.48%], [80.04 +- 0.30%], [80.37 +- 0.30%],
)

#let table_2_precision = table(
  columns: (1.6fr, 1.8fr, 1.8fr, 1.1fr, 1.5fr, 1.5fr, 1.1fr),
  fill: (col, row) => if row == 0 { rgb("#f1f5f9") } else { none },
  stroke: 0.5pt + rgb("#cbd5e1"),
  inset: 4pt,
  [*Aggregator*], [*Float32 Acc*], [*INT8 Acc*], [*Delta*], [*F32 Uplink*], [*INT8 Uplink*], [*Cut*],
  [FedAvg], [40.51 +- 28.25%], [41.60 +- 29.28%], [+1.09%], [397.5 KB], [99.4 KB], [-75.0%],
  [Multi-Krum], [84.89 +- 0.24%], [84.90 +- 0.26%], [+0.01%], [397.5 KB], [99.4 KB], [-75.0%],
  [Coordinate median], [83.21 +- 0.62%], [82.97 +- 0.62%], [-0.23%], [397.5 KB], [99.4 KB], [-75.0%],
  [Trimmed mean], [82.92 +- 0.55%], [83.00 +- 0.45%], [+0.08%], [397.5 KB], [99.4 KB], [-75.0%],
)

#let table_3_wrong_f = table(
  columns: (1.1fr, 1.4fr, 1.8fr, 1.8fr, 3.2fr),
  fill: (col, row) => if row == 0 { rgb("#f1f5f9") } else { none },
  stroke: 0.5pt + rgb("#cbd5e1"),
  inset: 4pt,
  [*Assumed $f$*], [*Assumed Ratio*], [*Multi-Krum Acc*], [*Trimmed Mean Acc*], [*Vulnerability Profile*],
  [2], [2/20 (10%)], [51.65 +- 9.44%], [17.61 +- 5.09%], [Underestimates attackers: poisoned updates leak into aggregation],
  [4], [4/20 (20%)], [81.41 +- 3.15%], [77.13 +- 3.03%], [Partial underestimation: vulnerable to coordinated shift],
  [6], [6/20 (30%)], [84.75 +- 0.48%], [80.37 +- 0.30%], [Exact configuration: optimal rejection envelope],
  [8], [8/20 (40%)], [84.25 +- 0.90%], [80.08 +- 0.26%], [Overestimates attackers: trims honest samples without collapsing],
)

#let table_4_multikrum = table(
  columns: (1.8fr, 1.4fr, 2.8fr, 2.8fr),
  fill: (col, row) => if row == 0 { rgb("#f1f5f9") } else { none },
  stroke: 0.5pt + rgb("#cbd5e1"),
  inset: 4.5pt,
  [*Attack Vector*], [*Attacker Share*], [*Attacker Rejection Rate*], [*Honest False-Rejection Rate*],
  [Gaussian], [10%], [100.00 +- 0.00%], [0.00 +- 0.00%],
  [Gaussian], [20%], [100.00 +- 0.00%], [0.00 +- 0.00%],
  [Gaussian], [30%], [100.00 +- 0.00%], [0.00 +- 0.00%],
  [sign-flip], [10%], [99.80 +- 0.42%], [0.02 +- 0.05%],
  [sign-flip], [20%], [90.80 +- 11.66%], [2.30 +- 2.92%],
  [sign-flip], [30%], [95.50 +- 4.74%], [1.93 +- 2.03%],
)

