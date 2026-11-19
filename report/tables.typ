// Auto-generated Typst tables from results/results.duckdb
// Written by experiments/make_report_assets.py. Do not edit by hand.

#let table_1_robustness = table(
  columns: (1.4fr, 1fr, 2fr, 2fr, 2.2fr, 2fr),
  fill: (col, row) => if row == 0 { rgb("#f1f5f9") } else { none },
  stroke: 0.5pt + rgb("#cbd5e1"),
  inset: 4pt,
  [*Attack Vector*], [*Share*], [*FedAvg*], [*Multi-Krum*], [*Coord Median*], [*Trimmed Mean*],
  [None (Clean)], [0%], [85.30 +- 0.31%], [85.35 +- 0.25%], [84.65 +- 0.44%], [85.36 +- 0.27%],
  [Gaussian], [10%], [48.22 +- 3.36%], [85.30 +- 0.39%], [84.54 +- 0.34%], [85.07 +- 0.40%],
  [Gaussian], [20%], [41.74 +- 5.94%], [85.17 +- 0.32%], [84.57 +- 0.36%], [84.89 +- 0.23%],
  [Gaussian], [30%], [43.00 +- 3.88%], [85.13 +- 0.29%], [84.67 +- 0.53%], [84.85 +- 0.40%],
  [sign-flip], [10%], [78.44 +- 4.01%], [85.28 +- 0.38%], [83.97 +- 0.71%], [84.07 +- 0.63%],
  [sign-flip], [20%], [41.67 +- 27.62%], [84.87 +- 0.28%], [83.03 +- 0.67%], [83.01 +- 0.44%],
  [sign-flip], [30%], [10.00 +- 0.00%], [84.79 +- 0.45%], [79.99 +- 0.47%], [80.21 +- 0.40%],
)

#let table_2_precision = table(
  columns: (1.6fr, 1.8fr, 1.8fr, 1.1fr, 1.5fr, 1.5fr, 1.1fr),
  fill: (col, row) => if row == 0 { rgb("#f1f5f9") } else { none },
  stroke: 0.5pt + rgb("#cbd5e1"),
  inset: 4pt,
  [*Aggregator*], [*Float32 Acc*], [*INT8 Acc*], [*Delta*], [*F32 Uplink*], [*INT8 Uplink*], [*Cut*],
  [FedAvg], [40.51 +- 28.25%], [41.67 +- 27.62%], [+1.15%], [397.5 KB], [99.4 KB], [-75.0%],
  [Multi-Krum], [84.89 +- 0.24%], [84.87 +- 0.28%], [-0.02%], [397.5 KB], [99.4 KB], [-75.0%],
  [Coordinate median], [83.21 +- 0.62%], [83.03 +- 0.67%], [-0.18%], [397.5 KB], [99.4 KB], [-75.0%],
  [Trimmed mean], [82.92 +- 0.55%], [83.01 +- 0.44%], [+0.09%], [397.5 KB], [99.4 KB], [-75.0%],
)

#let table_3_wrong_f = table(
  columns: (1.1fr, 1.4fr, 1.8fr, 1.8fr, 3.2fr),
  fill: (col, row) => if row == 0 { rgb("#f1f5f9") } else { none },
  stroke: 0.5pt + rgb("#cbd5e1"),
  inset: 4pt,
  [*Assumed $f$*], [*Assumed Ratio*], [*Multi-Krum Acc*], [*Trimmed Mean Acc*], [*Setting*],
  [2], [2/20 (10%)], [52.12 +- 8.51%], [21.62 +- 14.02%], [Underestimates attackers (assumed f < actual f = 6)],
  [4], [4/20 (20%)], [81.85 +- 2.21%], [77.09 +- 2.96%], [Underestimates attackers (assumed f < actual f = 6)],
  [6], [6/20 (30%)], [84.79 +- 0.45%], [80.21 +- 0.40%], [Assumed f equals actual f = 6],
  [8], [8/20 (40%)], [84.21 +- 0.92%], [80.13 +- 0.07%], [Overestimates attackers (assumed f > actual f = 6)],
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

