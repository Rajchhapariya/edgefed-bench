"""Tables, figures and report numbers for the extended study.

Called by make_report_assets.py. Every number comes from results/results.duckdb.
Sections whose experiments have no runs yet are skipped, so the report can be
regenerated at any point while experiments are still running.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import stats
from engine import compression

AGGS = ["FedAvg", "Multi-Krum", "Coordinate median", "Trimmed mean"]
ROBUST = ["Multi-Krum", "Coordinate median", "Trimmed mean"]
MLP_PARAMS = 101_770
# Equivalence margin in accuracy points. INT8 counts as "no worse than Float32" for a
# setting when the whole 95% CI of the paired difference lies within +-EQUIV_MARGIN.
# We chose this value ourselves; readers can apply their own margin to the CI column.
EQUIV_MARGIN = 0.5


# ---------------------------------------------------------------- formatting


def _fmt(v: Optional[float], digits: int = 2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


def _ms(vals: Sequence[float]) -> str:
    if not vals:
        return "n/a"
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return f"{np.mean(vals):.2f} +- {sd:.2f}"


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _tables(name: str, header: List[str], rows: List[List[str]], widths: str) -> Tuple[str, str]:
    md = ["| " + " | ".join(header) + " |", "| " + " | ".join([":---"] * len(header)) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    esc = lambda s: s.replace("#", "\\#").replace("$", "\\$").replace("*", "\\*")
    typ = [
        f"#let {name} = table(",
        f"  columns: ({widths}),",
        '  fill: none,',
        '  stroke: (x, y) => if y == 0 { (top: 0.8pt, bottom: 0.4pt) },',
        "  inset: 3.5pt,",
        "  " + ", ".join(f"[*{esc(h)}*]" for h in header) + ",",
    ]
    typ += ["  " + ", ".join(f"[{esc(c)}]" for c in r) + "," for r in rows]
    typ.append(")")
    return "\n".join(md), "\n".join(typ) + "\n"


def _scen(attack: str, share: float) -> str:
    return "clean" if attack == "none" else f"{attack} {int(round(share * 100))}%"


# ---------------------------------------------------------------- data access


class Data:
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    def acc(self, where: str) -> Dict[Tuple, Dict[int, float]]:
        """{(experiment_type, aggregator, attack, share, precision, alpha, model, assumed_f): {seed: acc}}"""
        out: Dict[Tuple, Dict[int, float]] = defaultdict(dict)
        for row in self.con.execute(f"""
            SELECT experiment_type, aggregator, attack, attacker_share, wire_precision,
                   dirichlet_alpha, model, assumed_f, seed, final_accuracy
            FROM runs WHERE {where}""").fetchall():
            out[tuple(row[:8])][row[8]] = row[9]
        return out

    def count(self, experiment_type: str) -> int:
        return self.con.execute("SELECT count(*) FROM runs WHERE experiment_type = ?",
                                [experiment_type]).fetchone()[0]


def _seeds_acc(data: Data, where: str) -> Dict[int, float]:
    rows = data.con.execute(f"SELECT seed, final_accuracy FROM runs WHERE {where}").fetchall()
    return {s: a for s, a in rows}


def _paired(a: Dict[int, float], b: Dict[int, float]) -> Optional[Dict[str, float]]:
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    return stats.paired_comparison([a[s] for s in seeds], [b[s] for s in seeds])


# ---------------------------------------------------------------- sections


def precision_stats(data: Data, m: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """INT8 minus Float32 on the main grid, paired by seed, with Holm correction."""
    tests = []
    for agg in AGGS:
        for attack, share in [("none", 0.0)] + [(a, s) for a in ("sign-flip", "Gaussian") for s in (0.1, 0.2, 0.3)]:
            base = (f"experiment_type='grid' AND aggregator='{agg}' AND attack='{attack}' "
                    f"AND abs(attacker_share - {share}) < 1e-9")
            f32 = _seeds_acc(data, base + " AND wire_precision='Float32'")
            i8 = _seeds_acc(data, base + " AND wire_precision='INT8'")
            res = _paired(i8, f32)
            if res:
                tests.append((agg, attack, share, f32, i8, res))
    if not tests:
        return None
    # Holm within each aggregator: 7 settings per family. Across all 28 tests, an exact
    # permutation test on n seeds cannot reach p below 2 / 2^n, so a single 28-test family
    # could never reject anything with 10 seeds.
    adj = [0.0] * len(tests)
    for agg in AGGS:
        idx = [i for i, t in enumerate(tests) if t[0] == agg]
        for i, p in zip(idx, stats.holm([tests[i][5]["p_perm"] for i in idx])):
            adj[i] = p
    rows = []
    for (agg, attack, share, f32, i8, r), p_adj in zip(tests, adj):
        within = -EQUIV_MARGIN < r["ci_low"] and r["ci_high"] < EQUIV_MARGIN
        rows.append([agg, _scen(attack, share), str(r["n"]), f"{np.mean(list(f32.values())):.2f}",
                     f"{np.mean(list(i8.values())):.2f}", f"{r['mean_diff']:+.2f}",
                     f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]", _p(r["p_perm"]), _p(p_adj),
                     "yes" if within else "no"])
    sig = [(t, p) for t, p in zip(tests, adj) if p < 0.05]
    robust = [(t, p) for t, p in zip(tests, adj) if t[0] != "FedAvg"]
    n_seeds = max(t[5]["n"] for t in tests)
    m["stat_n_tests"] = str(len(tests))
    m["stat_n_seeds"] = str(n_seeds)
    m["stat_min_p"] = f"{2 / 2 ** n_seeds:.4f}"
    m["stat_margin"] = f"{EQUIV_MARGIN:.1f}"
    m["stat_n_robust"] = str(len(robust))
    m["stat_n_equiv_robust"] = str(sum(
        1 for t, _ in robust if -EQUIV_MARGIN < t[5]["ci_low"] and t[5]["ci_high"] < EQUIV_MARGIN))
    m["stat_not_equiv_list"] = "; ".join(
        f"{t[0]}, {_scen(t[1], t[2])}" for t, _ in robust
        if not (-EQUIV_MARGIN < t[5]["ci_low"] and t[5]["ci_high"] < EQUIV_MARGIN)) or "none"

    # The margin was chosen after seeing seeds 0-4, so seeds 5-9 alone act as a confirmation.
    confirm = []
    for agg, attack, share, f32, i8, _ in tests:
        if agg == "FedAvg":
            continue
        res = _paired({s: v for s, v in i8.items() if s >= 5}, {s: v for s, v in f32.items() if s >= 5})
        if res:
            confirm.append(-EQUIV_MARGIN < res["ci_low"] and res["ci_high"] < EQUIV_MARGIN)
    if confirm:
        m["stat_confirm_n"] = str(len(confirm))
        m["stat_confirm_equiv"] = str(sum(confirm))
    m["stat_n_sig"] = str(len(sig))
    m["stat_n_sig_robust"] = str(sum(1 for t, p in robust if p < 0.05))
    m["stat_max_abs_robust"] = f"{max(abs(t[5]['mean_diff']) for t, _ in robust):.2f}"
    worst = max(robust, key=lambda tp: max(abs(tp[0][5]["ci_low"]), abs(tp[0][5]["ci_high"])))[0]
    m["stat_widest_ci_robust"] = f"{max(abs(worst[5]['ci_low']), abs(worst[5]['ci_high'])):.2f}"
    m["stat_widest_ci_case"] = f"{worst[0]}, {_scen(worst[1], worst[2])}"
    sig_aggs = sorted({t[0] for t, _ in sig})
    m["stat_sig_aggs"] = ", ".join(sig_aggs) or "none"
    for agg in AGGS:
        k = agg.split()[0].lower().replace("-", "")
        m[f"stat_{k}_n_sig"] = str(sum(1 for t, _ in sig if t[0] == agg))
        m[f"stat_{k}_n_neg"] = str(sum(1 for t in tests if t[0] == agg and t[5]["mean_diff"] < 0))
        m[f"stat_{k}_n"] = str(sum(1 for t in tests if t[0] == agg))
    m["stat_sig_list"] = "; ".join(
        f"{t[0]}, {_scen(t[1], t[2])} ({t[5]['mean_diff']:+.2f})" for t, _ in sig) or "none"
    return _tables("table_stats", ["Aggregator", "Setting", "n", "Float32", "INT8", "Diff",
                                   "95% CI", "p", "p (Holm)", f"Within +-{EQUIV_MARGIN:g}"], rows,
                   "1.6fr, 1.3fr, 0.4fr, 0.8fr, 0.8fr, 0.7fr, 1.4fr, 0.7fr, 0.8fr, 0.8fr")


def mechanism(data: Data, m: Dict[str, str], figure_dirs: List[Path]) -> Optional[Tuple[str, str]]:
    """Distances between updates, Multi-Krum, 20% attackers, averaged over rounds and seeds."""
    rows = data.con.execute("""
        SELECT r.attack, r.wire_precision, count(DISTINCT r.config_hash),
               avg(x.honest_pair_dist), avg(x.honest_attacker_dist), avg(x.honest_codec_error),
               avg(x.honest_update_norm)
        FROM round_mechanism x JOIN runs r USING (config_hash)
        WHERE r.aggregator = 'Multi-Krum' AND abs(r.attacker_share - 0.2) < 1e-9
          AND r.experiment_type IN ('grid', 'strong_attacks', 'bit_width')
          AND r.dirichlet_alpha = 0.5 AND r.model = 'mlp'
        GROUP BY 1, 2
    """).fetchall()
    if not rows:
        return None
    order_a = ["sign-flip", "Gaussian", "ALIE", "IPM"]
    order_p = ["Float32", "INT8", "INT4", "Sign"]
    rows.sort(key=lambda r: (order_a.index(r[0]), order_p.index(r[1])))
    table = []
    for atk, prec, n, hh, ha, err, norm in rows:
        table.append([atk, prec, str(n), _fmt(hh), _fmt(ha, 2), _fmt(err, 3),
                      _fmt(err / hh if hh else None, 3), _fmt(ha / hh if hh else None, 2)])
        key = f"{atk.replace('-', '').lower()}_{prec.lower()}"
        m[f"mech_{key}_pair"] = _fmt(hh)
        m[f"mech_{key}_att"] = _fmt(ha)
        m[f"mech_{key}_err"] = _fmt(err, 3)
        m[f"mech_{key}_err_ratio"] = _fmt(err / hh if hh else None, 3)
        m[f"mech_{key}_att_ratio"] = _fmt(ha / hh if hh else None, 2)

    # Figure: per attack, honest-honest vs honest-attacker distance and INT8 error.
    by = {(r[0], r[1]): r for r in rows}
    attacks = [a for a in order_a if (a, "INT8") in by]
    if attacks:
        fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=300)
        x = np.arange(len(attacks))
        w = 0.25
        series = [("honest to honest", 3, "#16a34a"), ("honest to attacker", 4, "#dc2626"),
                  ("INT8 error of an honest update", 5, "#2563eb")]
        for i, (label, col, color) in enumerate(series):
            ax.bar(x + (i - 1) * w, [by[(a, "INT8")][col] for a in attacks], w, label=label, color=color)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(attacks)
        ax.set_ylabel("Mean L2 distance (log scale)")
        ax.legend(fontsize=8, frameon=True)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        for d in figure_dirs:
            d.mkdir(parents=True, exist_ok=True)
            plt.savefig(d / "fig3_mechanism_distances.png")
            plt.savefig(d / "fig3_mechanism_distances.pdf")
        plt.close()
    return _tables("table_mechanism", ["Attack", "Wire", "Runs", "Honest-honest", "Honest-attacker",
                                       "Codec error", "Error / honest-honest", "Attacker / honest-honest"],
                   table, "1fr, 0.8fr, 0.6fr, 1.1fr, 1.2fr, 1fr, 1.3fr, 1.4fr")


def strong_attacks(data: Data, m: Dict[str, str]) -> Optional[Tuple[str, str]]:
    if not data.count("strong_attacks"):
        return None
    rows = []
    for agg in AGGS:
        for atk in ("ALIE", "IPM"):
            for share in (0.1, 0.2, 0.3):
                base = (f"experiment_type='strong_attacks' AND aggregator='{agg}' AND attack='{atk}' "
                        f"AND abs(attacker_share - {share}) < 1e-9")
                f32 = _seeds_acc(data, base + " AND wire_precision='Float32'")
                i8 = _seeds_acc(data, base + " AND wire_precision='INT8'")
                if not f32 and not i8:
                    continue
                res = _paired(i8, f32)
                rej = data.con.execute(
                    f"SELECT avg(attacker_rejection_rate) * 100 FROM runs WHERE {base}").fetchone()[0]
                rows.append([agg, f"{atk} {int(share * 100)}%", _ms(list(f32.values())),
                             _ms(list(i8.values())),
                             f"{res['mean_diff']:+.2f}" if res else "n/a",
                             _p(res["p_perm"]) if res else "n/a", _fmt(rej, 1) if rej is not None else "-"])
                key = f"{atk.lower()}{int(share * 100)}_{agg.split()[0].lower().replace('-', '')}"
                if i8:
                    m[f"sa_{key}_int8"] = f"{np.mean(list(i8.values())):.2f}"
                if f32:
                    m[f"sa_{key}_f32"] = f"{np.mean(list(f32.values())):.2f}"
                if rej is not None:
                    m[f"sa_{key}_rej"] = f"{rej:.1f}"
    return _tables("table_strong", ["Aggregator", "Attack", "Float32", "INT8", "Diff", "p",
                                    "Multi-Krum rejection %"], rows,
                   "1.5fr, 1fr, 1.3fr, 1.3fr, 0.7fr, 0.7fr, 1.3fr")


def alpha_sweep(data: Data, m: Dict[str, str]) -> Optional[Tuple[str, str]]:
    if not data.count("alpha_sweep"):
        return None
    skew = {a: (c, s) for a, c, s in data.con.execute("""
        SELECT dirichlet_alpha, avg(classes_present), avg(max_class_share)
        FROM partition_stats WHERE validation_size = 0 AND seed BETWEEN 0 AND 4""" + (
        " AND dataset = 'fashion_mnist'" if "dataset" in [r[0] for r in data.con.execute(
            "DESCRIBE partition_stats").fetchall()] else "") + " GROUP BY 1 ORDER BY 1").fetchall()} \
        if _has_table(data.con, "partition_stats") else {}
    for a, (c, s) in skew.items():
        m[f"skew_a{a:g}_classes".replace(".", "p")] = f"{c:.1f}"
        m[f"skew_a{a:g}_maxshare".replace(".", "p")] = f"{s * 100:.0f}"
    rows = []
    for agg in AGGS:
        for atk, share in (("none", 0.0), ("sign-flip", 0.2), ("ALIE", 0.2)):
            cells = []
            for alpha in (0.1, 0.5, 1.0):
                if alpha == 0.5:
                    exp = "grid" if atk != "ALIE" else "strong_attacks"
                    extra = " AND seed BETWEEN 0 AND 4"
                else:
                    exp, extra = "alpha_sweep", ""
                base = (f"experiment_type='{exp}' AND aggregator='{agg}' AND attack='{atk}' "
                        f"AND abs(attacker_share - {share}) < 1e-9 AND dirichlet_alpha = {alpha}{extra}")
                f32 = _seeds_acc(data, base + " AND wire_precision='Float32'")
                i8 = _seeds_acc(data, base + " AND wire_precision='INT8'")
                res = _paired(i8, f32)
                cells += [_fmt(float(np.mean(list(i8.values()))) if i8 else None),
                          f"{res['mean_diff']:+.2f}" if res else "n/a"]
                if i8:
                    key = f"al_{agg.split()[0].lower().replace('-', '')}_{atk.replace('-', '').lower()}_a{alpha:g}"
                    m[key.replace(".", "p")] = f"{np.mean(list(i8.values())):.2f}"
                    if res:
                        m[key.replace(".", "p") + "_diff"] = f"{res['mean_diff']:+.2f}"
                        m[key.replace(".", "p") + "_p"] = f"{res['p_perm']:.3f}"
                        m[key.replace(".", "p") + "_n"] = str(res["n"])
            rows.append([agg, _scen(atk, share)] + cells)
    header = ["Aggregator", "Setting", "a=0.1 INT8", "diff", "a=0.5 INT8", "diff", "a=1.0 INT8", "diff"]
    return _tables("table_alpha", header, rows, "1.5fr, 1.1fr, 1fr, 0.7fr, 1fr, 0.7fr, 1fr, 0.7fr")


def bit_width(data: Data, m: Dict[str, str], figure_dirs: List[Path]) -> Optional[Tuple[str, str]]:
    if not data.count("bit_width"):
        return None
    precs = ["Float32", "INT8", "INT4", "Sign"]
    for p in precs:
        m[f"bw_kb_{p.lower()}"] = f"{compression.payload_bytes(p, MLP_PARAMS) / 1024:.1f}"
    rows, plot = [], {}
    for agg in AGGS:
        for atk, share in (("none", 0.0), ("sign-flip", 0.2), ("ALIE", 0.2)):
            cells = []
            for p in precs:
                exp = "bit_width" if p in ("INT4", "Sign") else ("strong_attacks" if atk == "ALIE" else "grid")
                where = (f"experiment_type='{exp}' AND aggregator='{agg}' AND attack='{atk}' AND "
                         f"abs(attacker_share - {share}) < 1e-9 AND wire_precision='{p}' AND seed BETWEEN 0 AND 4")
                vals = list(_seeds_acc(data, where).values())
                cells.append(_ms(vals))
                plot[(agg, atk, p)] = float(np.mean(vals)) if vals else np.nan
                if vals:
                    key = f"bw_{agg.split()[0].lower().replace('-', '')}_{atk.replace('-', '').lower()}_{p.lower()}"
                    m[key] = f"{np.mean(vals):.2f}"
            rows.append([agg, _scen(atk, share)] + cells)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=300, sharey=True)
    for ax, (atk, title) in zip(axes, (("none", "clean"), ("sign-flip", "sign-flip 20%"), ("ALIE", "ALIE 20%"))):
        x = np.arange(len(precs))
        for agg, color in zip(AGGS, ("#dc2626", "#2563eb", "#16a34a", "#9333ea")):
            ax.plot(x, [plot[(agg, atk, p)] for p in precs], marker="o", label=agg, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}\n{m['bw_kb_' + p.lower()]} KB" for p in precs], fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 90)
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Final test accuracy (%)")
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    for d in figure_dirs:
        d.mkdir(parents=True, exist_ok=True)
        plt.savefig(d / "fig4_bit_width.png")
        plt.savefig(d / "fig4_bit_width.pdf")
    plt.close()
    header = ["Aggregator", "Setting"] + [f"{p} ({m['bw_kb_' + p.lower()]} KB)" for p in precs]
    return _tables("table_bits", header, rows, "1.5fr, 1.1fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr")


def cnn(data: Data, m: Dict[str, str]) -> Optional[Tuple[str, str]]:
    if not data.count("cnn"):
        return None
    info = data.con.execute("SELECT max(num_rounds), count(DISTINCT seed) FROM runs WHERE experiment_type='cnn'").fetchone()
    m["cnn_rounds"], m["cnn_seeds"] = str(info[0]), str(info[1])
    cost = data.con.execute("""
        SELECT avg(wall_clock_sec / num_rounds) FILTER (WHERE model = 'cnn'),
               avg(wall_clock_sec / num_rounds) FILTER (WHERE model = 'mlp' AND experiment_type <> 'tuning')
        FROM runs WHERE torch_threads = 2""").fetchone()
    if cost[0] and cost[1]:
        m["cnn_cost_ratio"] = f"{cost[0] / cost[1]:.1f}"
    rows = []
    for agg in AGGS:
        for atk, share in (("none", 0.0), ("sign-flip", 0.2), ("ALIE", 0.2)):
            base = (f"experiment_type='cnn' AND aggregator='{agg}' AND attack='{atk}' "
                    f"AND abs(attacker_share - {share}) < 1e-9")
            f32 = _seeds_acc(data, base + " AND wire_precision='Float32'")
            i8 = _seeds_acc(data, base + " AND wire_precision='INT8'")
            res = _paired(i8, f32)
            rows.append([agg, _scen(atk, share), _ms(list(f32.values())), _ms(list(i8.values())),
                         f"{res['mean_diff']:+.2f}" if res else "n/a"])
            if i8:
                m[f"cnn_{agg.split()[0].lower().replace('-', '')}_{atk.replace('-', '').lower()}_int8"] = \
                    f"{np.mean(list(i8.values())):.2f}"
            if f32:
                m[f"cnn_{agg.split()[0].lower().replace('-', '')}_{atk.replace('-', '').lower()}_f32"] = \
                    f"{np.mean(list(f32.values())):.2f}"
    return _tables("table_cnn", ["Aggregator", "Setting", "Float32", "INT8", "Diff"], rows,
                   "1.6fr, 1.2fr, 1.4fr, 1.4fr, 0.8fr")


def long_run(data: Data, m: Dict[str, str]) -> Optional[Tuple[str, str]]:
    if not data.count("long_run"):
        return None
    rows = []
    gains, late = [], []
    for agg in ROBUST:
        for atk, share in (("none", 0.0), ("sign-flip", 0.2)):
            for prec in ("Float32", "INT8"):
                r = data.con.execute(f"""
                    SELECT round_num, avg(test_accuracy), count(*) FROM round_metrics
                    JOIN runs USING (config_hash)
                    WHERE experiment_type='long_run' AND aggregator='{agg}' AND attack='{atk}'
                      AND abs(attacker_share - {share}) < 1e-9 AND wire_precision='{prec}'
                      AND round_num IN (50, 100, 150)
                    GROUP BY 1 ORDER BY 1""").fetchall()
                at = {rn: a for rn, a, _ in r}
                if 150 not in at:
                    continue
                gains.append(at[150] - at[50])
                late.append(at[150] - at[100])
                rows.append([agg, _scen(atk, share), prec, _fmt(at.get(50)), _fmt(at.get(100)),
                             _fmt(at[150]), f"{at[150] - at[50]:+.2f}", f"{at[150] - at[100]:+.2f}"])
    if not rows:
        return None
    m["long_gain_min"] = f"{min(gains):+.2f}"
    m["long_gain_max"] = f"{max(gains):+.2f}"
    m["long_late_min"] = f"{min(late):+.2f}"
    m["long_late_max"] = f"{max(late):+.2f}"
    # INT8 minus Float32 at round 150, paired by seed, Multi-Krum under sign-flip.
    for agg in ROBUST:
        base = (f"experiment_type='long_run' AND aggregator='{agg}' AND attack='sign-flip'")
        res = _paired(_seeds_acc(data, base + " AND wire_precision='INT8'"),
                      _seeds_acc(data, base + " AND wire_precision='Float32'"))
        if res:
            m[f"long_diff_{agg.split()[0].lower().replace('-', '')}"] = f"{res['mean_diff']:+.2f}"
    return _tables("table_long", ["Aggregator", "Setting", "Wire", "Round 50", "Round 100",
                                  "Round 150", "Gain 50 to 150", "Gain 100 to 150"], rows,
                   "1.4fr, 1fr, 0.8fr, 0.8fr, 0.8fr, 0.8fr, 1fr, 1fr")


def tuning(data: Data, m: Dict[str, str]) -> Optional[Tuple[str, str]]:
    if not data.count("tuning"):
        return None
    res = data.con.execute("""
        SELECT lr, momentum, batch_size, avg(final_accuracy), stddev(final_accuracy), count(*)
        FROM runs WHERE experiment_type='tuning' GROUP BY 1, 2, 3 ORDER BY 4 DESC""").fetchall()
    rows = []
    for rank, (lr, mom, bs, mean, sd, n) in enumerate(res, start=1):
        rows.append([str(rank), f"{lr:g}", f"{mom:g}", str(bs), f"{mean:.2f} +- {(sd or 0):.2f}", str(n)])
        if (lr, mom, bs) == (0.05, 0.9, 128):
            m["tune_default_rank"], m["tune_default_acc"] = str(rank), f"{mean:.2f}"
    best = res[0]
    m["tune_n_configs"] = str(len(res))
    m["tune_best"] = f"lr {best[0]:g}, momentum {best[1]:g}, batch {best[2]}"
    m["tune_best_acc"] = f"{best[3]:.2f}"
    m["tune_seeds"] = str(res[0][5])
    if "tune_default_acc" in m:
        m["tune_gap"] = f"{best[3] - float(m['tune_default_acc']):.2f}"
    return _tables("table_tuning", ["Rank", "lr", "Momentum", "Batch", "Validation acc", "Seeds"],
                   rows, "0.6fr, 0.7fr, 1fr, 0.8fr, 1.6fr, 0.7fr")


def _key(agg: str, atk: str) -> str:
    return f"{agg.split()[0].lower().replace('-', '')}_{atk.replace('-', '').lower()}"


SCENARIOS_3 = (("none", 0.0), ("sign-flip", 0.2), ("ALIE", 0.2))


def cifar10(data: Data, m: Dict[str, str]) -> Optional[Tuple[str, str]]:
    if not data.count("cifar10"):
        return None
    info = data.con.execute(
        "SELECT max(num_rounds), count(DISTINCT seed), min(seed), max(seed) FROM runs WHERE experiment_type='cifar10'"
    ).fetchone()
    m["cifar_rounds"], m["cifar_seeds"] = str(info[0]), str(info[1])
    m["cifar_seed_range"] = f"{info[2]} to {info[3]}"
    kb = data.con.execute("SELECT avg(bytes_per_client_round) / 1024 FROM runs WHERE experiment_type='cifar10' "
                          "AND wire_precision='Float32'").fetchone()[0]
    kb8 = data.con.execute("SELECT avg(bytes_per_client_round) / 1024 FROM runs WHERE experiment_type='cifar10' "
                           "AND wire_precision='INT8' AND attack='none'").fetchone()[0]
    if kb and kb8:
        m["cifar_kb_f32"], m["cifar_kb_int8"] = f"{kb:.1f}", f"{kb8:.1f}"
    cost = data.con.execute("""
        SELECT avg(wall_clock_sec / num_rounds) FILTER (WHERE experiment_type = 'cifar10'),
               avg(wall_clock_sec / num_rounds) FILTER (WHERE experiment_type = 'cnn')
        FROM runs WHERE torch_threads = 2""").fetchone()
    if cost[0] and cost[1]:
        m["cifar_cost_ratio"] = f"{cost[0] / cost[1]:.1f}"
    rows, robust_diffs = [], []
    for agg in AGGS:
        for atk, share in SCENARIOS_3:
            base = (f"experiment_type='cifar10' AND aggregator='{agg}' AND attack='{atk}' "
                    f"AND abs(attacker_share - {share}) < 1e-9")
            f32 = _seeds_acc(data, base + " AND wire_precision='Float32'")
            i8 = _seeds_acc(data, base + " AND wire_precision='INT8'")
            res = _paired(i8, f32)
            rows.append([agg, _scen(atk, share), _ms(list(f32.values())), _ms(list(i8.values())),
                         f"{res['mean_diff']:+.2f}" if res else "n/a"])
            k = _key(agg, atk)
            if i8:
                m[f"cifar_{k}_int8"] = f"{np.mean(list(i8.values())):.2f}"
            if f32:
                m[f"cifar_{k}_f32"] = f"{np.mean(list(f32.values())):.2f}"
            if res:
                m[f"cifar_{k}_diff"] = f"{res['mean_diff']:+.2f}"
                if agg != "FedAvg":
                    robust_diffs.append((abs(res["mean_diff"]), agg, _scen(atk, share), res["mean_diff"]))
    if robust_diffs:
        top = max(robust_diffs)
        m["cifar_max_abs_robust"] = f"{top[0]:.2f}"
        m["cifar_max_abs_case"] = f"{top[1]}, {top[2]}"
        m["cifar_n_robust"] = str(len(robust_diffs))
        m["cifar_n_over_margin"] = str(sum(1 for d in robust_diffs if d[0] > EQUIV_MARGIN))
        m["cifar_n_neg"] = str(sum(1 for d in robust_diffs if d[3] < 0))
        m["cifar_min_p"] = f"{2 / 2 ** int(m['cifar_seeds']):.2f}"
    return _tables("table_cifar", ["Aggregator", "Setting", "Float32", "INT8", "Diff"], rows,
                   "1.6fr, 1.2fr, 1.4fr, 1.4fr, 0.8fr")


def tuned(data: Data, m: Dict[str, str]) -> Optional[Tuple[str, str]]:
    if not data.count("tuned"):
        return None
    lr, mom, bs, n_seeds = data.con.execute(
        "SELECT any_value(lr), any_value(momentum), any_value(batch_size), count(DISTINCT seed) "
        "FROM runs WHERE experiment_type='tuned'").fetchone()
    m["tuned_setting"] = f"lr {lr:g}, momentum {mom:g}, batch {bs}"
    m["tuned_seeds"] = str(n_seeds)
    rows, gains, robust_diffs = [], [], []
    for agg in AGGS:
        for atk, share in SCENARIOS_3:
            cond = f"aggregator='{agg}' AND attack='{atk}' AND abs(attacker_share - {share}) < 1e-9"
            old_exp = "strong_attacks" if atk == "ALIE" else "grid"
            old = f"experiment_type='{old_exp}' AND dirichlet_alpha = 0.5 AND seed BETWEEN 0 AND 4 AND {cond}"
            new = f"experiment_type='tuned' AND {cond}"
            o32 = _seeds_acc(data, old + " AND wire_precision='Float32'")
            o8 = _seeds_acc(data, old + " AND wire_precision='INT8'")
            n32 = _seeds_acc(data, new + " AND wire_precision='Float32'")
            n8 = _seeds_acc(data, new + " AND wire_precision='INT8'")
            res = _paired(n8, n32)
            rows.append([agg, _scen(atk, share),
                         _fmt(float(np.mean(list(o32.values())))) if o32 else "n/a",
                         _fmt(float(np.mean(list(o8.values())))) if o8 else "n/a",
                         _ms(list(n32.values())), _ms(list(n8.values())),
                         f"{res['mean_diff']:+.2f}" if res else "n/a",
                         _p(res["p_perm"]) if res else "n/a"])
            k = _key(agg, atk)
            if n8:
                m[f"tuned_{k}_int8"] = f"{np.mean(list(n8.values())):.2f}"
            if n32:
                m[f"tuned_{k}_f32"] = f"{np.mean(list(n32.values())):.2f}"
            if o32 and n32:
                m[f"tuned_{k}_gain"] = f"{np.mean(list(n32.values())) - np.mean(list(o32.values())):+.2f}"
                if agg != "FedAvg" or atk != "sign-flip":
                    gains.append(np.mean(list(n32.values())) - np.mean(list(o32.values())))
            if res:
                m[f"tuned_{k}_diff"] = f"{res['mean_diff']:+.2f}"
                m[f"tuned_{k}_p"] = _p(res["p_perm"])
                if agg != "FedAvg":
                    robust_diffs.append((abs(res["mean_diff"]), agg, _scen(atk, share), res["mean_diff"]))
    if gains:
        m["tuned_gain_min"], m["tuned_gain_max"] = f"{min(gains):+.2f}", f"{max(gains):+.2f}"
    if robust_diffs:
        top = max(robust_diffs)
        m["tuned_max_abs_robust"] = f"{top[0]:.2f}"
        m["tuned_max_abs_case"] = f"{top[1]}, {top[2]}"
        m["tuned_coord_n_neg"] = str(sum(1 for d in robust_diffs if d[1] == "Coordinate median" and d[3] < 0))
        m["tuned_coord_n"] = str(sum(1 for d in robust_diffs if d[1] == "Coordinate median"))
    return _tables("table_tuned", ["Aggregator", "Setting", f"Batch 128 Float32", f"Batch 128 INT8",
                                   f"Batch {bs} Float32", f"Batch {bs} INT8", "Diff", "p"], rows,
                   "1.5fr, 1.1fr, 0.9fr, 0.9fr, 1.2fr, 1.2fr, 0.7fr, 0.6fr")


def _has_table(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return bool(con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]).fetchone()[0])


# ---------------------------------------------------------------- entry point


SECTIONS = [
    ("table_stats", "3.6. Statistical Test of the INT8 Effect (Main Grid, Paired by Seed)",
     "INT8 accuracy minus Float32 accuracy for the same aggregator, attack and seed. The 95% CI is a "
     "paired t interval; p is an exact two-sided sign-flip permutation test; p (Holm) corrects for the "
     "seven settings of each aggregator. The last column says whether the whole CI lies within the "
     "equivalence margin in accuracy points; we chose that margin ourselves."),
    ("table_mechanism", "3.7. Why INT8 Does Not Hide Attackers: Distances Between Updates",
     "Multi-Krum, 20% attackers, alpha 0.5, MLP. Mean L2 distances per round, averaged over rounds and "
     "seeds. Codec error is the L2 norm of an honest update's compression error (0 for Float32). Only "
     "runs executed after distance logging was added are included (grid seeds 5 to 9 and the extended "
     "experiments)."),
    ("table_strong", "3.8. Attacks Built to Look Honest: ALIE and IPM",
     "Final test accuracy, mean +- standard deviation over seeds 0 to 4. Diff is INT8 minus Float32, "
     "paired by seed; p is the exact permutation p-value."),
    ("table_alpha", "3.9. Label Skew: Dirichlet Alpha 0.1, 0.5 and 1.0",
     "Final INT8 accuracy (mean over seeds 0 to 4) and INT8 minus Float32 (diff). Alpha 0.5 values come "
     "from the main grid and the ALIE runs with the same seeds."),
    ("table_bits", "3.10. Fewer Bits: INT4 and 1-Bit Sign",
     "Final test accuracy, mean +- standard deviation over seeds 0 to 4. Payload per client per round in "
     "the header."),
    ("table_cnn", "3.11. A Second Model: Small CNN",
     "Final test accuracy of a 20,522-parameter CNN, mean +- standard deviation. This study uses fewer "
     "seeds and rounds than the MLP grid to limit CPU time."),
    ("table_cifar", "3.12. A Second Dataset: CIFAR-10",
     "Final test accuracy of a 30,138-parameter CNN on CIFAR-10 (20 clients, alpha 0.5), mean +- "
     "standard deviation. Diff is INT8 minus Float32, paired by seed. Fewer seeds and rounds than the "
     "MLP grid to limit CPU time."),
    ("table_long", "3.13. Longer Training: 150 Rounds",
     "Mean test accuracy over seeds 0 to 4 at rounds 50, 100 and 150, and the gain over the last 50 "
     "rounds as a convergence check."),
    ("table_tuning", "3.14. Hyperparameter Check on a Validation Split",
     "Clean FedAvg, Float32. 10,000 training images are held out as a validation set before "
     "partitioning; accuracy here is validation accuracy, not test accuracy."),
    ("table_tuned", "3.15. Rerun at the Best Validation Setting",
     "MLP, alpha 0.5, seeds 0 to 4. Batch 128 columns are the means of the matching main-grid and ALIE "
     "runs; the other columns rerun the same settings at the best setting from Section 3.14. Diff is "
     "INT8 minus Float32 at the new setting, paired by seed; p is the exact permutation p-value."),
]


def build(con: duckdb.DuckDBPyConnection, figure_dirs: List[Path]) -> Tuple[str, str, Dict[str, str]]:
    """Returns (README markdown, Typst table definitions, report metrics)."""
    data = Data(con)
    m: Dict[str, str] = {}
    built = {
        "table_stats": precision_stats(data, m),
        "table_mechanism": mechanism(data, m, figure_dirs),
        "table_strong": strong_attacks(data, m),
        "table_alpha": alpha_sweep(data, m),
        "table_bits": bit_width(data, m, figure_dirs),
        "table_cnn": cnn(data, m),
        "table_long": long_run(data, m),
        "table_tuning": tuning(data, m),
        "table_cifar": cifar10(data, m),
        "table_tuned": tuned(data, m),
    }
    md_parts, typ_parts = [], []
    for name, title, caption in SECTIONS:
        out = built.get(name)
        if out is None:
            continue
        md, typ = out
        md_parts.append(f"### {title}\n{caption}\n\n{md}\n")
        if name == "table_mechanism" and (figure_dirs and (figure_dirs[0] / "fig3_mechanism_distances.png").exists()):
            md_parts.append('<p align="center"><img src="results/figures/fig3_mechanism_distances.png" '
                            'alt="Distances between updates" width="70%" /></p>\n')
        if name == "table_bits":
            md_parts.append('<p align="center"><img src="results/figures/fig4_bit_width.png" '
                            'alt="Accuracy by bit width" width="95%" /></p>\n')
        typ_parts.append(typ)
    # Empty placeholders keep report/edgefed_bench_v2.typ compiling before every experiment has run.
    for name, _, _ in SECTIONS:
        if built.get(name) is None:
            typ_parts.append(f"#let {name} = [_No runs yet._]\n")
    return "\n".join(md_parts), "\n".join(typ_parts), m
