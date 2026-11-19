"""Paired statistics for comparing two wire formats on the same seeds.

Each seed gives one paired difference (for example INT8 accuracy minus Float32
accuracy for the same aggregator, attack and seed). We report:

- the mean difference and a 95% t confidence interval;
- an exact two-sided sign-flip permutation p-value, which does not assume normal
  differences (with 10 seeds there are 2^10 = 1,024 sign patterns);
- Holm-adjusted p-values across a family of comparisons.

Only NumPy and the standard library are used, so no SciPy dependency is added. The t
quantile is found by bisection on a numerically integrated t density.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence

import numpy as np


def _t_pdf(x: float, df: float) -> float:
    c = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    return math.exp(c - (df + 1) / 2 * math.log1p(x * x / df))


def _t_cdf(x: float, df: float, steps: int = 4000) -> float:
    """P(T <= x) by Simpson's rule on [0, |x|], using symmetry around 0."""
    if x == 0:
        return 0.5
    a, b = 0.0, abs(x)
    h = (b - a) / steps
    s = _t_pdf(a, df) + _t_pdf(b, df)
    for i in range(1, steps):
        s += (4 if i % 2 else 2) * _t_pdf(a + i * h, df)
    area = s * h / 3
    return 0.5 + area if x > 0 else 0.5 - area


def t_ppf(p: float, df: float) -> float:
    """Inverse CDF of Student's t distribution, for 0.5 < p < 1."""
    if not 0.5 < p < 1:
        raise ValueError("t_ppf supports 0.5 < p < 1")
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, df) < p:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def paired_comparison(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Compares paired samples a and b (same seeds, same order): statistics of a - b."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(d)
    if n < 2:
        raise ValueError("need at least two pairs")
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    half = t_ppf(0.975, n - 1) * sd / math.sqrt(n)

    # Exact permutation test: under no difference, every sign pattern is equally likely.
    observed = abs(d.sum())
    abs_d = np.abs(d)
    hits = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        if abs(float(np.dot(signs, abs_d))) >= observed - 1e-12:
            hits += 1
    return {
        "n": n,
        "mean_diff": mean,
        "sd_diff": sd,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "p_perm": hits / 2 ** n,
    }


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()
