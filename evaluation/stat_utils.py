# -*- coding: utf-8 -*-
"""
Shared statistical utilities for ablation analysis.

Exported functions
------------------
Descriptive statistics
  mean_std(xs)       — (mean, std) tuple
  fmt_mean_sd(xs)    — "mean±sd" formatted string
  median(xs)         — median value
  fmt_median_iqr(xs) — "median [q1–q3]" formatted string

Effect size
  cliffs_delta(a, b) — Cliff's delta

Hypothesis tests (pure Python, no SciPy required)
  mann_whitney_p(a, b)  — two-sided Mann-Whitney U p-value
  wilcoxon_p(a, b)      — two-sided Wilcoxon signed-rank p-value

Resampling
  bootstrap_ci(xs, ...) — bootstrap 95% CI for the mean

Formatting helpers
  fmt_p(p) — p-value with significance stars
  fmt(x, nd=2) — number → string with NaN/None handling
"""
from __future__ import annotations

import math
import random
from typing import List, Optional


# ==================================================================
# Descriptive statistics
# ==================================================================

def mean_std(xs: List[float]) -> tuple:
    """Return (mean, std) tuple. Returns (nan, nan) for empty input."""
    if not xs:
        return (float("nan"), float("nan"))
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return (m, 0.0)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return (m, math.sqrt(var))


def fmt_mean_sd(xs: List[float]) -> str:
    """Return formatted ``mean±sd`` string. Returns ``n/a`` for empty input."""
    if not xs:
        return "n/a"
    m, s = mean_std(xs)
    if math.isnan(s) or s == 0.0:
        return f"{m:.2f}"
    return f"{m:.2f}\u00b1{s:.2f}"


def median(xs: List[float]) -> float:
    """Return median of a list. Returns nan for empty input."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def fmt_median_iqr(xs: List[float]) -> str:
    """Return formatted ``median [q1–q3]`` string. Returns ``n/a`` for empty input."""
    if not xs:
        return "n/a"
    s = sorted(xs)
    n = len(s)

    def _pct(p):
        idx = (n - 1) * p
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        if lo == hi:
            return s[lo]
        frac = idx - lo
        return s[lo] + (s[hi] - s[lo]) * frac

    med = _pct(0.5)
    q1, q3 = _pct(0.25), _pct(0.75)
    return f"{med:.1f} [{q1:.1f}\u2013{q3:.1f}]"


# ==================================================================
# Effect size
# ==================================================================

def cliffs_delta(a: List[float], b: List[float]) -> float:
    """Compute Cliff's delta between two samples. Returns nan if either is empty."""
    if not a or not b:
        return float("nan")
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    return (gt - lt) / (len(a) * len(b))


# ==================================================================
# Rank-based test helpers (module-private)
# ==================================================================

def _rank(values: List[float]) -> List[float]:
    """Assign ranks with tie correction. Returns list of same length as *values*."""
    idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[idx[j + 1]] == values[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def _ncdf(z: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ==================================================================
# Hypothesis tests (pure Python)
# ==================================================================

def mann_whitney_p(a: List[float], b: List[float]) -> float:
    """Two-sided Mann-Whitney U test p-value (normal approximation with continuity correction).

    Returns nan if either sample is empty, 1.0 if all values are identical.
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    combined_ranks = _rank(a + b)
    u1 = sum(combined_ranks[:n1]) - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma == 0:
        return 1.0
    z = (abs(u1 - mu) - 0.5) / sigma
    return max(0.0, min(1.0, 2.0 * (1.0 - _ncdf(z))))


def wilcoxon_p(a: List[float], b: List[float]) -> float:
    """Two-sided Wilcoxon signed-rank test p-value (normal approximation with continuity correction).

    Requires paired samples of equal length. Returns nan if lengths differ or empty.
    """
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    diffs = [x - y for x, y in zip(a, b) if (x - y) != 0]
    n = len(diffs)
    if n == 0:
        return 1.0
    ranks = _rank([abs(d) for d in diffs])
    w_plus = sum(r for d, r in zip(diffs, ranks) if d > 0)
    w_minus = sum(r for d, r in zip(diffs, ranks) if d < 0)
    w = min(w_plus, w_minus)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return 1.0
    z = (abs(w - mu) - 0.5) / sigma
    return max(0.0, min(1.0, 2.0 * (1.0 - _ncdf(z))))


# ==================================================================
# Resampling
# ==================================================================

def bootstrap_ci(xs: List[float], n_boot: int = 2000,
                 alpha: float = 0.05) -> tuple:
    """Bootstrap confidence interval for the mean.

    Parameters
    ----------
    xs : list of float
        Sample data.
    n_boot : int
        Number of bootstrap replicates (default 2000).
    alpha : float
        Significance level (default 0.05 → 95 % CI).

    Returns
    -------
    (lo, hi) tuple. Returns (nan, nan) for empty input.
    """
    if not xs:
        return (float("nan"), float("nan"))
    if len(xs) == 1:
        return (xs[0], xs[0])
    rng = random.Random(42)
    n = len(xs)
    means = sorted(
        sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    return (means[int((alpha / 2) * n_boot)],
            means[int((1 - alpha / 2) * n_boot) - 1])


# ==================================================================
# Formatting helpers
# ==================================================================

def fmt_p(p: Optional[float]) -> str:
    """Format p-value with significance stars.

    Rules
    -----
    p < 0.001 → ``<0.001***``
    p < 0.01  → ``0.003**``
    p < 0.05  → ``0.030*``
    otherwise → ``0.100 (n.s.)``
    None      → ``n/a``
    """
    if p is None:
        return "n/a"
    if p < 0.001:
        return "<0.001***"
    if p < 0.01:
        return f"{p:.3f}**"
    if p < 0.05:
        return f"{p:.3f}*"
    return f"{p:.3f} (n.s.)"


def fmt(x: Optional[float], nd: int = 2) -> str:
    """Format a number to *nd* decimal places. Returns ``-`` for NaN/None."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"