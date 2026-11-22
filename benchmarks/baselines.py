"""Pure-Python reference baselines for the ``eproc`` benchmark suite.

Zero third-party dependencies (Python 3 stdlib only: ``array``, ``bisect``,
``collections``, ``heapq``, ``math``, ``random``).

Each baseline is an idiomatic, fairly optimized pure-Python implementation of
the exact published algorithm or market incumbent it represents:

- ``PyONSBettingEProcess``: ONS betting supermartingale (Waudby-Smith & Ramdas
  2024; matching ``gostevehoward/confseq`` pure-Python betting).
- ``PyEmpiricalBernsteinCS``: Predictable plug-in Empirical-Bernstein
  Confidence Sequence (Waudby-Smith & Ramdas 2024).
- ``PyFixedHorizonPeekTest``: Standard fixed-horizon Student-t / z-test
  checked at every tick (continuous peeking failure control).
- ``PyMultiRateAgACI``: Multi-rate crossing-free AgACI + Conformal PID
  (matching ``salesforce/online_conformal`` / ``MAPIE``).
- ``PyRawScoreDistributedConformal``: Raw float64 score shipping baseline for
  distributed conformal calibration.
- ``PyNaiveAverageFuser`` & ``PyNaiveAgentVoting``: Unweighted ensemble
  averaging and un-deduplicated log-odds voting (vulnerable to echoes &
  Byzantine outliers).
- ``PySafeBOA`` & ``PyEchoPool``: Pure-Python counterparts of
  ``eproc.SafeFuser`` and ``eproc.EchoPool`` for apples-to-apples speed.
- ``PyVarOptSampler``, ``PyOnlineRLS``, ``PySiftVOI``: Pure-Python
  counterparts of ``eproc.VarOptSampler``, ``eproc.OnlineRLSSampler``, and
  ``eproc.SiftVOISelector``.
"""

from __future__ import annotations

import array
import bisect
import heapq
import math
import random
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

__all__ = [
    "BetResult",
    "CSResult",
    "PeekResult",
    "ConformalOut",
    "BOAStep",
    "EchoOut",
    "RLSOut",
    "SiftAction",
    "PyONSBettingEProcess",
    "PyEmpiricalBernsteinCS",
    "PyFixedHorizonPeekTest",
    "PyMultiRateAgACI",
    "PyRawScoreDistributedConformal",
    "PyNaiveAverageFuser",
    "PyNaiveAgentVoting",
    "PySafeBOA",
    "PyEchoPool",
    "PyVarOptSampler",
    "PyOnlineRLS",
    "PySiftVOI",
    "norm_ppf",
    "t_crit",
]

_ONS_STEP = 2.0 / (2.0 - math.log(3.0))  # ~2.2187946958176
_LOG_2 = math.log(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Lightweight result containers (slots for fair, low-overhead Python speed)
# ---------------------------------------------------------------------------


class BetResult:
    __slots__ = ("n", "e_value", "log_e", "max_log_e", "p_value", "mean", "rejected")

    def __init__(
        self,
        n: int,
        e_value: float,
        log_e: float,
        max_log_e: float,
        p_value: float,
        mean: float,
        rejected: bool,
    ) -> None:
        self.n = n
        self.e_value = e_value
        self.log_e = log_e
        self.max_log_e = max_log_e
        self.p_value = p_value
        self.mean = mean
        self.rejected = rejected


class CSResult:
    __slots__ = ("n", "mean", "low", "high", "width")

    def __init__(self, n: int, mean: float, low: float, high: float, width: float) -> None:
        self.n = n
        self.mean = mean
        self.low = low
        self.high = high
        self.width = width


class PeekResult:
    __slots__ = ("n", "mean", "stat", "rejected", "rejected_at")

    def __init__(
        self, n: int, mean: float, stat: float, rejected: bool, rejected_at: int
    ) -> None:
        self.n = n
        self.mean = mean
        self.stat = stat
        self.rejected = rejected
        self.rejected_at = rejected_at


class ConformalOut:
    __slots__ = ("quantiles", "eff_alphas", "emp_coverage")

    def __init__(
        self,
        quantiles: List[float],
        eff_alphas: List[float],
        emp_coverage: List[float],
    ) -> None:
        self.quantiles = quantiles
        self.eff_alphas = eff_alphas
        self.emp_coverage = emp_coverage


class BOAStep:
    __slots__ = (
        "fused_pred",
        "anchor_pred",
        "fallback_active",
        "n_quarantined",
        "sup_log_e",
        "deg_log_e",
        "n_updates",
    )

    def __init__(
        self,
        fused_pred: float,
        anchor_pred: float,
        fallback_active: bool,
        n_quarantined: int,
        sup_log_e: float,
        deg_log_e: float,
        n_updates: int,
    ) -> None:
        self.fused_pred = fused_pred
        self.anchor_pred = anchor_pred
        self.fallback_active = fallback_active
        self.n_quarantined = n_quarantined
        self.sup_log_e = sup_log_e
        self.deg_log_e = deg_log_e
        self.n_updates = n_updates


class EchoOut:
    __slots__ = (
        "accepted",
        "posterior_prob",
        "log_odds",
        "source_reliability",
        "unique_count",
    )

    def __init__(
        self,
        accepted: bool,
        posterior_prob: float,
        log_odds: float,
        source_reliability: float,
        unique_count: int,
    ) -> None:
        self.accepted = accepted
        self.posterior_prob = posterior_prob
        self.log_odds = log_odds
        self.source_reliability = source_reliability
        self.unique_count = unique_count


class RLSOut:
    __slots__ = ("keep", "leverage", "prob", "ht_weight")

    def __init__(self, keep: bool, leverage: float, prob: float, ht_weight: float) -> None:
        self.keep = keep
        self.leverage = leverage
        self.prob = prob
        self.ht_weight = ht_weight


class SiftAction:
    __slots__ = ("action_type", "target_idx", "evoi_per_cost")

    def __init__(self, action_type: int, target_idx: int, evoi_per_cost: float) -> None:
        self.action_type = action_type
        self.target_idx = target_idx
        self.evoi_per_cost = evoi_per_cost


# ---------------------------------------------------------------------------
# Pure-Python math helpers (zero scipy)
# ---------------------------------------------------------------------------


def _logsumexp2(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    if a == -math.inf:
        return -math.inf
    return a + math.log1p(math.exp(b - a))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x) if x < 700.0 else 0.0
        return 1.0 / (1.0 + z)
    z = math.exp(x) if x > -700.0 else 0.0
    return z / (1.0 + z)


def _norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / _SQRT_2PI


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, <1.15e-9 error)."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be strictly in (0, 1)")
    a1 = -3.969683028665376e01
    a2 = 2.209460984245205e02
    a3 = -2.759285104469687e02
    a4 = 1.383577518672690e02
    a5 = -3.066479806614716e01
    a6 = 2.506628277459239e00

    b1 = -5.447609879822406e01
    b2 = 1.615858368580409e02
    b3 = -1.556989798598866e02
    b4 = 6.680131188771972e01
    b5 = -1.328068155288572e01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e00
    c4 = -2.549732539343734e00
    c5 = 4.374664141464968e00
    c6 = 2.938163982698783e00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e00
    d4 = 3.754408661907416e00

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
            (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
            * q
            / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
        (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
    )


def t_crit(df: int, alpha: float, two_sided: bool = False) -> float:
    """Student-t upper critical value via Cornish-Fisher expansion (exact for df >= 5)."""
    p = 1.0 - (0.5 * alpha if two_sided else alpha)
    z = norm_ppf(p)
    if df <= 0:
        return math.inf
    z2 = z * z
    z3 = z2 * z
    z5 = z3 * z2
    z7 = z5 * z2
    g1 = (z3 + z) / 4.0
    g2 = (5.0 * z5 + 16.0 * z3 + 3.0 * z) / 96.0
    g3 = (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / 384.0
    inv_df = 1.0 / df
    return z + inv_df * (g1 + inv_df * (g2 + inv_df * g3))


def _pava_nondecreasing(vals: List[float]) -> List[float]:
    """Pool Adjacent Violators Algorithm for monotone non-decreasing quantiles."""
    # Each block: [weighted_sum, count]
    blocks: List[List[float]] = []
    for v in vals:
        s = v
        c = 1.0
        while blocks and (blocks[-1][0] / blocks[-1][1]) > (s / c):
            prev = blocks.pop()
            s += prev[0]
            c += prev[1]
        blocks.append([s, c])
    out: List[float] = []
    for s, c in blocks:
        avg = s / c
        out.extend([avg] * int(c))
    return out


def _normalize_side(side: str | int) -> int:
    if isinstance(side, int):
        if side in (0, 1, 2):
            return side
        raise ValueError(f"invalid side code: {side}")
    s = side.lower().replace("-", "_")
    if s == "upper":
        return 0
    if s == "lower":
        return 1
    if s in ("two_sided", "twosided", "both"):
        return 2
    raise ValueError(f"invalid side: {side}")


# ---------------------------------------------------------------------------
# 1. Anytime-Valid Betting E-Process, EB Confidence Sequence & Peeking Control
# ---------------------------------------------------------------------------


class PyONSBettingEProcess:
    """Pure-Python ONS betting supermartingale matching ``confseq`` betting."""

    __slots__ = (
        "lower",
        "upper",
        "inv_span",
        "mu0_norm",
        "alpha",
        "log_thresh",
        "c_max",
        "side",
        "lam_pos_max",
        "lam_neg_min",
        "lam_pos",
        "a_pos",
        "log_e_pos",
        "lam_neg",
        "a_neg",
        "log_e_neg",
        "log_e",
        "max_log_e",
        "sum_x",
        "n",
        "rejected_at",
        "n_invalid",
    )

    def __init__(
        self,
        mu0: float = 0.5,
        alpha: float = 0.05,
        lower: float = 0.0,
        upper: float = 1.0,
        side: str | int = "upper",
        c_max: float = 0.75,
    ) -> None:
        if upper <= lower or not (0.0 < alpha < 1.0):
            raise ValueError("invalid bounds or alpha")
        self.lower = float(lower)
        self.upper = float(upper)
        self.inv_span = 1.0 / (self.upper - self.lower)
        self.mu0_norm = (float(mu0) - self.lower) * self.inv_span
        if not (0.0 < self.mu0_norm < 1.0):
            raise ValueError("mu0 must lie strictly inside (lower, upper)")
        self.alpha = float(alpha)
        self.log_thresh = math.log(1.0 / self.alpha)
        self.c_max = float(c_max)
        self.side = _normalize_side(side)
        self.lam_pos_max = self.c_max / self.mu0_norm
        self.lam_neg_min = -self.c_max / (1.0 - self.mu0_norm)

        self.lam_pos = 0.0
        self.a_pos = 1.0
        self.log_e_pos = 0.0

        self.lam_neg = 0.0
        self.a_neg = 1.0
        self.log_e_neg = 0.0

        self.log_e = 0.0
        self.max_log_e = 0.0
        self.sum_x = 0.0
        self.n = 0
        self.rejected_at = 0
        self.n_invalid = 0

    def update(self, x: float) -> BetResult:
        if not math.isfinite(x):
            self.n_invalid += 1
            return self.query()

        if x < self.lower:
            x = self.lower
        elif x > self.upper:
            x = self.upper

        y = (x - self.lower) * self.inv_span
        diff = y - self.mu0_norm
        side = self.side
        _log = math.log

        if side == 0 or side == 2:
            lp = self.lam_pos
            w = 1.0 + lp * diff
            if w < 1e-15:
                w = 1e-15
            self.log_e_pos += _log(w)
            z = diff / w
            ap = self.a_pos + z * z
            self.a_pos = ap
            nlp = lp + _ONS_STEP * z / ap
            if nlp < 0.0:
                nlp = 0.0
            elif nlp > self.lam_pos_max:
                nlp = self.lam_pos_max
            self.lam_pos = nlp

        if side == 1 or side == 2:
            ln = self.lam_neg
            w = 1.0 + ln * diff
            if w < 1e-15:
                w = 1e-15
            self.log_e_neg += _log(w)
            z = diff / w
            an = self.a_neg + z * z
            self.a_neg = an
            nln = ln + _ONS_STEP * z / an
            if nln > 0.0:
                nln = 0.0
            elif nln < self.lam_neg_min:
                nln = self.lam_neg_min
            self.lam_neg = nln

        if side == 0:
            le = self.log_e_pos
        elif side == 1:
            le = self.log_e_neg
        else:
            le = _logsumexp2(self.log_e_pos, self.log_e_neg) - _LOG_2

        self.log_e = le
        if le > self.max_log_e:
            self.max_log_e = le

        self.sum_x += x
        n = self.n + 1
        self.n = n
        if self.rejected_at == 0 and self.max_log_e >= self.log_thresh:
            self.rejected_at = n

        ev = math.exp(le if le < 700.0 else 700.0)
        pv = math.exp(-self.max_log_e) if self.max_log_e > 0.0 else 1.0
        return BetResult(
            n=n,
            e_value=ev,
            log_e=le,
            max_log_e=self.max_log_e,
            p_value=pv,
            mean=self.sum_x / n,
            rejected=(self.rejected_at > 0),
        )

    def update_batch(self, xs: Iterable[float]) -> BetResult:
        # Idiomatic Python loop caching locals (fair, well-optimized baseline)
        lower = self.lower
        upper = self.upper
        inv_span = self.inv_span
        mu0_norm = self.mu0_norm
        side = self.side
        lam_pos_max = self.lam_pos_max
        lam_neg_min = self.lam_neg_min
        log_thresh = self.log_thresh

        lp = self.lam_pos
        ap = self.a_pos
        lep = self.log_e_pos
        ln = self.lam_neg
        an = self.a_neg
        leneg = self.log_e_neg
        le = self.log_e
        max_le = self.max_log_e
        sum_x = self.sum_x
        n = self.n
        rej_at = self.rejected_at
        n_inv = self.n_invalid

        _isfinite = math.isfinite
        _log = math.log
        _ons = _ONS_STEP

        for x in xs:
            if not _isfinite(x):
                n_inv += 1
                continue
            if x < lower:
                x = lower
            elif x > upper:
                x = upper
            diff = (x - lower) * inv_span - mu0_norm
            if side == 0 or side == 2:
                w = 1.0 + lp * diff
                if w < 1e-15:
                    w = 1e-15
                lep += _log(w)
                z = diff / w
                ap += z * z
                lp += _ons * z / ap
                if lp < 0.0:
                    lp = 0.0
                elif lp > lam_pos_max:
                    lp = lam_pos_max
            if side == 1 or side == 2:
                w = 1.0 + ln * diff
                if w < 1e-15:
                    w = 1e-15
                leneg += _log(w)
                z = diff / w
                an += z * z
                ln += _ons * z / an
                if ln > 0.0:
                    ln = 0.0
                elif ln < lam_neg_min:
                    ln = lam_neg_min
            if side == 0:
                le = lep
            elif side == 1:
                le = leneg
            else:
                le = _logsumexp2(lep, leneg) - _LOG_2
            if le > max_le:
                max_le = le
            sum_x += x
            n += 1
            if rej_at == 0 and max_le >= log_thresh:
                rej_at = n

        self.lam_pos = lp
        self.a_pos = ap
        self.log_e_pos = lep
        self.lam_neg = ln
        self.a_neg = an
        self.log_e_neg = leneg
        self.log_e = le
        self.max_log_e = max_le
        self.sum_x = sum_x
        self.n = n
        self.rejected_at = rej_at
        self.n_invalid = n_inv
        return self.query()

    def query(self) -> BetResult:
        le = self.log_e
        ev = math.exp(le if le < 700.0 else 700.0)
        pv = math.exp(-self.max_log_e) if self.max_log_e > 0.0 else 1.0
        mean = self.sum_x / self.n if self.n > 0 else 0.0
        return BetResult(
            n=self.n,
            e_value=ev,
            log_e=le,
            max_log_e=self.max_log_e,
            p_value=pv,
            mean=mean,
            rejected=(self.rejected_at > 0),
        )


class PyEmpiricalBernsteinCS:
    """Pure-Python predictable plug-in Empirical-Bernstein CS (Waudby-Smith & Ramdas 2024)."""

    __slots__ = (
        "lower",
        "upper",
        "span",
        "inv_span",
        "alpha",
        "log_2_over_alpha",
        "c_max",
        "sum_y",
        "sum_sq_dev",
        "sum_lam",
        "sum_lam_y",
        "sum_psi_v",
        "cs_low",
        "cs_high",
        "sum_x",
        "n",
        "n_invalid",
    )

    def __init__(
        self,
        alpha: float = 0.05,
        lower: float = 0.0,
        upper: float = 1.0,
        c_max: float = 0.5,
    ) -> None:
        if upper <= lower or not (0.0 < alpha < 1.0):
            raise ValueError("invalid bounds or alpha")
        self.lower = float(lower)
        self.upper = float(upper)
        self.span = self.upper - self.lower
        self.inv_span = 1.0 / self.span
        self.alpha = float(alpha)
        self.log_2_over_alpha = math.log(2.0 / self.alpha)
        self.c_max = float(c_max)

        self.sum_y = 0.0
        self.sum_sq_dev = 0.0
        self.sum_lam = 0.0
        self.sum_lam_y = 0.0
        self.sum_psi_v = 0.0
        self.cs_low = self.lower
        self.cs_high = self.upper
        self.sum_x = 0.0
        self.n = 0
        self.n_invalid = 0

    def update(self, x: float) -> CSResult:
        if not math.isfinite(x):
            self.n_invalid += 1
            return self.query()

        if x < self.lower:
            x = self.lower
        elif x > self.upper:
            x = self.upper

        y = (x - self.lower) * self.inv_span
        t = self.n + 1
        mu_hat = (0.5 + self.sum_y) / t
        var_hat = (0.25 + self.sum_sq_dev) / t

        denom = var_hat * t * math.log(t + 1.0)
        lam = math.sqrt(2.0 * self.log_2_over_alpha / denom) if denom > 0.0 else self.c_max
        if lam > self.c_max:
            lam = self.c_max

        psi_e = -math.log1p(-lam) - lam
        dev = y - mu_hat
        v = dev * dev

        self.sum_y += y
        self.sum_sq_dev += v
        self.sum_lam += lam
        self.sum_lam_y += lam * y
        self.sum_psi_v += psi_e * v
        self.sum_x += x
        self.n = t

        center = self.sum_lam_y / self.sum_lam
        margin = (self.log_2_over_alpha + self.sum_psi_v) / self.sum_lam
        raw_lo = center - margin
        raw_hi = center + margin
        if raw_lo < 0.0:
            raw_lo = 0.0
        if raw_hi > 1.0:
            raw_hi = 1.0
        lo = self.lower + self.span * raw_lo
        hi = self.lower + self.span * raw_hi
        if lo > self.cs_low:
            self.cs_low = lo
        if hi < self.cs_high:
            self.cs_high = hi
        if self.cs_low > self.cs_high:
            mid = 0.5 * (self.cs_low + self.cs_high)
            self.cs_low = mid
            self.cs_high = mid

        return CSResult(
            n=t,
            mean=self.sum_x / t,
            low=self.cs_low,
            high=self.cs_high,
            width=self.cs_high - self.cs_low,
        )

    def update_batch(self, xs: Iterable[float]) -> CSResult:
        lower = self.lower
        upper = self.upper
        span = self.span
        inv_span = self.inv_span
        log2a = self.log_2_over_alpha
        two_log2a = 2.0 * log2a
        c_max = self.c_max

        sum_y = self.sum_y
        sum_sq_dev = self.sum_sq_dev
        sum_lam = self.sum_lam
        sum_lam_y = self.sum_lam_y
        sum_psi_v = self.sum_psi_v
        cs_low = self.cs_low
        cs_high = self.cs_high
        sum_x = self.sum_x
        n = self.n
        n_inv = self.n_invalid

        _isfinite = math.isfinite
        _log = math.log
        _log1p = math.log1p
        _sqrt = math.sqrt

        for x in xs:
            if not _isfinite(x):
                n_inv += 1
                continue
            if x < lower:
                x = lower
            elif x > upper:
                x = upper
            y = (x - lower) * inv_span
            n += 1
            inv_t = 1.0 / n
            mu_hat = (0.5 + sum_y) * inv_t
            var_hat = (0.25 + sum_sq_dev) * inv_t
            lam = _sqrt(two_log2a / (var_hat * n * _log(n + 1.0)))
            if lam > c_max:
                lam = c_max
            psi_e = -_log1p(-lam) - lam
            dev = y - mu_hat
            v = dev * dev
            sum_y += y
            sum_sq_dev += v
            sum_lam += lam
            sum_lam_y += lam * y
            sum_psi_v += psi_e * v
            sum_x += x

            inv_slam = 1.0 / sum_lam
            center = sum_lam_y * inv_slam
            margin = (log2a + sum_psi_v) * inv_slam
            raw_lo = center - margin
            raw_hi = center + margin
            if raw_lo < 0.0:
                raw_lo = 0.0
            if raw_hi > 1.0:
                raw_hi = 1.0
            lo = lower + span * raw_lo
            hi = lower + span * raw_hi
            if lo > cs_low:
                cs_low = lo
            if hi < cs_high:
                cs_high = hi

        if cs_low > cs_high:
            mid = 0.5 * (cs_low + cs_high)
            cs_low = mid
            cs_high = mid

        self.sum_y = sum_y
        self.sum_sq_dev = sum_sq_dev
        self.sum_lam = sum_lam
        self.sum_lam_y = sum_lam_y
        self.sum_psi_v = sum_psi_v
        self.cs_low = cs_low
        self.cs_high = cs_high
        self.sum_x = sum_x
        self.n = n
        self.n_invalid = n_inv
        return self.query()

    def query(self) -> CSResult:
        mean = self.sum_x / self.n if self.n > 0 else 0.0
        return CSResult(
            n=self.n,
            mean=mean,
            low=self.cs_low,
            high=self.cs_high,
            width=self.cs_high - self.cs_low,
        )


class PyFixedHorizonPeekTest:
    """Standard fixed-horizon one-sample Student-t / z-test checked at every tick.

    Demonstrates why naive continuous peeking (without an anytime-valid
    supermartingale) inflates Type-I error from nominal alpha=0.05 to ~30-45%
    over T=1,000 ticks.
    """

    __slots__ = (
        "mu0",
        "alpha",
        "side",
        "min_n",
        "use_t_crit",
        "z_crit",
        "n",
        "mean",
        "m2",
        "last_stat",
        "rejected",
        "rejected_at",
    )

    def __init__(
        self,
        mu0: float = 0.5,
        alpha: float = 0.05,
        side: str | int = "upper",
        min_n: int = 10,
        use_t_crit: bool = True,
    ) -> None:
        self.mu0 = float(mu0)
        self.alpha = float(alpha)
        self.side = _normalize_side(side)
        self.min_n = int(max(2, min_n))
        self.use_t_crit = bool(use_t_crit)
        p = 1.0 - (0.5 * self.alpha if self.side == 2 else self.alpha)
        self.z_crit = norm_ppf(p)

        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.last_stat = 0.0
        self.rejected = False
        self.rejected_at = 0

    def update(self, x: float) -> PeekResult:
        n = self.n + 1
        self.n = n
        delta = x - self.mean
        self.mean += delta / n
        self.m2 += delta * (x - self.mean)

        stat = 0.0
        if n >= self.min_n:
            var = self.m2 / (n - 1)
            if var > 1e-18:
                stat = (self.mean - self.mu0) / math.sqrt(var / n)
                crit = t_crit(n - 1, self.alpha, two_sided=(self.side == 2)) if self.use_t_crit else self.z_crit
                hit = (
                    (self.side == 0 and stat >= crit)
                    or (self.side == 1 and stat <= -crit)
                    or (self.side == 2 and abs(stat) >= crit)
                )
                if hit and not self.rejected:
                    self.rejected = True
                    self.rejected_at = n
        self.last_stat = stat
        return PeekResult(
            n=n,
            mean=self.mean,
            stat=stat,
            rejected=self.rejected,
            rejected_at=self.rejected_at,
        )

    def update_batch(self, xs: Iterable[float]) -> PeekResult:
        for x in xs:
            self.update(x)
        return PeekResult(
            n=self.n,
            mean=self.mean,
            stat=self.last_stat,
            rejected=self.rejected,
            rejected_at=self.rejected_at,
        )


# ---------------------------------------------------------------------------
# 2. Multi-Rate Online Conformal (AgACI + PID) & Raw-Score Distributed Baseline
# ---------------------------------------------------------------------------


class PyMultiRateAgACI:
    """Pure-Python multi-rate crossing-free AgACI + Conformal PID calibrator.

    Matches the algorithm of ``salesforce/online_conformal`` / ``MAPIE`` AgACI
    extended across multiple coverage rates with PAVA isotonic projection and
    an incrementally maintained ``bisect`` sorted window.
    """

    GAMMAS: Tuple[float, ...] = (0.005, 0.02, 0.08, 0.20)

    def __init__(
        self,
        alphas: Sequence[float] = (0.50, 0.20, 0.10, 0.05, 0.01),
        window_size: int = 128,
        pid_ki: float = 0.05,
        pid_csat: float = 1.0,
    ) -> None:
        self.target_alphas = sorted((float(a) for a in alphas), reverse=True)
        self.n_rates = len(self.target_alphas)
        self.n_experts = len(self.GAMMAS)
        self.win_cap = int(window_size)
        self.pid_ki = float(pid_ki)
        self.pid_csat = float(pid_csat)
        self.n_updates = 0

        self.alpha_t: List[List[float]] = [
            [a for _ in self.GAMMAS] for a in self.target_alphas
        ]
        self.exp_log_w: List[List[float]] = [
            [0.0 for _ in self.GAMMAS] for _ in self.target_alphas
        ]
        self.exp_var: List[List[float]] = [
            [1e-4 for _ in self.GAMMAS] for _ in self.target_alphas
        ]
        self.pid_err_sum: List[float] = [0.0 for _ in self.target_alphas]
        self.eff_alpha: List[float] = list(self.target_alphas)
        self.current_q: List[float] = [1.0 for _ in self.target_alphas]
        self.emp_cov_ema: List[float] = [1.0 - a for a in self.target_alphas]
        self.hit_counts: List[int] = [0 for _ in self.target_alphas]

        self.ring: List[float] = []
        self.ring_pos: int = 0
        self.sorted_win: List[float] = []

    def _quantile_from_sorted(self, alpha_eff: float) -> float:
        wlen = len(self.sorted_win)
        if wlen == 0:
            return 1.0
        a = min(max(alpha_eff, 1e-6), 1.0 - 1e-6)
        idx = int(math.ceil((1.0 - a) * (wlen + 1))) - 1
        if idx < 0:
            idx = 0
        elif idx >= wlen:
            idx = wlen - 1
        return self.sorted_win[idx]

    def update(self, nonconformity_score: float) -> ConformalOut:
        score = float(nonconformity_score)
        if not math.isfinite(score):
            score = 0.0
        if score < 0.0:
            score = abs(score)

        self.n_updates += 1
        wlen = len(self.sorted_win)
        gammas = self.GAMMAS
        n_exp = self.n_experts
        _exp = math.exp
        _sqrt = math.sqrt
        _tanh = math.tanh

        # 1. Evaluate previous quantiles & update AgACI experts + PID
        for r in range(self.n_rates):
            target_a = self.target_alphas[r]
            miss = 1.0 if score > self.current_q[r] else 0.0
            if miss == 0.0:
                self.hit_counts[r] += 1
            self.emp_cov_ema[r] = 0.95 * self.emp_cov_ema[r] + 0.05 * (1.0 - miss)

            # BOA weights across the 4 learning-rate experts
            lw_row = self.exp_log_w[r]
            max_lw = max(lw_row)
            ws = [_exp(lw - max_lw) for lw in lw_row]
            w_sum = sum(ws)
            inv_wsum = 1.0 / w_sum if w_sum > 0.0 else 1.0 / n_exp
            probs = [w * inv_wsum for w in ws]

            at_row = self.alpha_t[r]
            ev_row = self.exp_var[r]
            exp_losses = [0.0] * n_exp
            exp_qs = [0.0] * n_exp
            for k in range(n_exp):
                qk = self._quantile_from_sorted(at_row[k])
                exp_qs[k] = qk
                diff = score - qk
                # Pinball loss at level (1 - target_a)
                exp_losses[k] = (1.0 - target_a) * diff if diff >= 0.0 else -target_a * diff

            mix_loss = 0.0
            for k in range(n_exp):
                mix_loss += probs[k] * exp_losses[k]

            scale = max(1.0, self.current_q[r], score)
            inv_scale = 1.0 / scale
            for k in range(n_exp):
                g = (exp_losses[k] - mix_loss) * inv_scale
                if g > 1.0:
                    g = 1.0
                elif g < -1.0:
                    g = -1.0
                ev_row[k] += g * g
                eta = min(0.5, 1.0 / _sqrt(ev_row[k]))
                lw_row[k] -= eta * (g + eta * g * g)
                # ACI step for expert k
                err_k = 1.0 if score > exp_qs[k] else 0.0
                new_ak = at_row[k] + gammas[k] * (target_a - err_k)
                if new_ak < 1e-4:
                    new_ak = 1e-4
                elif new_ak > 0.999:
                    new_ak = 0.999
                at_row[k] = new_ak

            # Recompute normalized expert weights for combined eff_alpha
            max_lw = max(lw_row)
            ws = [_exp(lw - max_lw) for lw in lw_row]
            w_sum = sum(ws)
            eff_a = 0.0
            for k in range(n_exp):
                eff_a += (ws[k] / w_sum) * at_row[k]

            # Optional Conformal PID integral saturation correction
            if self.pid_ki > 0.0:
                self.pid_err_sum[r] += target_a - miss
                pid_adj = self.pid_ki * _tanh(self.pid_err_sum[r] / max(self.pid_csat, 1e-6))
                eff_a += pid_adj

            if eff_a < 1e-4:
                eff_a = 1e-4
            elif eff_a > 0.999:
                eff_a = 0.999
            self.eff_alpha[r] = eff_a

        # 2. Update sliding ring & incrementally maintained sorted list
        if wlen < self.win_cap:
            self.ring.append(score)
            bisect.insort(self.sorted_win, score)
        else:
            old = self.ring[self.ring_pos]
            self.ring[self.ring_pos] = score
            self.ring_pos = (self.ring_pos + 1) % self.win_cap
            old_idx = bisect.bisect_left(self.sorted_win, old)
            del self.sorted_win[old_idx]
            bisect.insort(self.sorted_win, score)

        # 3. Query quantiles & apply PAVA isotonic projection (zero crossing)
        raw_qs = [self._quantile_from_sorted(a) for a in self.eff_alpha]
        self.current_q = _pava_nondecreasing(raw_qs)

        return ConformalOut(
            quantiles=list(self.current_q),
            eff_alphas=list(self.eff_alpha),
            emp_coverage=list(self.emp_cov_ema),
        )

    def update_batch(self, scores: Iterable[float]) -> ConformalOut:
        out: ConformalOut | None = None
        for s in scores:
            out = self.update(s)
        if out is None:
            return ConformalOut(
                quantiles=list(self.current_q),
                eff_alphas=list(self.eff_alpha),
                emp_coverage=list(self.emp_cov_ema),
            )
        return out

    def predict(self, point_pred: float, scale: float = 1.0) -> Tuple[List[float], List[float]]:
        s = abs(float(scale))
        lows = [point_pred - q * s for q in self.current_q]
        highs = [point_pred + q * s for q in self.current_q]
        return lows, highs


class PyRawScoreDistributedConformal:
    """Naive distributed conformal baseline shipping raw float64 score arrays."""

    __slots__ = ("scores",)

    def __init__(self, scores: Iterable[float] = ()) -> None:
        self.scores = array.array("d", scores)

    def add(self, score: float) -> None:
        self.scores.append(float(score))

    def add_batch(self, scores: Iterable[float]) -> None:
        if isinstance(scores, array.array) and scores.typecode == "d":
            self.scores.extend(scores)
        else:
            self.scores.extend(float(s) for s in scores)

    @property
    def count(self) -> int:
        return len(self.scores)

    def wire_size(self) -> int:
        return 8 * len(self.scores)

    def serialize(self) -> bytes:
        return self.scores.tobytes()

    @classmethod
    def deserialize(cls, buf: bytes | memoryview) -> "PyRawScoreDistributedConformal":
        obj = cls()
        obj.scores.frombytes(buf)
        return obj

    def merge(self, other: "PyRawScoreDistributedConformal") -> "PyRawScoreDistributedConformal":
        out = PyRawScoreDistributedConformal()
        out.scores.extend(self.scores)
        out.scores.extend(other.scores)
        return out

    def subtract(self, local: "PyRawScoreDistributedConformal") -> "PyRawScoreDistributedConformal":
        rem = Counter(local.scores)
        out = array.array("d")
        for s in self.scores:
            if rem[s] > 0:
                rem[s] -= 1
            else:
                out.append(s)
        return PyRawScoreDistributedConformal(out)

    def quantile(self, q: float) -> float:
        n = len(self.scores)
        if n == 0:
            return 0.0
        vals = sorted(self.scores)
        q_clamped = min(max(float(q), 0.0), 1.0)
        idx = int(math.ceil(q_clamped * (n + 1))) - 1
        if idx < 0:
            idx = 0
        elif idx >= n:
            idx = n - 1
        return vals[idx]


# ---------------------------------------------------------------------------
# 3. Multi-Agent Fusion Baselines (Naive vs Pure-Python BOA & EchoPool)
# ---------------------------------------------------------------------------


class PyNaiveAverageFuser:
    """Standard unweighted multi-agent average (vulnerable to Byzantine spikes)."""

    __slots__ = ("n_experts", "n_updates", "sum_sq_err", "max_abs_err")

    def __init__(self, n_experts: int) -> None:
        self.n_experts = int(n_experts)
        self.n_updates = 0
        self.sum_sq_err = 0.0
        self.max_abs_err = 0.0

    def predict(self, expert_preds: Sequence[float]) -> float:
        return sum(expert_preds) / len(expert_preds)

    def update(self, expert_preds: Sequence[float], y_true: float) -> Tuple[float, float]:
        pred = sum(expert_preds) / len(expert_preds)
        err = abs(pred - y_true)
        sq = err * err
        self.n_updates += 1
        self.sum_sq_err += sq
        if err > self.max_abs_err:
            self.max_abs_err = err
        return pred, sq

    @property
    def mse(self) -> float:
        return self.sum_sq_err / self.n_updates if self.n_updates > 0 else 0.0


class PyNaiveAgentVoting:
    """Un-deduplicated, unweighted Bayesian log-odds voting (echo-vulnerable)."""

    __slots__ = ("n_claims", "prior_lo", "claim_lo", "n_submitted")

    def __init__(self, n_claims: int, prior_prob: float = 0.5) -> None:
        self.n_claims = int(n_claims)
        p = min(max(float(prior_prob), 1e-6), 1.0 - 1e-6)
        self.prior_lo = math.log(p / (1.0 - p))
        self.claim_lo = [self.prior_lo] * self.n_claims
        self.n_submitted = 0

    def submit(
        self, source_id: int, claim_id: int, evidence_id: int, llr: float
    ) -> float:
        self.claim_lo[claim_id] += llr
        self.n_submitted += 1
        return _sigmoid(self.claim_lo[claim_id])

    def resolve(self, claim_id: int, outcome_positive: bool) -> None:
        # Naive voting does not track peer reliability.
        return None

    def posterior(self, claim_id: int) -> float:
        return _sigmoid(self.claim_lo[claim_id])


class PySafeBOA:
    """Pure-Python Scale-Free BOA with 3-Gate Ville Fallback & Poisoning Isolation."""

    def __init__(
        self,
        n_experts: int,
        anchor_idx: int = 0,
        alpha_deg: float = 0.05,
        clip_mult: float = 25.0,
    ) -> None:
        self.n_experts = int(n_experts)
        self.anchor_idx = int(anchor_idx)
        self.clip_mult = float(clip_mult)
        self.fallback_active = False
        self.scale_ema = 1.0
        self.n_updates = 0
        self.n_poison_clipped = 0
        self.n_fallback_trips = 0

        init_w = 1.0 / self.n_experts
        self.weights = [init_w] * self.n_experts
        self.log_w = [0.0] * self.n_experts
        self.grad_sq_sum = [1e-4] * self.n_experts
        self.cum_loss = [0.0] * self.n_experts
        self.quarantined = [False] * self.n_experts
        self.poison_strikes = [0] * self.n_experts

        # 3 Ville E-processes over normalized loss advantage in [0, 1], null mu0=0.5
        self.sup_e = PyONSBettingEProcess(mu0=0.5, alpha=alpha_deg, lower=0.0, upper=1.0, side="upper")
        self.deg_e = PyONSBettingEProcess(mu0=0.5, alpha=alpha_deg, lower=0.0, upper=1.0, side="upper")
        self.rec_e = PyONSBettingEProcess(mu0=0.5, alpha=alpha_deg, lower=0.0, upper=1.0, side="upper")

    def predict(self, expert_preds: Sequence[float]) -> float:
        anchor_pred = expert_preds[self.anchor_idx]
        if not math.isfinite(anchor_pred):
            anchor_pred = 0.0
        if self.fallback_active:
            return anchor_pred
        bound = self.clip_mult * max(self.scale_ema, 1.0)
        total = 0.0
        w_sum = 0.0
        for i in range(self.n_experts):
            if self.quarantined[i]:
                continue
            w = self.weights[i]
            p = expert_preds[i]
            if not math.isfinite(p) or abs(p - anchor_pred) > bound:
                p = anchor_pred
            total += w * p
            w_sum += w
        return total / w_sum if w_sum > 0.0 else anchor_pred

    def update(self, expert_preds: Sequence[float], y_true: float) -> BOAStep:
        anchor_pred = expert_preds[self.anchor_idx]
        if not math.isfinite(anchor_pred):
            anchor_pred = 0.0

        bound = self.clip_mult * max(self.scale_ema, 1.0)
        sanitized = [0.0] * self.n_experts
        for i in range(self.n_experts):
            p = expert_preds[i]
            if (not math.isfinite(p)) or (i != self.anchor_idx and abs(p - anchor_pred) > bound):
                self.n_poison_clipped += 1
                self.poison_strikes[i] += 1
                if self.poison_strikes[i] >= 3:
                    self.quarantined[i] = True
                sanitized[i] = anchor_pred
            else:
                sanitized[i] = p

        # Compute BOA candidate prediction & active prediction
        boa_pred = 0.0
        w_sum = 0.0
        for i in range(self.n_experts):
            if not self.quarantined[i]:
                boa_pred += self.weights[i] * sanitized[i]
                w_sum += self.weights[i]
        boa_pred = boa_pred / w_sum if w_sum > 0.0 else anchor_pred
        fused_pred = anchor_pred if self.fallback_active else boa_pred

        # Squared errors clipped to predictable scale^2
        scale_sq = max(self.scale_ema * self.scale_ema, 1e-4)
        loss_cap = self.clip_mult * scale_sq
        boa_loss = min((boa_pred - y_true) ** 2, loss_cap)
        anc_loss = min((anchor_pred - y_true) ** 2, loss_cap)

        # Update 3 Ville E-processes on normalized difference in [0, 1]
        denom = max(boa_loss + anc_loss, scale_sq)
        # deg: boa_loss > anc_loss => z_deg > 0.5
        z_deg = 0.5 + 0.5 * max(-1.0, min(1.0, (boa_loss - anc_loss) / denom))
        z_sup = 1.0 - z_deg
        sup_res = self.sup_e.update(z_sup)
        deg_res = self.deg_e.update(z_deg)
        if self.fallback_active:
            rec_res = self.rec_e.update(z_sup)
            if rec_res.rejected:
                self.fallback_active = False
                self.deg_e = PyONSBettingEProcess(
                    mu0=0.5, alpha=self.deg_e.alpha, lower=0.0, upper=1.0, side="upper"
                )
        else:
            if deg_res.rejected:
                self.fallback_active = True
                self.n_fallback_trips += 1
                self.rec_e = PyONSBettingEProcess(
                    mu0=0.5, alpha=self.rec_e.alpha, lower=0.0, upper=1.0, side="upper"
                )

        # Scale-free BOA exponential weight update
        inv_scale_sq = 1.0 / max(loss_cap, 1e-6)
        max_lw = -math.inf
        for i in range(self.n_experts):
            if self.quarantined[i]:
                continue
            li = min((sanitized[i] - y_true) ** 2, loss_cap)
            self.cum_loss[i] += li
            g = (li - boa_loss) * inv_scale_sq
            if g > 1.0:
                g = 1.0
            elif g < -1.0:
                g = -1.0
            self.grad_sq_sum[i] += g * g
            eta = min(0.5, 1.0 / math.sqrt(self.grad_sq_sum[i]))
            self.log_w[i] -= eta * (g + eta * g * g)
            if self.log_w[i] > max_lw:
                max_lw = self.log_w[i]

        w_tot = 0.0
        for i in range(self.n_experts):
            if self.quarantined[i]:
                self.weights[i] = 0.0
            else:
                wi = math.exp(self.log_w[i] - max_lw)
                self.weights[i] = wi
                w_tot += wi
        if w_tot > 0.0:
            inv_wtot = 1.0 / w_tot
            for i in range(self.n_experts):
                self.weights[i] *= inv_wtot
        else:
            self.quarantined[self.anchor_idx] = False
            self.weights[self.anchor_idx] = 1.0

        # Update predictable scale EMA from anchor residual
        anc_err = abs(anchor_pred - y_true)
        self.scale_ema = 0.95 * self.scale_ema + 0.05 * max(anc_err, 0.1)
        self.n_updates += 1

        return BOAStep(
            fused_pred=fused_pred,
            anchor_pred=anchor_pred,
            fallback_active=self.fallback_active,
            n_quarantined=sum(self.quarantined),
            sup_log_e=sup_res.log_e,
            deg_log_e=deg_res.log_e,
            n_updates=self.n_updates,
        )


class PyEchoPool:
    """Pure-Python provenance-deduplicated multi-agent evidence pool."""

    def __init__(
        self, n_sources: int, n_claims: int, prior_prob: float = 0.5
    ) -> None:
        self.n_sources = int(n_sources)
        self.n_claims = int(n_claims)
        p = min(max(float(prior_prob), 1e-6), 1.0 - 1e-6)
        self.prior_lo = math.log(p / (1.0 - p))

        self.src_alpha = [2.0] * self.n_sources
        self.src_beta = [2.0] * self.n_sources
        self.src_reliability = [0.5] * self.n_sources

        self.claim_lo = [self.prior_lo] * self.n_claims
        self.claim_unique = [0] * self.n_claims
        self.claim_entries: List[List[Tuple[int, float]]] = [[] for _ in range(self.n_claims)]
        self.seen: set[Tuple[int, int]] = set()
        self.n_accepted = 0
        self.n_echo_dropped = 0

    def submit(
        self, source_id: int, claim_id: int, evidence_id: int, llr: float
    ) -> EchoOut:
        key = (claim_id, evidence_id)
        rel = self.src_reliability[source_id]
        if key in self.seen:
            self.n_echo_dropped += 1
            lo = self.claim_lo[claim_id]
            return EchoOut(
                accepted=False,
                posterior_prob=_sigmoid(lo),
                log_odds=lo,
                source_reliability=rel,
                unique_count=self.claim_unique[claim_id],
            )
        self.seen.add(key)
        self.n_accepted += 1
        w_llr = rel * llr
        self.claim_lo[claim_id] += w_llr
        self.claim_unique[claim_id] += 1
        self.claim_entries[claim_id].append((source_id, llr))
        lo = self.claim_lo[claim_id]
        return EchoOut(
            accepted=True,
            posterior_prob=_sigmoid(lo),
            log_odds=lo,
            source_reliability=rel,
            unique_count=self.claim_unique[claim_id],
        )

    def resolve(self, claim_id: int, outcome_positive: bool) -> None:
        entries = self.claim_entries[claim_id]
        sign = 1.0 if outcome_positive else -1.0
        for src, llr in entries:
            if llr * sign > 0.0:
                self.src_alpha[src] += 1.0
            elif llr * sign < 0.0:
                self.src_beta[src] += 1.0
            a = self.src_alpha[src]
            b = self.src_beta[src]
            acc = a / (a + b)
            # Map accuracy [0.5, 1.0] -> reliability [0.0, 1.0]; inverted/random -> 0.0
            self.src_reliability[src] = max(0.0, min(1.0, 2.0 * acc - 1.0)) if (a + b) > 5.0 else acc
        entries.clear()

    def posterior(self, claim_id: int) -> float:
        return _sigmoid(self.claim_lo[claim_id])

    def source_reliability(self, source_id: int) -> float:
        return self.src_reliability[source_id]


# ---------------------------------------------------------------------------
# 4. Causal Online Subsampling & Multi-Resolution Value-of-Information (VOI)
# ---------------------------------------------------------------------------


class PyVarOptSampler:
    """Pure-Python Cohen et al. (2011) Variance-Optimal Weighted Reservoir Sampler.

    Implements exact Horvitz-Thompson adjusted weights with eviction probability
    ``q_i = 1 - a_i / tau_new`` so ``E[adj_weight_i] == orig_weight_i`` and
    ``sum(adj_weights) == total_weight`` holds identically at every step.
    """

    def __init__(self, k: int, seed: int = 42) -> None:
        self.k = int(k)
        self.rng = random.Random(seed)
        self.tau = 0.0
        self.total_weight = 0.0
        self.n_seen = 0
        # Heavy items (w > tau): min-heap of (weight, tie_idx, item_id, value)
        self._heavy: List[Tuple[float, int, int, float]] = []
        # Previously-light items (all have current adjusted weight == self.tau):
        # stored in a flat list for O(1) uniform random eviction
        self._light: List[Tuple[int, float, float]] = []  # (item_id, orig_weight, value)

    def push(self, item_id: int, weight: float, value: float) -> None:
        if weight <= 0.0 or not math.isfinite(weight):
            return
        self.n_seen += 1
        self.total_weight += weight
        tie = self.n_seen

        if len(self._heavy) + len(self._light) < self.k:
            heapq.heappush(self._heavy, (weight, tie, item_id, value))
            return

        # Track newly-light items with their pre-eviction adjusted weights a_i:
        # tuple = (pre_adj_weight, item_id, orig_weight, value)
        prev_tau = self.tau
        n_prev_light = len(self._light)
        sum_light = prev_tau * n_prev_light
        newly_light: List[Tuple[float, int, float, float]] = []

        if weight > prev_tau:
            heapq.heappush(self._heavy, (weight, tie, item_id, value))
        else:
            newly_light.append((weight, item_id, weight, value))
            sum_light += weight

        m_light = n_prev_light + len(newly_light)

        # Ensure at least 2 light items (or 1 if all k remaining are heavy) so tau_new is well-defined
        if m_light == 0:
            hw, _htie, hid, hval = heapq.heappop(self._heavy)
            newly_light.append((hw, hid, hw, hval))
            sum_light += hw
            m_light = 1

        if m_light == 1 and self._heavy:
            hw, _htie, hid, hval = heapq.heappop(self._heavy)
            newly_light.append((hw, hid, hw, hval))
            sum_light += hw
            m_light = 2

        # Demote any heavy items with weight <= current candidate tau_new = sum_light / (m_light - 1)
        while self._heavy and m_light > 1:
            tau_cand = sum_light / (m_light - 1)
            if self._heavy[0][0] <= tau_cand:
                hw, _htie, hid, hval = heapq.heappop(self._heavy)
                newly_light.append((hw, hid, hw, hval))
                sum_light += hw
                m_light += 1
            else:
                break

        if m_light <= 1:
            # Degenerate k == 0 guard
            self.tau = sum_light
            self._light.clear()
            return

        tau_new = sum_light / (m_light - 1)
        self.tau = tau_new

        # Exact Cohen et al. eviction: draw r in [0, 1); item i evicted with prob (1 - a_i / tau_new)
        r = self.rng.random()
        prev_evict_mass = n_prev_light * (1.0 - prev_tau / tau_new) if n_prev_light > 0 else 0.0

        if r < prev_evict_mass and n_prev_light > 0:
            # Evict one uniformly random previously-light item in O(1) via swap-pop
            ev_idx = self.rng.randrange(n_prev_light)
            self._light[ev_idx] = self._light[-1]
            self._light.pop()
            # All newly-light items survive with new adjusted weight tau_new
            for _a, nid, nw, nval in newly_light:
                self._light.append((nid, nw, nval))
        else:
            r -= prev_evict_mass
            evicted_idx = len(newly_light) - 1
            for idx, (a_i, _nid, _nw, _nval) in enumerate(newly_light):
                q_i = 1.0 - a_i / tau_new
                if r < q_i:
                    evicted_idx = idx
                    break
                r -= q_i
            for idx, (_a, nid, nw, nval) in enumerate(newly_light):
                if idx != evicted_idx:
                    self._light.append((nid, nw, nval))

    def estimate_sum(self) -> float:
        total = 0.0
        for hw, _tie, _hid, hval in self._heavy:
            total += hw * hval
        tau = self.tau
        for _lid, _lw, lval in self._light:
            total += tau * lval
        return total

    def estimate_mean(self) -> float:
        if self.total_weight <= 0.0:
            return 0.0
        return self.estimate_sum() / self.total_weight


class PyOnlineRLS:
    """Pure-Python Online Ridge Leverage Score (RLS) causal subsampler."""

    def __init__(
        self,
        d: int,
        lambda_reg: float = 1.0,
        oversample: float = 4.0,
        min_prob: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.d = int(d)
        self.lambda_reg = float(lambda_reg)
        self.oversample = float(oversample)
        self.min_prob = float(min_prob)
        self.n_seen = 0
        self.n_kept = 0
        self.rng = random.Random(seed)
        inv_l = 1.0 / self.lambda_reg
        self.inv_a = [0.0] * (self.d * self.d)
        for i in range(self.d):
            self.inv_a[i * self.d + i] = inv_l

    def step(self, x: Sequence[float]) -> RLSOut:
        d = self.d
        inv_a = self.inv_a
        v = [0.0] * d
        lev = 0.0
        for i in range(d):
            row_off = i * d
            s = 0.0
            for j in range(d):
                s += inv_a[row_off + j] * x[j]
            v[i] = s
            lev += x[i] * s

        if lev < 0.0:
            lev = 0.0
        prob = self.oversample * lev
        if prob < self.min_prob:
            prob = self.min_prob
        elif prob > 1.0:
            prob = 1.0

        self.n_seen += 1
        keep = self.rng.random() < prob
        if keep:
            self.n_kept += 1
            denom = prob + lev
            if denom > 1e-12:
                inv_denom = 1.0 / denom
                for i in range(d):
                    vi = v[i] * inv_denom
                    row_off = i * d
                    for j in range(d):
                        inv_a[row_off + j] -= vi * v[j]
            return RLSOut(keep=True, leverage=lev, prob=prob, ht_weight=1.0 / prob)
        return RLSOut(keep=False, leverage=lev, prob=prob, ht_weight=0.0)


class PySiftVOI:
    """Pure-Python Hierarchical Multi-Resolution Bayes Value-of-Information Selector."""

    def __init__(
        self,
        n_sites: int,
        per_site: int,
        signal_mean: float = 2.0,
        noise_sd: float = 1.0,
        prior_hot_site: float = 0.05,
        prior_active_cold: float = 0.002,
        prior_active_hot: float = 0.25,
    ) -> None:
        self.n_sites = int(n_sites)
        self.per_site = int(per_site)
        self.n_channels = self.n_sites * self.per_site
        self.signal_mean = float(signal_mean)
        self.noise_sd = float(noise_sd)
        self.prior_hot_site = float(prior_hot_site)
        self.prior_active_cold = float(prior_active_cold)
        self.prior_active_hot = float(prior_active_hot)

        base_p = (
            self.prior_hot_site * self.prior_active_hot
            + (1.0 - self.prior_hot_site) * self.prior_active_cold
        )
        self.site_prob = [self.prior_hot_site] * self.n_sites
        self.chan_prob = [base_p] * self.n_channels
        self.chan_noise_scale = [1.0] * self.n_channels

    def set_noise_scale(self, channel_idx: int, noise_scale: float) -> None:
        self.chan_noise_scale[channel_idx] = max(float(noise_scale), 1e-3)

    def observe_channel(self, channel_idx: int, obs_mean: float, n_ticks: int = 1) -> float:
        p = self.chan_prob[channel_idx]
        p = min(max(p, 1e-9), 1.0 - 1e-9)
        sd = (self.noise_sd * self.chan_noise_scale[channel_idx]) / math.sqrt(max(1, n_ticks))
        var = sd * sd
        mu = self.signal_mean
        # Gaussian log-likelihood ratio: log(p(obs|H1) / p(obs|H0))
        llr = (obs_mean * mu - 0.5 * mu * mu) / var
        lo = math.log(p / (1.0 - p)) + llr
        new_p = _sigmoid(lo)
        self.chan_prob[channel_idx] = new_p

        # Propagate channel evidence upward to site hot posterior
        site = channel_idx // self.per_site
        sp = min(max(self.site_prob[site], 1e-9), 1.0 - 1e-9)
        # Likelihood ratio of channel outcome under hot vs cold site
        lh_hot = self.prior_active_hot * math.exp(min(llr, 40.0)) + (1.0 - self.prior_active_hot)
        lh_cold = self.prior_active_cold * math.exp(min(llr, 40.0)) + (1.0 - self.prior_active_cold)
        site_lo = math.log(sp / (1.0 - sp)) + math.log(max(lh_hot, 1e-12) / max(lh_cold, 1e-12))
        new_sp = _sigmoid(site_lo)
        self.site_prob[site] = new_sp
        return new_p

    def observe_site(self, site_idx: int, site_mean: float) -> float:
        sp = min(max(self.site_prob[site_idx], 1e-9), 1.0 - 1e-9)
        m = self.per_site
        base = site_idx * m
        # Effective site-mean variance accounting for heterogeneous channel noise scales
        sum_var = 0.0
        for j in range(m):
            sc = self.chan_noise_scale[base + j]
            sum_var += (self.noise_sd * sc) ** 2
        site_var = sum_var / (m * m)
        mu_hot = self.prior_active_hot * self.signal_mean
        mu_cold = self.prior_active_cold * self.signal_mean
        llr = (
            -0.5 * ((site_mean - mu_hot) ** 2) / site_var
            + 0.5 * ((site_mean - mu_cold) ** 2) / site_var
        )
        new_sp = _sigmoid(math.log(sp / (1.0 - sp)) + llr)
        old_sp = self.site_prob[site_idx]
        self.site_prob[site_idx] = new_sp

        # Propagate site posterior update to sibling channels in this site
        old_base = old_sp * self.prior_active_hot + (1.0 - old_sp) * self.prior_active_cold
        new_base = new_sp * self.prior_active_hot + (1.0 - new_sp) * self.prior_active_cold
        delta_lo = math.log((new_base / (1.0 - new_base)) / (old_base / (1.0 - old_base)))
        for j in range(m):
            idx = base + j
            cp = min(max(self.chan_prob[idx], 1e-9), 1.0 - 1e-9)
            self.chan_prob[idx] = _sigmoid(math.log(cp / (1.0 - cp)) + delta_lo)
        return new_sp

    def select_next(
        self,
        cost_site: float = 1.0,
        cost_channel: float = 1.0,
        cost_ticks: float = 2.0,
        ticks_per_read: int = 8,
    ) -> SiftAction:
        best_type = 0
        best_idx = 0
        best_score = -1.0

        mu = self.signal_mean
        noise_sd = self.noise_sd
        per_site = self.per_site
        chan_prob = self.chan_prob
        chan_ns = self.chan_noise_scale
        site_prob = self.site_prob
        _sqrt = math.sqrt
        _exp = math.exp

        # Action 0: Read site aggregate mean
        inv_cs = 1.0 / max(cost_site, 1e-9)
        delta_mu_site = (self.prior_active_hot - self.prior_active_cold) * mu
        for s in range(self.n_sites):
            sp = site_prob[s]
            unc = sp * (1.0 - sp)
            if unc <= 1e-12:
                continue
            base = s * per_site
            sum_v = 0.0
            for j in range(per_site):
                ns = chan_ns[base + j]
                sum_v += (noise_sd * ns) ** 2
            site_sd = _sqrt(sum_v) / per_site
            snr = delta_mu_site / max(site_sd, 1e-9)
            # Expected info gain proxy: uncertainty * SNR^2 / (1 + SNR^2) * per_site
            evoi = unc * per_site * (1.0 - _exp(-0.5 * snr * snr))
            score = evoi * inv_cs
            if score > best_score:
                best_score = score
                best_type = 0
                best_idx = s

        # Action 1 (single channel mean) & Action 2 (raw ticks)
        inv_cc = 1.0 / max(cost_channel, 1e-9)
        inv_ct = 1.0 / max(cost_ticks, 1e-9)
        sqrt_ticks = _sqrt(max(1, ticks_per_read))
        for c in range(self.n_channels):
            cp = chan_prob[c]
            unc = cp * (1.0 - cp)
            if unc <= 1e-12:
                continue
            sd1 = noise_sd * chan_ns[c]
            snr1 = mu / max(sd1, 1e-9)
            evoi1 = unc * (1.0 - _exp(-0.25 * snr1 * snr1))
            s1 = evoi1 * inv_cc
            if s1 > best_score:
                best_score = s1
                best_type = 1
                best_idx = c

            snr2 = snr1 * sqrt_ticks
            evoi2 = unc * (1.0 - _exp(-0.25 * snr2 * snr2))
            s2 = evoi2 * inv_ct
            if s2 > best_score:
                best_score = s2
                best_type = 2
                best_idx = c

        return SiftAction(action_type=best_type, target_idx=best_idx, evoi_per_cost=best_score)
