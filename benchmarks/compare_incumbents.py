#!/usr/bin/env python3
"""Reproducible [LIVE-CPU] head-to-head comparison of ``eproc`` vs incumbent reference algorithms.

Zero third-party dependencies (Python 3 stdlib + ``eproc`` + ``benchmarks/baselines.py``).

Compares latency, memory/wire footprint, and statistical error across four workloads:
  1. Continuous peeking: ``eproc.EProcess`` vs every-tick Student-t peeking vs
     fixed-horizon ``N`` vs pure-Python ONS betting (``confseq``-style reference).
  2. Online conformal: ``eproc.MultiRateConformal`` (5 crossing-free rates) vs
     pure-Python AgACI + Conformal-PID (``online_conformal``/``MAPIE`` online loop)
     and naive independent 5-rate ACI (crossing check).
  3. Distributed fleet conformal: ``eproc.ConformalMonoid`` (640 B fixed wire) vs
     raw float64 score array exchange + exact LOO recomputation.
  4. Multi-agent fusion: ``eproc.SafeFuser`` vs naive mean vs coordinate-wise
     median vs anchor baseline under 1e6 Byzantine spikes + regime shift.

Usage:
  PYTHONPATH=python python3 benchmarks/compare_incumbents.py
  PYTHONPATH=python python3 benchmarks/compare_incumbents.py --json /tmp/compare.json
"""

from __future__ import annotations

import argparse
import array
import ctypes
import datetime as _dt
import gc
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from typing import Any, Callable, Sequence

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
if _PYTHON_DIR.is_dir() and str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import baselines as base  # noqa: E402
import eproc  # noqa: E402
from eproc import _ffi  # noqa: E402


SEEDS: dict[str, int] = {
    "peeking_null": 20261201,
    "peeking_alt": 20261202,
    "conformal_stream": 20261203,
    "fleet": 20261204,
    "fusion": 20261205,
}


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _percentile(vals: Sequence[float], q: float) -> float:
    s = sorted(vals)
    if not s:
        return 0.0
    if len(s) == 1:
        return float(s[0])
    pos = (q / 100.0) * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return (max(0.0, center - half), min(1.0, center + half))


def _paired_ns(
    fn_a: Callable[[], None],
    fn_b: Callable[[], None],
    ops_per_run: int,
    n_runs: int = 5,
) -> tuple[dict[str, float], dict[str, float]]:
    """Measure paired alternating runs (AB/BA) returning median and p95 ns/op."""
    # Warmup
    fn_a()
    fn_b()
    a_samples: list[float] = []
    b_samples: list[float] = []
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for r in range(n_runs):
            if r % 2 == 0:
                t0 = time.perf_counter_ns()
                fn_a()
                t1 = time.perf_counter_ns()
                fn_b()
                t2 = time.perf_counter_ns()
                a_samples.append((t1 - t0) / ops_per_run)
                b_samples.append((t2 - t1) / ops_per_run)
            else:
                t0 = time.perf_counter_ns()
                fn_b()
                t1 = time.perf_counter_ns()
                fn_a()
                t2 = time.perf_counter_ns()
                b_samples.append((t1 - t0) / ops_per_run)
                a_samples.append((t2 - t1) / ops_per_run)
    finally:
        if gc_was_enabled:
            gc.enable()
    return (
        {"median_ns": statistics.median(a_samples), "p95_ns": _percentile(a_samples, 95.0)},
        {"median_ns": statistics.median(b_samples), "p95_ns": _percentile(b_samples, 95.0)},
    )


def _time_ns(fn: Callable[[], None], ops_per_run: int, n_runs: int = 5) -> dict[str, float]:
    fn()
    samples: list[float] = []
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(n_runs):
            t0 = time.perf_counter_ns()
            fn()
            t1 = time.perf_counter_ns()
            samples.append((t1 - t0) / ops_per_run)
    finally:
        if gc_was_enabled:
            gc.enable()
    return {"median_ns": statistics.median(samples), "p95_ns": _percentile(samples, 95.0)}


# ---------------------------------------------------------------------------
# Additional Incumbent Reference Implementations (pure stdlib)
# ---------------------------------------------------------------------------


class _PyIndependentMultiRateACI:
    """5 independent single-rate ACI loops (without PAVA isotonic projection).

    Represents running 5 separate single-rate ACI instances (as in standard
    single-rate online conformal libraries), which can cross under distribution shift.
    """

    def __init__(
        self,
        alphas: Sequence[float] = (0.50, 0.20, 0.10, 0.05, 0.01),
        window_size: int = 128,
        gamma: float = 0.05,
    ) -> None:
        self.target_alphas = sorted((float(a) for a in alphas), reverse=True)
        self.n_rates = len(self.target_alphas)
        self.win_cap = int(window_size)
        self.gamma = float(gamma)
        self.eff_alpha = list(self.target_alphas)
        self.current_q = [1.0] * self.n_rates
        self.ring: list[float] = []
        self.ring_pos = 0
        self.sorted_win: list[float] = []

    def _q(self, alpha_eff: float) -> float:
        wlen = len(self.sorted_win)
        if wlen == 0:
            return 1.0
        a = min(max(alpha_eff, 1e-6), 1.0 - 1e-6)
        idx = int(math.ceil((1.0 - a) * (wlen + 1))) - 1
        idx = max(0, min(wlen - 1, idx))
        return self.sorted_win[idx]

    def update(self, score: float) -> list[float]:
        import bisect

        for r in range(self.n_rates):
            err = 1.0 if score > self.current_q[r] else 0.0
            a = self.eff_alpha[r] + self.gamma * (self.target_alphas[r] - err)
            self.eff_alpha[r] = min(max(a, 1e-4), 0.999)

        if len(self.sorted_win) < self.win_cap:
            self.ring.append(score)
            bisect.insort(self.sorted_win, score)
        else:
            old = self.ring[self.ring_pos]
            self.ring[self.ring_pos] = score
            self.ring_pos = (self.ring_pos + 1) % self.win_cap
            del self.sorted_win[bisect.bisect_left(self.sorted_win, old)]
            bisect.insort(self.sorted_win, score)

        self.current_q = [self._q(a) for a in self.eff_alpha]
        return self.current_q


class _PyCoordinateMedianFuser:
    """Coordinate-wise median multi-agent fuser (classic Byzantine-robust static rule)."""

    __slots__ = ("n_experts", "n_updates")

    def __init__(self, n_experts: int) -> None:
        self.n_experts = int(n_experts)
        self.n_updates = 0

    def predict(self, preds: Sequence[float]) -> float:
        s = sorted(preds)
        m = len(s)
        mid = m // 2
        if m % 2 == 1:
            return float(s[mid])
        return 0.5 * (float(s[mid - 1]) + float(s[mid]))

    def update(self, preds: Sequence[float], y_true: float) -> float:
        self.n_updates += 1
        return self.predict(preds)


# ---------------------------------------------------------------------------
# 1. Continuous Peeking Comparison
# ---------------------------------------------------------------------------


def run_peeking_comparison() -> dict[str, Any]:
    n_null_trials = 1000
    horizon = 1000
    fixed_n = 857  # 1-sided alpha=0.05, 90% power at MDE=0.05 around p0=0.50
    alpha = 0.05
    rng_null = random.Random(SEEDS["peeking_null"])

    ep_fp = 0
    peek_t_fp = 0
    fixed_t_fp = 0
    for _ in range(n_null_trials):
        stream = [1.0 if rng_null.random() < 0.50 else 0.0 for _ in range(horizon)]
        ep = eproc.EProcess(mu0=0.50, alpha=alpha, lower=0.0, upper=1.0, side="upper")
        if ep.update_batch(array.array("d", stream)).rejected:
            ep_fp += 1

        pt = base.PyFixedHorizonPeekTest(mu0=0.50, alpha=alpha, side="upper", min_n=10, use_t_crit=True)
        if pt.update_batch(stream).rejected:
            peek_t_fp += 1

        # Fixed-horizon one-time check at exactly N = fixed_n
        sub = stream[:fixed_n]
        mean = sum(sub) / fixed_n
        var = sum((x - mean) ** 2 for x in sub) / (fixed_n - 1)
        if var > 1e-18:
            t_stat = (mean - 0.50) / math.sqrt(var / fixed_n)
            if t_stat >= base.t_crit(fixed_n - 1, alpha, two_sided=False):
                fixed_t_fp += 1

    # Stopping power & savings on real uplifts vs fixed-horizon N=857
    rng_alt = random.Random(SEEDS["peeking_alt"])
    n_alt_trials = 400
    uplift_rows: list[dict[str, Any]] = []
    for delta in (0.08, 0.12, 0.15):
        p_true = 0.50 + delta
        stop_ns: list[int] = []
        ep_hits = 0
        fixed_hits = 0
        for _ in range(n_alt_trials):
            stream = [1.0 if rng_alt.random() < p_true else 0.0 for _ in range(fixed_n)]
            ep = eproc.EProcess(mu0=0.50, alpha=alpha, lower=0.0, upper=1.0, side="upper")
            res = ep.update_batch(array.array("d", stream))
            if res.rejected:
                ep_hits += 1
                stop_ns.append(ep.rejected_at)
            else:
                stop_ns.append(fixed_n)

            mean = sum(stream) / fixed_n
            var = sum((x - mean) ** 2 for x in stream) / (fixed_n - 1)
            if var > 1e-18:
                t_stat = (mean - 0.50) / math.sqrt(var / fixed_n)
                if t_stat >= base.t_crit(fixed_n - 1, alpha, two_sided=False):
                    fixed_hits += 1

        med_stop = statistics.median(stop_ns)
        mean_stop = statistics.mean(stop_ns)
        uplift_rows.append(
            {
                "delta": delta,
                "p_true": p_true,
                "fixed_n": fixed_n,
                "eproc_power_pct": round(100.0 * ep_hits / n_alt_trials, 2),
                "fixed_power_pct": round(100.0 * fixed_hits / n_alt_trials, 2),
                "eproc_median_stop_n": float(med_stop),
                "eproc_mean_stop_n": round(mean_stop, 1),
                "median_speedup_x": round(fixed_n / med_stop, 2),
                "mean_samples_saved_pct": round(100.0 * (1.0 - mean_stop / fixed_n), 2),
            }
        )

    # Latency benchmark (5 paired runs over 50,000 steps)
    lat_n = 50_000
    lat_rng = random.Random(SEEDS["peeking_null"] + 7)
    lat_list = [1.0 if lat_rng.random() < 0.55 else 0.0 for _ in range(lat_n)]
    lat_arr = array.array("d", lat_list)

    def _run_eproc_batch() -> None:
        ep = eproc.EProcess(mu0=0.50, alpha=alpha)
        ep.update_batch(lat_arr)

    def _run_py_ons_batch() -> None:
        py = base.PyONSBettingEProcess(mu0=0.50, alpha=alpha)
        py.update_batch(lat_list)

    def _run_eproc_scalar() -> None:
        ep = eproc.EProcess(mu0=0.50, alpha=alpha)
        upd = ep.update
        for x in lat_list:
            upd(x)

    def _run_peek_t_scalar() -> None:
        pt = base.PyFixedHorizonPeekTest(mu0=0.50, alpha=alpha, min_n=10, use_t_crit=True)
        upd = pt.update
        for x in lat_list:
            upd(x)

    eproc_batch_t, py_ons_batch_t = _paired_ns(_run_eproc_batch, _run_py_ons_batch, lat_n)
    eproc_scalar_t, peek_t_scalar_t = _paired_ns(_run_eproc_scalar, _run_peek_t_scalar, lat_n)

    return {
        "null_trials": n_null_trials,
        "horizon": horizon,
        "fixed_horizon_n": fixed_n,
        "alpha": alpha,
        "eproc_null_fpr": ep_fp / n_null_trials,
        "eproc_null_ci": _wilson_ci(ep_fp, n_null_trials),
        "student_t_peek_null_fpr": peek_t_fp / n_null_trials,
        "student_t_peek_null_ci": _wilson_ci(peek_t_fp, n_null_trials),
        "fixed_horizon_t_null_fpr": fixed_t_fp / n_null_trials,
        "fixed_horizon_t_null_ci": _wilson_ci(fixed_t_fp, n_null_trials),
        "uplift_early_stop": uplift_rows,
        "latency": {
            "eproc_batch_ns": eproc_batch_t,
            "eproc_scalar_ns": eproc_scalar_t,
            "py_ons_batch_ns": py_ons_batch_t,
            "student_t_peek_scalar_ns": peek_t_scalar_t,
            "speedup_batch_vs_py_ons_x": round(
                py_ons_batch_t["median_ns"] / eproc_batch_t["median_ns"], 2
            ),
            "speedup_scalar_vs_peek_t_x": round(
                peek_t_scalar_t["median_ns"] / eproc_scalar_t["median_ns"], 2
            ),
        },
        "state_bytes": {
            "eproc_c_struct_bytes": ctypes.sizeof(_ffi.eproc_bet_t),
        },
    }


# ---------------------------------------------------------------------------
# 2. Online Conformal Comparison (5-rate MultiRateConformal vs Pure-Python ACI/PID)
# ---------------------------------------------------------------------------


def run_conformal_comparison() -> dict[str, Any]:
    alphas = (0.50, 0.20, 0.10, 0.05, 0.01)
    rng = random.Random(SEEDS["conformal_stream"])
    n_ticks = 2000
    # Non-stationary stream: sigma=1.0 for t<1000, sudden 4x variance shock for t>=1000
    scores = [abs(rng.gauss(0.0, 1.0 if t < 1000 else 4.0)) for t in range(n_ticks)]
    scores_arr = array.array("d", scores)

    def _eval_coverage_and_crossings(
        name: str,
        step_fn: Callable[[float], tuple[float, ...]],
    ) -> dict[str, Any]:
        hits = [0] * len(alphas)
        post_hits = [0] * len(alphas)
        crossings = 0
        eval_start = 200
        for t, s in enumerate(scores):
            qs = step_fn(s)
            if t >= eval_start:
                for r in range(len(alphas)):
                    if s <= qs[r]:
                        hits[r] += 1
                        if t >= 1200:
                            post_hits[r] += 1
            for r in range(len(alphas) - 1):
                if qs[r] > qs[r + 1] + 1e-12:
                    crossings += 1
                    break
        n_eval = n_ticks - eval_start
        n_post = n_ticks - 1200
        covs = [hits[r] / n_eval for r in range(len(alphas))]
        post_covs = [post_hits[r] / n_post for r in range(len(alphas))]
        targets = [1.0 - a for a in alphas]
        mae = sum(abs(c - tgt) for c, tgt in zip(covs, targets)) / len(alphas)
        post_mae = sum(abs(c - tgt) for c, tgt in zip(post_covs, targets)) / len(alphas)
        return {
            "method": name,
            "empirical_coverage": [round(100.0 * c, 2) for c in covs],
            "post_shock_coverage": [round(100.0 * c, 2) for c in post_covs],
            "mean_abs_cov_err_pct": round(100.0 * mae, 2),
            "post_shock_mae_pct": round(100.0 * post_mae, 2),
            "crossing_ticks": crossings,
        }

    cp_eproc = eproc.MultiRateConformal(alphas=alphas, window_size=128, pid_ki=0.05)

    def _step_eproc(s: float) -> tuple[float, ...]:
        qs = cp_eproc.quantiles
        cp_eproc.update(s)
        return qs

    cp_py_agaci = base.PyMultiRateAgACI(alphas=alphas, window_size=128, pid_ki=0.05)

    def _step_py_agaci(s: float) -> tuple[float, ...]:
        qs = tuple(cp_py_agaci.current_q)
        cp_py_agaci.update(s)
        return qs

    cp_indep = _PyIndependentMultiRateACI(alphas=alphas, window_size=128, gamma=0.08)

    def _step_indep(s: float) -> tuple[float, ...]:
        qs = tuple(cp_indep.current_q)
        cp_indep.update(s)
        return qs

    acc_eproc = _eval_coverage_and_crossings("eproc.MultiRateConformal", _step_eproc)
    acc_py_agaci = _eval_coverage_and_crossings("PyMultiRateAgACI (online_conformal/MAPIE ref)", _step_py_agaci)
    acc_indep = _eval_coverage_and_crossings("Independent 5-rate ACI (no PAVA)", _step_indep)

    # Latency benchmark (5 paired runs)
    lat_n = 2000

    def _bench_eproc_batch() -> None:
        c = eproc.MultiRateConformal(alphas=alphas, window_size=128, pid_ki=0.05)
        c.update_batch(scores_arr)

    def _bench_eproc_scalar() -> None:
        c = eproc.MultiRateConformal(alphas=alphas, window_size=128, pid_ki=0.05)
        upd = c.update
        for s in scores:
            upd(s)

    def _bench_py_agaci() -> None:
        c = base.PyMultiRateAgACI(alphas=alphas, window_size=128, pid_ki=0.05)
        c.update_batch(scores)

    def _bench_py_indep_5rate() -> None:
        c = _PyIndependentMultiRateACI(alphas=alphas, window_size=128, gamma=0.08)
        upd = c.update
        for s in scores:
            upd(s)

    eproc_batch_t, py_agaci_t = _paired_ns(_bench_eproc_batch, _bench_py_agaci, lat_n)
    eproc_scalar_t, py_indep_t = _paired_ns(_bench_eproc_scalar, _bench_py_indep_5rate, lat_n)

    return {
        "alphas": list(alphas),
        "n_ticks": n_ticks,
        "accuracy": {
            "eproc": acc_eproc,
            "py_agaci_pid": acc_py_agaci,
            "independent_5rate_aci": acc_indep,
        },
        "latency": {
            "eproc_batch_ns": eproc_batch_t,
            "eproc_scalar_ns": eproc_scalar_t,
            "py_agaci_pid_ns": py_agaci_t,
            "py_independent_5rate_aci_ns": py_indep_t,
            "speedup_batch_vs_py_agaci_x": round(
                py_agaci_t["median_ns"] / eproc_batch_t["median_ns"], 2
            ),
            "speedup_scalar_vs_py_agaci_x": round(
                py_agaci_t["median_ns"] / eproc_scalar_t["median_ns"], 2
            ),
            "speedup_batch_vs_py_indep_aci_x": round(
                py_indep_t["median_ns"] / eproc_batch_t["median_ns"], 2
            ),
        },
        "state_bytes": {
            "eproc_c_struct_bytes": ctypes.sizeof(_ffi.eproc_conformal_t),
        },
    }


# ---------------------------------------------------------------------------
# 3. Distributed Fleet Conformal Comparison (ConformalMonoid vs Raw Float64 Arrays)
# ---------------------------------------------------------------------------


def run_fleet_comparison() -> dict[str, Any]:
    n_agents = 256
    scores_per_agent = 500
    total_scores = n_agents * scores_per_agent
    rng = random.Random(SEEDS["fleet"])

    shard_lists: list[list[float]] = [
        [abs(rng.gauss(0.0, 1.0)) for _ in range(scores_per_agent)]
        for _ in range(n_agents)
    ]
    shard_arrs = [array.array("d", s) for s in shard_lists]

    monoids: list[eproc.ConformalMonoid] = []
    raw_shards: list[base.PyRawScoreDistributedConformal] = []
    for arr in shard_arrs:
        m = eproc.ConformalMonoid(scale_hint=1.0)
        m.add_batch(arr)
        monoids.append(m)
        r = base.PyRawScoreDistributedConformal()
        r.add_batch(arr)
        raw_shards.append(r)

    def _tree_merge_monoids() -> eproc.ConformalMonoid:
        lvl = list(monoids)
        while len(lvl) > 1:
            lvl = [
                lvl[i].merge(lvl[i + 1]) if i + 1 < len(lvl) else lvl[i]
                for i in range(0, len(lvl), 2)
            ]
        return lvl[0]

    def _tree_merge_raw() -> base.PyRawScoreDistributedConformal:
        lvl = list(raw_shards)
        while len(lvl) > 1:
            lvl = [
                lvl[i].merge(lvl[i + 1]) if i + 1 < len(lvl) else lvl[i]
                for i in range(0, len(lvl), 2)
            ]
        return lvl[0]

    global_m = _tree_merge_monoids()
    global_r = _tree_merge_raw()

    # Check merge time (total ms for 256-agent tree reduction)
    m_merge_t, r_merge_t = _paired_ns(_tree_merge_monoids, _tree_merge_raw, 1, n_runs=5)

    # Check exact LOO peer calibration (subtract + quantile(0.90) for 16 agents)
    n_loo_probes = 16

    def _loo_monoid() -> None:
        for i in range(n_loo_probes):
            peer = global_m.subtract(monoids[i])
            _ = peer.quantile(0.90)

    def _loo_raw() -> None:
        for i in range(n_loo_probes):
            peer = global_r.subtract(raw_shards[i])
            _ = peer.quantile(0.90)

    m_loo_t, r_loo_t = _paired_ns(_loo_monoid, _loo_raw, n_loo_probes, n_runs=5)

    # Quantile accuracy vs exact sorted float64
    q_rows: list[dict[str, Any]] = []
    for q in (0.50, 0.80, 0.90, 0.95, 0.99):
        exact = global_r.quantile(q)
        approx = global_m.quantile(q)
        abs_err = abs(approx - exact)
        rel_pct = 100.0 * abs_err / max(abs(exact), 1e-12)
        q_rows.append(
            {
                "q": q,
                "exact_raw": round(exact, 5),
                "monoid_640b": round(approx, 5),
                "abs_err": round(abs_err, 5),
                "rel_err_pct": round(rel_pct, 3),
            }
        )

    # Verify 0 integer bin drift on LOO across 16 agents
    loo_bin_drift = 0
    for i in range(n_loo_probes):
        peer = global_m.subtract(monoids[i])
        rebuilt = eproc.ConformalMonoid(scale_hint=1.0)
        for j in range(n_agents):
            if j != i:
                rebuilt = rebuilt.merge(monoids[j])
        if peer.bin_counts != rebuilt.bin_counts:
            loo_bin_drift += 1

    monoid_wire_b = global_m.wire_size
    raw_agent_wire_b = raw_shards[0].wire_size()
    raw_global_wire_b = global_r.wire_size()

    return {
        "n_agents": n_agents,
        "scores_per_agent": scores_per_agent,
        "total_scores": total_scores,
        "wire_bytes": {
            "monoid_per_agent": monoid_wire_b,
            "monoid_global": monoid_wire_b,
            "raw_per_agent": raw_agent_wire_b,
            "raw_global": raw_global_wire_b,
            "per_agent_compression_x": round(raw_agent_wire_b / monoid_wire_b, 2),
            "global_compression_x": round(raw_global_wire_b / monoid_wire_b, 1),
        },
        "latency": {
            "monoid_tree_merge_ms": round(m_merge_t["median_ns"] / 1e6, 4),
            "raw_tree_merge_ms": round(r_merge_t["median_ns"] / 1e6, 4),
            "tree_merge_speedup_x": round(r_merge_t["median_ns"] / m_merge_t["median_ns"], 2),
            "monoid_loo_per_agent_us": round(m_loo_t["median_ns"] / 1e3, 3),
            "raw_loo_per_agent_us": round(r_loo_t["median_ns"] / 1e3, 3),
            "loo_speedup_x": round(r_loo_t["median_ns"] / m_loo_t["median_ns"], 1),
        },
        "loo_integer_bin_drift": loo_bin_drift,
        "quantiles": q_rows,
    }


# ---------------------------------------------------------------------------
# 4. Multi-Agent Fusion Comparison (SafeFuser vs Naive Mean vs Coordinate Median)
# ---------------------------------------------------------------------------


def run_fusion_comparison() -> dict[str, Any]:
    n_experts = 8
    n_ticks = 1000
    attack_start = 300
    shift_start = 600
    rng = random.Random(SEEDS["fusion"])

    # Generate deterministic 8-agent trajectory with:
    # - Agent 0: conservative anchor (sigma=0.45)
    # - Agents 1-2: sharp pre-shift experts (sigma=0.10), degrade to sigma=0.75 after shift
    # - Agents 3-4: moderate pre-shift (sigma=0.55), become sharp (sigma=0.10) after shift
    # - Agents 5-6: noisy peers (sigma=0.60)
    # - Agent 7: clean before t=300 (sigma=0.25), Byzantine +1e6 spike attacker for t>=300
    ys: list[float] = []
    pred_rows: list[list[float]] = []
    for t in range(n_ticks):
        y = 2.0 * math.sin(0.03 * t) + 0.5 * math.cos(0.07 * t)
        ys.append(y)
        if t < shift_start:
            sigmas = [0.45, 0.10, 0.12, 0.55, 0.58, 0.60, 0.62, 0.25]
        else:
            sigmas = [0.45, 0.75, 0.80, 0.10, 0.12, 0.60, 0.62, 0.25]
        row = [y + rng.gauss(0.0, s) for s in sigmas]
        if t >= attack_start:
            row[7] = y + (1e6 if (t % 2 == 0) else -1e6)
        pred_rows.append(row)

    sf = eproc.SafeFuser(n_experts=n_experts, anchor_idx=0, alpha_deg=0.05)
    naive = base.PyNaiveAverageFuser(n_experts=n_experts)
    med = _PyCoordinateMedianFuser(n_experts=n_experts)

    def _accum() -> dict[str, Any]:
        return {"pre": [], "attack": [], "post": [], "max_err": 0.0}

    stats = {
        "eproc_safe_fuser": _accum(),
        "naive_mean": _accum(),
        "coordinate_median": _accum(),
        "anchor_only": _accum(),
    }

    for t in range(n_ticks):
        row = pred_rows[t]
        y = ys[t]
        step = sf.update(row, y)
        p_naive, _ = naive.update(row, y)
        p_med = med.update(row, y)
        p_anc = row[0]

        phase = "pre" if t < attack_start else ("attack" if t < shift_start else "post")
        for key, pred in (
            ("eproc_safe_fuser", step.fused_pred),
            ("naive_mean", p_naive),
            ("coordinate_median", p_med),
            ("anchor_only", p_anc),
        ):
            err = abs(pred - y)
            stats[key][phase].append(err * err)
            if err > stats[key]["max_err"]:
                stats[key]["max_err"] = err

    summary: dict[str, Any] = {}
    for key, st in stats.items():
        all_after_attack = st["attack"] + st["post"]
        summary[key] = {
            "pre_attack_mse": round(statistics.mean(st["pre"]), 5),
            "attack_phase_mse": round(statistics.mean(st["attack"]), 5),
            "post_shift_mse": round(statistics.mean(st["post"]), 5),
            "combined_attack_post_mse": round(statistics.mean(all_after_attack), 5),
            "max_abs_err": round(st["max_err"], 4),
        }

    # Latency benchmark (5 paired runs over 1,000 steps)
    def _bench_eproc_sf() -> None:
        f = eproc.SafeFuser(n_experts=n_experts, anchor_idx=0, alpha_deg=0.05)
        upd = f.update
        for row, y in zip(pred_rows, ys):
            upd(row, y)

    def _bench_py_boa() -> None:
        f = base.PySafeBOA(n_experts=n_experts, anchor_idx=0, alpha_deg=0.05)
        upd = f.update
        for row, y in zip(pred_rows, ys):
            upd(row, y)

    def _bench_med() -> None:
        f = _PyCoordinateMedianFuser(n_experts=n_experts)
        upd = f.update
        for row, y in zip(pred_rows, ys):
            upd(row, y)

    def _bench_naive() -> None:
        f = base.PyNaiveAverageFuser(n_experts=n_experts)
        upd = f.update
        for row, y in zip(pred_rows, ys):
            upd(row, y)

    sf_t, py_boa_t = _paired_ns(_bench_eproc_sf, _bench_py_boa, n_ticks)
    med_t, naive_t = _paired_ns(_bench_med, _bench_naive, n_ticks)

    return {
        "n_experts": n_experts,
        "n_ticks": n_ticks,
        "attack_start": attack_start,
        "shift_start": shift_start,
        "safe_fuser_diagnostics": {
            "quarantined_count": sum(sf.quarantined),
            "poison_clipped": sf.n_poison_clipped,
            "fallback_trips": sf.n_fallback_trips,
        },
        "accuracy": summary,
        "latency": {
            "eproc_safe_fuser_ns": sf_t,
            "py_safe_boa_ns": py_boa_t,
            "coordinate_median_ns": med_t,
            "naive_mean_ns": naive_t,
            "speedup_vs_py_safe_boa_x": round(py_boa_t["median_ns"] / sf_t["median_ns"], 2),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="eproc vs incumbent reference benchmarks")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    t0 = time.perf_counter()
    report: dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "cpu": _cpu_model(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "eproc_version": eproc.__version__,
            "abi_version": eproc.abi_version(),
            "simd_flags": eproc.simd_flags(),
            "simd_names": eproc.simd_flag_names(eproc.simd_flags()),
        },
        "peeking": run_peeking_comparison(),
        "conformal": run_conformal_comparison(),
        "fleet": run_fleet_comparison(),
        "fusion": run_fusion_comparison(),
    }
    report["wall_time_s"] = round(time.perf_counter() - t0, 2)

    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
