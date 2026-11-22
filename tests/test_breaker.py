"""Independent Adversarial Breaker Red-Team Audit Suite for ``eproc``.

Attacks ALL 6 adversarial fronts without modifying implementation files:
  - Front A: Anytime-valid Type-I error & simultaneous coverage across 5,000+ nulls
  - Front B: Strict causality (``assert_causal``, 0.0 exact bitwise diff) across all engines
  - Front C: Exact algebraic monoid & group inverse invariants across 1,000 random shards
  - Front D: Extreme numerical, Byzantine poisoning, echo saturation & monotonicity torture
  - Front E: C11 ASan+UBSan, TSan, 1M concurrent seqlock stress & Python thread safety
  - Front F: Packaging, cold import latency, hygiene, single-header C/C++, MCP & CLI torture

Run via:
    PYTHONPATH=python python3 -m unittest tests/test_breaker.py -v
"""

from __future__ import annotations

import ast
import base64
import ctypes
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = REPO_ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

import eproc  # noqa: E402
from eproc import cli, mcp_server  # noqa: E402


def _f64_bits(x: float) -> bytes:
    """Exact IEEE-754 binary64 bit pattern (treats NaN payloads strictly)."""
    return struct.pack("!d", float(x))


def _assert_bit_identical(tc: unittest.TestCase, a: object, b: object, msg: str = "") -> None:
    """Recursively assert 0.0 exact bitwise identity across scalars/tuples/lists/dicts."""
    if isinstance(a, float) and isinstance(b, float):
        tc.assertEqual(_f64_bits(a), _f64_bits(b), f"Bitwise float mismatch {a!r} != {b!r}: {msg}")
    elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        tc.assertEqual(len(a), len(b), f"Length mismatch: {msg}")
        for idx, (x, y) in enumerate(zip(a, b)):
            _assert_bit_identical(tc, x, y, f"{msg}[{idx}]")
    elif isinstance(a, dict) and isinstance(b, dict):
        tc.assertEqual(set(a.keys()), set(b.keys()), f"Key mismatch: {msg}")
        for k in a:
            _assert_bit_identical(tc, a[k], b[k], f"{msg}.{k}")
    else:
        tc.assertEqual(a, b, msg)


# ============================================================================
# FRONT A — ANYTIME-VALID TYPE-I ERROR TORTURE (5,000+ NULLS)
# ============================================================================


class TestFrontA_AnytimeValidTypeIError(unittest.TestCase):
    def test_a1_bernoulli_05_upper_and_two_sided_2000_trials(self) -> None:
        """Bernoulli(0.5) upper & two-sided continuous peeking: N=2000, T=500, alpha=0.05."""
        rng = random.Random(20260101)
        n_trials = 2000
        horizon = 500
        alpha = 0.05
        rej_upper = 0
        rej_two = 0

        for _ in range(n_trials):
            ep_u = eproc.EProcess(mu0=0.5, alpha=alpha, lower=0.0, upper=1.0, side="upper")
            ep_2 = eproc.EProcess(mu0=0.5, alpha=alpha, lower=0.0, upper=1.0, side="two_sided")
            hit_u = False
            hit_2 = False
            for _t in range(horizon):
                x = 1.0 if rng.random() < 0.5 else 0.0
                if not hit_u and ep_u.update(x).rejected:
                    hit_u = True
                if not hit_2 and ep_2.update(x).rejected:
                    hit_2 = True
                if hit_u and hit_2:
                    break
            rej_upper += int(hit_u)
            rej_two += int(hit_2)

        fpr_u = rej_upper / n_trials
        fpr_2 = rej_two / n_trials
        self.assertLessEqual(
            fpr_u, 0.055, f"Bern(0.5) upper FPR={fpr_u:.4f} ({rej_upper}/{n_trials}) > 0.055"
        )
        self.assertLessEqual(
            fpr_2, 0.055, f"Bern(0.5) two_sided FPR={fpr_2:.4f} ({rej_two}/{n_trials}) > 0.055"
        )

        # Positive control: verify naive fixed-horizon z-test peeking inflates FPR >= 0.25
        naive_rej = 0
        for _ in range(300):
            s = 0.0
            for t in range(1, horizon + 1):
                s += 1.0 if rng.random() < 0.5 else 0.0
                if t >= 10:
                    z = (s / t - 0.5) / (0.5 / math.sqrt(t))
                    if z >= 1.6448536269514722:
                        naive_rej += 1
                        break
        self.assertGreaterEqual(naive_rej / 300.0, 0.25, "Naive peeking control should exceed 25% FPR")

    def test_a2_boundary_bernoulli_001_and_099_1000_trials(self) -> None:
        """Boundary Bernoulli(0.01) & Bernoulli(0.99): N=1000 each, T=500, alpha=0.05."""
        rng = random.Random(20260102)
        n_trials = 1000
        horizon = 500
        alpha = 0.05

        for p in (0.01, 0.99):
            rej_u = 0
            rej_l = 0
            rej_2 = 0
            for _ in range(n_trials):
                ep_u = eproc.EProcess(mu0=p, alpha=alpha, lower=0.0, upper=1.0, side="upper")
                ep_l = eproc.EProcess(mu0=p, alpha=alpha, lower=0.0, upper=1.0, side="lower")
                ep_2 = eproc.EProcess(mu0=p, alpha=alpha, lower=0.0, upper=1.0, side="two_sided")
                hu = hl = h2 = False
                for _t in range(horizon):
                    x = 1.0 if rng.random() < p else 0.0
                    if not hu and ep_u.update(x).rejected:
                        hu = True
                    if not hl and ep_l.update(x).rejected:
                        hl = True
                    if not h2 and ep_2.update(x).rejected:
                        h2 = True
                    if hu and hl and h2:
                        break
                rej_u += int(hu)
                rej_l += int(hl)
                rej_2 += int(h2)
            for label, cnt in (("upper", rej_u), ("lower", rej_l), ("two_sided", rej_2)):
                fpr = cnt / n_trials
                self.assertLessEqual(
                    fpr, 0.055, f"Bern({p}) {label} FPR={fpr:.4f} ({cnt}/{n_trials}) > 0.055"
                )

    def test_a3_asymmetric_two_point_null(self) -> None:
        """Adversarial 2-point asymmetric null P(X=0)=0.9, P(X=1)=0.1 testing mu0=0.1."""
        rng = random.Random(20260103)
        n_trials = 1000
        horizon = 500
        alpha = 0.05
        rej_u = rej_l = rej_2 = 0

        for _ in range(n_trials):
            ep_u = eproc.EProcess(mu0=0.1, alpha=alpha, lower=0.0, upper=1.0, side="upper")
            ep_l = eproc.EProcess(mu0=0.1, alpha=alpha, lower=0.0, upper=1.0, side="lower")
            ep_2 = eproc.EProcess(mu0=0.1, alpha=alpha, lower=0.0, upper=1.0, side="two_sided")
            hu = hl = h2 = False
            for _t in range(horizon):
                x = 1.0 if rng.random() < 0.1 else 0.0
                if not hu and ep_u.update(x).rejected:
                    hu = True
                if not hl and ep_l.update(x).rejected:
                    hl = True
                if not h2 and ep_2.update(x).rejected:
                    h2 = True
                if hu and hl and h2:
                    break
            rej_u += int(hu)
            rej_l += int(hl)
            rej_2 += int(h2)

        for label, cnt in (("upper", rej_u), ("lower", rej_l), ("two_sided", rej_2)):
            self.assertLessEqual(cnt / n_trials, 0.055, f"Asymmetric 2-pt {label} FPR={cnt/n_trials:.4f}")

    def test_a4_nonstationary_adaptive_martingale_null(self) -> None:
        """Non-stationary adaptive martingale null: E[X_t | F_{t-1}] <= mu0 varies with X_{t-1}."""
        rng = random.Random(20260104)
        n_trials = 1000
        horizon = 500
        alpha = 0.05

        # Regime 1: Markov adaptive conditional mean mu_t = 0.50 if X_{t-1}=1 else 0.30
        rej_markov = 0
        for _ in range(n_trials):
            ep = eproc.EProcess(mu0=0.50, alpha=alpha, lower=0.0, upper=1.0, side="upper")
            prev = 1
            for _t in range(horizon):
                p_t = 0.50 if prev == 1 else 0.30
                x = 1.0 if rng.random() < p_t else 0.0
                prev = int(x)
                if ep.update(x).rejected:
                    rej_markov += 1
                    break
        self.assertLessEqual(
            rej_markov / n_trials,
            0.055,
            f"Adaptive Markov martingale null FPR={rej_markov/n_trials:.4f} > 0.055",
        )

        # Regime 2: Exact boundary conditional mean E[X_t|F_{t-1}] == 0.50 with time-varying distribution
        # (oscillating between maximal-variance Bern(0.5) and low-variance Uniform(0.3, 0.7))
        rej_het = 0
        for _ in range(n_trials):
            ep = eproc.EProcess(mu0=0.50, alpha=alpha, lower=0.0, upper=1.0, side="two_sided")
            for t in range(horizon):
                if t % 2 == 0:
                    x = 1.0 if rng.random() < 0.5 else 0.0
                else:
                    x = rng.uniform(0.3, 0.7)
                if ep.update(x).rejected:
                    rej_het += 1
                    break
        self.assertLessEqual(
            rej_het / n_trials,
            0.055,
            f"Heteroskedastic martingale null FPR={rej_het/n_trials:.4f} > 0.055",
        )

    def test_a5_eb_cs_simultaneous_miscoverage_1000_trials(self) -> None:
        """Empirical-Bernstein CS simultaneous miscoverage over 1,000 trials at alpha=0.05."""
        rng = random.Random(20260105)
        n_trials = 1000
        horizon = 350
        alpha = 0.05

        configs = [
            ("Bern(0.5)", 0.5, lambda r: 1.0 if r.random() < 0.5 else 0.0),
            ("Bern(0.05)", 0.05, lambda r: 1.0 if r.random() < 0.05 else 0.0),
            ("Beta(0.2,0.2)", 0.5, lambda r: r.betavariate(0.2, 0.2)),
        ]
        for name, true_mu, draw in configs:
            misses = 0
            for _ in range(n_trials):
                cs = eproc.ConfidenceSequence(alpha=alpha, lower=0.0, upper=1.0, mode="eb")
                prev_w = float("inf")
                missed = False
                for _t in range(horizon):
                    r = cs.update(draw(rng))
                    self.assertLessEqual(r.width, prev_w + 1e-12, f"{name} running-intersection non-monotone")
                    prev_w = r.width
                    if r.low > true_mu or r.high < true_mu:
                        missed = True
                        break
                misses += int(missed)
            miscov = misses / n_trials
            self.assertLessEqual(
                miscov, 0.055, f"EB-CS {name} simultaneous miscoverage={miscov:.4f} ({misses}/{n_trials}) > 0.055"
            )

    def test_a6_agent_race_global_null_fwer(self) -> None:
        """AgentRace global null (8 identical arms p=0.5, 500 trials): false-winner rate <= 0.055."""
        rng = random.Random(20260106)
        n_trials = 500
        alpha = 0.05
        false_winners = 0
        any_elim = 0

        for _ in range(n_trials):
            race = eproc.AgentRace(n_arms=8, alpha=alpha, lower=0.0, upper=1.0, margin=0.0)
            for _step in range(320):
                arm = race.suggest_arm()
                r = 1.0 if rng.random() < 0.5 else 0.0
                st = race.update(arm, r)
                if st.winner is not None:
                    false_winners += 1
                    break
            if race.status().n_active < 8:
                any_elim += 1

        self.assertLessEqual(
            false_winners / n_trials,
            0.055,
            f"AgentRace global null false-winner rate={false_winners/n_trials:.4f} > 0.055",
        )
        self.assertLessEqual(
            any_elim / n_trials,
            0.055,
            f"AgentRace global null any-elimination FWER={any_elim/n_trials:.4f} > 0.055",
        )

    def test_a7_edetector_null_arl_and_reset_empty_batch_witness(self) -> None:
        """EDetector under null (T=1000, N=500, alpha=0.05): verify ARL >= 1/alpha & clean reset."""
        rng = random.Random(20260107)
        n_trials = 500
        horizon = 1000
        alpha = 0.05
        alarmed = 0
        run_lengths = []

        for _ in range(n_trials):
            ed = eproc.EDetector(mu0=0.5, alpha=alpha, lower=0.0, upper=1.0, side="upper")
            hit_t = horizon
            for t in range(1, horizon + 1):
                x = 1.0 if rng.random() < 0.5 else 0.0
                if ed.update(x).alarm:
                    alarmed += 1
                    hit_t = t
                    break
            run_lengths.append(hit_t)

        censored_arl = sum(run_lengths) / n_trials
        # Page-CUSUM e-detector guarantees ARL >= 1/alpha = 20.0 under H0
        self.assertGreaterEqual(
            censored_arl,
            1.0 / alpha,
            f"EDetector empirical censored ARL={censored_arl:.2f} violated floor 1/alpha={1.0/alpha:.1f}",
        )

        # FAIL WITNESS: After alarm + reset_alarm(), update_batch([]) must report stat == exp(0) == 1.0
        ed_bug = eproc.EDetector(mu0=0.5, alpha=0.05, lower=0.0, upper=1.0, side="upper")
        for _ in range(60):
            ed_bug.update(1.0)
        self.assertGreater(ed_bug.n_alarms, 0)
        ed_bug.reset_alarm()
        res_empty = ed_bug.update_batch([])
        self.assertAlmostEqual(
            res_empty.stat,
            1.0,
            places=12,
            msg=(
                f"BUG WITNESS: EDetector.update_batch([]) after reset_alarm() leaked stale "
                f"pre-reset stat={res_empty.stat} while log_stat={res_empty.log_stat}"
            ),
        )


# ============================================================================
# FRONT B — STRICT CAUSALITY (assert_causal, 0.0 EXACT BITWISE DIFF)
# ============================================================================


class TestFrontB_StrictCausality(unittest.TestCase):
    PERTURBATIONS = (1e15, -1e15, float("nan"), float("inf"), float("-inf"))

    def test_b1_eprocess_strict_causality(self) -> None:
        rng = random.Random(101)
        t0 = 40
        prefix = [rng.random() for _ in range(t0)]
        clean_future = [rng.random() for _ in range(30)]

        for side in ("upper", "lower", "two_sided"):
            ep_ref = eproc.EProcess(mu0=0.5, alpha=0.05, side=side)
            ref_out = [ep_ref.update(x).to_dict() for x in prefix]
            for x in clean_future:
                ep_ref.update(x)

            for pval in self.PERTURBATIONS:
                ep_alt = eproc.EProcess(mu0=0.5, alpha=0.05, side=side)
                alt_out = [ep_alt.update(x).to_dict() for x in prefix]
                for _ in clean_future:
                    ep_alt.update(pval)
                _assert_bit_identical(self, ref_out, alt_out, f"EProcess({side}) perturb={pval}")

    def test_b2_confidence_sequence_strict_causality(self) -> None:
        rng = random.Random(102)
        t0 = 40
        prefix = [rng.random() for _ in range(t0)]
        for mode in ("eb", "normal"):
            cs_ref = eproc.ConfidenceSequence(alpha=0.05, lower=0.0, upper=1.0, mode=mode)
            ref_out = [cs_ref.update(x).to_dict() for x in prefix]
            for _ in range(20):
                cs_ref.update(0.5)

            for pval in self.PERTURBATIONS:
                cs_alt = eproc.ConfidenceSequence(alpha=0.05, lower=0.0, upper=1.0, mode=mode)
                alt_out = [cs_alt.update(x).to_dict() for x in prefix]
                for _ in range(20):
                    cs_alt.update(pval)
                _assert_bit_identical(self, ref_out, alt_out, f"CS({mode}) perturb={pval}")

    def test_b3_edetector_strict_causality(self) -> None:
        rng = random.Random(103)
        t0 = 40
        prefix = [rng.random() for _ in range(t0)]
        ed_ref = eproc.EDetector(mu0=0.5, alpha=0.05, side="upper")
        ref_out = [ed_ref.update(x).to_dict() for x in prefix]
        for _ in range(20):
            ed_ref.update(0.9)

        for pval in self.PERTURBATIONS:
            ed_alt = eproc.EDetector(mu0=0.5, alpha=0.05, side="upper")
            alt_out = [ed_alt.update(x).to_dict() for x in prefix]
            for _ in range(20):
                ed_alt.update(pval)
            _assert_bit_identical(self, ref_out, alt_out, f"EDetector perturb={pval}")

    def test_b4_multirate_conformal_strict_causality(self) -> None:
        rng = random.Random(104)
        t0 = 50
        prefix = [abs(rng.gauss(0.0, 1.0)) for _ in range(t0)]

        cp_ref = eproc.MultiRateConformal(alphas=(0.50, 0.20, 0.10, 0.05, 0.01), window_size=64, pid_ki=0.05)
        ref_out = [(cp_ref.predict(10.0, 1.5).to_dict(), cp_ref.update(s).to_dict()) for s in prefix]
        for _ in range(20):
            cp_ref.update(0.5)

        for pval in self.PERTURBATIONS:
            cp_alt = eproc.MultiRateConformal(alphas=(0.50, 0.20, 0.10, 0.05, 0.01), window_size=64, pid_ki=0.05)
            alt_out = [(cp_alt.predict(10.0, 1.5).to_dict(), cp_alt.update(s).to_dict()) for s in prefix]
            for _ in range(20):
                cp_alt.update(pval)
            _assert_bit_identical(self, ref_out, alt_out, f"MultiRateConformal perturb={pval}")

    def test_b5_delayed_conformal_100_tickets_out_of_order_causality(self) -> None:
        """Issue 100 tickets, reveal out-of-order & late; assert 0.0 diff on prior predictions."""
        rng = random.Random(105)
        preds = [rng.gauss(10.0, 2.0) for _ in range(100)]
        scales = [0.5 + rng.random() * 2.0 for _ in range(100)]
        true_ys = [p + rng.gauss(0.0, s) for p, s in zip(preds, scales)]
        # Reveal order: first 30 revealed interleaved (out-of-order) before tick 60;
        # remaining 70 revealed AFTER all 100 tickets have been issued.
        early_indices = list(range(29, -1, -1))
        late_indices = list(range(99, 29, -1))

        def run_sim(future_override: float | None) -> list[list[tuple[float, float]]]:
            dc = eproc.DelayedConformal(alphas=(0.20, 0.10, 0.05), window_size=64, pid_ki=0.05)
            tickets: list[int] = []
            emitted: list[list[tuple[float, float]]] = []
            for i in range(100):
                tid, ivs = dc.issue(preds[i], scales[i])
                tickets.append(tid)
                emitted.append(ivs)
                if i == 50:
                    for idx in early_indices:
                        dc.reveal(tickets[idx], true_ys[idx])
            # Now reveal all remaining tickets (t_reveal > all t_issue) with clean or poisoned labels
            for idx in late_indices:
                y_val = true_ys[idx] if future_override is None else future_override
                dc.reveal(tickets[idx], y_val)
            self.assertEqual(dc.pending_count, 0)
            return emitted

        ref_emitted = run_sim(None)
        for pval in self.PERTURBATIONS:
            alt_emitted = run_sim(pval)
            _assert_bit_identical(self, ref_emitted, alt_emitted, f"DelayedConformal perturb={pval}")

    def test_b6_budget_conformal_strict_causality(self) -> None:
        rng = random.Random(106)
        t0 = 40
        prefix = [(100.0 + rng.uniform(-10, 10), 105.0 + rng.uniform(0, 30), i % 4 == 0) for i in range(t0)]

        bc_ref = eproc.BudgetConformal(alpha=0.10, window_size=64, censor_penalty=1.4)
        ref_out = []
        for est, cost, cens in prefix:
            ref_out.append(bc_ref.bound(est))
            bc_ref.update(est, cost, censored=cens)
        bc_ref.update(100.0, 200.0, censored=True)

        for pval in (1e15, float("nan"), float("inf")):
            bc_alt = eproc.BudgetConformal(alpha=0.10, window_size=64, censor_penalty=1.4)
            alt_out = []
            for est, cost, cens in prefix:
                alt_out.append(bc_alt.bound(est))
                bc_alt.update(est, cost, censored=cens)
            bc_alt.update(100.0, pval, censored=True)
            _assert_bit_identical(self, ref_out, alt_out, f"BudgetConformal perturb={pval}")

    def test_b7_safe_fuser_strict_causality(self) -> None:
        rng = random.Random(107)
        t0 = 40
        prefix = [([rng.gauss(1.0, 0.3) for _ in range(5)], rng.gauss(1.0, 0.1)) for _ in range(t0)]

        sf_ref = eproc.SafeFuser(n_experts=5, anchor_idx=0, alpha_deg=0.05)
        ref_out = [(sf_ref.predict(ps), sf_ref.update(ps, y).to_dict(), sf_ref.weights()) for ps, y in prefix]
        sf_ref.update([1.0] * 5, 1.0)

        for pval in self.PERTURBATIONS:
            sf_alt = eproc.SafeFuser(n_experts=5, anchor_idx=0, alpha_deg=0.05)
            alt_out = [(sf_alt.predict(ps), sf_alt.update(ps, y).to_dict(), sf_alt.weights()) for ps, y in prefix]
            sf_alt.update([pval] * 5, pval)
            _assert_bit_identical(self, ref_out, alt_out, f"SafeFuser perturb={pval}")

    def test_b8_echopool_strict_causality(self) -> None:
        rng = random.Random(108)
        t0 = 35
        events = [(i % 6, i % 8, f"ev-{i}", rng.gauss(0.0, 1.5)) for i in range(t0)]

        ep_ref = eproc.EchoPool(n_sources=6, n_claims=8, prior_prob=0.5)
        ref_out = [ep_ref.submit(s, c, e, l).to_dict() for s, c, e, l in events]
        ep_ref.submit(0, 0, "future", 3.0)
        ep_ref.resolve(0, True)

        for pval in self.PERTURBATIONS:
            ep_alt = eproc.EchoPool(n_sources=6, n_claims=8, prior_prob=0.5)
            alt_out = [ep_alt.submit(s, c, e, l).to_dict() for s, c, e, l in events]
            ep_alt.submit(0, 0, "future", pval)
            ep_alt.resolve(0, False)
            _assert_bit_identical(self, ref_out, alt_out, f"EchoPool perturb={pval}")

    def test_b9_online_rls_and_varopt_strict_causality(self) -> None:
        rng = random.Random(109)
        t0 = 40
        vecs = [[rng.gauss(0.0, 1.0) for _ in range(6)] for _ in range(t0)]
        rls_ref = eproc.OnlineRLSSampler(d=6, lambda_reg=1.0, oversample=2.0, min_prob=0.05, seed=77)
        ref_rls = [rls_ref.step(v).to_dict() for v in vecs]
        rls_ref.step([1.0] * 6)

        for pval in self.PERTURBATIONS:
            rls_alt = eproc.OnlineRLSSampler(d=6, lambda_reg=1.0, oversample=2.0, min_prob=0.05, seed=77)
            alt_rls = [rls_alt.step(v).to_dict() for v in vecs]
            rls_alt.step([pval] * 6)
            _assert_bit_identical(self, ref_rls, alt_rls, f"OnlineRLS perturb={pval}")

        items = [(i, 0.5 + rng.random() * 10.0, rng.gauss(0.0, 5.0)) for i in range(t0)]
        vo_ref = eproc.VarOptSampler(k=12, seed=88)
        ref_vo = []
        for iid, w, v in items:
            vo_ref.push(iid, w, v)
            ref_vo.append((vo_ref.estimate_sum(), vo_ref.estimate_mean(), vo_ref.tau))
        vo_ref.push(999, 100.0, 5.0)

        for pval in (1e15, 0.0):
            vo_alt = eproc.VarOptSampler(k=12, seed=88)
            alt_vo = []
            for iid, w, v in items:
                vo_alt.push(iid, w, v)
                alt_vo.append((vo_alt.estimate_sum(), vo_alt.estimate_mean(), vo_alt.tau))
            vo_alt.push(999, 1e15, pval)
            _assert_bit_identical(self, ref_vo, alt_vo, f"VarOpt perturb={pval}")


# ============================================================================
# FRONT C — EXACT ALGEBRAIC MONOID & GROUP INVERSE TORTURE (ConformalMonoid)
# ============================================================================


class TestFrontC_ConformalMonoidAlgebra(unittest.TestCase):
    def _rand_scores(self, rng: random.Random, n: int) -> list[float]:
        specials = [0.0, -0.0, 1e-12, -1e-12, 1e12, -1e12, float("nan"), float("inf"), float("-inf")]
        out = []
        for _ in range(n):
            u = rng.random()
            if u < 0.25:
                out.append(rng.choice(specials))
            elif u < 0.65:
                out.append(rng.uniform(-20.0, 40.0))
            else:
                out.append(rng.gauss(0.0, 5.0))
        return out

    def test_c1_1000_random_shards_bin_commutativity_associativity_loo_wire(self) -> None:
        """1,000 random shards: bit-exact bin_counts commutativity, associativity, LOO, wire."""
        rng = random.Random(20260301)
        for trial in range(1000):
            a = eproc.ConformalMonoid(scale_hint=1.0)
            b = eproc.ConformalMonoid(scale_hint=1.0)
            c = eproc.ConformalMonoid(scale_hint=1.0)
            a.add_batch(self._rand_scores(rng, rng.randint(0, 25)))
            b.add_batch(self._rand_scores(rng, rng.randint(0, 25)))
            c.add_batch(self._rand_scores(rng, rng.randint(0, 25)))

            ab = a.merge(b)
            ba = b.merge(a)
            self.assertEqual(ab.count, ba.count, f"trial {trial}: count commutativity")
            self.assertEqual(ab.bin_counts, ba.bin_counts, f"trial {trial}: bin commutativity")
            for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0):
                self.assertEqual(ab.quantile(q), ba.quantile(q), f"trial {trial}: q={q} commutativity")

            lhs = a.merge(b).merge(c)
            rhs = a.merge(b.merge(c))
            self.assertEqual(lhs.count, rhs.count, f"trial {trial}: count associativity")
            self.assertEqual(lhs.bin_counts, rhs.bin_counts, f"trial {trial}: bin associativity")
            for q in (0.0, 0.5, 0.95, 1.0):
                self.assertEqual(lhs.quantile(q), rhs.quantile(q), f"trial {trial}: q={q} associativity")

            loo_b = ab.subtract(a)
            self.assertEqual(loo_b.count, b.count, f"trial {trial}: LOO count")
            self.assertEqual(loo_b.bin_counts, b.bin_counts, f"trial {trial}: LOO bin_counts")

            # Wire roundtrip exactness
            wire = lhs.to_bytes()
            self.assertEqual(len(wire), 640)
            decoded = eproc.ConformalMonoid.from_bytes(wire)
            self.assertEqual(decoded.to_bytes(), wire, f"trial {trial}: wire roundtrip")

    def test_c2_wire_bytes_commutativity_and_associativity_witness(self) -> None:
        """FAIL WITNESS: merge(A,B).to_bytes() must equal merge(B,A).to_bytes() (same scale_hint)."""
        a = eproc.ConformalMonoid(scale_hint=1.0)
        b = eproc.ConformalMonoid(scale_hint=1.0)
        a.add_batch([1e12, 1e-12, 0.5, -1e12, 0.05823])
        b.add_batch([1e12, 2e-12, 0.2, -1e12, -5.0])
        ab = a.merge(b)
        ba = b.merge(a)
        self.assertEqual(
            ab.to_bytes(),
            ba.to_bytes(),
            "BUG WITNESS: ConformalMonoid.merge(A,B).to_bytes() != merge(B,A).to_bytes() "
            "due to asymmetric Neumaier Kahan merge on (sum, sum_comp) and (sum_sq, sum_sq_comp)",
        )

    def test_c3_loo_subtract_quantile_outlier_leak_witness(self) -> None:
        """FAIL WITNESS: subtract(merge(A,B), A).quantile(1.0) leaks removed shard A's 1e12 outlier!"""
        a = eproc.ConformalMonoid(scale_hint=1.0)
        b = eproc.ConformalMonoid(scale_hint=1.0)
        a.add_batch([1e12])
        b.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])
        loo_b = a.merge(b).subtract(a)
        self.assertEqual(loo_b.bin_counts, b.bin_counts)
        self.assertAlmostEqual(
            loo_b.quantile(1.0),
            b.quantile(1.0),
            places=6,
            msg=(
                f"BUG WITNESS: LOO subtract(merge(A,B), A) retains removed shard A's max_val=1e12! "
                f"Expected quantile(1.0)={b.quantile(1.0)}, got {loo_b.quantile(1.0)}"
            ),
        )

    def test_c4_add_vs_add_batch_bin_mismatch_witness(self) -> None:
        """FAIL WITNESS: add(x) vs add_batch([x]) place the same score into different bins."""
        edge = 1.1042051166702023  # exact EPROC_CMONOID_EDGES[25]
        scale = 7.0
        val = edge * scale  # 7.729435816691415 -> scalar bin 26, batch bin 25
        m_scalar = eproc.ConformalMonoid(scale_hint=scale)
        m_batch = eproc.ConformalMonoid(scale_hint=scale)
        m_scalar.add(val)
        m_batch.add_batch([val])
        self.assertEqual(
            m_scalar.bin_counts,
            m_batch.bin_counts,
            "BUG WITNESS: eproc_cmonoid_add uses (score / scale) -> bin 26 while "
            "eproc_cmonoid_add_batch uses (score * (1.0 / scale)) -> bin 25 for identical score 7.729435816691415 at scale=7.0",
        )

    def test_c5_merge_scale_hint_mismatch_witness(self) -> None:
        """FAIL WITNESS: merge() silently ignores scale_hint mismatch (unlike subtract())."""
        ma = eproc.ConformalMonoid(scale_hint=1.0)
        mb = eproc.ConformalMonoid(scale_hint=10.0)
        ma.add(1.0)
        mb.add(10.0)
        with self.assertRaises(
            ValueError,
            msg="BUG WITNESS: ConformalMonoid.merge() silently merges incompatible scale_hint shards",
        ):
            _ = ma.merge(mb)

    def test_c6_underflow_and_corrupted_wire_rejection(self) -> None:
        """Invalid subtract raises ValueError; flipped bit, bad magic, truncated & trailing wire rejected."""
        a = eproc.ConformalMonoid()
        b = eproc.ConformalMonoid()
        a.add_batch([1.0, 2.0])
        b.add_batch([1.0, 2.0, 3.0])

        # Underflow protection: subtracting larger shard MUST raise ValueError
        with self.assertRaises(ValueError):
            a.subtract(b)
        # Disjoint bin underflow (same count, different bin)
        c = eproc.ConformalMonoid()
        c.add_batch([20.0, 25.0])
        with self.assertRaises(ValueError):
            a.subtract(c)

        wire = b.to_bytes()
        # Bit flip across representative offsets (count, bins, floats, magic, checksum)
        for off in (0, 8, 64, 512, 592, 600, 632, 639):
            mut = bytearray(wire)
            mut[off] ^= 0x01
            with self.assertRaises(ValueError, msg=f"Bit flip at offset {off} was not rejected"):
                eproc.ConformalMonoid.from_bytes(bytes(mut))

        # Bad magic
        bad_magic = bytearray(wire)
        bad_magic[592:600] = b"BADMAGIC"
        with self.assertRaises(ValueError):
            eproc.ConformalMonoid.from_bytes(bytes(bad_magic))

        # Empty buffer
        with self.assertRaises(ValueError):
            eproc.ConformalMonoid.from_bytes(b"")

        # Truncated buffer (note: C returns EPROC_ERR_CAPACITY -> OverflowError instead of ValueError!)
        with self.assertRaises((ValueError, OverflowError)):
            eproc.ConformalMonoid.from_bytes(wire[:320])

        # FAIL WITNESS: trailing garbage beyond 640 bytes MUST be rejected!
        with self.assertRaises(
            ValueError,
            msg="BUG WITNESS: ConformalMonoid.from_bytes() silently accepted trailing garbage bytes (len > 640)",
        ):
            eproc.ConformalMonoid.from_bytes(wire + b"TRAILING_GARBAGE")


# ============================================================================
# FRONT D — EXTREME NUMERICAL, BYZANTINE & MONOTONICITY TORTURE
# ============================================================================


class TestFrontD_NumericalByzantineMonotonicity(unittest.TestCase):
    EXTREMES = (
        float("nan"),
        float("inf"),
        float("-inf"),
        1e-310,
        -1e-310,
        1e15,
        -1e15,
        0.0,
        -0.0,
    )

    def test_d1_extreme_numerical_torture_all_engines(self) -> None:
        """Feed NaN, ±Inf, subnormal 1e-310, ±1e15, ±0.0 into every engine: 0 crashes, 0 NaNs."""
        # EProcess
        ep = eproc.EProcess(mu0=0.5, alpha=0.05, lower=-10.0, upper=10.0, side="two_sided")
        for v in self.EXTREMES:
            r = ep.update(v)
            for f in (r.e_value, r.log_e, r.max_log_e, r.p_value, r.mean):
                self.assertTrue(math.isfinite(f), f"EProcess non-finite output {f} on input {v}")

        # ConfidenceSequence (EB and Normal)
        for mode in ("eb", "normal"):
            cs = eproc.ConfidenceSequence(alpha=0.05, lower=-10.0, upper=10.0, mode=mode)
            for v in self.EXTREMES:
                r = cs.update(v)
                self.assertFalse(math.isnan(r.mean), f"CS({mode}) NaN mean on {v}")
                self.assertFalse(math.isnan(r.low), f"CS({mode}) NaN low on {v}")
                self.assertFalse(math.isnan(r.high), f"CS({mode}) NaN high on {v}")
                if mode == "eb":
                    self.assertTrue(math.isfinite(r.low) and math.isfinite(r.high))

        # EDetector
        ed = eproc.EDetector(mu0=0.5, alpha=0.05, lower=0.0, upper=1.0, side="two_sided")
        for v in self.EXTREMES:
            r = ed.update(v)
            self.assertTrue(math.isfinite(r.log_stat) and math.isfinite(r.stat))

        # AgentRace
        race = eproc.AgentRace(n_arms=4, alpha=0.05, lower=0.0, upper=1.0)
        for idx, v in enumerate(self.EXTREMES):
            st = race.update(idx % 4, v)
            self.assertTrue(
                math.isfinite(st.leader_low)
                and math.isfinite(st.leader_high)
                and math.isfinite(st.runner_up_high)
            )

        # MultiRateConformal
        cp = eproc.MultiRateConformal(alphas=(0.50, 0.20, 0.10, 0.05, 0.01), window_size=32, pid_ki=0.05)
        for v in self.EXTREMES:
            m = cp.update(v)
            p = cp.predict(v, scale=v)
            for val in (*m.quantiles, *m.eff_alphas, *m.emp_coverage, *p.lows, *p.highs):
                self.assertTrue(math.isfinite(val), f"MultiRateConformal non-finite {val} on {v}")

        # DelayedConformal
        dc = eproc.DelayedConformal(alphas=(0.20, 0.10, 0.05), window_size=32)
        for v in self.EXTREMES:
            tid, ivs = dc.issue(v, scale=v)
            for lo, hi in ivs:
                self.assertTrue(math.isfinite(lo) and math.isfinite(hi))
            m = dc.reveal(tid, v)
            for val in (*m.quantiles, *m.eff_alphas):
                self.assertTrue(math.isfinite(val))

        # BudgetConformal
        bc = eproc.BudgetConformal(alpha=0.10, window_size=32, censor_penalty=1.5)
        for v in self.EXTREMES:
            b = bc.bound(v)
            self.assertTrue(math.isfinite(b))
            if not math.isfinite(v) or v >= 0.0:
                bc.update(v, v, censored=True)

        # ConformalMonoid
        cm = eproc.ConformalMonoid()
        for v in self.EXTREMES:
            cm.add(v)
        for q in (0.0, 0.5, 0.95, 1.0):
            self.assertTrue(math.isfinite(cm.quantile(q)))
        self.assertTrue(math.isfinite(cm.mean))

        # SafeFuser (warmed up)
        sf = eproc.SafeFuser(n_experts=4, anchor_idx=0, alpha_deg=0.05)
        sf.update([1.0, 1.0, 1.0, 1.0], 1.0)
        for v in self.EXTREMES:
            p = sf.predict([1.0, v, v, v])
            self.assertTrue(math.isfinite(p))
            st = sf.update([1.0, v, v, v], v)
            self.assertTrue(math.isfinite(st.fused_pred) and math.isfinite(st.anchor_pred))

        # EchoPool
        pool = eproc.EchoPool(n_sources=4, n_claims=4, prior_prob=0.5)
        for i, v in enumerate(self.EXTREMES):
            res = pool.submit(i % 4, i % 4, i + 1, v)
            self.assertTrue(math.isfinite(res.posterior_prob) and math.isfinite(res.log_odds))

        # VarOptSampler (non-negative finite weights; NaN/Inf/neg must raise ValueError cleanly)
        vo = eproc.VarOptSampler(k=4, seed=1)
        for i, v in enumerate(self.EXTREMES):
            if math.isfinite(v) and v >= 0.0:
                vo.push(i, v, v)
            else:
                with self.assertRaises(ValueError):
                    vo.push(i, v, 1.0)
        self.assertTrue(math.isfinite(vo.estimate_sum()) and math.isfinite(vo.estimate_mean()))

        # OnlineRLSSampler
        rls = eproc.OnlineRLSSampler(d=4, lambda_reg=1.0, oversample=2.0, min_prob=0.05, seed=1)
        for v in self.EXTREMES:
            out = rls.step([v, 0.0, 0.0, 0.0])
            self.assertTrue(
                math.isfinite(out.leverage) and math.isfinite(out.prob) and math.isfinite(out.ht_weight)
            )

        # SiftVOISelector
        sift = eproc.SiftVOISelector(n_sites=4, per_site=4, signal_mean=1.5, noise_sd=1.0)
        for v in self.EXTREMES:
            sift.observe_site(0, v)
            sift.observe_channel(0, v, n_ticks=1)
        act = sift.select_next()
        self.assertTrue(math.isfinite(act.evoi_per_cost))

    def test_d2_kahan_overflow_nan_witness(self) -> None:
        """FAIL WITNESS: Finite 1e308 inputs overflow Neumaier Kahan sum -> +Inf + (-Inf) = NaN."""
        with self.subTest("ConfidenceSequence(mode='normal')"):
            cs = eproc.ConfidenceSequence(alpha=0.05, mode="normal", sigma=1.0, v_opt=100.0)
            cs.update(1e308)
            r2 = cs.update(1e308)
            self.assertFalse(
                math.isnan(r2.mean),
                f"BUG WITNESS: ConfidenceSequence(mode='normal') produced mean=NaN on two finite 1e308 inputs: {r2}",
            )

        with self.subTest("ConformalMonoid"):
            cm = eproc.ConformalMonoid()
            cm.add(1e308)
            cm.add(1e308)
            self.assertFalse(
                math.isnan(cm.mean),
                "BUG WITNESS: ConformalMonoid produced mean=NaN on two finite 1e308 inputs",
            )

        with self.subTest("VarOptSampler"):
            vo = eproc.VarOptSampler(k=2, seed=1)
            vo.push(0, 1e308, 0.0)
            vo.push(1, 1e308, 0.0)
            vo.push(2, 1e308, 0.0)
            self.assertFalse(
                math.isnan(vo.estimate_sum()),
                f"BUG WITNESS: VarOptSampler produced estimate_sum()=NaN on finite (1e308, 0.0) inputs (tau={vo.tau})",
            )

    def test_d3_conformal_extreme_finite_nan_witness(self) -> None:
        """FAIL WITNESS: Finite (-1e308, +1e308, +1e308) permanently corrupts MultiRateConformal to NaN."""
        cp = eproc.MultiRateConformal(alphas=(0.50, 0.20, 0.10, 0.05, 0.01), window_size=32, pid_ki=0.05)
        cp.update(-1e308)
        cp.update(1e308)
        meta = cp.update(1e308)
        self.assertTrue(
            all(math.isfinite(a) for a in meta.eff_alphas),
            f"BUG WITNESS: MultiRateConformal eff_alphas permanently corrupted to NaN: {meta.eff_alphas}",
        )

    def test_d4_rls_subnormal_1e310_bogus_leverage_witness(self) -> None:
        """FAIL WITNESS: Subnormal vector [1e-310, 0, 0, 0] causes 1/scale=+Inf -> 0*Inf=NaN -> leverage=1e300!"""
        rls = eproc.OnlineRLSSampler(d=4, lambda_reg=1.0, oversample=2.0, min_prob=0.02, seed=42)
        out = rls.step([1e-310, 0.0, 0.0, 0.0])
        self.assertLess(
            out.leverage,
            1e-100,
            f"BUG WITNESS: OnlineRLSSampler returned bogus leverage={out.leverage}, prob={out.prob} "
            f"on subnormal input [1e-310, 0, 0, 0] (expected leverage ~0, prob=0.02)",
        )
        self.assertAlmostEqual(out.prob, 0.02, places=12)

    def test_d5_multirate_conformal_2000_ticks_zero_crossings_and_shock_recovery(self) -> None:
        """2,000 ticks with 10x variance shocks & 1e15 spikes: 0 quantile crossings & coverage recovery."""
        rng = random.Random(20260405)
        alphas = (0.50, 0.20, 0.10, 0.05, 0.01)
        cp = eproc.MultiRateConformal(alphas=alphas, window_size=128, pid_ki=0.05)
        crossings = 0
        post_shock_hits = [0] * 5
        post_shock_n = 0

        for t in range(2000):
            if t in (300, 900, 1400):
                score = 1e15
            elif 500 <= t < 1200:
                score = abs(rng.gauss(0.0, 10.0))  # 10x variance shock
            else:
                score = abs(rng.gauss(0.0, 1.0))

            pred = cp.predict(0.0, scale=1.0)
            for i in range(4):
                if pred.quantiles[i] > pred.quantiles[i + 1]:
                    crossings += 1
                if pred.lows[i] < pred.lows[i + 1] or pred.highs[i] > pred.highs[i + 1]:
                    crossings += 1

            if 800 <= t < 1200:
                post_shock_n += 1
                for i in range(5):
                    if score <= pred.quantiles[i]:
                        post_shock_hits[i] += 1

            meta = cp.update(score)
            for i in range(4):
                if meta.quantiles[i] > meta.quantiles[i + 1]:
                    crossings += 1

        self.assertEqual(crossings, 0, f"Detected {crossings} quantile/interval crossings!")
        # Verify 90% and 95% coverage recover reasonably during sustained 10x shock [800..1200)
        cov90 = post_shock_hits[2] / post_shock_n
        cov95 = post_shock_hits[3] / post_shock_n
        self.assertGreater(cov90, 0.80, f"90% coverage failed to recover under 10x shock: {cov90:.3f}")
        self.assertGreater(cov95, 0.87, f"95% coverage failed to recover under 10x shock: {cov95:.3f}")

    def test_d6_safe_fuser_100pct_poisoned_steady_state_and_tick0_spread_witness(self) -> None:
        """100% poisoned non-anchor experts: steady-state immunity AND tick-0 spread attack witness."""
        # Part 1: Post-warmup 100% poisoned non-anchor experts -> +0.00% catastrophic blast
        sf = eproc.SafeFuser(n_experts=5, anchor_idx=0, alpha_deg=0.05)
        for t in range(10):
            sf.update([1.0, 1.0, 1.0, 1.0, 1.0], 1.0)

        # Attack with 100% poisoned non-anchor experts: +1e12, -1e12, NaN, +Inf
        attack_preds = [1.0, 1e12, -1e12, float("nan"), float("inf")]
        p = sf.predict(attack_preds)
        self.assertAlmostEqual(p, 1.0, places=9, msg="Steady-state 100% poison leaked into predict")
        st = sf.update(attack_preds, 1.0)
        self.assertAlmostEqual(st.fused_pred, 1.0, places=9)
        self.assertEqual(st.n_quarantined, 4)

        # Part 2: Moderate sustained degradation (below quarantine threshold) trips Ville fallback
        sf_deg = eproc.SafeFuser(n_experts=4, anchor_idx=0, alpha_deg=0.05)
        for _ in range(30):
            sf_deg.update([1.0, 0.0, 0.0, 0.0], 0.0)
        tripped = False
        for _ in range(150):
            st_d = sf_deg.update([0.0, 3.0, 3.0, 3.0], 0.0)
            if st_d.fallback_active:
                tripped = True
                break
        self.assertTrue(tripped, "Ville degradation gate failed to trip anchor fallback")

        # Part 3 (FAIL WITNESS): Tick-0 100% poisoned spread attack [0, 1e12, 2e12, 3e12, 4e12]
        sf_t0 = eproc.SafeFuser(n_experts=5, anchor_idx=0, alpha_deg=0.05)
        st_t0 = sf_t0.update([0.0, 1e12, 2e12, 3e12, 4e12], y_true=0.0)
        self.assertLess(
            abs(st_t0.fused_pred - st_t0.anchor_pred),
            100.0,
            f"BUG WITNESS: Tick-0 Byzantine spread attack bypassed MAD gate with catastrophic "
            f"blast radius fused_pred={st_t0.fused_pred:.3e} vs anchor={st_t0.anchor_pred}",
        )

    def test_d7_echopool_50x_echoes_colliding_claims_liar_and_4096_saturation(self) -> None:
        """50x echoes, colliding claims, 100% lying peer, and >4096 table saturation."""
        pool = eproc.EchoPool(n_sources=16, n_claims=32, prior_prob=0.5)
        first = pool.submit(source_id=0, claim_id=0, evidence_id="doc-alpha", llr=2.0)
        self.assertTrue(first.accepted)
        base_post = first.posterior_prob

        # 50x echoed message across different sources -> 0 double-counting
        for i in range(50):
            dup = pool.submit(source_id=i % 16, claim_id=0, evidence_id="doc-alpha", llr=2.0)
            self.assertFalse(dup.accepted)
            self.assertEqual(_f64_bits(dup.posterior_prob), _f64_bits(base_post))
        self.assertEqual(pool.n_echo_dropped, 50)

        # Colliding evidence_id on DIFFERENT claim_id MUST be accepted independently
        other_claim = pool.submit(source_id=0, claim_id=1, evidence_id="doc-alpha", llr=-2.0)
        self.assertTrue(other_claim.accepted)
        self.assertLess(other_claim.posterior_prob, 0.5)

        # 100% lying adversarial peer (source 15) across 10 resolved claims -> reliability == 0.0
        for c in range(2, 12):
            pool.submit(source_id=1, claim_id=c, evidence_id=f"good-{c}", llr=+2.0)
            pool.submit(source_id=15, claim_id=c, evidence_id=f"liar-{c}", llr=-2.0)
            pool.resolve(claim_id=c, outcome_positive=True)
        self.assertGreater(pool.source_reliability(1), 0.95)
        self.assertEqual(pool.source_reliability(15), 0.0)

        # Table saturation (> 4096 unique items) degrades gracefully without crash/corruption
        sat = eproc.EchoPool(n_sources=8, n_claims=8, prior_prob=0.5)
        for i in range(4096):
            r = sat.submit(i % 8, i % 8, i + 1, 0.05)
            self.assertTrue(r.accepted)
        post_before = [sat.posterior(c) for c in range(8)]
        # Echo of existing item still deduplicates cleanly when full
        dup_full = sat.submit(0, 0, 1, 0.05)
        self.assertFalse(dup_full.accepted)
        # 4097th unique item raises OverflowError cleanly and preserves all posteriors
        with self.assertRaises(OverflowError):
            sat.submit(0, 0, 999999, 5.0)
        self.assertEqual(sat.n_table_full, 1)
        self.assertEqual([sat.posterior(c) for c in range(8)], post_before)

    def test_d8_varopt_500_trials_pareto_horvitz_thompson_unbiasedness(self) -> None:
        """500 trials verifying Horvitz-Thompson unbiasedness under heavy-tailed Pareto weights [1, 10000]."""
        rng = random.Random(20260408)
        n_items = 250
        weights = []
        values = []
        for i in range(n_items):
            # Pareto(alpha=1.2) truncated to [1, 10000]
            u = max(rng.random(), 1e-9)
            w = min(10000.0, u ** (-1.0 / 1.2))
            v = 1.0 + 0.05 * i + rng.gauss(0.0, 2.0)
            weights.append(w)
            values.append(v)

        exact_w = sum(weights)
        exact_sum = sum(w * v for w, v in zip(weights, values))
        exact_mean = exact_sum / exact_w

        n_trials = 500
        est_sums = []
        est_means = []
        for tr in range(n_trials):
            vo = eproc.VarOptSampler(k=32, seed=1000 + tr)
            for i in range(n_items):
                vo.push(i, weights[i], values[i])
            # Exact weight conservation invariant on every single trial
            adj_w_tot = sum(item.adj_weight for item in vo.items())
            self.assertAlmostEqual(adj_w_tot, exact_w, delta=1e-6 * exact_w)
            est_sums.append(vo.estimate_sum())
            est_means.append(vo.estimate_mean())

        mc_sum = sum(est_sums) / n_trials
        mc_mean = sum(est_means) / n_trials
        rel_err_sum = abs(mc_sum - exact_sum) / abs(exact_sum)
        rel_err_mean = abs(mc_mean - exact_mean) / abs(exact_mean)
        self.assertLess(rel_err_sum, 0.02, f"VarOpt HT sum bias rel_err={rel_err_sum:.4f}")
        self.assertLess(rel_err_mean, 0.02, f"VarOpt HT mean bias rel_err={rel_err_mean:.4f}")

    def test_d9_sift_voi_3x_loud_distractor_does_not_beat_quiet_channel(self) -> None:
        """3x loud distractor channel with +4.0 mean MUST NOT beat quiet channel with +2.5 mean."""
        sift = eproc.SiftVOISelector(n_sites=4, per_site=4, signal_mean=1.5, noise_sd=1.0)
        loud_ch = 0   # site 0, ch 0 (3x noise)
        quiet_ch = 1  # site 0, ch 1 (1x noise)
        sift.set_noise_scale(loud_ch, 3.0)
        sift.set_noise_scale(quiet_ch, 1.0)

        sift.observe_channel(loud_ch, obs_mean=4.0, n_ticks=1)
        sift.observe_channel(quiet_ch, obs_mean=2.5, n_ticks=1)

        p_loud = float(sift._state.chan_prob[loud_ch])
        p_quiet = float(sift._state.chan_prob[quiet_ch])
        self.assertGreater(
            p_quiet,
            p_loud,
            f"Quiet channel (+2.5, 1x noise, p={p_quiet:.4f}) must beat 3x loud distractor (+4.0, p={p_loud:.4f})",
        )

        # Next channel/tick selection (disable site reads via huge cost) must NOT pick loud_ch over quiet_ch
        act = sift.select_next(cost_site=1e9, cost_channel=1.0, cost_ticks=4.0, ticks_per_read=4)
        self.assertEqual(act.target_idx, quiet_ch, f"Sift selected {act} instead of quiet_ch={quiet_ch}")


# ============================================================================
# FRONT E — CONCURRENCY & SANITIZER TORTURE (ASan+UBSan, TSan, Python Threading)
# ============================================================================


class TestFrontE_ConcurrencyAndSanitizers(unittest.TestCase):
    def test_e1_c_torture_asan_ubsan_and_tsan_1m_seqlock(self) -> None:
        """Compile & execute tests/breaker_c_torture.c under BOTH ASan+UBSan and TSan."""
        gcc = shutil.which("gcc")
        if gcc is None:
            self.skipTest("gcc not available")
        c_torture = REPO_ROOT / "tests" / "breaker_c_torture.c"
        core_srcs = [
            str(REPO_ROOT / "csrc" / f)
            for f in ("eproc_eprocess.c", "eproc_conformal.c", "eproc_fusion.c", "eproc_subsample.c")
        ]
        with tempfile.TemporaryDirectory() as td:
            asan_bin = os.path.join(td, "breaker_asan")
            tsan_bin = os.path.join(td, "breaker_tsan")

            # ASan + UBSan build & run
            subprocess.run(
                [
                    gcc, "-std=c11", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                    "-O2", "-g", "-march=native", "-fno-omit-frame-pointer",
                    "-fsanitize=address,undefined", f"-I{REPO_ROOT / 'csrc'}",
                    str(c_torture), *core_srcs, "-o", asan_bin, "-pthread", "-lm",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            env_asan = os.environ.copy()
            env_asan["ASAN_OPTIONS"] = "detect_leaks=1:abort_on_error=1"
            env_asan["UBSAN_OPTIONS"] = "print_stacktrace=1:halt_on_error=1"
            res_asan = subprocess.run([asan_bin], env=env_asan, capture_output=True, text=True, check=False)
            self.assertEqual(res_asan.returncode, 0, f"ASan+UBSan failed:\n{res_asan.stdout}\n{res_asan.stderr}")

            # TSan build & run
            subprocess.run(
                [
                    gcc, "-std=c11", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                    "-O2", "-g", "-march=native", "-fno-omit-frame-pointer",
                    "-fsanitize=thread", "-Wno-tsan", f"-I{REPO_ROOT / 'csrc'}",
                    str(c_torture), *core_srcs, "-o", tsan_bin, "-pthread", "-lm",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            env_tsan = os.environ.copy()
            env_tsan["TSAN_OPTIONS"] = "halt_on_error=1:second_deadlock_stack=1"
            res_tsan = subprocess.run([tsan_bin], env=env_tsan, capture_output=True, text=True, check=False)
            self.assertEqual(res_tsan.returncode, 0, f"TSan failed:\n{res_tsan.stdout}\n{res_tsan.stderr}")

    def test_e2_python_safe_fuser_snapshot_concurrent_thread_safety_witness(self) -> None:
        """FAIL WITNESS: Python SafeFuser.snapshot() shares self._weights_buf across threads -> torn reads!"""
        sf = eproc.SafeFuser(n_experts=64, anchor_idx=0, alpha_deg=0.05)
        for _ in range(40):
            sf.update([5.0, 0.0] + [10.0] * 62, 0.0)

        torn = 0
        torn_lock = threading.Lock()
        stop = threading.Event()

        def writer() -> None:
            for _ in range(40):
                for _t in range(25):
                    sf.update([0.0] + [2.0] * 63, 0.0)
                for _t in range(25):
                    sf.update([2.0] + [0.0] * 63, 0.0)
            stop.set()

        def reader() -> None:
            nonlocal torn
            local_torn = 0
            while not stop.is_set():
                w, fb = sf.snapshot()
                s = sum(w)
                if fb:
                    if w[0] != 1.0 or any(x != 0.0 for x in w[1:]):
                        local_torn += 1
                elif abs(s - 1.0) > 1e-6:
                    local_torn += 1
            if local_torn:
                with torn_lock:
                    torn += local_torn

        readers = [threading.Thread(target=reader) for _ in range(8)]
        wt = threading.Thread(target=writer)
        for t in readers:
            t.start()
        wt.start()
        wt.join()
        for t in readers:
            t.join()

        self.assertEqual(
            torn,
            0,
            f"BUG WITNESS: SafeFuser.snapshot() produced {torn} torn reads across 8 concurrent "
            f"Python reader threads because self._weights_buf is shared across calls!",
        )

    def test_e3_conformal_ubsan_float_cast_overflow_witness(self) -> None:
        """FAIL WITNESS: C11 UB (float-cast-overflow) in eproc_sorted_quantile on finite (-1e308, 1e308, 1e308)."""
        gcc = shutil.which("gcc")
        if gcc is None:
            self.skipTest("gcc not available")
        c_torture = REPO_ROOT / "tests" / "breaker_c_torture.c"
        core_srcs = [
            str(REPO_ROOT / "csrc" / f)
            for f in ("eproc_eprocess.c", "eproc_conformal.c", "eproc_fusion.c", "eproc_subsample.c")
        ]
        with tempfile.TemporaryDirectory() as td:
            fco_bin = os.path.join(td, "breaker_fco")
            subprocess.run(
                [
                    gcc, "-std=c11", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                    "-O2", "-g", "-march=native", "-fno-omit-frame-pointer",
                    "-fsanitize=address,undefined,float-cast-overflow", f"-I{REPO_ROOT / 'csrc'}",
                    str(c_torture), *core_srcs, "-o", fco_bin, "-pthread", "-lm",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            env = os.environ.copy()
            env["UBSAN_OPTIONS"] = "print_stacktrace=1:halt_on_error=1"
            res = subprocess.run([fco_bin, "--ub-witness"], env=env, capture_output=True, text=True, check=False)
            self.assertEqual(
                res.returncode,
                0,
                f"BUG WITNESS: UBSan caught live C11 Undefined Behavior (float-cast-overflow) in csrc/eproc_conformal.c:41:\n{res.stderr}",
            )


# ============================================================================
# FRONT F — PACKAGING, HYGIENE, SINGLE-HEADER & MCP/CLI ADVERSARIAL TORTURE
# ============================================================================


class TestFrontF_PackagingHygieneAndMcpCli(unittest.TestCase):
    def test_f1_cold_import_under_15ms_and_no_eager_ctypes_or_typing(self) -> None:
        """Cold import < 15 ms with zero eager ctypes, typing, or _ffi loading."""
        code = (
            "import sys, time; "
            "t0 = time.perf_counter(); "
            "import eproc; "
            "dt = (time.perf_counter() - t0) * 1000.0; "
            "leaked = [m for m in ('ctypes', 'typing', 'eproc._ffi', 'numpy') if m in sys.modules]; "
            "assert not leaked, f'Eagerly imported modules: {leaked}'; "
            "print(f'{dt:.3f}')"
        )
        env = {"PYTHONPATH": str(PY_ROOT), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        times = []
        for _ in range(5):
            proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
            times.append(float(proc.stdout.strip()))
        times.sort()
        med_ms = times[len(times) // 2]
        self.assertLess(med_ms, 15.0, f"Median cold import {med_ms:.2f} ms >= 15.0 ms")

    def test_f2_zero_third_party_deps_and_no_private_leaks(self) -> None:
        """Zero third-party imports in python/eproc/ and zero leaked private paths/names."""
        stdlib = sys.stdlib_module_names
        pkg_dir = PY_ROOT / "eproc"
        for py_file in sorted(pkg_dir.rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        self.assertTrue(
                            top in stdlib or top == "eproc",
                            f"Non-stdlib import {alias.name!r} in {py_file.relative_to(REPO_ROOT)}",
                        )
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    top = node.module.split(".")[0]
                    self.assertTrue(
                        top in stdlib or top == "eproc",
                        f"Non-stdlib from-import {node.module!r} in {py_file.relative_to(REPO_ROOT)}",
                    )

        import re
        abs_home_re = re.compile(r"(?:/Users/[A-Za-z0-9._-]+/|/home/[A-Za-z0-9._-]+/)")
        # Scan distributable code & docs for leaked developer workstation paths
        skip_names = {"verify_release_hygiene.py", "test_breaker.py"}
        for sub in ("csrc", "python", "docs", "dist", "skills", "examples"):
            base = REPO_ROOT / sub
            if not base.exists():
                continue
            for p in sorted(base.rglob("*")):
                if not p.is_file() or p.name in skip_names:
                    continue
                if p.suffix not in {".c", ".h", ".py", ".pyi", ".md", ".txt", ".json", ".toml"}:
                    continue
                txt = p.read_text(encoding="utf-8", errors="ignore")
                m = abs_home_re.search(txt)
                self.assertIsNone(
                    m,
                    f"Leaked workstation path {m.group(0)!r} in {p.relative_to(REPO_ROOT)}" if m else "",
                )

    def test_f3_single_header_c11_and_cpp17_compilation_witness(self) -> None:
        """dist/eproc_single.h compiles in C11 AND (FAIL WITNESS) as C++17 as advertised in header."""
        gcc = shutil.which("gcc")
        gpp = shutil.which("g++")
        if gcc is None:
            self.skipTest("gcc not available")
        with tempfile.TemporaryDirectory() as td:
            c_src = Path(td) / "test_sh.c"
            c_src.write_text(
                '#define EPROC_IMPLEMENTATION\n#include "eproc_single.h"\n'
                "int main(void) { eproc_bet_t b; return eproc_bet_init(&b, 0.5, 0.05, 0.0, 1.0, 0); }\n",
                encoding="utf-8",
            )
            r_c = subprocess.run(
                [gcc, "-std=c11", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                 f"-I{REPO_ROOT / 'dist'}", str(c_src), "-o", os.path.join(td, "t_c"), "-lm", "-pthread"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(r_c.returncode, 0, f"Single-header C11 build failed:\n{r_c.stderr}")

            if gpp is not None:
                cpp_src = Path(td) / "test_sh.cpp"
                cpp_src.write_text(
                    '#define EPROC_IMPLEMENTATION\n#include "eproc_single.h"\n'
                    "int main() { eproc_bet_t b; return eproc_bet_init(&b, 0.5, 0.05, 0.0, 1.0, 0); }\n",
                    encoding="utf-8",
                )
                r_cpp = subprocess.run(
                    [gpp, "-std=c++17", "-O2", "-ffp-contract=off",
                     f"-I{REPO_ROOT / 'dist'}", str(cpp_src), "-o", os.path.join(td, "t_cpp"), "-lm", "-pthread"],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(
                    r_cpp.returncode,
                    0,
                    f"BUG WITNESS: dist/eproc_single.h advertises 'In exactly ONE C/C++ translation unit' "
                    f"but fails g++ -std=c++17:\n{r_cpp.stderr[:600]}",
                )

    def test_f4_mcp_2026_07_28_adversarial_protocol_and_session_eviction(self) -> None:
        """MCP 2026-07-28 compliance: malformed JSON, unknown tools, oversized payload, LRU eviction."""
        srv = mcp_server.McpServer(max_sessions=8)

        # Empty / whitespace lines produce None
        self.assertIsNone(srv.handle_line(""))
        self.assertIsNone(srv.handle_line("   \t\n"))

        # Invalid JSON -> -32700
        err_parse = json.loads(srv.handle_line("{bad json") or "")
        self.assertEqual(err_parse["error"]["code"], mcp_server.ERR_PARSE)

        # JSON array batch -> -32600 (MCP 2026-07-28 forbids batching)
        err_arr = json.loads(srv.handle_line("[]") or "")
        self.assertEqual(err_arr["error"]["code"], mcp_server.ERR_INVALID_REQUEST)

        # Non-object params -> -32602
        err_params = json.loads(
            srv.handle_line(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": [1, 2]})) or ""
        )
        self.assertEqual(err_params["error"]["code"], mcp_server.ERR_INVALID_PARAMS)

        # Unknown tool -> -32602
        err_tool = json.loads(
            srv.handle_line(
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "evil_tool"}})
            )
            or ""
        )
        self.assertEqual(err_tool["error"]["code"], mcp_server.ERR_INVALID_PARAMS)

        # Non-object tool arguments -> -32602
        err_args = json.loads(
            srv.handle_line(
                json.dumps(
                    {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "eproc_etest", "arguments": [1]}}
                )
            )
            or ""
        )
        self.assertEqual(err_args["error"]["code"], mcp_server.ERR_INVALID_PARAMS)

        # NaN in mu0 must be rejected cleanly with isError=True (no crash, valid strict JSON)
        nan_mu0 = json.loads(
            srv.handle_line(
                '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"eproc_etest","arguments":{"mu0":NaN,"values":[0.5]}}}'
            )
            or ""
        )
        self.assertTrue(nan_mu0["result"]["isError"])

        # Oversized valid payload (~1 MB) handled cleanly without crash
        big_vals = [0.5] * 25000
        big_req = json.dumps(
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "eproc_etest", "arguments": {"mu0": 0.5, "values": big_vals}}}
        )
        big_resp = json.loads(srv.handle_line(big_req) or "")
        self.assertFalse(big_resp["result"]["isError"])
        self.assertEqual(big_resp["result"]["structuredContent"]["n"], 25000)

        # Session eviction overflow: max_sessions=8, insert 20 sessions -> oldest 12 evicted cleanly
        for i in range(20):
            resp = json.loads(
                srv.handle_line(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 100 + i,
                            "method": "tools/call",
                            "params": {
                                "name": "eproc_etest",
                                "arguments": {"mu0": 0.5, "values": [0.6], "session_id": f"sess-{i}"},
                            },
                        }
                    )
                )
                or ""
            )
            self.assertFalse(resp["result"]["isError"])
        self.assertEqual(len(srv._sessions), 8)
        # Oldest session sess-0 must have been evicted -> querying without mu0 must fail
        evicted_resp = json.loads(
            srv.handle_line(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 200,
                        "method": "tools/call",
                        "params": {"name": "eproc_etest", "arguments": {"session_id": "sess-0"}},
                    }
                )
            )
            or ""
        )
        self.assertTrue(evicted_resp["result"]["isError"])
        # Newest session sess-19 must still be alive
        alive_resp = json.loads(
            srv.handle_line(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 201,
                        "method": "tools/call",
                        "params": {"name": "eproc_etest", "arguments": {"session_id": "sess-19"}},
                    }
                )
            )
            or ""
        )
        self.assertFalse(alive_resp["result"]["isError"])
        self.assertEqual(alive_resp["result"]["structuredContent"]["n"], 1)

    def test_f5a_mcp_reset_string_false_witness(self) -> None:
        """FAIL WITNESS: MCP 'reset':'false' coerces via bool('false')==True and wipes session."""
        srv = mcp_server.McpServer()
        srv.handle_line(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "eproc_etest", "arguments": {"mu0": 0.5, "values": [0.8, 0.9], "session_id": "s1"}},
                }
            )
        )
        resp_reset_str = json.loads(
            srv.handle_line(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "eproc_etest",
                            "arguments": {"mu0": 0.5, "values": [0.7], "session_id": "s1", "reset": "false"},
                        },
                    }
                )
            )
            or ""
        )
        is_err = resp_reset_str["result"]["isError"]
        n_after = resp_reset_str["result"]["structuredContent"].get("n")
        self.assertTrue(
            is_err or n_after == 3,
            f"BUG WITNESS: MCP tools/call with reset='false' silently wiped session s1 (n={n_after} instead of 3 or error)!",
        )

    def test_f5b_mcp_invalid_request_missing_id_witness(self) -> None:
        """FAIL WITNESS: Invalid JSON-RPC 2.0 request missing 'id' silently dropped instead of returning -32600."""
        srv = mcp_server.McpServer()
        bad_ver = srv.handle_line(json.dumps({"jsonrpc": "1.0", "method": "ping"}))
        self.assertIsNotNone(
            bad_ver,
            "BUG WITNESS: Invalid JSON-RPC request (jsonrpc='1.0') without 'id' was silently dropped instead of returning -32600",
        )

    def test_f5c_mcp_bool_coerced_to_float_witness(self) -> None:
        """FAIL WITNESS: Boolean passed as numeric in eproc_safe_fuse/eproc_echopool/eproc_conformal silently coerced to 1.0."""
        srv = mcp_server.McpServer()
        bool_y = json.loads(
            srv.handle_line(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {
                            "name": "eproc_safe_fuse",
                            "arguments": {"n_experts": 2, "steps": [{"preds": [1.0, 1.0], "y": True}]},
                        },
                    }
                )
            )
            or ""
        )
        self.assertTrue(
            bool_y["result"]["isError"],
            "BUG WITNESS: eproc_safe_fuse silently coerced boolean y=True to 1.0 instead of rejecting non-numeric",
        )

    def test_f6_cli_conformal_zero_scale_crash_witness(self) -> None:
        """FAIL WITNESS: `eproc conformal` crashes with unhandled ZeroDivisionError on scale=0.0."""
        out = io.StringIO()
        err = io.StringIO()
        try:
            rc = cli.main(
                ["conformal", "--json"],
                stdin=io.StringIO('{"pred": 1.0, "y": 2.0, "scale": 0.0}\n'),
                stdout=out,
                stderr=err,
            )
        except ZeroDivisionError as exc:
            self.fail(f"BUG WITNESS: CLI crashed with unhandled ZeroDivisionError on scale=0.0: {exc}")
        self.assertEqual(rc, 2, f"Expected clean CLI error exit code 2, got {rc}")


if __name__ == "__main__":
    unittest.main()
