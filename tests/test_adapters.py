"""Comprehensive unittest suite for zero-dependency drop-in ``eproc.adapters``."""

from __future__ import annotations

import array
import ast
import json
import math
from pathlib import Path
import random
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = REPO_ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

import eproc  # noqa: E402
from eproc.adapters import (  # noqa: E402
    ConformalWrapper,
    EvalGate,
    EvalGateSummary,
    FleetCalibrationResult,
    IntervalResult,
    PairedEvalGate,
    assert_eprocess_uplift,
    assert_no_regression,
    fleet_calibrate,
)


class _MockSklearnLinearRegressor:
    """Minimal sklearn-like regressor exposing .predict(X) without importing sklearn."""

    def __init__(self, weights: tuple[float, ...], intercept: float = 0.0) -> None:
        self.weights = weights
        self.intercept = intercept
        self.predict_calls = 0

    def predict(self, X: list[list[float]]) -> list[float]:
        self.predict_calls += 1
        out: list[float] = []
        for row in X:
            val = self.intercept
            for w, x in zip(self.weights, row):
                val += w * x
            out.append(val)
        return out


class TestEvalGate(unittest.TestCase):
    def test_early_stopping_and_samples_saved_pct(self) -> None:
        rng = random.Random(202601)
        gate = EvalGate(mu0=0.50, alpha=0.05, lower=0.0, upper=1.0, side="upper", fixed_horizon_n=857)
        self.assertFalse(gate.should_stop)
        self.assertEqual(gate.summary().verdict, "CONTINUE")
        self.assertEqual(gate.summary().samples_saved_pct, 0.0)

        for _ in range(857):
            score = 1.0 if rng.random() < 0.72 else 0.0
            gate.observe(score)
            if gate.should_stop:
                break

        self.assertTrue(gate.should_stop)
        self.assertTrue(gate.rejected)
        self.assertLess(gate.n, 300)
        summary = gate.summary()
        self.assertIsInstance(summary, EvalGateSummary)
        self.assertEqual(summary.verdict, "REJECT")
        self.assertEqual(summary["verdict"], "REJECT")
        self.assertIn("samples_saved_pct", summary)
        self.assertIsNotNone(summary.samples_saved_pct)
        assert summary.samples_saved_pct is not None
        expected_saved = 100.0 * (1.0 - gate.rejected_at / 857.0)
        self.assertAlmostEqual(summary.samples_saved_pct, expected_saved, places=9)
        self.assertGreater(summary.samples_saved_pct, 60.0)
        self.assertLessEqual(summary.p_value, 0.05)
        self.assertGreaterEqual(summary.e_value, 20.0)

        # JSON-serializable to_dict()
        as_dict = summary.to_dict()
        self.assertEqual(as_dict["verdict"], "REJECT")
        self.assertAlmostEqual(as_dict["samples_saved_pct"], expected_saved, places=9)
        json.dumps(as_dict)

    def test_without_fixed_horizon_omits_savings(self) -> None:
        gate = EvalGate(mu0=0.50, alpha=0.05)
        gate.observe_batch([0.9] * 50)
        summary = gate.summary()
        self.assertEqual(summary.verdict, "REJECT")
        self.assertIsNone(summary.samples_saved_pct)
        self.assertIsNone(summary.fixed_horizon_n)
        self.assertNotIn("samples_saved_pct", summary)
        self.assertNotIn("samples_saved_pct", summary.to_dict())
        self.assertIsNone(summary.get("samples_saved_pct"))
        with self.assertRaises(KeyError):
            _ = summary["samples_saved_pct"]

    def test_scalar_vs_batch_equivalence_and_rejected_at_savings(self) -> None:
        rng = random.Random(202602)
        stream = [1.0 if rng.random() < 0.70 else 0.0 for _ in range(400)]

        g_scalar = EvalGate(mu0=0.5, alpha=0.05, fixed_horizon_n=500)
        for x in stream:
            g_scalar.observe(x)

        g_batch = EvalGate(mu0=0.5, alpha=0.05, fixed_horizon_n=500)
        g_batch.observe_batch(array.array("d", stream))

        s1 = g_scalar.summary()
        s2 = g_batch.summary()
        self.assertEqual(s1.verdict, s2.verdict)
        self.assertEqual(s1.n, s2.n)
        self.assertEqual(s1.rejected_at, s2.rejected_at)
        self.assertAlmostEqual(s1.e_value, s2.e_value, places=8)
        self.assertAlmostEqual(s1.p_value, s2.p_value, places=11)
        self.assertAlmostEqual(s1.mean, s2.mean, places=12)
        self.assertEqual(s1.samples_saved_pct, s2.samples_saved_pct)
        self.assertGreater(s2.samples_saved_pct or 0.0, 40.0)

    def test_continuous_peeking_null_safety(self) -> None:
        # Ville inequality: FPR <= alpha under continuous every-tick peeking
        n_trials = 300
        horizon = 200
        false_positives = 0
        for trial in range(n_trials):
            rng = random.Random(10_000 + trial)
            gate = EvalGate(mu0=0.50, alpha=0.05, lower=0.0, upper=1.0, side="upper")
            for _ in range(horizon):
                gate.observe(1.0 if rng.random() < 0.50 else 0.0)
                if gate.should_stop:
                    false_positives += 1
                    break
        fpr = false_positives / n_trials
        # With alpha=0.05 and n_trials=300, expected <= 15; generous finite-sample bound <= 0.08
        self.assertLessEqual(fpr, 0.08, f"Null continuous-peeking FPR too high: {fpr:.3f}")

    def test_sides_budget_nan_and_validation(self) -> None:
        g_lo = EvalGate(mu0=0.5, alpha=0.05, side="lower", fixed_horizon_n=100)
        g_lo.observe_batch([0.1] * 100)
        self.assertTrue(g_lo.should_stop)
        self.assertTrue(g_lo.budget_reached)

        g_two = EvalGate(mu0=0.5, alpha=0.05, side="two_sided")
        g_two.observe_batch([float("nan"), float("inf"), 0.05, 0.05] * 40)
        self.assertTrue(g_two.should_stop)
        self.assertTrue(math.isfinite(g_two.e_value))
        self.assertTrue(math.isfinite(g_two.p_value))

        with self.assertRaises(ValueError):
            EvalGate(mu0=0.5, fixed_horizon_n=0)
        with self.assertRaises(ValueError):
            EvalGate(mu0=0.5, fixed_horizon_n=-10)
        with self.assertRaises(TypeError):
            EvalGate(mu0=0.5, fixed_horizon_n=True)  # type: ignore[arg-type]


class TestPairedEvalGate(unittest.TestCase):
    def test_paired_ab_early_stop_and_stats(self) -> None:
        rng = random.Random(202603)
        gate = PairedEvalGate(alpha=0.05, span=1.0, fixed_horizon_n=600)
        for _ in range(600):
            base = 1.0 if rng.random() < 0.45 else 0.0
            cand = 1.0 if rng.random() < 0.70 else 0.0
            gate.observe_pair(cand, base)
            if gate.should_stop:
                break

        self.assertTrue(gate.should_stop)
        self.assertLess(gate.n, 350)
        self.assertGreater(gate.mean, 0.0)
        self.assertGreater(gate.candidate_mean, gate.baseline_mean)
        self.assertEqual(gate.wins + gate.losses + gate.ties, gate.n)
        self.assertGreater(gate.win_rate, 0.0)

        summary = gate.summary()
        self.assertEqual(summary.verdict, "REJECT")
        self.assertGreater(summary.samples_saved_pct or 0.0, 40.0)
        self.assertIn("PairedEvalGate", repr(gate))

    def test_paired_ab_null_safety(self) -> None:
        n_trials = 200
        false_positives = 0
        for trial in range(n_trials):
            rng = random.Random(20_000 + trial)
            gate = PairedEvalGate(alpha=0.05, span=1.0)
            for _ in range(150):
                # Identical marginal score distributions -> E[cand - base] == 0
                c = rng.random()
                b = rng.random()
                gate.observe_pair(c, b)
                if gate.should_stop:
                    false_positives += 1
                    break
        fpr = false_positives / n_trials
        self.assertLessEqual(fpr, 0.08, f"Paired null FPR too high: {fpr:.3f}")

    def test_paired_batch_modes_and_custom_span(self) -> None:
        rng = random.Random(202604)
        cands = [rng.uniform(6.0, 9.5) for _ in range(120)]
        bases = [rng.uniform(2.0, 5.5) for _ in range(120)]
        pairs = list(zip(cands, bases))
        diffs = [c - b for c, b in pairs]

        g1 = PairedEvalGate(alpha=0.05, span=10.0)
        g1.observe_batch(cands, bases)

        g2 = PairedEvalGate(alpha=0.05, span=10.0)
        g2.observe_batch(pairs)

        g3 = PairedEvalGate(alpha=0.05, span=10.0)
        g3.observe_batch(diffs)

        self.assertTrue(g1.should_stop)
        self.assertAlmostEqual(g1.e_value, g2.e_value, places=8)
        self.assertAlmostEqual(g1.e_value, g3.e_value, places=8)
        self.assertAlmostEqual(g1.mean, g3.mean, places=12)

        with self.assertRaises(ValueError):
            PairedEvalGate(span=0.0)
        with self.assertRaises(ValueError):
            g1.observe_batch([1.0, 2.0], [1.0])

    def test_paired_eval_gate_nan_resilience(self) -> None:
        gate = PairedEvalGate(alpha=0.05, span=1.0)
        gate.observe_pair(float("nan"), 0.5)
        gate.observe_pair(0.8, 0.2)
        self.assertEqual(gate.n, 1)
        self.assertAlmostEqual(gate.candidate_mean, 0.8, places=12)
        self.assertAlmostEqual(gate.baseline_mean, 0.2, places=12)
        self.assertTrue(math.isfinite(gate.e_value))


class TestConformalWrapper(unittest.TestCase):
    def test_mock_sklearn_regressor_fit_partial_fit_and_zero_crossing(self) -> None:
        rng = random.Random(202605)
        reg = _MockSklearnLinearRegressor(weights=(2.0, -0.5), intercept=1.0)
        wrapper = ConformalWrapper(reg)

        def make_batch(n: int, noise_std: float) -> tuple[list[list[float]], list[float]]:
            X = [[rng.uniform(-3.0, 3.0), rng.uniform(-2.0, 2.0)] for _ in range(n)]
            y = [2.0 * r[0] - 0.5 * r[1] + 1.0 + rng.gauss(0.0, noise_std) for r in X]
            return X, y

        X_cal, y_cal = make_batch(200, 0.5)
        fitted = wrapper.fit(X_cal, y_cal)
        self.assertIs(fitted, wrapper)
        self.assertEqual(wrapper.n_updates, 200)
        self.assertEqual(wrapper.alphas, (0.50, 0.20, 0.10, 0.05, 0.01))

        X_test, y_test = make_batch(100, 0.5)
        res = wrapper.predict_interval(X_test)
        self.assertIsInstance(res, IntervalResult)
        self.assertEqual(len(res), 100)
        self.assertFalse(res.has_crossings())

        # Verify 3-tuple unpacking compatibility
        y_hat, lows_by_rate, highs_by_rate = res
        self.assertEqual(len(y_hat), 100)
        self.assertEqual(set(lows_by_rate.keys()), set(wrapper.alphas))
        self.assertEqual(set(highs_by_rate.keys()), set(wrapper.alphas))

        # Verify strict nesting for every single test point across all 5 rates
        ordered = wrapper.alphas  # descending alpha -> ascending coverage (50%..99%)
        for i in range(len(y_hat)):
            for r in range(len(ordered) - 1):
                a_lo, a_hi = ordered[r], ordered[r + 1]
                self.assertLessEqual(lows_by_rate[a_hi][i], lows_by_rate[a_lo][i] + 1e-12)
                self.assertGreaterEqual(highs_by_rate[a_hi][i], highs_by_rate[a_lo][i] - 1e-12)
                self.assertLessEqual(lows_by_rate[a_lo][i], y_hat[i] + 1e-12)
                self.assertGreaterEqual(highs_by_rate[a_lo][i], y_hat[i] - 1e-12)

        # Verify monotonic empirical coverage (wider rates cover >= narrower rates)
        covs = []
        for a in ordered:
            lo, hi = res.interval_for(a)
            covered = sum(1 for yi, l, h in zip(y_test, lo, hi) if l <= yi <= h) / len(y_test)
            covs.append(covered)
        for r in range(len(covs) - 1):
            self.assertLessEqual(covs[r], covs[r + 1] + 1e-12)
        self.assertGreaterEqual(covs[-1], 0.90)  # 99% interval on stationary test set

        # Online adaptation under sudden 10x variance shock + partial_fit
        for _ in range(10):
            X_shock, y_shock = make_batch(20, 5.0)
            meta = wrapper.partial_fit(X_shock, y_shock)
            self.assertEqual(meta.n_rates, 5)
            shock_res = wrapper.predict_interval(X_shock)
            self.assertFalse(shock_res.has_crossings())

        self.assertEqual(wrapper.n_updates, 400)
        # Quantiles must have expanded after 10x shock
        self.assertGreater(wrapper.quantiles[-1], res.quantiles[-1])
        json.dumps(res.to_dict())

    def test_callable_predictors_and_heteroskedastic_scales(self) -> None:
        # 1. Per-row callable on 2D rows where len(X) == 4 (guards against list concat trap)
        fn_row = lambda row: row[0] + row[1]
        X_rows = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        y_rows = [3.1, 7.2, 10.8, 15.3]
        w_row = ConformalWrapper(fn_row, alphas=(0.20, 0.05))
        w_row.fit(X_rows, y_rows)
        self.assertEqual(w_row.predict(X_rows), [3.0, 7.0, 11.0, 15.0])

        # 2. Per-scalar callable fn(x)
        w_scalar = ConformalWrapper(lambda x: 3.0 * x - 1.0)
        w_scalar.fit([1.0, 2.0, 3.0, 4.0], [2.2, 4.8, 8.1, 10.9])
        res_s = w_scalar.predict_interval([5.0, 6.0], scale=[1.0, 2.0])
        self.assertFalse(res_s.has_crossings())
        # Second sample has 2x scale -> 2x interval half-width
        lo_05, hi_05 = res_s.interval_for(0.05)
        hw0 = hi_05[0] - res_s.y_hat[0]
        hw1 = hi_05[1] - res_s.y_hat[1]
        self.assertAlmostEqual(hw1, 2.0 * hw0, places=10)

        # 3. Batch callable fn(X) + scale_fn
        fn_batch = lambda X: [2.0 * r[0] for r in X]
        w_batch = ConformalWrapper(
            fn_batch,
            scale_fn=lambda X: [1.0 + abs(r[0]) for r in X],
        )
        w_batch.fit([[1.0], [2.0], [3.0]], [2.5, 4.2, 6.9])
        res_b = w_batch.predict_interval([[0.0], [4.0]])
        lo_b, hi_b = res_b.interval_for(0.10)
        # scale(4.0) = 5.0 vs scale(0.0) = 1.0 -> 5x half-width
        self.assertAlmostEqual(
            (hi_b[1] - res_b.y_hat[1]),
            5.0 * (hi_b[0] - res_b.y_hat[0]),
            places=10,
        )

        # 4. Empty X handling, reset=False, scalar scale, KeyError on unknown alpha, and validation
        empty_res = w_batch.predict_interval([])
        self.assertEqual(len(empty_res), 0)
        self.assertEqual(empty_res.y_hat, ())
        prev_updates = w_scalar.n_updates
        w_scalar.fit([1.0, 2.0], [2.1, 4.9], reset=False)
        self.assertEqual(w_scalar.n_updates, prev_updates + 2)
        res_const_scale = w_scalar.predict_interval([5.0], scale=2.5)
        self.assertEqual(len(res_const_scale), 1)
        with self.assertRaises(KeyError):
            res_const_scale.interval_for(0.33)
        with self.assertRaises(TypeError):
            ConformalWrapper(12345)
        with self.assertRaises(ValueError):
            w_batch.predict_interval([[1.0]], scale=0.0)
        with self.assertRaises(ValueError):
            w_batch.fit([[1.0], [2.0]], [1.0])


class TestPytestAssertionHelpers(unittest.TestCase):
    def test_assert_eprocess_uplift_passes_and_fails_with_diagnostics(self) -> None:
        rng = random.Random(202606)
        good_scores = [1.0 if rng.random() < 0.78 else 0.0 for _ in range(250)]
        res = assert_eprocess_uplift(good_scores, mu0=0.50, alpha=0.05)
        self.assertTrue(res.rejected)
        self.assertEqual(res.n, 250)

        # Lower-sided uplift check (e.g., error rate reduction below 0.50)
        low_errs = [0.10] * 100
        res_lo = assert_eprocess_uplift(low_errs, mu0=0.50, alpha=0.05, side="lower")
        self.assertTrue(res_lo.rejected)

        # Null stream should fail with rich diagnostic AssertionError
        null_scores = [0.50] * 80
        with self.assertRaises(AssertionError) as ctx:
            assert_eprocess_uplift(null_scores, mu0=0.50, alpha=0.05)
        msg = str(ctx.exception)
        for token in ("n=80", "mean=", "mu0=0.5", "e_value=", "p_value="):
            self.assertIn(token, msg)

    def test_assert_no_regression_passes_and_catches_regression(self) -> None:
        # Neutral / positive diffs pass
        ok_res = assert_no_regression([0.0, 0.05, -0.02, 0.04, 0.01] * 40, alpha=0.05)
        self.assertFalse(ok_res.rejected)

        # Significant negative regression raises AssertionError with diagnostics
        rng = random.Random(202607)
        regressed = [rng.uniform(-0.8, -0.1) for _ in range(120)]
        with self.assertRaises(AssertionError) as ctx:
            assert_no_regression(regressed, alpha=0.05, lower=-1.0, upper=1.0)
        msg = str(ctx.exception)
        for token in ("regression gate FAILED", "mean_diff=", "e_value=", "p_value=", "rejected_at="):
            self.assertIn(token, msg)


class TestFleetCalibrate(unittest.TestCase):
    def test_exact_loo_and_mixed_shard_inputs(self) -> None:
        rng = random.Random(202608)
        raw_shards: list[list[float]] = [
            [abs(rng.gauss(0.0, 0.5 + 0.15 * i)) for _ in range(80 + 15 * i)]
            for i in range(8)
        ]

        # Mix float lists, serialized 640-byte wire payloads, memoryview, and ConformalMonoid
        mixed_shards: list[object] = []
        for idx, vals in enumerate(raw_shards):
            if idx in (0, 1):
                mixed_shards.append(vals)
            elif idx in (2, 3):
                m = eproc.ConformalMonoid(scale_hint=1.0)
                m.add_batch(vals)
                mixed_shards.append(m.to_bytes())
            elif idx == 4:
                m = eproc.ConformalMonoid(scale_hint=1.0)
                m.add_batch(vals)
                mixed_shards.append(memoryview(m.to_bytes()))
            else:
                m = eproc.ConformalMonoid(scale_hint=1.0)
                m.add_batch(vals)
                mixed_shards.append(m)

        q_levels = (0.50, 0.90, 0.95)
        result = fleet_calibrate(mixed_shards, quantiles=q_levels, scale_hint=1.0)  # type: ignore[arg-type]
        self.assertIsInstance(result, FleetCalibrationResult)
        self.assertEqual(result.wire_size, 640)
        self.assertEqual(result.total_count, sum(len(s) for s in raw_shards))
        self.assertEqual(result.shard_counts, tuple(len(s) for s in raw_shards))

        # Check 3-tuple unpacking
        global_m, global_q, peer_q = result
        self.assertIs(global_m, result.global_monoid)
        self.assertEqual(global_q, result.global_quantiles)
        self.assertEqual(peer_q, result.peer_quantiles)

        # Verify bit-exact LOO against manual rebuild of all j != i for EVERY shard
        for i in range(len(raw_shards)):
            manual = eproc.ConformalMonoid(scale_hint=1.0)
            for j, vals in enumerate(raw_shards):
                if j != i:
                    manual.add_batch(vals)
            self.assertEqual(result.peer_monoids[i].count, manual.count)
            self.assertEqual(result.peer_monoids[i].bin_counts, manual.bin_counts)
            for q in q_levels:
                self.assertAlmostEqual(peer_q[i][q], manual.quantile(q), places=12)

        # Permutation invariance (associativity + commutativity)
        rev = fleet_calibrate(list(reversed(raw_shards)), quantiles=q_levels, scale_hint=1.0)
        self.assertEqual(global_m.to_bytes(), rev.global_monoid.to_bytes())
        for q in q_levels:
            self.assertAlmostEqual(global_q[q], rev.global_quantiles[q], places=15)

        json.dumps(result.to_dict())

    def test_single_shard_generator_and_error_cases(self) -> None:
        res = fleet_calibrate([[0.1, 0.5, 0.9]], quantiles=(0.50, 0.90))
        self.assertEqual(res.total_count, 3)
        self.assertEqual(res.peer_monoids[0].count, 0)
        self.assertEqual(res.peer_quantiles[0][0.50], 0.0)

        # Generator of shards
        gen_res = fleet_calibrate((s for s in ([0.1, 0.2], [0.5, 0.8])), quantiles=(0.50, 0.90))  # type: ignore[arg-type]
        self.assertEqual(gen_res.total_count, 4)

        with self.assertRaises(ValueError):
            fleet_calibrate([])
        with self.assertRaises(ValueError):
            fleet_calibrate((x for x in ()))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fleet_calibrate([[0.1, 0.2]], quantiles=(1.5,))


class TestZeroThirdPartyImports(unittest.TestCase):
    def test_adapters_uses_only_stdlib_and_eproc(self) -> None:
        adapters_path = PY_ROOT / "eproc" / "adapters.py"
        tree = ast.parse(adapters_path.read_text(encoding="utf-8"))
        stdlib = sys.stdlib_module_names
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    self.assertTrue(
                        top in stdlib or top == "eproc",
                        f"Disallowed import {alias.name!r} in adapters.py",
                    )
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top = node.module.split(".")[0]
                self.assertTrue(
                    top in stdlib or top == "eproc",
                    f"Disallowed from-import {node.module!r} in adapters.py",
                )


if __name__ == "__main__":
    unittest.main()
