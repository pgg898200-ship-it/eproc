"""Comprehensive unit and integration tests for the eproc Python API, CLI, and MCP server."""

from __future__ import annotations

import array
import base64
import ctypes
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = REPO_ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

import eproc  # noqa: E402
from eproc import _ffi, cli, mcp_server  # noqa: E402


class TestColdImportAndABI(unittest.TestCase):
    def test_lazy_import_and_cold_startup_under_15ms(self) -> None:
        code = (
            "import sys, time; "
            "t0 = time.perf_counter(); "
            "import eproc; "
            "dt_ms = (time.perf_counter() - t0) * 1000.0; "
            "assert 'eproc._ffi' not in sys.modules, 'eproc._ffi eagerly loaded!'; "
            "assert 'ctypes' not in sys.modules, 'ctypes eagerly loaded!'; "
            "print(f'{dt_ms:.3f}')"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PY_ROOT)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        dt_ms = float(proc.stdout.strip())
        self.assertLess(dt_ms, 15.0, f"Cold import took {dt_ms:.2f} ms (limit 15 ms)")

    def test_abi_version_and_wire_size(self) -> None:
        self.assertEqual(eproc.abi_version(), _ffi.EPROC_ABI_VERSION)
        self.assertEqual(ctypes.sizeof(_ffi.eproc_cmonoid_t), 640)
        self.assertEqual(eproc.ConformalMonoid.wire_size, 640)
        flags = eproc.simd_flags()
        self.assertIsInstance(flags, int)
        names = eproc.simd_flag_names(flags)
        self.assertIsInstance(names, list)


class TestEProcessAndConfidenceSequence(unittest.TestCase):
    def test_eprocess_upper_rejects_strong_signal_and_matches_batch(self) -> None:
        rng = random.Random(1234)
        xs = [1.0 if rng.random() < 0.72 else 0.0 for _ in range(300)]

        ep_scalar = eproc.EProcess(mu0=0.5, alpha=0.05, lower=0.0, upper=1.0, side="upper")
        for x in xs:
            ep_scalar.update(x)

        ep_batch = eproc.EProcess(mu0=0.5, alpha=0.05, lower=0.0, upper=1.0, side="upper")
        res_batch = ep_batch.update_batch(array.array("d", xs))

        res_scalar = ep_scalar.result()
        self.assertTrue(res_scalar.rejected)
        self.assertTrue(res_batch.rejected)
        self.assertEqual(res_scalar.n, 300)
        self.assertEqual(res_batch.n, 300)
        self.assertAlmostEqual(res_scalar.log_e, res_batch.log_e, places=11)
        self.assertAlmostEqual(res_scalar.p_value, res_batch.p_value, places=11)
        self.assertAlmostEqual(res_scalar.mean, res_batch.mean, places=12)
        self.assertLessEqual(res_batch.p_value, 0.05)
        self.assertGreater(ep_batch.e_value, 20.0)
        self.assertGreater(ep_batch.rejected_at, 0)

    def test_eprocess_lower_and_two_sided_and_zero_copy_buffers(self) -> None:
        xs = [0.2] * 150
        ep_lo = eproc.EProcess(mu0=0.5, alpha=0.05, side="lower")
        buf = array.array("d", xs)
        mv = memoryview(buf)
        res_lo = ep_lo.update_batch(mv)
        self.assertTrue(res_lo.rejected)

        ep_two = eproc.EProcess(mu0=0.5, alpha=0.05, side="two_sided")
        c_arr = (ctypes.c_double * len(xs))(*xs)
        res_two = ep_two.update_batch(c_arr)
        self.assertTrue(res_two.rejected)

        # Empty batch should be safe and non-mutating
        n_before = ep_two.n
        res_empty = ep_two.update_batch([])
        self.assertEqual(res_empty.n, n_before)

        # Optional NumPy zero-copy check if numpy is installed
        try:
            import numpy as np  # type: ignore[import-not-found]
        except ImportError:
            np = None
        if np is not None:
            arr = np.asarray(xs, dtype=np.float64)
            ptr, n, owner = _ffi._as_double_ptr(arr)
            self.assertEqual(n, len(xs))
            self.assertIs(owner, arr)
            self.assertEqual(ctypes.cast(ptr, ctypes.c_void_p).value, arr.__array_interface__["data"][0])

    def test_eprocess_nan_inf_sanitization_and_combine(self) -> None:
        ep = eproc.EProcess(mu0=0.5, alpha=0.05)
        ep.update_batch([0.6, float("nan"), float("inf"), float("-inf"), 0.7])
        self.assertEqual(ep.n_invalid, 3)
        self.assertTrue(math.isfinite(ep.log_e))
        self.assertTrue(math.isfinite(ep.p_value))

        # Independent combine = sum of log e-values
        ind = eproc.combine_evalues([1.5, 2.0, -0.5], independent=True)
        self.assertAlmostEqual(ind, 3.0, places=12)

        # Dependent combine = log((1/K) * sum exp(log_e_k))
        dep = eproc.combine_evalues([0.0, math.log(3.0)], independent=False)
        self.assertAlmostEqual(dep, math.log(2.0), places=12)

    def test_confidence_sequence_eb_and_normal_modes(self) -> None:
        rng = random.Random(7)
        true_mu = 0.65
        xs = [1.0 if rng.random() < true_mu else 0.0 for _ in range(400)]

        cs_eb = eproc.ConfidenceSequence(alpha=0.05, lower=0.0, upper=1.0, mode="eb")
        prev_width = float("inf")
        for x in xs[:50]:
            r = cs_eb.update(x)
            self.assertLessEqual(r.width, prev_width + 1e-12)
            prev_width = r.width
        r_final = cs_eb.update_batch(xs[50:])
        self.assertEqual(r_final.n, 400)
        self.assertLessEqual(r_final.low, true_mu)
        self.assertGreaterEqual(r_final.high, true_mu)
        self.assertEqual(cs_eb.bounds, (r_final.low, r_final.high))
        self.assertAlmostEqual(cs_eb.width, r_final.high - r_final.low, places=12)

        # Single-path monotonicity & containment at strict alpha=1e-4, plus
        # empirical time-uniform coverage across 100 independent trials at alpha=0.05.
        cs_strict = eproc.ConfidenceSequence(alpha=1e-4, mode="normal", sigma=1.0, v_opt=100.0)
        g_xs = [rng.gauss(2.5, 1.0) for _ in range(300)]
        r_strict = cs_strict.update_batch(g_xs)
        self.assertLessEqual(r_strict.low, 2.5)
        self.assertGreaterEqual(r_strict.high, 2.5)

        misses = 0
        for trial_seed in range(100):
            tr = random.Random(1000 + trial_seed)
            cs_norm = eproc.ConfidenceSequence(alpha=0.05, mode="normal", sigma=1.0, v_opt=100.0)
            res = cs_norm.update_batch([tr.gauss(2.5, 1.0) for _ in range(200)])
            if res.low > 2.5 or res.high < 2.5:
                misses += 1
        self.assertLessEqual(misses, 8)  # nominal <= 5%, generous finite-sample bound for 100 trials

    def test_edetector_drift_and_reset(self) -> None:
        ed = eproc.EDetector(mu0=0.5, alpha=0.05, lower=0.0, upper=1.0, side="upper")
        alarmed = False
        for _ in range(250):
            r = ed.update(0.95)
            if r.alarm:
                alarmed = True
                break
        self.assertTrue(alarmed)
        self.assertGreaterEqual(ed.n_alarms, 1)
        ed.reset_alarm()
        self.assertAlmostEqual(ed.log_stat, 0.0, places=12)

    def test_agent_race_elimination_and_winner(self) -> None:
        rng = random.Random(99)
        probs = [0.35, 0.85, 0.50]
        race = eproc.AgentRace(
            n_arms=3,
            alpha=0.05,
            lower=0.0,
            upper=1.0,
            names=["weak", "champion", "mid"],
        )
        for _ in range(600):
            arm = race.suggest_arm()
            if arm < 0 or race.winner is not None:
                break
            reward = 1.0 if rng.random() < probs[arm] else 0.0
            race.update(arm, reward)

        self.assertEqual(race.winner, 1)
        self.assertEqual(race.winner_name, "champion")
        self.assertEqual(race.active_arms, [1])
        st = race.arm_state("champion")
        self.assertTrue(st.active)
        self.assertEqual(st.name, "champion")
        self.assertFalse(race.arm_state("weak").active)

        # Also test update_batch with arm names
        race2 = eproc.AgentRace(n_arms=2, names=["a", "b"])
        status2 = race2.update_batch(["a", "b", "a", "b"], [0.9, 0.1, 0.8, 0.2])
        self.assertEqual(status2.total_obs, 4)


class TestConformalFamily(unittest.TestCase):
    def test_multirate_conformal_non_crossing(self) -> None:
        rng = random.Random(2026)
        cp = eproc.MultiRateConformal(alphas=(0.50, 0.20, 0.10, 0.05, 0.01), window_size=128, pid_ki=0.05)
        scores = [abs(rng.gauss(0.0, 1.0)) for _ in range(300)]
        meta = cp.update_batch(scores)
        self.assertEqual(meta.n_rates, 5)
        # Non-crossing guarantee: quantiles must be non-decreasing
        for i in range(4):
            self.assertLessEqual(meta.quantiles[i], meta.quantiles[i + 1] + 1e-12)

        pred = cp.predict(10.0, scale=2.0)
        self.assertEqual(len(pred.intervals()), 5)
        for i in range(4):
            self.assertGreaterEqual(pred.lows[i], pred.lows[i + 1] - 1e-12)
            self.assertLessEqual(pred.highs[i], pred.highs[i + 1] + 1e-12)

    def test_delayed_conformal_strict_causality(self) -> None:
        dc1 = eproc.DelayedConformal(alphas=(0.20, 0.10, 0.05), window_size=64, pid_ki=0.05)
        dc2 = eproc.DelayedConformal(alphas=(0.20, 0.10, 0.05), window_size=64, pid_ki=0.05)

        t1_a, iv1_a = dc1.issue(5.0, scale=1.0)
        t1_b, iv1_b = dc1.issue(6.0, scale=1.0)
        t2_a, iv2_a = dc2.issue(5.0, scale=1.0)
        t2_b, iv2_b = dc2.issue(6.0, scale=1.0)

        self.assertEqual(iv1_a, iv2_a)
        self.assertEqual(iv1_b, iv2_b)
        self.assertEqual(dc1.pending_count, 2)

        # Reveal out of order with drastically different future labels
        dc1.reveal(t1_b, 6.2)
        dc2.reveal(t2_b, 100.0)
        self.assertEqual(dc1.pending_count, 1)
        dc1.reveal(t1_a, 5.1)
        self.assertEqual(dc1.pending_count, 0)
        self.assertEqual(dc1.n_updates, 2)

    def test_budget_conformal_right_censoring(self) -> None:
        bc = eproc.BudgetConformal(alpha=0.10, window_size=64, censor_penalty=1.5)
        for _ in range(40):
            bc.update(base_estimate=100.0, observed_cost=120.0, censored=False)
        bound_clean = bc.bound(100.0)
        for _ in range(20):
            bc.update(base_estimate=100.0, observed_cost=200.0, censored=True)
        bound_after_censor = bc.bound(100.0)
        self.assertEqual(bc.n_completed, 40)
        self.assertEqual(bc.n_censored, 20)
        self.assertGreaterEqual(bound_after_censor, bound_clean)

    def test_conformal_monoid_algebraic_exactness_and_wire(self) -> None:
        rng = random.Random(55)
        s_a = [abs(rng.gauss(0.0, 1.0)) for _ in range(120)]
        s_b = [abs(rng.gauss(0.5, 1.2)) for _ in range(85)]
        s_c = [abs(rng.gauss(1.0, 0.7)) for _ in range(60)]

        ma = eproc.ConformalMonoid(scale_hint=1.0)
        mb = eproc.ConformalMonoid(scale_hint=1.0)
        mc = eproc.ConformalMonoid(scale_hint=1.0)
        ma.add_batch(s_a)
        mb.add_batch(s_b)
        mc.add_batch(s_c)

        # Commutativity: bit-exact bin counts
        ab = ma.merge(mb)
        ba = mb.merge(ma)
        self.assertEqual(ab.count, 205)
        self.assertEqual(ab.bin_counts, ba.bin_counts)

        # Associativity: (A + B) + C == A + (B + C) bit-exact bin counts
        lhs = (ma + mb) + mc
        rhs = ma + (mb + mc)
        self.assertEqual(lhs.count, rhs.count)
        self.assertEqual(lhs.bin_counts, rhs.bin_counts)
        self.assertEqual(lhs.quantile(0.9), rhs.quantile(0.9))

        # Exact leave-one-out subtraction: (A + B) - A == B bit-exact
        loo_b = ab.subtract(ma)
        self.assertEqual(loo_b.count, mb.count)
        self.assertEqual(loo_b.bin_counts, mb.bin_counts)

        # Wire serialization roundtrip (fixed 640 bytes)
        wire = lhs.to_bytes()
        self.assertEqual(len(wire), 640)
        restored = eproc.ConformalMonoid.from_bytes(wire)
        self.assertEqual(restored.count, lhs.count)
        self.assertEqual(restored.bin_counts, lhs.bin_counts)
        self.assertEqual(restored.quantile(0.95), lhs.quantile(0.95))

        # Invalid LOO subtraction (subtracting larger monoid) raises ValueError
        with self.assertRaises(ValueError):
            ma.subtract(lhs)


class TestFusionAndSubsample(unittest.TestCase):
    def test_safe_fuser_poisoning_isolation_and_seqlock_weights(self) -> None:
        rng = random.Random(314)
        fuser = eproc.SafeFuser(n_experts=4, anchor_idx=0, alpha_deg=0.05)

        for t in range(120):
            y = math.sin(t * 0.1)
            # Expert 0: good anchor; Expert 1: best; Expert 2: noisy; Expert 3: Byzantine attacker!
            preds = [
                y + rng.gauss(0.0, 0.15),
                y + rng.gauss(0.0, 0.03),
                y + rng.gauss(0.0, 0.6),
                1e6 if t >= 20 else y,
            ]
            step = fuser.update(preds, y)
            self.assertTrue(math.isfinite(step.fused_pred))
            self.assertLess(abs(step.fused_pred - y), 50.0)

        w = fuser.weights()
        self.assertEqual(len(w), 4)
        self.assertAlmostEqual(sum(w), 1.0, places=6)
        # Poisoned expert 3 should have negligible weight or be quarantined
        self.assertTrue(fuser.quarantined[3] or w[3] < 0.05)
        self.assertGreater(fuser.n_poison_clipped, 0)
        # Predict path works without mutating n_updates
        n_before = fuser.n_updates
        p = fuser.predict([1.0, 1.0, 1.0, 1e6])
        self.assertEqual(fuser.n_updates, n_before)
        self.assertTrue(math.isfinite(p))

    def test_echopool_dedup_and_reliability(self) -> None:
        pool = eproc.EchoPool(n_sources=4, n_claims=8, prior_prob=0.5)
        r1 = pool.submit(source_id=0, claim_id=0, evidence_id="arxiv:2611.0001", llr=2.0)
        self.assertTrue(r1.accepted)
        self.assertGreater(r1.posterior_prob, 0.5)

        # Same evidence echoed by another source must be dropped idempotently
        r2 = pool.submit(source_id=1, claim_id=0, evidence_id="arxiv:2611.0001", llr=2.0)
        self.assertFalse(r2.accepted)
        self.assertEqual(pool.n_echo_dropped, 1)
        self.assertAlmostEqual(r2.posterior_prob, r1.posterior_prob, places=12)

        # Train source 0 accurate, source 2 adversarial across claims 1..6
        for c in range(1, 7):
            pool.submit(source_id=0, claim_id=c, evidence_id=c * 100 + 1, llr=2.0)
            pool.submit(source_id=2, claim_id=c, evidence_id=c * 100 + 2, llr=-2.0)
            pool.resolve(claim_id=c, outcome_positive=True)

        self.assertGreater(pool.source_reliability(0), pool.source_reliability(2))

    def test_varopt_rls_and_sift_voi(self) -> None:
        vo = eproc.VarOptSampler(k=32, seed=42)
        true_sum = 0.0
        for i in range(200):
            w = 10.0 if i % 25 == 0 else 1.0
            v = float((i % 7) + 1)
            vo.push(i, w, v)
            true_sum += w * v
        self.assertEqual(vo.count, 32)
        self.assertEqual(vo.n_seen, 200)
        est_sum = vo.estimate_sum()
        self.assertGreater(est_sum, 0.0)
        self.assertLess(abs(est_sum - true_sum) / true_sum, 0.35)
        self.assertGreater(vo.estimate_mean(), 0.0)
        self.assertEqual(len(vo.items()), 32)

        rls = eproc.OnlineRLSSampler(d=4, lambda_reg=1.0, oversample=2.0, min_prob=0.05, seed=7)
        for i in range(100):
            vec = [1.0 if (i % 4) == j else 0.1 for j in range(4)]
            out = rls.step(vec)
            self.assertGreaterEqual(out.leverage, 0.0)
            self.assertGreaterEqual(out.prob, 0.05)
            self.assertLessEqual(out.prob, 1.0)
            if out.keep:
                self.assertAlmostEqual(out.ht_weight, 1.0 / out.prob, places=10)
            else:
                self.assertEqual(out.ht_weight, 0.0)
        self.assertEqual(rls.n_seen, 100)
        self.assertGreater(rls.n_kept, 0)

        sift = eproc.SiftVOISelector(n_sites=4, per_site=4, signal_mean=2.0, noise_sd=1.0)
        sift.set_noise_scale(0, 1.0)
        sift.observe_site(1, 3.5)
        sift.observe_channel(4, 2.8, n_ticks=4)
        act = sift.select_next(cost_site=1.0, cost_channel=1.0, cost_ticks=8.0, ticks_per_read=8)
        self.assertIn(act.action_type, (0, 1, 2))
        self.assertIn(act.action_name, ("site", "channel", "ticks"))
        self.assertGreaterEqual(act.evoi_per_cost, 0.0)


class TestCLI(unittest.TestCase):
    def _run_cli(self, argv: list[str], stdin_data: str = "") -> tuple[int, str, str]:
        out = io.StringIO()
        err = io.StringIO()
        rc = cli.main(argv, stdin=io.StringIO(stdin_data), stdout=out, stderr=err)
        return rc, out.getvalue(), err.getvalue()

    def test_version_and_gate_and_race_and_conformal_and_fuse_and_subsample(self) -> None:
        rc, out, err = self._run_cli(["version", "--json"])
        self.assertEqual(rc, 0, err)
        ver = json.loads(out)
        self.assertEqual(ver["abi_version"], 1)

        # gate (JSONL)
        jsonl_gate = "\n".join(json.dumps({"score": 0.9}) for _ in range(120)) + "\n"
        rc, out, err = self._run_cli(
            ["gate", "--mu0", "0.5", "--alpha", "0.05", "--stop-early", "--json"],
            stdin_data=jsonl_gate,
        )
        self.assertEqual(rc, 0, err)
        g = json.loads(out)
        self.assertEqual(g["verdict"], "PASS")
        self.assertTrue(g["rejected"])
        self.assertLess(g["n"], 120)

        # race (JSONL)
        rows = []
        for _ in range(150):
            rows.append(json.dumps({"arm": "fast", "reward": 0.95}))
            rows.append(json.dumps({"arm": "slow", "reward": 0.20}))
        rc, out, err = self._run_cli(
            ["race", "--names", "fast,slow", "--stop-early", "--json"],
            stdin_data="\n".join(rows) + "\n",
        )
        self.assertEqual(rc, 0, err)
        r = json.loads(out)
        self.assertEqual(r["status"]["winner_name"], "fast")

        # conformal
        conf_in = "\n".join(json.dumps({"pred": 10.0, "y": 10.0 + (i % 5) * 0.2}) for i in range(60))
        rc, out, err = self._run_cli(
            ["conformal", "--predict", "10.0", "--json"],
            stdin_data=conf_in,
        )
        self.assertEqual(rc, 0, err)
        c = json.loads(out)
        self.assertEqual(c["n_updates"], 60)
        self.assertIn("prediction", c)

        # fuse
        fuse_in = "\n".join(json.dumps({"preds": [1.0, 1.05, 1e6], "y": 1.02}) for _ in range(30))
        rc, out, err = self._run_cli(["fuse", "--anchor", "0", "--json"], stdin_data=fuse_in)
        self.assertEqual(rc, 0, err)
        f = json.loads(out)
        self.assertEqual(f["n_updates"], 30)
        self.assertAlmostEqual(sum(f["weights"]), 1.0, places=5)

        # subsample varopt & rls
        vo_in = "\n".join(json.dumps({"id": i, "weight": 1.0, "value": 2.0}) for i in range(40))
        rc, out, err = self._run_cli(["subsample", "--method", "varopt", "--k", "16", "--json"], stdin_data=vo_in)
        self.assertEqual(rc, 0, err)
        v = json.loads(out)
        self.assertEqual(v["n_seen"], 40)
        self.assertAlmostEqual(v["estimate_sum"], 80.0, places=5)

        rls_in = "\n".join(json.dumps({"x": [1.0, 0.5, 0.25]}) for _ in range(20))
        rc, out, err = self._run_cli(["subsample", "--method", "rls", "--json"], stdin_data=rls_in)
        self.assertEqual(rc, 0, err)
        rs = json.loads(out)
        self.assertEqual(rs["n_seen"], 20)


class TestMcpServer(unittest.TestCase):
    def _rpc(self, srv: mcp_server.McpServer, msg: dict) -> dict | None:
        line = srv.handle_line(json.dumps(msg))
        return None if line is None else json.loads(line)

    def test_handshake_2025_and_2026_and_discovery_and_tools_list(self) -> None:
        for proto in ("2025-11-25", "2026-07-28"):
            srv = mcp_server.McpServer()
            init_resp = self._rpc(
                srv,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": proto, "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
                },
            )
            assert init_resp is not None
            self.assertEqual(init_resp["result"]["protocolVersion"], proto)
            self.assertEqual(init_resp["result"]["serverInfo"]["name"], "eproc")

            # Notification must produce no response
            notif = self._rpc(srv, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
            self.assertIsNone(notif)

            # 2026-07-28 server/discover
            disc = self._rpc(srv, {"jsonrpc": "2.0", "id": 2, "method": "server/discover", "params": {}})
            assert disc is not None
            self.assertEqual(disc["result"]["ttlMs"], 300000)
            self.assertEqual(disc["result"]["cacheScope"], "public")

            # tools/list caching headers
            tl = self._rpc(srv, {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}})
            assert tl is not None
            self.assertEqual(tl["result"]["ttlMs"], 300000)
            self.assertEqual(tl["result"]["cacheScope"], "public")
            names = {t["name"] for t in tl["result"]["tools"]}
            self.assertEqual(
                names,
                {
                    "eproc_etest",
                    "eproc_confidence_sequence",
                    "eproc_race",
                    "eproc_conformal",
                    "eproc_cmonoid_merge",
                    "eproc_safe_fuse",
                    "eproc_echopool",
                },
            )

            # ping
            ping = self._rpc(srv, {"jsonrpc": "2.0", "id": 4, "method": "ping"})
            assert ping is not None
            self.assertEqual(ping["result"], {})

    def test_all_7_tools_and_multi_turn_sessions(self) -> None:
        srv = mcp_server.McpServer()

        def call_tool(req_id: int, name: str, arguments: dict) -> dict:
            resp = self._rpc(
                srv,
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                },
            )
            assert resp is not None
            self.assertIn("result", resp)
            self.assertFalse(resp["result"]["isError"], resp["result"])
            return resp["result"]["structuredContent"]

        # 1. Stateful multi-turn eproc_etest
        t1 = call_tool(10, "eproc_etest", {"mu0": 0.5, "values": [0.9] * 10, "session_id": "eval-1"})
        self.assertEqual(t1["n"], 10)
        t2 = call_tool(11, "eproc_etest", {"values": [0.9] * 80, "session_id": "eval-1"})
        self.assertEqual(t2["n"], 90)
        self.assertTrue(t2["rejected"])
        self.assertEqual(t2["verdict"], "REJECT")

        # 2. eproc_confidence_sequence
        cs = call_tool(12, "eproc_confidence_sequence", {"values": [0.6, 0.7, 0.8, 0.65], "session_id": "cs-1"})
        self.assertEqual(cs["n"], 4)
        self.assertLessEqual(cs["low"], cs["high"])

        # 3. eproc_race
        rc = call_tool(
            13,
            "eproc_race",
            {
                "names": ["a", "b"],
                "observations": [{"arm": "a", "reward": 0.95}, {"arm": "b", "reward": 0.1}] * 60,
                "session_id": "race-1",
            },
        )
        self.assertEqual(rc["status"]["winner_name"], "a")

        # 4. eproc_conformal
        cp = call_tool(
            14,
            "eproc_conformal",
            {
                "scores": [0.1, 0.2, 0.3, 0.4, 0.5] * 10,
                "predict": {"pred": 10.0, "scale": 1.0},
                "session_id": "cp-1",
            },
        )
        self.assertEqual(cp["n_updates"], 50)
        self.assertIn("prediction", cp)

        # 5. eproc_cmonoid_merge (raw shards + wire_b64 + exact LOO subtract)
        cm1 = call_tool(15, "eproc_cmonoid_merge", {"shards": [[0.1, 0.2, 0.3], [0.4, 0.5]]})
        self.assertEqual(cm1["count"], 5)
        self.assertEqual(cm1["wire_size"], 640)
        self.assertEqual(len(base64.b64decode(cm1["wire_b64"])), 640)
        cm2 = call_tool(
            16,
            "eproc_cmonoid_merge",
            {"shards": [cm1["wire_b64"]], "subtract_shard": [0.4, 0.5]},
        )
        self.assertEqual(cm2["count"], 3)

        # 6. eproc_safe_fuse
        sf = call_tool(
            17,
            "eproc_safe_fuse",
            {
                "steps": [{"preds": [1.0, 1.01, 1e6], "y": 1.0}] * 25,
                "predict": [2.0, 2.0, 1e6],
                "session_id": "fuse-1",
            },
        )
        self.assertEqual(sf["n_updates"], 25)
        self.assertTrue(math.isfinite(sf["prediction"]))

        # 7. eproc_echopool
        ep = call_tool(
            18,
            "eproc_echopool",
            {
                "submissions": [
                    {"source_id": 0, "claim_id": 0, "evidence_id": "ev-1", "llr": 1.5},
                    {"source_id": 1, "claim_id": 0, "evidence_id": "ev-1", "llr": 1.5},
                ],
                "resolutions": [{"claim_id": 0, "outcome_positive": True}],
                "session_id": "echo-1",
            },
        )
        self.assertEqual(ep["n_accepted"], 1)
        self.assertEqual(ep["n_echo_dropped"], 1)

    def test_mcp_error_handling_and_nan_safety(self) -> None:
        srv = mcp_server.McpServer()

        # Invalid JSON -> -32700
        bad_json = json.loads(srv.handle_line("{not valid json") or "")
        self.assertEqual(bad_json["error"]["code"], mcp_server.ERR_PARSE)

        # Unknown method -> -32601
        unk_method = self._rpc(srv, {"jsonrpc": "2.0", "id": 1, "method": "foo/bar"})
        assert unk_method is not None
        self.assertEqual(unk_method["error"]["code"], mcp_server.ERR_METHOD_NOT_FOUND)

        # Unknown tool -> -32602
        unk_tool = self._rpc(
            srv,
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "nope", "arguments": {}}},
        )
        assert unk_tool is not None
        self.assertEqual(unk_tool["error"]["code"], mcp_server.ERR_INVALID_PARAMS)

        # Missing required arg -> clean isError response without crashing
        missing_mu0 = self._rpc(
            srv,
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "eproc_etest", "arguments": {"values": [0.5]}}},
        )
        assert missing_mu0 is not None
        self.assertTrue(missing_mu0["result"]["isError"])

        # NaN/Inf in values must never crash or produce invalid JSON
        nan_line = '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"eproc_etest","arguments":{"mu0":0.5,"values":[0.6,NaN,Infinity]}}}'
        raw_out = srv.handle_line(nan_line)
        assert raw_out is not None
        parsed = json.loads(raw_out)
        self.assertFalse(parsed["result"]["isError"])
        self.assertEqual(parsed["result"]["structuredContent"]["n_invalid"], 2)


class TestCycle3RegressionsAndNativeParity(unittest.TestCase):
    def test_w02_safe_fuser_concurrent_snapshot_zero_torn_reads(self) -> None:
        import threading

        sf = eproc.SafeFuser(n_experts=32, anchor_idx=0, alpha_deg=0.05)
        for _ in range(30):
            sf.update([4.0, 0.0] + [8.0] * 30, 0.0)
        torn = 0
        lock = threading.Lock()
        stop = threading.Event()

        def writer() -> None:
            for _ in range(25):
                for _ in range(15):
                    sf.update([0.0] + [2.0] * 31, 0.0)
                for _ in range(15):
                    sf.update([2.0] + [0.0] * 31, 0.0)
            stop.set()

        def reader() -> None:
            nonlocal torn
            bad = 0
            while not stop.is_set():
                w, fb = sf.snapshot()
                if fb:
                    if w[0] != 1.0 or any(x != 0.0 for x in w[1:]):
                        bad += 1
                elif abs(sum(w) - 1.0) > 1e-6:
                    bad += 1
            if bad:
                with lock:
                    torn += bad

        threads = [threading.Thread(target=reader) for _ in range(8)]
        wt = threading.Thread(target=writer)
        for t in threads:
            t.start()
        wt.start()
        wt.join()
        for t in threads:
            t.join()
        self.assertEqual(torn, 0)

    def test_w08_and_w11_cmonoid_scale_mismatch_and_exact_wire_length(self) -> None:
        ma = eproc.ConformalMonoid(scale_hint=1.0)
        mb = eproc.ConformalMonoid(scale_hint=10.0)
        ma.add(1.0)
        mb.add(10.0)
        with self.assertRaises(ValueError):
            ma.merge(mb)

        wire = ma.to_bytes()
        self.assertEqual(len(wire), 640)
        with self.assertRaises(ValueError):
            eproc.ConformalMonoid.from_bytes(wire[:320])
        with self.assertRaises(ValueError):
            eproc.ConformalMonoid.from_bytes(wire + b"EXTRA")

    def test_w09_cli_conformal_zero_or_negative_scale(self) -> None:
        for bad_scale in (0.0, -1.0):
            out = io.StringIO()
            err = io.StringIO()
            rc = cli.main(
                ["conformal", "--json"],
                stdin=io.StringIO(json.dumps({"pred": 1.0, "y": 2.0, "scale": bad_scale}) + "\n"),
                stdout=out,
                stderr=err,
            )
            self.assertEqual(rc, 2)

    def test_w12_edetector_reset_then_empty_batch(self) -> None:
        ed = eproc.EDetector(mu0=0.5, alpha=0.05, lower=0.0, upper=1.0, side="upper")
        for _ in range(60):
            ed.update(1.0)
        self.assertGreater(ed.n_alarms, 0)
        ed.reset_alarm()
        res = ed.update_batch([])
        self.assertAlmostEqual(res.log_stat, 0.0, places=12)
        self.assertAlmostEqual(res.stat, 1.0, places=12)
        self.assertFalse(res.alarm)

    def test_w14_w15_w16_mcp_strict_validation_and_invalid_request(self) -> None:
        srv = mcp_server.McpServer()

        # W15: invalid request lacking id must return -32600 with id=None
        raw = srv.handle_line(json.dumps({"jsonrpc": "1.0", "method": "ping"}))
        assert raw is not None
        resp = json.loads(raw)
        self.assertIsNone(resp["id"])
        self.assertEqual(resp["error"]["code"], mcp_server.ERR_INVALID_REQUEST)

        # W14: reset="false" must NOT wipe session and must return isError=True
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
        r_bad = json.loads(
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
        self.assertTrue(r_bad["result"]["isError"])
        # Confirm session s1 is intact (still n=2)
        r_chk = json.loads(
            srv.handle_line(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {"name": "eproc_etest", "arguments": {"session_id": "s1"}},
                    }
                )
            )
            or ""
        )
        self.assertEqual(r_chk["result"]["structuredContent"]["n"], 2)

        # W16: bool must be rejected as numeric across all tools
        bool_cases = [
            ("eproc_safe_fuse", {"n_experts": 2, "steps": [{"preds": [1.0, 1.0], "y": True}]}),
            ("eproc_echopool", {"submissions": [{"source_id": 0, "claim_id": 0, "evidence_id": 1, "llr": True}]}),
            ("eproc_conformal", {"observations": [{"pred": True, "y": 1.0}]}),
            ("eproc_etest", {"mu0": 0.5, "values": [True]}),
            ("eproc_confidence_sequence", {"values": [False]}),
            ("eproc_race", {"n_arms": 2, "observations": [{"arm": 0, "reward": True}]}),
            ("eproc_cmonoid_merge", {"shards": [[True]]}),
        ]
        for idx, (tool_name, tool_args) in enumerate(bool_cases, start=10):
            r = json.loads(
                srv.handle_line(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": idx,
                            "method": "tools/call",
                            "params": {"name": tool_name, "arguments": tool_args},
                        }
                    )
                )
                or ""
            )
            self.assertTrue(r["result"]["isError"], f"{tool_name} accepted bool as numeric")

    def test_native_extension_loaded_by_default(self) -> None:
        if not os.environ.get("EPROC_DISABLE_NATIVE"):
            from eproc import _native  # type: ignore[attr-defined]

            self.assertIs(eproc.EProcess, _native.EProcess)
            self.assertIs(eproc.ConformalMonoid, _native.ConformalMonoid)
            self.assertIs(eproc.SafeFuser, _native.SafeFuser)
            self.assertIs(eproc.OnlineRLSSampler, _native.OnlineRLSSampler)


if __name__ == "__main__":
    unittest.main()

