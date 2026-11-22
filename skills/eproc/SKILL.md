---
name: eproc
description: Anytime-valid sequential hypothesis testing (e-processes, confidence sequences, early-eliminating model/agent racing), crossing-free multi-rate conformal calibration, 640-byte distributed conformal monoids, Byzantine-safe multi-agent fusion, echo-deduplicated evidence pooling, and causal subsampling via C11/Python/CLI/MCP. Use whenever continuously peeking at evals, gating rollouts, racing prompts/models, calibrating uncertainty intervals, or fusing untrusted agent outputs.
---

# `eproc` — Anytime-Valid Inference & Multi-Agent Statistical Guardrails

Use `eproc` instead of fixed-horizon z-tests, naive p-value peeking, uncalibrated ensembles, or naive multi-agent voting. Every core runs in deterministic C11 (`libeproc`) with zero third-party runtime dependencies.

## When to Reach for `eproc`

| Scenario | Tool / Class | Guarantee |
|---|---|---|
| Live eval / canary gate with continuous peeking | `EProcess` / `eproc gate` / `eproc_etest` | Ville inequality: `P_H0(∃t: E_t >= 1/α) <= α` |
| Always-valid running error bars on a metric | `ConfidenceSequence` / `eproc_confidence_sequence` | `O(1)` Empirical-Bernstein or Normal-Mixture CS |
| Streaming silent regression / drift alarm | `EDetector` | Resettable Page-CUSUM e-detector (`ARL >= 1/α`) |
| Comparing `K` models/prompts/agents cheaply | `AgentRace` / `eproc race` / `eproc_race` | Bonferroni EB-CS with automatic early elimination |
| Multi-rate prediction intervals without crossing | `MultiRateConformal` / `eproc conformal` / `eproc_conformal` | PAVA crossing-free AgACI + Conformal PID |
| Delayed labels arriving out-of-order | `DelayedConformal` | Strict causality (`issue` before `reveal`) |
| Right-censored agent token/step/latency budgets | `BudgetConformal` | Conservative upper predictive bound under timeouts |
| Distributed multi-agent conformal aggregation | `ConformalMonoid` / `eproc_cmonoid_merge` | Fixed 640-byte abelian group; bit-exact LOO subtraction |
| Fusing noisy/poisoned agent predictions | `SafeFuser` / `eproc fuse` / `eproc_safe_fuse` | Scale-free BOA + 3-gate Ville fallback + lock-free seqlock |
| Aggregating claims with echo/sycophancy risk | `EchoPool` / `eproc_echopool` | Deduplicates `(claim_id, evidence_id)` + Beta-Bernoulli peer trust |
| Subsampling massive streams / tool budgets | `VarOptSampler`, `OnlineRLSSampler`, `SiftVOISelector` | Exact Horvitz-Thompson unbiasedness & EVOI/cost |

## CLI Quick Reference

```bash
# Anytime-valid promotion gate over JSONL/CSV/numbers (stops early on rejection)
cat eval_scores.jsonl | eproc gate --mu0 0.70 --alpha 0.05 --side upper --stop-early --json

# Multi-arm prompt/model race with automatic early elimination
cat arm_rewards.jsonl | eproc race --names baseline,candidate_a,candidate_b --alpha 0.05 --stop-early --json

# Crossing-free 50/80/90/95/99% online conformal calibration
cat residuals.jsonl | eproc conformal --predict 12.5 --json

# Byzantine-safe BOA fusion with anchor fallback
cat expert_stream.jsonl | eproc fuse --anchor 0 --alpha-deg 0.05 --json

# Launch MCP 2026-07-28 stdio server (also supports 2025-11-25)
eproc mcp

# Inspect ABI version and SIMD capabilities
eproc version --json
```

## Python API Quick Reference

```python
import eproc  # < 5 ms lazy import (PEP 562)

# 1. Anytime-valid eval gate
ep = eproc.EProcess(mu0=0.70, alpha=0.05, lower=0.0, upper=1.0, side="upper")
res = ep.update_batch([0.85, 0.90, 0.78, 0.92])
if res.rejected:
    print(f"Certified pass at n={res.n}, p={res.p_value:.4g}")

# 2. Multi-arm racing (zero wasted eval tokens on losing arms)
race = eproc.AgentRace(n_arms=3, alpha=0.05, names=["v1", "v2", "v3"])
arm = race.suggest_arm()
status = race.update(arm, 0.91)

# 3. Distributed 640-byte conformal monoid with exact leave-one-out
m_a = eproc.ConformalMonoid(); m_a.add_batch([0.1, 0.3, 0.2])
m_b = eproc.ConformalMonoid(); m_b.add_batch([0.4, 0.2, 0.5])
global_m = m_a.merge(m_b)
loo_a = global_m.subtract(m_a)  # bit-exact bin counts equal to m_b

# 4. Byzantine-safe multi-agent fusion & echo deduplication
fuser = eproc.SafeFuser(n_experts=4, anchor_idx=0, alpha_deg=0.05)
step = fuser.update([1.0, 1.02, 0.98, 1e6], y_true=1.01)
weights = fuser.weights()  # lock-free seqlock snapshot

pool = eproc.EchoPool(n_sources=8, n_claims=32)
r1 = pool.submit(source_id=0, claim_id=1, evidence_id="doc:42", llr=1.8)
r2 = pool.submit(source_id=1, claim_id=1, evidence_id="doc:42", llr=1.8)  # r2.accepted == False
```

## MCP Server Tips

When calling MCP tools (`eproc_etest`, `eproc_confidence_sequence`, `eproc_race`, `eproc_conformal`, `eproc_safe_fuse`, `eproc_echopool`) across multiple turns:
1. Pass a stable `session_id` string to stream incremental ticks without resending history.
2. Check `verdict == "REJECT"` (or `status.winner != null` in `eproc_race`) to stop early and save eval budget.
3. Pass `reset: true` with an existing `session_id` to start a new experiment under the same handle.
4. Use `eproc_cmonoid_merge` with `wire_b64` payloads to merge calibration summaries across parallel worker agents in 640 bytes per shard.
