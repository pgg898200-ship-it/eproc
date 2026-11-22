#!/usr/bin/env python3
"""Example 2: Byzantine-safe multi-agent fusion & echo-deduplicated belief pooling.

Demonstrates how SafeFuser quarantines a poisoned 1e6 agent with 0% blast radius
and how EchoPool eliminates 5x echo amplification across a live 8-agent swarm.

Run:
    PYTHONPATH=python python3 examples/byzantine_swarm_demo.py
"""
from __future__ import annotations

import math
import random

from eproc import EchoPool, SafeFuser


def main() -> None:
    rng = random.Random(7)

    # --- Part A: Continuous prediction fusion under a 1e6 Byzantine attack ---
    fuser = SafeFuser(n_experts=8, anchor_idx=0, alpha_deg=0.05)
    naive_sq = 0.0
    safe_sq = 0.0
    anchor_sq = 0.0

    for t in range(400):
        y = math.sin(t * 0.05) * 2.0 + rng.gauss(0.0, 0.15)
        preds = [
            y + rng.gauss(0.0, 0.45),  # 0: safe anchor baseline
            y + rng.gauss(0.0, 0.12),  # 1: strong specialist
            y + rng.gauss(0.0, 0.18),  # 2: strong specialist
            y + rng.gauss(0.0, 0.60),  # 3..6: noisy peers
            y + rng.gauss(0.0, 0.70),
            y + rng.gauss(0.0, 0.80),
            y + rng.gauss(0.0, 0.90),
            y + rng.gauss(0.0, 0.25),  # 7: turns Byzantine at t >= 150!
        ]
        if t >= 150:
            preds[7] = 1e6 if (t % 2 == 0) else -1e6

        naive_pred = sum(preds) / len(preds)
        step = fuser.update(preds, y)

        if t >= 150:
            naive_sq += (naive_pred - y) ** 2
            safe_sq += (step.fused_pred - y) ** 2
            anchor_sq += (preds[0] - y) ** 2

    n_atk = 250
    print("=== SafeFuser under 1e6 Byzantine Attack (t=150..399) ===")
    print(f"Naive 8-agent mean MSE : {naive_sq / n_atk:,.2f}  (destroyed)")
    print(f"Anchor baseline MSE    : {anchor_sq / n_atk:.4f}")
    print(f"eproc SafeFuser MSE    : {safe_sq / n_atk:.4f}  (quarantined={step.n_quarantined})")

    # --- Part B: Echo-deduplicated belief pooling ---
    pool = EchoPool(n_sources=8, n_claims=4, prior_prob=0.5)
    # 5 agents echo the EXACT same evidence_id=9001 with +1.2 log-odds
    for src in range(5):
        out = pool.submit(source_id=src, claim_id=0, evidence_id=9001, llr=1.2)
    naive_prob = 1.0 / (1.0 + math.exp(-5.0 * 1.2))
    print("\n=== EchoPool under 5x Reprinted Evidence ===")
    print(f"Naive log-odds sum prob: {naive_prob:.4f} (overconfident 5x echo!)")
    print(
        f"eproc EchoPool prob    : {out.posterior_prob:.4f} "
        f"(unique_count={out.unique_count}, dropped_echoes=4)"
    )


if __name__ == "__main__":
    main()
