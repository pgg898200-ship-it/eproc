#!/usr/bin/env python3
"""Example 1: Stop an 8-agent LLM evaluation tournament 4x-8x earlier with exact FWER <= 0.05.

Run:
    PYTHONPATH=python python3 examples/multi_agent_eval_demo.py
"""
from __future__ import annotations

import random

from eproc import AgentRace, EProcess


def main() -> None:
    rng = random.Random(20261122)

    # 8 candidate agents/prompts with unknown true task pass rates
    agents = [
        ("prompt-baseline", 0.50),
        ("prompt-cot-v1",   0.54),
        ("prompt-cot-v2",   0.58),
        ("agent-tool-v1",   0.46),
        ("agent-tool-v2",   0.61),
        ("agent-hybrid",    0.74),  # true winner
        ("agent-fast-mini", 0.41),
        ("agent-strict",    0.52),
    ]

    race = AgentRace(
        n_arms=len(agents),
        alpha=0.05,
        lower=0.0,
        upper=1.0,
        names=[name for name, _ in agents],
    )

    fixed_horizon_budget = len(agents) * 600  # 4,800 eval calls if naive
    calls = 0

    while race.winner is None and calls < fixed_horizon_budget:
        arm = race.suggest_arm()
        _, true_p = agents[arm]
        passed = 1.0 if rng.random() < true_p else 0.0
        status = race.update(arm, passed)
        calls += 1
        if calls % 250 == 0 or status.winner is not None:
            print(
                f"[call {calls:4d}] active={status.n_active}/{status.n_arms} "
                f"leader={agents[status.leader][0]} "
                f"[{status.leader_low:.3f}, {status.leader_high:.3f}]"
            )

    if race.winner is not None:
        saved_pct = 100.0 * (1.0 - calls / fixed_horizon_budget)
        print(
            f"\nCertified winner: {agents[race.winner][0]} "
            f"in {calls} calls vs {fixed_horizon_budget} fixed-horizon "
            f"({fixed_horizon_budget / calls:.2f}x faster, {saved_pct:.1f}% eval tokens saved)"
        )

    # Also verify head-to-head against baseline with an exact ONS betting E-Process
    ep = EProcess(mu0=0.50, alpha=0.05, lower=0.0, upper=1.0, side="upper")
    for step in range(1, 501):
        res = ep.update(1.0 if rng.random() < 0.74 else 0.0)
        if res.rejected:
            print(
                f"Head-to-head EProcess certified >0.50 at step {step} "
                f"(e_value={res.e_value:.1f}, anytime p={res.p_value:.4f})"
            )
            break


if __name__ == "__main__":
    main()
