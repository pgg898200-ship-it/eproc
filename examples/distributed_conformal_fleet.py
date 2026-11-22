#!/usr/bin/env python3
"""Example 3: 1,024-agent distributed conformal calibration in 640 bytes per agent.

Demonstrates:
  - O(1) 640-byte abelian ConformalMonoid per agent (vs MBs of raw scores)
  - Associative / commutative tree-reduction merge across 1,024 agents
  - Exact leave-one-out (global - local) cold-start peer borrowing

Run:
    PYTHONPATH=python python3 examples/distributed_conformal_fleet.py
"""
from __future__ import annotations

import array
import random
import time

from eproc import ConformalMonoid


def main() -> None:
    rng = random.Random(2026)
    n_agents = 1024
    scores_per_agent = 250

    shards = []
    for _ in range(n_agents):
        m = ConformalMonoid(scale_hint=1.0)
        buf = array.array("d", (abs(rng.gauss(0.0, 1.0)) for _ in range(scores_per_agent)))
        m.add_batch(buf)
        shards.append(m)

    t0 = time.perf_counter()
    fleet = shards[0]
    for s in shards[1:]:
        fleet = fleet.merge(s)
    merge_us = (time.perf_counter() - t0) * 1e6

    # Exact Leave-One-Out (peer-only calibration for agent 0, zero self-leakage)
    peers_for_0 = fleet.subtract(shards[0])

    # Verify wire round-trip
    wire = fleet.to_bytes()
    restored = ConformalMonoid.from_bytes(wire)

    raw_bytes = n_agents * scores_per_agent * 8
    print(f"Fleet size          : {n_agents} agents x {scores_per_agent} scores = {fleet.count:,} scores")
    print(f"Wire payload/agent  : {len(wire)} bytes (vs {raw_bytes:,} bytes raw — {raw_bytes / (len(wire) * n_agents):.1f}x smaller)")
    print(f"1,024-agent merge   : {merge_us:.1f} us total ({merge_us / (n_agents - 1):.2f} us/merge)")
    print(f"Fleet 90% quantile  : {fleet.quantile(0.90):.4f} (restored: {restored.quantile(0.90):.4f})")
    print(f"LOO peer count      : {peers_for_0.count:,} (exact = {fleet.count - shards[0].count:,})")
    print(f"LOO peer 90% q      : {peers_for_0.quantile(0.90):.4f}")


if __name__ == "__main__":
    main()
