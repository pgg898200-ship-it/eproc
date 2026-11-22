# eproc

**Sub-microsecond anytime-valid statistics, online conformal calibration, and Byzantine-safe multi-agent fusion.**
Zero runtime dependencies. Pure C11/AVX2 + native Python C-API + single-header C/C++17 + CLI + MCP 2026-07-28 server + Agent Skill.

> **Stop peeking blindly. Stop burning 80% of eval tokens. Stop letting one poisoned or echoing agent wreck your swarm.**

```text
Continuous peeking false-positive rate (2,000 nulls, T=1,000, nominal alpha=0.05)
  Student-t peeked every tick |███████████████████████████████████████░| 39.2%  (broken)
  eproc.EProcess (ONS bet)    |████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|  3.9%  (exact <= 5%)

Eval calls to certify +0.15 uplift vs fixed-horizon N=857 (100% power)
  Fixed-horizon plan          |████████████████████████████████████████|  857 calls
  eproc.EProcess early stop   |████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|   88 calls (9.74x faster, 88.7% saved)

8-agent swarm MSE under 1e6 Byzantine attack + regime shift (B4a)
  Naive 8-agent mean          |████████████████████████████████████████| 10,424,109,416.18
  Anchor baseline             |█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|          0.1951
  eproc.SafeFuser (BOA+Ville) |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|          0.0220 (-88.7% vs anchor)
```

---

## Why `eproc`

Live multi-agent systems, online evaluators, model routers, and streaming ML pipelines break classical fixed-horizon statistics in four ways:

1. **Continuous peeking breaks p-values:** Checking a t-test/z-test on live LLM evals or rollouts inflates a 5% test to **39.2% false positives** (`N=2,000` live Monte Carlo). Waiting for fixed horizons wastes **56%–89% of tokens**.
2. **Distribution shift & delayed labels break static conformal intervals:** Existing Python/R libraries (`online_conformal` archived 2025, `MAPIE`, `confseq`, `conformalForecast`) are pure Python/R (`85 µs–5 ms`/step), lack multi-rate non-crossing guarantees, and cannot merge calibration state across distributed fleets.
3. **Multi-agent swarms collapse under echoes and outliers:** Naive voting double-counts reprinted evidence, fails to down-weight sycophantic/unreliable peers, and explodes under a single poisoned agent.
4. **Firehose monitoring drowns in loud distractors:** Ranking raw channel means wastes 96% of attention on high-variance channels while missing quiet true anomalies.

`eproc` solves all four in **one auditable, zero-allocation C11/AVX2 engine** (`0` calls to `malloc`/`free` in `libeproc.so`, `0` third-party Python dependencies, **0.78 ms** cold import, **0-ULP** bit-identity between AVX2 and portable scalar builds).

---

## Install (Zero Dependencies)

### Python (3.10+, zero external deps)

```bash
pip install eproc

# Or build directly from source (compiles self-contained C11 _native wheel):
git clone https://github.com/pgg898200-ship-it/eproc.git
cd eproc
pip install .                           # portable C11 build
EPROC_NATIVE_ARCH=1 pip install .       # host-tuned AVX2/FMA/NEON build
```

### Single-Header C11 / C++17 (stb-style, 1 file)

Copy [`dist/eproc_single.h`](dist/eproc_single.h) into your project:

```c
#define EPROC_IMPLEMENTATION
#include "eproc_single.h"
// Compile: cc -std=c11 -O3 -ffp-contract=off main.c -lm
// Or C++17: c++ -std=c++17 -O3 -ffp-contract=off main.cpp -lm
```

### Coding Agents (MCP 2026-07-28 Server + Agent Skill)

```bash
# Run zero-dependency stdio MCP server (supports 2025-11-25 & 2026-07-28)
eproc mcp
```

Drop-in manifests included for:
- **Claude Code:** `.claude-plugin/plugin.json` + `skills/eproc/SKILL.md`
- **Gemini CLI:** `gemini-extension.json` + `skills/eproc/SKILL.md`
- **Codex / Cursor / Generic MCP Hosts:** `eproc mcp` (`tools/list` exposes `eproc_etest`, `eproc_confidence_sequence`, `eproc_race`, `eproc_conformal`, `eproc_cmonoid_merge`, `eproc_safe_fuse`, `eproc_echopool` with stateful `session_id` streaming).

---

## 30-Second Quickstarts

### 1. Anytime-Valid Eval Gate & Multi-Agent Tournament (`EProcess`, `AgentRace`)

```python
from eproc import EProcess, AgentRace

# Safe to peek on EVERY tick — exact Ville inequality P(exists t: E_t >= 1/alpha) <= alpha
ep = EProcess(mu0=0.50, alpha=0.05, lower=0.0, upper=1.0, side="upper")
for score in live_eval_stream:
    res = ep.update(score)
    if res.rejected:
        print(f"Certified uplift at step {res.n}: e={res.e_value:.1f}, p={res.p_value:.4f}")
        break

# Race K=8 agents/prompts with automatic early elimination & FWER <= 0.05
race = AgentRace(n_arms=8, alpha=0.05, lower=0.0, upper=1.0)
while race.winner is None:
    arm = race.suggest_arm()
    race.update(arm, run_agent_task(arm))
```

### 2. Multi-Rate Crossing-Free Online Conformal + 640-Byte Distributed Monoid

```python
from eproc import MultiRateConformal, ConformalMonoid

# Calibrate 50/80/90/95/99% intervals simultaneously in 316 ns/step (zero quantile crossing)
cp = MultiRateConformal(alphas=(0.50, 0.20, 0.10, 0.05, 0.01), window_size=128, pid_ki=0.05)
pred = cp.predict(10.0, scale=1.0)       # pred.lows, pred.highs, pred.radii (strictly nested)
cp.update(abs(y_true - 10.0))

# Distributed fleet calibration: 640 bytes/agent, bit-exact abelian merge + LOO peer borrow
shard_a = ConformalMonoid(); shard_a.add_batch(scores_a)
shard_b = ConformalMonoid(); shard_b.add_batch(scores_b)
fleet = shard_a.merge(shard_b)           # associative & commutative (0 integer drift)
peers_for_a = fleet.subtract(shard_a)    # exact Leave-One-Out peer calibration
q90 = peers_for_a.quantile(0.90)
wire_640b = fleet.to_bytes()             # fixed 640 bytes regardless of N scores
```

### 3. Byzantine-Safe Multi-Agent Fusion & Echo-Deduplicated Belief Pooling

```python
from eproc import SafeFuser, EchoPool

# Scale-Free BOA + 3-Gate Ville Non-Inferiority + Boehm-2012 Lock-Free Seqlock
fuser = SafeFuser(n_experts=8, anchor_idx=0, alpha_deg=0.05)
step = fuser.update(expert_preds, y_true)   # quarantines 1e6/NaN Byzantine peers automatically
weights, fallback = fuser.snapshot()        # wait-free concurrent read (0 torn reads)

# Provenance-deduplicated log-odds pool + online Beta-Bernoulli peer reliability
pool = EchoPool(n_sources=8, n_claims=64, prior_prob=0.5)
out = pool.submit(source_id=3, claim_id=0, evidence_id="story-9001", llr=1.2)  # echoes dropped
pool.resolve(claim_id=0, outcome_positive=True)  # updates per-source reliability online
```

### 4. Causal Horvitz-Thompson Subsampling & 8,192-Channel `SiftVOI`

```python
from eproc import VarOptSampler, OnlineRLSSampler, SiftVOISelector

vo = VarOptSampler(k=256, seed=42)                # exact Cohen-2011 unbiased HT weights
rls = OnlineRLSSampler(d=16, oversample=2.0)      # 118 ns Sherman-Morrison ridge leverage
sift = SiftVOISelector(n_sites=1024, per_site=8)  # exact Gaussian max-entropy B&B VOI
```

### 5. Zero-Dependency CLI (`< 25 ms` startup)

```bash
cat evals.jsonl | eproc gate --mu0 0.50 --alpha 0.05 --side upper
cat tournament.jsonl | eproc race --arms 8 --alpha 0.05
cat residuals.jsonl | eproc conformal --alphas 0.5,0.2,0.1,0.05,0.01
eproc bench --quick
eproc version
```

---

## Verified Benchmarks `[LIVE-CPU]`

All numbers measured live on `Intel(R) Xeon(R) CPU @ 2.30GHz` (2 cores, AVX2+FMA, Python 3.12.16, GCC 13.3.0 `-std=c11 -O3 -ffp-contract=off -march=native`). Full provenance, Wilson/bootstrap CIs, and raw paired trials in [`benchmarks/receipts.json`](benchmarks/receipts.json) and [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md).

### Kernel Latency & Speedup vs Pure-Python Baselines (Median of 5 Paired Runs)

| Kernel | Pure Python (`ns/op`) | `eproc` Python API (`ns/op`) | `eproc` C11 Native (`ns/op`) | Throughput | Speedup vs Python |
|---|---:|---:|---:|---:|---:|
| **`ConformalMonoid.subtract` (exact LOO)** | 578,975 | **267.8** | — | 3.73 M/s | **2,161.7×** |
| **`MultiRateConformal` (5-rate AgACI+PID, batch)** | 86,490 | **316.8** | **315.3** | 3.17 M/s | **273.8×** |
| **`MultiRateConformal` (5-rate AgACI+PID, scalar)** | 86,330 | **446.3** | **315.3** | 2.24 M/s | **193.4×** |
| **`OnlineRLSSampler` (`d=16` Sherman-Morrison)** | 26,176 | **230.1** | **118.6** | 8.43 M/s | **220.7×** (C11) / **113.8×** (Py) |
| **`SiftVOISelector` (`2,048` ch exact B&B MI)** | 2,049,024 | **11,004** (`11.0 µs`) | **25,940** (select+obs) | 90.9k/s | **186.2×** |
| **`EProcess` (ONS betting, batch / C11)** | 2,125 | **27.5** (scalar **112.0**) | **30.1** | **36.3 M/s** | **70.6×** (C11) / **19.0×** (Py) |
| **`SafeFuser` (8 experts + 3 Ville gates + Seqlock)** | 25,531 | **614.0** | **566.0** | 1.77 M/s | **45.1×** |
| **`EchoPool` (dedup + Beta-Bernoulli log-odds)** | 1,572 | **205.9** | **30.8** | **32.5 M/s** | **51.0×** (C11) / **7.6×** (Py) |
| **`ConformalMonoid.merge` (640 B AVX2)** | 1,460 | **208.8** | **38.0** | **26.3 M/s** | **38.5×** (C11) / **7.0×** (Py) |
| **`ConfidenceSequence` (Empirical-Bernstein)** | 2,583 | **63.9** (scalar **121.2**) | **67.9** | **15.6 M/s** | **38.0×** (C11) / **21.3×** (Py) |
| **`VarOptSampler` (`k=256` Cohen-2011 HT)** | 2,002 | **247.8** | **134.0** | 7.46 M/s | **14.9×** (C11) / **8.1×** (Py) |

### End-to-End Multi-Agent & Distributed Benchmarks

| Benchmark | Incumbent / Naive Control | **`eproc`** | Verified Margin |
|---|---|---|---|
| **B1a: Continuous Peeking Null Safety (`N=2,000, T=1,000`)** | Student-t peeked every tick: **39.20% FPR** | **3.90% FPR** (Beta skewed: **4.00%**) | **10.1× fewer false positives** (strictly `<= 5%`) |
| **B1b: Early Stopping (`+0.15` uplift vs `N=857` fixed)** | `857` calls (0% saved) | **88 median calls** (`100%` power) | **9.74× faster** (**88.7% tokens saved**) |
| **B1c: `K=8` Agent Tournament (`100` runs)** | `6,400` calls fixed | **3,994 median calls**, **0.00% wrong winners / 0.00% null FWER** | **1.60× faster** (**36.1% mean saved**) |
| **B3: 1,024-Agent Fleet Conformal (`512k` scores)** | `4.10 MB` raw, `11.19 ms` merge | **640 B**, **0.51 ms** Py / **74.9 µs** C11, **0 bin drift** | **6,400× smaller wire**, **149× faster C11** |
| **B4a: 8-Agent `1e6` Byzantine + Regime Shift** | Naive mean MSE `10.42B`; Anchor `0.195` | Attack **`0.0220`**, post-shift **`0.0284`** | **-88.7% vs anchor** (`-85.0%` post-shift) |
| **B4b: 8-Agent 5× Echo + Byzantine Liar** | Naive vote Brier `0.3709` (skill `-0.484`, `1000` echoes) | Brier **`0.1483`** (skill **`+0.407`**, `0` echoes) | **`+0.89` Brier skill**, liar weight `-> 0.0` |
| **B5: 8,192-Channel Sparse Firehose (2% budget, 3× loud trap)** | Naive Top-Mean F1 **`0.100`** (`21.4` loud FA) | F1 **`0.752`** (`61.1%` recall, `98.1%` prec, **`0.0` loud FA**, `25.6 ms`) | **7.5× higher F1**, **0 wasted loud reads** |

### Optimization Ladder (Cycle 1 → Cycle 3)

| Step | Change | Target Kernel | Before | After | Gain |
|---|---|---|---:|---:|---:|
| 1 | Zero-dep CPython C-API `_native` (`METH_FASTCALL`, GIL-free `Py_buffer`, freelists) | `eprocess_scalar` (Py) | `6,629 ns` | **`112.0 ns`** | **59.2×** |
| 2 | Exact Gaussian Max-Entropy Branch-and-Bound (`I <= 0.5*p*(1-p)*SNR^2`) + AVX2 pre-filter | `sift_voi_2048ch` (Py) | `1,648 µs` | **`11.0 µs`** | **149.8×** |
| 3 | Single-pass inline AVX2 sorted shift + dual branchless bsearch + 4-wide AVX2 AgACI | `conformal_5rate_batch` | `896.5 ns` | **`316.8 ns`** | **2.83×** |
| 4 | Anchor-centered tick-0 guard + Fixed-Share BOA (`eps=0.01`) + resettable Page-CUSUM `deg_e` | `SafeFuser` post-shift MSE | `0.2983` | **`0.0284`** | **10.5× lower MSE** |

---

## Verification, Sanitizers & Adversarial Breaker Discipline

`eproc` is verified by an independent **42-test Adversarial Breaker** ([`tests/test_breaker.py`](tests/test_breaker.py), [`BREAKER_REPORT.md`](BREAKER_REPORT.md)) plus C11 sanitizers:

- **Zero heap allocations:** `nm -u libeproc.so` confirms `0` references to `malloc`/`calloc`/`realloc`/`free`.
- **0-ULP AVX2 vs portable bit-identity:** Verified identical 64-bit state digest between `-march=native` (AVX2+FMA) and portable scalar builds.
- **ASan + UBSan + `float-cast-overflow`:** `0` errors across extreme finite `±1e308`, subnormal `1e-310`, `NaN`, `±Inf`.
- **TSan + Concurrency Hammer:** `0` data races and `0` torn reads across `2,000,000` concurrent C11 Boehm-2012 seqlock reads and 8-thread Python `SafeFuser.snapshot()` stress.
- **Strict Causality (`assert_causal`):** `0.0` bitwise diff across all 10 engines under future `+1e15`/`NaN`/`Inf` perturbations.
- **11,500+ Adversarial Nulls:** Empirical Type-I error strictly `<= 0.05` across Bernoulli, boundary `0.01/0.99`, skewed Beta, and non-stationary adaptive Markov martingale nulls.

```bash
# Reproduce every check locally
make test && make test-asan && make test-tsan
PYTHONPATH=python python3 -m unittest discover -s tests -v
PYTHONPATH=python python3 benchmarks/bench_suite.py
python3 scripts/verify_release_hygiene.py --strict
```

---

## Documentation & Agent Discovery

- **[Interactive Browser Lab & Benchmarks](https://pgg898200-ship-it.github.io/eproc/)** ([`docs/index.html`](docs/index.html)) — Zero-CDN live peeking simulator, verified `[LIVE-CPU]` speedup explorer, and Byzantine/echo swarm lab.
- [`docs/MATHEMATICS.md`](docs/MATHEMATICS.md) — Exact theorems, martingale proofs, and literature citations.
- [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md) & [`benchmarks/receipts.json`](benchmarks/receipts.json) — Reproducible live benchmark report and raw JSON receipts.
- [`BREAKER_REPORT.md`](BREAKER_REPORT.md) — Adversarial red-team audit and full Cycle-3 resolution log.
- [`llms.txt`](llms.txt) & [`llms-full.txt`](llms-full.txt) — Machine-readable Markdown reference for AI coding agents and research assistants.
- [`skills/eproc/SKILL.md`](skills/eproc/SKILL.md) — Drop-in Agent Skill specification.
- [`SHA256SUMS`](SHA256SUMS) — Deterministic supply-chain digests for canonical headers, receipts, and license files.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
