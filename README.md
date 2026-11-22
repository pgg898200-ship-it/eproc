# eproc

[![CI](https://github.com/pgg898200-ship-it/eproc/actions/workflows/ci.yml/badge.svg)](https://github.com/pgg898200-ship-it/eproc/actions/workflows/ci.yml)
[![Live WASM Lab](https://img.shields.io/badge/Live_WASM_Lab-72_KiB_C11-2563eb)](https://pgg898200-ship-it.github.io/eproc/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/Runtime_Deps-0-brightgreen)](#install-zero-runtime-dependencies)

**Check a model evaluation after each result.** Feed pass/fail scores from your evaluator into `EvalGate`. When the evidence crosses your threshold, your runner can stop requesting more evaluations. The same C11 library also provides online calibration, agent racing and evidence fusion.

[Try the browser lab](https://pgg898200-ship-it.github.io/eproc/) · [Explore the interfaces](#install-zero-runtime-dependencies) · [Compare benchmarks](docs/COMPARISON.md)

## Stop a pass/fail evaluation

With Python 3.10+, Git and a C11 compiler, install the tagged source release.

```sh
python -m pip install "git+https://github.com/pgg898200-ship-it/eproc.git@v0.1.0"
```

Start with 60 synthetic results. A pass scores 1 and a failure scores 0. The one-sided gate tests a pass probability of at most 50% at a 5% significance level.

```python
from eproc import EvalGate

scores = [1, 1, 0, 1, 1] * 12
gate = EvalGate(mu0=0.50, alpha=0.05)
for score in scores:
    gate.observe(score)
    if gate.should_stop:
        break

result = gate.summary()
print(f"{result.verdict} after {result.n} results (e-value {result.e_value:.2f})")
```

```text
REJECT after 27 results (e-value 28.87)
```

On this sequence, the e-value first crosses 20 (1 / 0.05) on result 27. `REJECT` refers to the 50% null hypothesis. An evaluation that reaches its budget before crossing reports `CONTINUE`.

## Benchmark snapshots

```text
Continuous peeking false-positive rate (2,000 nulls, T=1,000, nominal alpha=0.05)
  Student-t peeked every tick |███████████████████████████████████████░| 39.2%  (broken)
  eproc.EProcess (ONS bet)    |████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|  3.9%  (exact <= 5%)

Eval calls to certify +0.15 uplift vs fixed-horizon N=857 (100% power)
  Fixed-horizon plan          |████████████████████████████████████████|  857 calls
  eproc.EProcess early stop   |████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|   88 calls (9.74x faster, 88.7% saved)

8-agent swarm MSE under 1e6 Byzantine attack + regime shift (docs/COMPARISON.md)
  Naive 8-agent mean          |████████████████████████████████████████| 15.62B  (destroyed)
  Anchor baseline             |█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 0.1926
  Coordinate-wise median      |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 0.0305
  eproc.SafeFuser (BOA+Ville) |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 0.0156  (-91.9% vs anchor, -48.8% vs median)
```

---

## Why `eproc`

Live multi-agent systems, online LLM evaluators, model routers, and streaming ML pipelines break classical fixed-horizon statistics in four ways:

1. **Continuous peeking breaks p-values:** Checking a t-test/z-test on live LLM evals or rollouts inflates a 5% test to **39.2% false positives** (`N=2,000` live Monte Carlo). Waiting for fixed horizons wastes **56%–89% of tokens**.
2. **Distribution shift & delayed labels break static conformal intervals:** Existing Python/R libraries (`online_conformal` archived 2025, `MAPIE`, `confseq`, `conformalForecast`) run pure-Python loops (`14–87 µs`/step), suffer **20.8% quantile-crossing ticks** across independent rates, and cannot merge calibration state across distributed fleets.
3. **Multi-agent swarms collapse under echoes and outliers:** Naive voting double-counts reprinted evidence, fails to down-weight sycophantic/unreliable peers, and explodes under a single poisoned agent.
4. **Firehose monitoring drowns in loud distractors:** Ranking raw channel means wastes 96% of attention on high-variance channels while missing quiet true anomalies.

`eproc` solves all four in **one auditable, zero-allocation C11 engine** (`0` calls to `malloc`/`free` in `libeproc.so`, **runtime CPUID AVX2 dispatch** inside portable wheels, **0.88 ms** Python cold import, **3-way 0-ULP bit-identity** across native AVX2, portable runtime-AVX2, and pure scalar/WASM).

See [`docs/COMPARISON.md`](docs/COMPARISON.md) for a fair head-to-head benchmark & capability matrix against `confseq`, `MAPIE`, `online_conformal`, `scipy.stats`, and `statsmodels`.

---

## Install (Zero Runtime Dependencies)

### Python (3.10–3.14, portable wheels with automatic runtime AVX2)

```bash
# Build from the tagged source release:
python -m pip install "git+https://github.com/pgg898200-ship-it/eproc.git@v0.1.0"
# Or grab a prebuilt wheel directly from Releases:
# https://github.com/pgg898200-ship-it/eproc/releases/tag/v0.1.0
# (And `pip install eproc` once PyPI OIDC Trusted Publishing is enabled)

# Or build from source:
git clone https://github.com/pgg898200-ship-it/eproc.git
cd eproc
pip install .                           # portable wheel (auto-dispatches AVX2 on x86_64 via CPUID)
EPROC_NATIVE_ARCH=1 pip install .       # host-tuned -march=native build
```

### Single-Header C11 / C++17 (1 file) or CMake `FetchContent`

Copy [`dist/eproc_single.h`](dist/eproc_single.h) into your project:

```c
#define EPROC_IMPLEMENTATION
#include "eproc_single.h"
// C11:   cc  -std=c11   -O3 -ffp-contract=off main.c   -lm   (runtime AVX2 auto-dispatched)
// C++17: c++ -std=c++17 -O3 -ffp-contract=off main.cpp -lm
```

Or link via CMake (`CMakeLists.txt` exposes `eproc::eproc` and `eproc::single_header`).

### WebAssembly + JS / TypeScript (`72 KiB`, zero Emscripten / WASI deps)

```js
import { initEproc, EProcess, ConformalMonoid } from "./js/eproc.mjs";

await initEproc("./dist/eproc.wasm"); // Works in Node.js, browsers, Cloudflare Workers, Bun, Deno
const ep = new EProcess({ mu0: 0.50, alpha: 0.05, lower: 0.0, upper: 1.0, side: "upper" });
const res = ep.updateBatch([1, 1, 0, 1, 1, 1]);
ep.free();
```

### Rust (zero Cargo deps) & Go (zero Go module deps)

- **Rust ([`rust/eproc/`](rust/eproc/)):** `cargo test --manifest-path rust/eproc/Cargo.toml --release` — compiles `csrc/` via `build.rs` with zero external crates and runtime AVX2.
- **Go ([`go/eproc/`](go/eproc/)):** `cd go/eproc && go test -race -v ./...` — self-contained CGo unity build with zero external modules and runtime AVX2.

### Coding Agents (MCP 2026-07-28 Server + Agent Skill)

```bash
eproc mcp   # Zero-dependency stdio MCP server (2025-11-25 & 2026-07-28, 7 stateful tools)
```

Drop-in manifests included for:
- **MCP Registry:** [`server.json`](server.json)
- **Claude Code:** [`.claude-plugin/plugin.json`](.claude-plugin/plugin.json) + [`skills/eproc/SKILL.md`](skills/eproc/SKILL.md)
- **Gemini CLI:** [`gemini-extension.json`](gemini-extension.json) + [`skills/eproc/SKILL.md`](skills/eproc/SKILL.md)
- **Codex / Cursor / Generic MCP Hosts:** `eproc mcp` (`eproc_etest`, `eproc_confidence_sequence`, `eproc_race`, `eproc_conformal`, `eproc_cmonoid_merge`, `eproc_safe_fuse`, `eproc_echopool`).

---

## 30-Second Quickstarts

### 1. Paired evaluations, CI assertions and agent racing

For a single score stream, run the [pass/fail evaluation](#stop-a-passfail-evaluation). To compare two models on the same prompts, pass their scores in pairs. `span=1.0` bounds the absolute difference between scores in [0, 1].

```python
from eproc import PairedEvalGate, assert_eprocess_uplift

paired = PairedEvalGate(alpha=0.05, span=1.0)
paired.observe_pair(candidate_score=0.92, baseline_score=0.71)
print(paired.summary().verdict)  # CONTINUE

# In CI, raise AssertionError if these scores never cross the threshold.
scores = [1, 1, 0, 1, 1] * 12
assert_eprocess_uplift(scores, mu0=0.50, alpha=0.05)
```

[Run the eight-agent tournament](examples/multi_agent_eval_demo.py) to see how `AgentRace` assigns the next evaluation and eliminates arms.

### 2. Crossing-Free Online Conformal, `sklearn` Wrapper & 640-Byte Fleet Monoid

```python
from eproc import MultiRateConformal, ConformalWrapper, ConformalMonoid, fleet_calibrate

# Calibrate 50/80/90/95/99% intervals simultaneously in 353 ns/step (0 quantile crossings)
cp = MultiRateConformal(alphas=(0.50, 0.20, 0.10, 0.05, 0.01), window_size=128, pid_ki=0.05)
pred = cp.predict(10.0, scale=1.0)       # pred.lows, pred.highs, pred.radii (strictly nested)
cp.update(abs(y_true - 10.0))

# Drop-in sklearn / MAPIE-style online conformal wrapper around any regressor
wrapped = ConformalWrapper(my_regressor).fit(X_cal, y_cal)
y_hat, lows, highs = wrapped.predict_interval(X_test)

# Distributed fleet calibration: 640 bytes/agent, bit-exact abelian merge + LOO peer borrow
cal = fleet_calibrate([scores_agent_a, scores_agent_b, scores_agent_c], quantiles=(0.50, 0.90, 0.95))
wire_640b = cal.global_monoid.to_bytes() # fixed 640 bytes regardless of N scores
```

### 3. Byzantine-Safe Multi-Agent Fusion & Echo-Deduplicated Belief Pooling

```python
from eproc import SafeFuser, EchoPool

# Scale-Free BOA + 3-Gate Ville Non-Inferiority + Slim Boehm-2012 Lock-Free Seqlock
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
rls = OnlineRLSSampler(d=16, oversample=2.0)      # 115 ns C11 Sherman-Morrison ridge leverage
sift = SiftVOISelector(n_sites=1024, per_site=8)  # division-free AVX2 Gaussian max-entropy B&B
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

All numbers measured live on `Intel(R) Xeon(R) CPU @ 2.30GHz` (2 cores, AVX2+FMA, Python 3.12.16, GCC 13.3.0 `-std=c11 -O3 -ffp-contract=off`). Full provenance, Wilson/bootstrap CIs, and raw paired trials in [`benchmarks/receipts.json`](benchmarks/receipts.json), [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md), and [`docs/COMPARISON.md`](docs/COMPARISON.md).

### Kernel Latency & Speedup vs Pure-Python Baselines (Median of 5 Paired Runs)

| Kernel | Pure Python (`ns/op`) | `eproc` Python API (`ns/op`) | `eproc` C11 Native (`ns/op`) | Throughput | Speedup vs Python |
|---|---:|---:|---:|---:|---:|
| **`ConformalMonoid.subtract` (exact LOO)** | 575,835 | **285.8** | — | 3.50 M/s | **2,015.0×** |
| **`MultiRateConformal` (5-rate AgACI+PID, batch)** | 83,691 | **353.5** | **350.1** | 2.86 M/s | **243.2×** |
| **`OnlineRLSSampler` (`d=16` Sherman-Morrison)** | 23,406 | **221.3** | **114.9** | 8.71 M/s | **203.8×** (C11) / **105.8×** (Py) |
| **`SiftVOISelector` (`2,048` ch exact B&B MI)** | 2,005,243 | **12,423** (`12.4 µs`) | **26,328** (sel+obs) | 80.5k/s | **161.4×** |
| **`EProcess` (ONS betting, batch / C11)** | 2,536 | **27.6** (scalar **118.5**) | **26.7** | **37.45 M/s** | **95.0×** (C11) / **21.4×** (Py) |
| **`EchoPool` (dedup + Beta-Bernoulli log-odds)** | 1,553 | **201.6** | **26.6** | **37.59 M/s** | **58.4×** (C11) / **7.7×** (Py) |
| **`SafeFuser` (8 experts + 3 Ville gates + Slim Seqlock)** | 25,832 | **569.9** | **445.7** | 2.24 M/s | **58.0×** (C11) / **45.3×** (Py) |
| **`ConformalMonoid.merge` (640 B AVX2)** | 1,609 | **219.5** | **34.2** | **29.26 M/s** | **47.1×** (C11) / **7.3×** (Py) |
| **`ConfidenceSequence` (Empirical-Bernstein)** | 2,615 | **65.0** (scalar **120.0**) | **62.8** | **15.94 M/s** | **41.7×** (C11) / **21.8×** (Py) |
| **`VarOptSampler` (`k=256` Cohen-2011 HT)** | 1,940 | **250.2** | **133.1** | 7.51 M/s | **14.6×** (C11) / **7.8×** (Py) |

*Portable wheel runtime CPUID AVX2 proof (`libeproc_portable.so`, compiled with zero `-mavx2` flags): `MultiRateConformal` **311.9 ns** vs `799.6 ns` forced scalar (**2.56× faster** out of the box, 0-ULP bit-identical).*

### End-to-End Multi-Agent & Distributed Benchmarks

| Benchmark | Incumbent / Naive Control | **`eproc`** | Verified Margin |
|---|---|---|---|
| **B1a: Continuous Peeking Null Safety (`N=2,000, T=1,000`)** | Student-t peeked every tick: **39.20% FPR** | **3.90% FPR** (Beta skewed: **4.00%**) | **10.1× fewer false positives** (strictly `<= 5%`) |
| **B1b: Early Stopping (`+0.15` uplift vs `N=857` fixed)** | `857` calls (0% saved) | **88 median calls** (`100%` power) | **9.74× faster** (**88.7% tokens saved**) |
| **B1c: `K=8` Agent Tournament (`100` runs)** | `6,400` calls fixed | **3,994 median calls**, **0.00% null FWER** | **1.60× faster** (**36.1% mean saved**) |
| **B3: 1,024-Agent Fleet Conformal (`512k` scores)** | `4.10 MB` raw, `11.85 ms` merge | **640 B**, **0.52 ms** Py / **62.2 µs** C11, **0 bin drift** | **6,400× smaller wire**, **190× faster C11** |
| **B4a: 8-Agent `1e6` Byzantine + Regime Shift** | Naive mean `10.42B`; Anchor `0.195` | Attack **`0.0220`**, post-shift **`0.0284`** | **-88.7% vs anchor** (`-85.0%` post-shift) |
| **B4b: 8-Agent 5× Echo + Byzantine Liar** | Naive vote Brier `0.3709` (skill `-0.484`, `1000` echoes) | Brier **`0.1483`** (skill **`+0.407`**, `0` echoes) | **`+0.89` Brier skill**, liar weight `-> 0.0` |
| **B5: 8,192-Channel Sparse Firehose (2% budget, 3× loud trap)** | Naive Top-Mean F1 **`0.100`** (`21.4` loud FA) | F1 **`0.752`** (`61.1%` recall, `98.1%` prec, **`0.0` loud FA**, `26.9 ms`) | **7.5× higher F1**, **0 wasted loud reads** |

---

## Verification, Sanitizers & Adversarial Breaker Discipline

`eproc` is verified across every language target in CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) and by an independent **42-test Adversarial Breaker** ([`tests/test_breaker.py`](tests/test_breaker.py), [`BREAKER_REPORT.md`](BREAKER_REPORT.md)):

- **Zero heap allocations:** `nm -u` confirms `0` references to `malloc`/`calloc`/`realloc`/`free` in `libeproc.so`, `libeproc_portable.so`, and `libeproc_scalar.so`.
- **3-way 0-ULP bit-identity:** Native `-march=native` AVX2, portable runtime-AVX2, and `-DEPROC_FORCE_SCALAR` pure scalar produce the exact same 64-bit state digest (`0xc1db7c3924a4bf25`), and `dist/eproc.wasm` produces bit-identical 640-byte `ConformalMonoid` wire payloads.
- **ASan + UBSan + `float-cast-overflow` + TSan + Go `-race`:** `0` errors across `±1e308`, subnormal `1e-310`, `NaN`, `±Inf`, and `2,000,000` concurrent Boehm-2012 seqlock reads (`0` torn reads).
- **Strict Causality (`assert_causal`):** `0.0` bitwise diff across all 10 engines under future `+1e15`/`NaN`/`Inf` perturbations.
- **11,500+ Adversarial Nulls:** Empirical Type-I error strictly `<= 0.05` across Bernoulli, boundary `0.01/0.99`, skewed Beta, and non-stationary adaptive Markov martingale nulls.

```bash
# Reproduce every check locally
make test && make test-asan && make test-tsan
PYTHONPATH=python python3 -m unittest discover -s tests -v
node --test js/test_wasm.mjs
RUSTFLAGS="-D warnings" cargo test --manifest-path rust/eproc/Cargo.toml --release
(cd go/eproc && go vet ./... && go test -race -v ./...)
python3 scripts/verify_release_hygiene.py --strict && sha256sum -c SHA256SUMS
```

---

## Documentation & Agent Discovery

- **[Live Interactive WASM Lab & Benchmarks](https://pgg898200-ship-it.github.io/eproc/)** ([`docs/index.html`](docs/index.html)) — Run real 72 KiB compiled C11 WebAssembly on your own data in the browser.
- [`docs/COMPARISON.md`](docs/COMPARISON.md) — Reproducible head-to-head comparison against `confseq`, `MAPIE`, `online_conformal`, `scipy.stats`, and `statsmodels`.
- [`docs/MATHEMATICS.md`](docs/MATHEMATICS.md) — Exact theorems, martingale proofs, and literature citations.
- [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md) & [`benchmarks/receipts.json`](benchmarks/receipts.json) — Reproducible live benchmark report and raw JSON receipts.
- [`BREAKER_REPORT.md`](BREAKER_REPORT.md) — Independent 42-test adversarial red-team audit and resolution log.
- [`llms.txt`](llms.txt) & [`llms-full.txt`](llms-full.txt) — Machine-readable Markdown reference for AI coding agents.
- [`skills/eproc/SKILL.md`](skills/eproc/SKILL.md) & [`server.json`](server.json) — Drop-in Agent Skill & MCP Registry manifest.
- [`SHA256SUMS`](SHA256SUMS), [`SECURITY.md`](SECURITY.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`CITATION.cff`](CITATION.cff), [`CHANGELOG.md`](CHANGELOG.md).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
