# `eproc` vs. Statistical & Conformal Incumbents `[LIVE-CPU]`

> Every benchmark number in this document was measured live on CPU by [`benchmarks/compare_incumbents.py`](../benchmarks/compare_incumbents.py). No numbers are extrapolated or copied from external publications.

## Reproduction & Execution Environment

```bash
PYTHONPATH=python python3 benchmarks/compare_incumbents.py
```

- **Timestamp (UTC):** `2026-11-22T19:06:42Z` (total wall time: `16.37 s`)
- **CPU:** `Intel(R) Xeon(R) CPU @ 2.30GHz`
- **Platform / Python:** `Linux-6.18.57-x86_64-with-glibc2.39` / Python `3.12.16`
- **Runtime:** `eproc v0.1.0`, ABI `v1`, SIMD flags `3` (`AVX2`, `FMA`)
- **Methodology:** 5 paired alternating (AB/BA) runs with GC disabled during timing windows; deterministic fixed seeds for all Monte Carlo and streaming comparisons.

---

## 1. Continuous Peeking & Sequential Evaluation Gates

**Workload:** `1,000` null Bernoulli(`0.50`) streams over `T = 1,000` ticks (`alpha = 0.05`) comparing `eproc.EProcess` (ONS betting supermartingale) against a Student-t test peeked every tick and a strict fixed-horizon Student-t test checked once at `N = 857` (the classical sample size for 1-sided `alpha = 0.05`, 90% power at `MDE = 0.05`). Stopping power and sample savings are measured over `400` trials per effect size; latency is measured over `50,000` steps.

### Null Safety (`N = 1,000` trials, `T = 1,000`, nominal `alpha = 0.05`)

| Method | Peeking Allowed? | Empirical FPR | 95% Wilson CI | Verdict |
|---|---|---:|---|---|
| Student-t peeked every tick | Every tick | **40.50%** | `[37.50%, 43.57%]` | Broken (`8.1×` nominal) |
| Fixed-horizon Student-t (`N = 857`) | Never (single look) | **4.70%** | `[3.55%, 6.19%]` | Valid only without peeking |
| **`eproc.EProcess` (ONS betting)** | **Every tick** | **4.10%** | `[3.04%, 5.51%]` | **Anytime-valid (`<= 5.0%`)** |

### Early Stopping vs. Fixed-Horizon `N = 857` (`400` trials per uplift)

| True Uplift (`p_true`) | Fixed-Horizon `N` (Power) | `eproc` Power | `eproc` Median Stop `N` | `eproc` Mean Stop `N` | Median Speedup | Mean Samples Saved |
|---|---|---:|---:|---:|---:|---:|
| `+0.08` (`p = 0.58`) | `857` (`100.0%`) | `92.25%` | **298.5** | `353.2` | **2.87×** | **58.78%** |
| `+0.12` (`p = 0.62`) | `857` (`100.0%`) | `100.0%` | **123.0** | `153.2` | **6.97×** | **82.12%** |
| `+0.15` (`p = 0.65`) | `857` (`100.0%`) | `100.0%` | **76.0** | `94.2` | **11.28×** | **89.01%** |

> **Trade-off note:** At modest effects near the fixed-horizon boundary (`+0.08`), committing strictly to `N = 857` without peeking achieves higher power (`100.0%` vs. `92.25%`) at the cost of always spending all `857` evaluations. `eproc.EProcess` trades a modest power margin on weak effects for continuous peeking safety and `58.8%–89.0%` sample savings.

### Per-Step Latency & State Footprint (`50,000` steps)

| Implementation | Median (`ns/op`) | p95 (`ns/op`) | Speedup vs. Reference | State Size |
|---|---:|---:|---:|---:|
| **`eproc.EProcess` (zero-copy batch)** | **28.4** | `32.6` | **16.94×** vs. Py ONS batch | **176 B** (0 heap allocs) |
| **`eproc.EProcess` (scalar Python call)** | **115.9** | `136.2` | **26.89×** vs. Student-t peek | **176 B** (0 heap allocs) |
| Pure-Python ONS betting (`confseq`-style batch) | `481.4` | `493.8` | `1.00×` (batch baseline) | Python object |
| Pure-Python Student-t peeked every tick | `3,115.5` | `3,436.8` | `1.00×` (scalar baseline) | Python object |

---

## 2. Online Multi-Rate Conformal Calibration

**Workload:** `T = 2,000` streaming nonconformity scores with a sudden `4×` standard-deviation regime shift at `t = 1,000` (`|N(0, 1)| -> |N(0, 4)|`), calibrated simultaneously across 5 coverage targets (`50%, 80%, 90%, 95%, 99%`). Compares `eproc.MultiRateConformal` (C11/AVX2 AgACI + Conformal PID + PAVA isotonic projection) against `PyMultiRateAgACI` (the same algorithm implemented in pure Python, representing archived `salesforce/online_conformal` / `MAPIE`-style online loops) and 5 independent single-rate ACI loops without isotonic projection.

### Coverage Accuracy, Post-Shock Recovery & Quantile Crossings (`T = 2,000`)

| Method | Empirical Coverage (`50/80/90/95/99%`) | Mean Abs Cov Err | Post-Shock MAE (`t >= 1200`) | Quantile Crossing Ticks |
|---|---|---:|---:|---:|
| **`eproc.MultiRateConformal`** | `50.06 / 79.94 / 90.06 / 95.00 / 98.89%` | **0.06%** | **0.10%** | **0 / 2,000** |
| `PyMultiRateAgACI` (pure-Python AgACI+PID ref) | `50.11 / 80.06 / 90.00 / 94.89 / 99.00%` | `0.06%` | `0.18%` | `0 / 2,000` |
| Independent 5-rate ACI (no PAVA projection) | `50.11 / 79.83 / 89.56 / 94.00 / 97.67%` | `0.61%` | `0.50%` | **415 / 2,000** (`20.8%`) |

### Per-Step Latency & State Footprint

| Implementation | Median (`ns/op`) | p95 (`ns/op`) | Speedup vs. Pure-Python AgACI+PID | State Size |
|---|---:|---:|---:|---:|
| **`eproc.MultiRateConformal` (batch)** | **377.5** | `381.9` | **231.64×** (`36.94×` vs. indep ACI) | **4,888 B** (0 heap allocs) |
| **`eproc.MultiRateConformal` (scalar)** | **521.9** | `889.8` | **167.55×** | **4,888 B** (0 heap allocs) |
| Pure-Python Independent 5-rate ACI (no PAVA) | `13,944.9` | `21,918.1` | `6.27×` (no BOA/PID/PAVA) | Python lists |
| Pure-Python AgACI + Conformal-PID (`online_conformal` ref) | `87,437.4` | `94,994.3` | `1.00×` (baseline) | Python lists |

---

## 3. Distributed Fleet Conformal Calibration

**Workload:** `256` distributed agents × `500` nonconformity scores per agent (`128,000` total scores), measuring wire footprint, full tree-reduction merge latency, exact Leave-One-Out (LOO) peer calibration (`global - local` + `quantile(0.90)`), integer bin drift, and quantile accuracy against raw float64 array exchange.

### Wire Size, Merge Latency & Leave-One-Out Speed

| Metric | Raw Float64 Array Exchange | `eproc.ConformalMonoid` | Advantage |
|---|---:|---:|---|
| Wire size per agent | `4,000 B` | **640 B** | **6.25× smaller** (`O(1)` fixed) |
| Global fleet summary payload | `1,024,000 B` (`1.024 MB`) | **640 B** | **1,600.0× smaller** |
| 256-agent tree-reduction merge | `2.8959 ms` | **0.0991 ms** (`99.1 µs`) | **29.22× faster** |
| Exact LOO peer calibration (`subtract` + `q90` / agent) | `95,229.6 µs` (`95.23 ms`) | **0.287 µs** (`287 ns`) | **332,244.6× faster** |
| LOO integer bin drift (`16` agents audited) | Exact (`O(N)`) | **0** | Bit-exact abelian group |

### Quantile Fidelity vs. Exact Sorted 128,000-Score Float64 Array

| Quantile | Exact Sorted Float64 | `ConformalMonoid` (640 B) | Absolute Error | Relative Error |
|---|---:|---:|---:|---:|
| `q50` | `0.67406` | `0.67460` | `0.00054` | `0.080%` |
| `q80` | `1.28478` | `1.28633` | `0.00155` | `0.121%` |
| `q90` | `1.64405` | `1.64558` | `0.00154` | `0.094%` |
| `q95` | `1.96097` | `1.97198` | `0.01101` | `0.561%` |
| `q99` | `2.56551` | `2.57659` | `0.01108` | `0.432%` |

---

## 4. Multi-Agent Prediction Fusion Under `1e6` Byzantine Attack + Regime Shift

**Workload:** `8` experts over `1,000` ticks (`anchor` has `sigma = 0.45`; experts `1–2` are sharp pre-shift `sigma = 0.10–0.12` then degrade to `0.75–0.80` at `t = 600`; experts `3–4` start at `0.55–0.58` and become sharp `0.10–0.12` at `t = 600`; expert `7` turns Byzantine at `t = 300`, emitting alternating `±1e6` spikes). Compares `eproc.SafeFuser` (Fixed-Share BOA + 3 Ville non-inferiority gates + quarantine + Boehm-2012 lock-free seqlock) against naive unweighted mean, coordinate-wise median, and anchor-only.

### Prediction Error (MSE & Max Single-Tick Error)

| Fuser | Pre-Attack MSE (`t < 300`) | Attack-Phase MSE (`300–599`) | Post-Shift MSE (`600–999`) | Combined MSE (`t >= 300`) | Max Abs Error |
|---|---:|---:|---:|---:|---:|
| Naive Mean | `0.02139` | `15,624,999,086.39` | `15,624,997,427.71` | `15,624,998,138.57` | `125,000.48` |
| Anchor Baseline (Agent 0) | `0.16940` | `0.17250` | `0.20769` | `0.19261` | `1.3432` |
| Coordinate-Wise Median | `0.01222` | `0.02721` | `0.03289` | `0.03045` | **0.8523** |
| **`eproc.SafeFuser` (BOA + Ville)** | **0.00886** | **0.00787** | **0.02135** | **0.01558** | `1.0789` |

`eproc.SafeFuser` quarantined the Byzantine peer (`quarantined=1`, `poison_clipped=1`, `fallback_trips=0`), cutting combined attack+shift MSE by **-91.9% vs. anchor** and **-48.8% vs. coordinate-wise median** (`3.46×` lower MSE during the attack phase, `1.54×` lower after the regime shift). Coordinate-wise median survives the `1e6` outlier (and has a slightly smaller worst single-tick error at the instant of regime shift), but cannot down-weight noisy peers or track which experts become sharpest.

### Per-Step Fusion Latency (`1,000` ticks, 8 experts)

| Implementation | Median (`ns/op`) | p95 (`ns/op`) | Notes |
|---|---:|---:|---|
| Naive Mean (pure Python) | `690.1` | `3,594.5` | Destroyed by `1e6` outlier |
| **`eproc.SafeFuser` (C11 + Seqlock)** | **749.5** | **1,209.7** | **35.49×** vs. `PySafeBOA`; faster than Python median |
| Coordinate-Wise Median (pure Python) | `951.3` | `2,283.7` | Static robust rule; no expert learning |
| `PySafeBOA` (pure-Python BOA + Ville) | `26,597.8` | `28,315.3` | Identical algorithm in pure Python |

---

## 5. Ecosystem Capability Matrix

Different libraries solve different parts of the statistical stack. The table below compares `eproc` fairly against widely used Python libraries and highlights where each tool excels.

| Capability | `eproc` | `confseq` | `MAPIE` | `online_conformal` *(archived)* | `scipy.stats` | `statsmodels` |
|---|---|---|---|---|---|---|
| **Anytime-valid e-processes & continuous peeking** | Yes (C11/AVX2 ONS betting, CS, e-detectors, racing) | Yes (broad catalog of CS/betting bounds) | No | No | No (fixed-horizon) | No (fixed-horizon) |
| **Online adaptive conformal (distribution shift)** | Yes (5-rate crossing-free AgACI + PID, `377 ns/step`) | No | Partial (ACI / EnbPI wrappers) | Yes (pure Python, archived 2025) | No | No |
| **Guaranteed non-crossing multi-rate intervals** | Yes (inline PAVA projection, 0 crossings) | No | No | No (single-rate) | No | No |
| **Distributed mergeable conformal + exact LOO** | Yes (640 B abelian monoid, `0` bin drift) | No | No | No | No | No |
| **Byzantine-safe multi-agent fusion & echo dedup** | Yes (`SafeFuser` seqlock + `EchoPool`) | No | No | No | No | No |
| **Offline batch GLMs, ARIMA, classical parametric tests** | **No** | No | No | No | **Yes (extensive)** | **Yes (extensive)** |
| **Batch `sklearn` CV+ / Jackknife+ / classification sets** | Regression wrapper (`ConformalWrapper`) | No | **Yes (extensive)** | No | No | No |
| **Runtime dependencies** | **0** (C11 + Python stdlib) | NumPy / SciPy | NumPy / SciPy / scikit-learn | NumPy / SciPy | NumPy / SciPy | NumPy / SciPy / pandas |
| **C11 / C++17 single-header & MCP server** | **Yes** (`eproc_single.h`, `eproc-mcp`) | C++ backend (Python-first) | Python only | Python only | Python/C/Fortran | Python only |

### When to Choose Each Library

- **Choose `scipy.stats` / `statsmodels`** for offline exploratory analysis, classical fixed-horizon hypothesis tests, parametric distributions, GLMs, mixed-effects models, econometrics, and ARIMA/state-space modeling.
- **Choose `MAPIE`** for offline batch conformal prediction tightly integrated with `scikit-learn` (CV+, Jackknife+-after-Bootstrap, Mondrian/classification prediction sets, conformalized quantile regression).
- **Choose `confseq`** when researching specialized academic confidence-sequence constructions (e.g., stitched boundaries, quantile CS, conjugate-mixture variants) in offline NumPy/SciPy notebooks.
- **Choose `eproc`** for **live online streams, LLM/agent eval gates, CI regression assertions, low-latency serving (`< 1 µs`), zero-dependency deployments, C/C++17 embedding, distributed fleet calibration (`640 B`), or adversarial multi-agent swarms**.
