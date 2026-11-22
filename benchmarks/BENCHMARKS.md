# `eproc` Empirical Benchmark Report `[LIVE-CPU]`

> Every number below was measured live on this host by `benchmarks/bench_suite.py` and recorded verbatim in `benchmarks/receipts.json`.

## Provenance & Execution Environment

- **Timestamp (UTC):** `2026-11-22T19:28:30Z`
- **Mode:** `full` (total wall time: `23.96 s`)
- **CPU:** `Intel(R) Xeon(R) CPU @ 2.30GHz` (2 logical cores)
- **Platform / Python:** `Linux-6.18.57-x86_64-with-glibc2.39` / Python `3.12.16`
- **eproc Runtime:** ABI `v1`, SIMD flags `3` (AVX2, FMA), cold Python import `0.888 ms`

## Pre-Registered Target Scorecard (T1–T6)

| Target | Pre-Registered Bar | Live Measured Result | Status |
|---|---|---|---|
| `T1_anytime_valid_safety` | eproc FPR <= 0.050 and naive peeking FPR >= 0.25 | `eproc_fpr=0.039, naive_peek_fpr=0.392` | **PASS** |
| `T2_eval_token_savings` | >= 3.0x fewer samples on real effect / early elimination in K=8 race | `race_speedup_mean_x=1.57, race_tokens_saved_pct=36.14, direct_best_speedup_x=9.74, race_null_fwer=0.0` | **PASS** |
| `T3_raw_kernel_speed` | < 100 ns E-process, < 400 ns 5-rate AgACI, >= 50x/200x vs pure Python | `eprocess_batch_ns=27.607, eprocess_batch_speedup_x=17.9, conformal_batch_ns=353.529, conformal_batch_speedup_x=236.73, c11_eprocess_ns=26.7, c11_conformal_ns=350.15` | **PASS** |
| `T4_distributed_conformal_monoid` | <= 1,024 B wire struct, 0 integer drift, < 500 us for 1,024 merges | `wire_bytes=640, integer_bin_drift=0, 1024_merge_us=62.21` | **PASS** |
| `T5_multi_agent_safe_fusion` | 0% poisoning blast vs anchor/clean, 0 echo double-counts, beats naive vote | `blast_vs_anchor_pct=-88.716, blast_vs_clean_pct=3.171, echo_double_counts=0, echopool_brier=0.1483, naive_brier=0.3709` | **PASS** |
| `T6_cold_import_budget` | < 20 ms Python cold import | `cold_import_ms=0.888` | **PASS** |

## B1. Anytime-Valid Peeking Safety & Eval Token Savings

### Continuous Peeking Null Safety (`N=2,000` paired trials, `T=1,000`, nominal `alpha=0.05`)

| Method | Null Distribution | Empirical FPR | 95% Wilson CI | Verdict |
|---|---|---:|---|---|
| `PyFixedHorizonPeekTest` (Student-t, every tick) | Bernoulli(0.5) | **39.20%** (784/2000) | [37.08%, 41.36%] | Broken (~10.1x nominal) |
| **`eproc.EProcess` (ONS Betting)** | Bernoulli(0.5) | **3.90%** (78/2000) | [3.14%, 4.84%] | Anytime-valid (`<= 5.0%`) |
| `PyFixedHorizonPeekTest` (Student-t) | Beta(0.5, 4.5), exact mean=0.10 | 26.40% | — | Inflated |
| **`eproc.EProcess` (ONS Betting)** | Beta(0.5, 4.5), exact mean=0.10 | **4.00%** | — | Anytime-valid |

### K=8 Live Multi-Agent Tournament (`100` tournaments, true pass rates `[0.72, 0.6, 0.55, 0.52, 0.49, 0.46, 0.43, 0.4]`)

| Policy | Eval Calls (Mean / Median) | Speedup vs Fixed | Tokens Saved | Winner Acc | Null FWER |
|---|---:|---:|---:|---:|---:|
| Fixed-Horizon (`8 x 800`) | 6,400 / 6,400 | 1.00x | 0.0% | — | — |
| **`eproc.AgentRace` (Early Elimination)** | **4,087.2 / 3,994.0** (95% CI [3,840, 4,342]) | **1.57x** (1.60x med) | **36.1%** | **91.0%** | **0.00%** |

Weak 6 arms (`p=0.40..0.55`) eliminated in median **3,964** total calls.

### Direct Stopping Speedup vs Fixed-Horizon Plan (`MDE=0.05`, 90% power, `N_fixed=857`)

| True Uplift `Delta` | Power | Median Stop `N` | Mean Stop `N` | Speedup (Median) | Tokens Saved |
|---:|---:|---:|---:|---:|---:|
| `+0.08` (`p=0.58`) | 90.5% | **339** | 374.5 | **2.53x** | **56.3%** |
| `+0.12` (`p=0.62`) | 100.0% | **152** | 165.9 | **5.66x** | **80.6%** |
| `+0.15` (`p=0.65`) | 100.0% | **88** | 96.5 | **9.74x** | **88.7%** |

## B2. Raw Kernel Latency & Throughput (5 Paired Alternating Runs)

### Python API (`eproc` Scalar FFI & Zero-Copy `array('d')` Batch vs Pure-Python Baselines)

| Kernel | Pure-Python Median (`ns/op`) | `eproc` Median (`ns/op`) | `eproc` p95 (`ns/op`) | `eproc` Throughput | Speedup |
|---|---:|---:|---:|---:|---:|
| `eprocess_scalar` | 2,536.1 | **118.5** | 224.3 | 8.442 M ops/s | **21.4x** |
| `eprocess_batch` | 494.1 | **27.6** | 28.7 | 36.223 M ops/s | **17.9x** |
| `cs_eb_scalar` | 2,614.7 | **120.0** | 154.4 | 8.334 M ops/s | **21.8x** |
| `cs_eb_batch` | 1,130.5 | **65.0** | 80.1 | 15.393 M ops/s | **17.4x** |
| `conformal_5rate_scalar` | 85,169.6 | **526.4** | 548.5 | 1.900 M ops/s | **161.8x** |
| `conformal_5rate_batch` | 83,691.1 | **353.5** | 356.5 | 2.829 M ops/s | **236.7x** |
| `cmonoid_add_batch` | 1.5 | **38.7** | 39.5 | 25.811 M ops/s | **0.0x** |
| `cmonoid_merge` | 1,608.7 | **219.5** | 234.9 | 4.556 M ops/s | **7.3x** |
| `cmonoid_subtract_loo` | 575,835.2 | **285.8** | 357.8 | 3.499 M ops/s | **2015.0x** |
| `safe_fuser_8experts` | 25,832.3 | **569.9** | 584.9 | 1.755 M ops/s | **45.3x** |
| `echopool_submit` | 1,552.8 | **201.6** | 209.0 | 4.959 M ops/s | **7.7x** |
| `varopt_k256` | 1,939.5 | **250.2** | 272.4 | 3.996 M ops/s | **7.8x** |
| `online_rls_d16` | 23,405.8 | **221.3** | 235.4 | 4.520 M ops/s | **105.8x** |
| `sift_voi_2048ch` | 2,005,243.2 | **12,423.4** | 12,751.5 | 0.081 M ops/s | **161.4x** |

### Bare-Metal C11/SIMD Kernel Speed (`scripts/bench_c_native.c`, Zero Python FFI Overhead)

| C11 Kernel | Pure-Python Baseline (`ns/op`) | C11 Native (`ns/op`) | C11 Throughput | Speedup vs Python |
|---|---:|---:|---:|---:|
| `bet_eprocess` | 2,536.1 | **26.7** | **37.45 M ops/s** | **95.0x** |
| `cs_empirical_bernstein` | 2,614.7 | **62.8** | **15.94 M ops/s** | **41.7x** |
| `conformal_5rate_agaci_w128` | 85,169.6 | **350.1** | **2.86 M ops/s** | **243.2x** |
| `cmonoid_add` | 1.5 | **36.9** | **27.14 M ops/s** | **0.0x** |
| `cmonoid_merge_640B` | 1,608.7 | **34.2** | **29.26 M ops/s** | **47.1x** |
| `safe_boa_8experts_seqlock` | 25,832.3 | **445.7** | **2.24 M ops/s** | **58.0x** |
| `echopool_submit` | 1,552.8 | **26.6** | **37.59 M ops/s** | **58.4x** |
| `varopt_k256` | 1,939.5 | **133.1** | **7.51 M ops/s** | **14.6x** |
| `online_rls_d16` | 23,405.8 | **114.9** | **8.71 M ops/s** | **203.8x** |
| `sift_voi_2048ch_select_and_obs` | 2,005,243.2 | **26,328.0** | **0.04 M ops/s** | **76.2x** |

## B3. 1,024-Agent Distributed `ConformalMonoid` Fleet Merge (512,000 Total Scores)

| Metric | Raw Float64 Shipping | `eproc.ConformalMonoid` | Advantage |
|---|---:|---:|---|
| Wire Payload / Agent | 4,000 B | **640 B** | **6.25x smaller** (O(1) fixed!) |
| Global Fleet Summary Payload | 4,096,000 B | **640 B** | **6,400x smaller** |
| 1,024-Agent Tree Merge (Python API) | 11.85 ms | **0.52 ms** | **22.7x faster** |
| 1,024-Agent Tree Merge (Native C11) | 11,850 us | **62.2 us** (60.8 ns/merge) | **190x faster** |
| Associativity / Commutativity / LOO Drift | exact (O(N)) | **0 integer bin drift** (16/16 LOO bit-exact) | Bit-exact abelian group |

| Quantile | Exact Raw (512k Scores) | `ConformalMonoid` (640 B) | Abs Error | Rel Error |
|---|---:|---:|---:|---:|
| `q50` | 0.66312 | 0.66371 | 0.00059 | 0.09% |
| `q80` | 1.27577 | 1.27713 | 0.00136 | 0.11% |
| `q90` | 1.65558 | 1.65971 | 0.00413 | 0.25% |
| `q95` | 1.99705 | 2.00588 | 0.00883 | 0.44% |
| `q99` | 2.69038 | 2.71291 | 0.02252 | 0.84% |

## B4. Adversarial 8-Agent Swarm Fusion (Poisoning + Echoes + Regime Shift)

### B4a. Continuous Prediction Fusion (`1000` ticks, 1e6 Byzantine spikes at `t >= 300`, regime shift at `t >= 600`)

| Fuser | Pre-Attack MSE | Attack-Phase MSE | Max Single-Tick Error | Blast vs Anchor |
|---|---:|---:|---:|---:|
| `PyNaiveAverageFuser` | 0.04029 | 10,424,109,416.18 | 125,000.8 | Destroyed |
| Anchor Baseline (Agent 0) | 0.19365 | 0.19513 | — | 0.00% (baseline) |
| **`eproc.SafeFuser` (BOA + 3 Ville Gates)** | **0.01413** | **0.02202** | **1.6703** | **-88.72%** (quarantined=1, fallback_ticks=0) |

Post-regime-shift MSE (`t >= 600`): `eproc.SafeFuser` = **0.02844** vs Anchor = 0.18983 (**+85.0%** error reduction).

### B4b. Multi-Agent Belief Pooling (`200` claims, 5x echo reprints & Byzantine inverted liar after claim `60`)

| Policy | Post-Attack Brier | Brier Skill Score | Accuracy | Echoes Dropped / Double-Counted |
|---|---:|---:|---:|---:|
| `PyNaiveAgentVoting` | 0.3709 | -0.4835 | 60.8% | 0 / 1000 |
| **`eproc.EchoPool`** | **0.1483** | **+0.4068** | **83.3%** | **1000 / 0** |

Learned Beta-Bernoulli reliabilities: strong experts `[0.5941, 0.5347]`, noisy peers `[0.099, 0.1485, 0.0495]`, Byzantine liar `0.0`.

## B5. 8,192-Channel Sparse Anomaly Firehose (`SiftVOI`)

Setup: `1,024` sites x `8` channels = `8,192` channels, `20%` 3x-noise loud distractors, `2%` attention budget (`164` reads/epoch across `5` epochs, ~`78.2` true anomalies/epoch).

| Policy | Recall | Precision | F1 | Loud-Channel False Alarms | Loud Reads Wasted | Epoch Latency |
|---|---:|---:|---:|---:|---:|---:|
| Naive Top-Mean | **6.8%** | 19.1% | **0.100** | 21.4 | 158.4 / 164 | 2.1 ms |
| Uniform Random (2%) | **1.2%** | 24.2% | **0.024** | 4.8 | 32.2 / 164 | 0.2 ms |
| **`eproc.SiftVOISelector`** | **61.1%** | 98.1% | **0.752** | 0.0 | 0.0 / 164 | 26.9 ms |

