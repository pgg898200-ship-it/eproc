# Changelog

All notable changes to `eproc` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-22

Initial public release of `eproc` — a sub-microsecond, zero-allocation C11/AVX2 and zero-dependency Python engine for anytime-valid online statistical inference, multi-rate conformal calibration, distributed fleet monoids, Byzantine-safe multi-agent fusion, and causal subsampling.

### Added

- **Zero-Allocation C11/SIMD Core (`csrc/`, `dist/eproc_single.h`):**
  - Anytime-valid ONS betting supermartingale (`EProcess`), O(1) Empirical-Bernstein & Normal-Mixture confidence sequences (`ConfidenceSequence`), resettable CUSUM e-detector (`EDetector`), multi-arm early-elimination racer (`AgentRace`), and Vovk–Wang e-value combination (`combine_evalues`).
  - Multi-rate crossing-free AgACI + Conformal PID online calibrator (`MultiRateConformal`, up to 5 simultaneous rates with PAVA isotonic projection), strictly causal out-of-order delayed feedback wrapper (`DelayedConformal`), right-censored budget upper predictive bounds (`BudgetConformal`), and fixed 640-byte abelian conformal monoid (`ConformalMonoid`) supporting associative/commutative merge and bit-exact leave-one-out subtraction.
  - Byzantine-safe multi-agent prediction fusion (`SafeFuser`: Scale-Free Fixed-Share BOA + 3-gate Ville non-inferiority fallback + strike quarantine + Boehm-2012 lock-free seqlock) and provenance-deduplicated belief pooling (`EchoPool` with BLAKE2b-64 IDs and online Beta-Bernoulli source reliability).
  - Causal Horvitz-Thompson subsampling (`VarOptSampler`, `OnlineRLSSampler`) and Gaussian max-entropy branch-and-bound multi-resolution anomaly selector (`SiftVOISelector`).
  - Single-header stb-style C11 & C++17 amalgamation (`dist/eproc_single.h`).
- **Zero-Dependency Python Package (`python/eproc/`):**
  - PEP 562 lazy top-level imports (`< 1 ms` cold import), native CPython C-API extension (`_native`) with automatic `ctypes` fallback (`EPROC_DISABLE_NATIVE=1`), zero-copy buffer protocol support, and full PEP 561 typing (`py.typed`, `__init__.pyi`).
  - Drop-in ML/LLM/CI adapters (`python/eproc/adapters.py`): `EvalGate`, `PairedEvalGate`, `ConformalWrapper`, `IntervalResult`, `assert_eprocess_uplift`, `assert_no_regression`, and `fleet_calibrate`.
- **CLI, MCP Server & Ecosystem Discovery:**
  - Zero-dependency streaming CLI (`eproc gate`, `eproc race`, `eproc conformal`, `eproc bench`, `eproc mcp`, `eproc version`).
  - Zero-dependency JSON-RPC 2.0 MCP stdio server (`eproc-mcp`, supporting protocols `2025-11-25` and `2026-07-28`) exposing 7 stateful/stateless tools (`eproc_etest`, `eproc_confidence_sequence`, `eproc_race`, `eproc_conformal`, `eproc_cmonoid_merge`, `eproc_safe_fuse`, `eproc_echopool`).
  - Agent & ecosystem metadata: `server.json`, `CITATION.cff`, `SECURITY.md`, `CONTRIBUTING.md`, `.claude-plugin/plugin.json`, `gemini-extension.json`, `skills/eproc/SKILL.md`, `llms.txt`, `llms-full.txt`, and `SHA256SUMS`.
- **Benchmarks & Documentation:**
  - Reproducible live benchmark suite (`benchmarks/bench_suite.py`, `benchmarks/BENCHMARKS.md`, `benchmarks/receipts.json`), head-to-head incumbent comparison (`benchmarks/compare_incumbents.py`, `docs/COMPARISON.md`), mathematical proofs (`docs/MATHEMATICS.md`), and interactive browser lab (`docs/index.html`).

### Security & Verification

- Verified `0` heap allocator symbols (`malloc`/`calloc`/`realloc`/`free`) in `libeproc.so` and `libeproc_portable.so`.
- Verified 0-ULP bit-identity between `-march=native` (AVX2+FMA) and portable scalar C11 builds.
- Passed full 42-test Adversarial Breaker audit (`tests/test_breaker.py`, `BREAKER_REPORT.md`) covering 11,500+ null trajectories, strict causality across all 10 engines, extreme finite `±1e308` / subnormal `1e-310` / `NaN` inputs, ASan + UBSan + `float-cast-overflow`, TSan, and 2M concurrent lock-free seqlock reads.
