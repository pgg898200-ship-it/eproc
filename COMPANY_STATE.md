# `eproc` Company & Programme State (`COMPANY_STATE.md`)

**Date:** `2026-11-22`
**Release:** `v0.1.0`
**Repository:** `https://github.com/pgg898200-ship-it/eproc`
**Live Browser WASM Lab & `llms.txt` Hub:** `https://pgg898200-ship-it.github.io/eproc/`
**License:** Apache-2.0

---

## 1. Thesis & Defensible Moat

Live multi-agent systems, LLM evaluation pipelines, model routers, and streaming ML monitors require **anytime-valid sequential inference**, **crossing-free adaptive conformal calibration**, **distributed fleet mergeability**, and **Byzantine/echo-safe multi-agent fusion** at sub-microsecond latency with zero runtime dependencies.

Existing Python/R libraries (`confseq`, `MAPIE`, archived `online_conformal`, `scipy.stats`, `statsmodels`) either lack anytime-valid guarantees, run pure-Python loops (`14–87 µs`/step), suffer quantile crossing across multiple coverage levels (`20.8%` crossing ticks), or require NumPy/SciPy/scikit-learn and cannot embed in C/C++, WebAssembly, Rust, or Go.

**Defensible Moat:**
1. **Single zero-allocation C11 engine (`0` `malloc`/`free`)** powering 7 zero-dependency targets: Python (`_native` C-API + `ctypes` fallback), Single-Header C11/C++17 + CMake, **72 KiB WebAssembly** (`0` Emscripten/WASI), Rust (`0` Cargo deps), Go (`0` Go module deps), CLI, and MCP 2026-07-28 + Agent Skills.
2. **Runtime CPUID AVX2 multiversioning inside portable wheels:** `pip install eproc` automatically runs AVX2 kernels on any Haswell+ x86_64 host (`2.56×` faster `MultiRateConformal` vs scalar, `311.9 ns` vs `799.6 ns`) with zero `-mavx2` flags and **3-way 0-ULP bit-identity** (`0xc1db7c3924a4bf25`).
3. **Abelian 640-byte `ConformalMonoid` wire format:** Bit-identical across C11, Python, WebAssembly, Rust, and Go (`0` integer bin drift, `6,400×` smaller global wire than 1,024-agent raw arrays, `62.2 µs` C11 tree merge, `2,015×–332,244×` faster exact LOO).
4. **Drop-in adoption & Ramp-style agent discoverability:** `EvalGate`, `PairedEvalGate`, `ConformalWrapper` (`sklearn`/`MAPIE`-style), `assert_eprocess_uplift`, `fleet_calibrate`, `llms.txt`, `llms-full.txt`, `docs/COMPARISON.md`, `server.json`, `CITATION.cff`, and a live zero-CDN C11 WebAssembly browser lab.

---

## 2. Iteration & Evidence Log

| Cycle | Focus | Key Verified Outcomes (`[LIVE-CPU]`) |
|---|---|---|
| **Cycle 1** | Core C11/AVX2 + Python/CLI/MCP v0.1 | 61 C11 ABI functions, 0 heap allocs, 0-ULP AVX2/scalar identity, 25/25 Python API tests PASS |
| **Cycle 2** | Independent Adversarial Breaker | 42 adversarial tests across 6 fronts surfaced 17 witnesses (`W01–W17`) |
| **Cycle 3** | Performance & Hardening Sprint | CPython `_native` (`59×`), AVX2 AgACI (`2.8×`), Gaussian max-entropy B&B `SiftVOI` (`150×`), Fixed-Share BOA + Page-CUSUM (`10.5×` lower post-shift MSE), **42/42 Breaker PASS**, **6/6 Gates PASS** |
| **Cycle 4** | Packaging & Release Hygiene | `setup.py` self-contained C11 wheel, OIDC PyPI/GitHub Release CI, `SHA256SUMS`, zero-leak strict hygiene |
| **Cycle 5** | Runtime AVX2, Slim Seqlock, Multi-Language & Live WASM Lab | Runtime CPUID AVX2 in portable wheels (`2.56×` speedup, 3-way 0-ULP `0xc1db7c3924a4bf25`); slim Boehm-2012 seqlock (`SafeFuser` C11 `566 → 445.7 ns`, `58×` vs Py); division-free AVX2 `SiftVOI`; real **72 KiB** `dist/eproc.wasm` (`13/13` `node --test` PASS, bit-exact 640 B wire parity); Rust (`10/10` PASS); Go (`9/9` `-race` PASS); CMake `ctest` PASS; Python drop-in adapters (`16/16` PASS native+ctypes); `docs/COMPARISON.md`; live GitHub Pages WASM lab + `/llms.txt` |

---

## 3. Current Verified Scorecard (`2026-11-22`, `[LIVE-CPU]`)

- **Pre-registered Benchmark Gates (`benchmarks/receipts.json`):** **6/6 PASS (`T1–T6`)**
- **C11 + Sanitizers:** 3-way 0-ULP bit-identity (`0xc1db7c3924a4bf25`), 0 heap allocs, ASan+UBSan+`float-cast-overflow` PASS, TSan 2M seqlock reads PASS (`0` torn reads)
- **Python:** 25/25 API + 16/16 Adapters (both `_native` and `EPROC_DISABLE_NATIVE=1`) + 42/42 Breaker PASS; `0.88 ms` cold import
- **Multi-Language:** WASM/Node `13/13` PASS (`73,795 B`), Rust `10/10` PASS (`-D warnings`), Go `9/9` PASS (`-race`), CMake static+shared `ctest` PASS
- **Hygiene & Supply Chain:** `scripts/verify_release_hygiene.py --strict` PASS (`0` issues), `sha256sum -c SHA256SUMS` PASS

---

## 4. Next Organic Traction Actions

1. Push Cycle 5 to `main` and verify both `verify-c-and-python` + `verify-multilang` CI jobs and the live WASM GitHub Pages deployment (`https://pgg898200-ship-it.github.io/eproc/`).
2. Tag and push `v0.1.0` to trigger `cibuildwheel@v4` (Linux `x86_64`/`aarch64`, macOS `arm64`/`x86_64`, Python 3.10–3.14), GitHub Release (`eproc_single.h`, `eproc.wasm`, `eproc.h`, wheels, `SHA256SUMS.txt`), and PyPI OIDC Trusted Publishing (`pip install eproc`).
3. Monitor GitHub issues/discussions, downloads, and ecosystem integrations; prioritize user-requested adapters and benchmarks with strict zero-hype reproducibility.
