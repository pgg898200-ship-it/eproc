# `eproc` — Autonomous Research & Engineering Log (`COMPANY_STATE.md`)

> **Mission:** Build the fastest, mathematically airtight, zero-dependency online & distributed statistical inference engine for live multi-agent systems, streaming ML, and sequential evaluation.
> **Verification Discipline:** Pre-registered quantitative gates (`T1–T6`), independent Adversarial Breaker auditing, zero-allocation C11 sanitizers (`ASan+UBSan+FCO`, `TSan`), 0-ULP SIMD-vs-scalar parity, and live reproducible receipts (`benchmarks/receipts.json`).

---

## 1. Niche Selection & Competitive Landscape (2026-11-22)

1. **Incumbent Vacuum:**
   - `salesforce/online_conformal`: archived (read-only since May 2025), pure Python/Pandas (`~100–800 µs/step`).
   - `gostevehoward/confseq`: C++14 + Boost headers, Python ≤3.10, early-stage; betting & without-replacement CS are pure Python (`~1 ms`).
   - `MAPIE`, `TorchCP`, `crepes`, `online-cp`, `River`: pure Python (`85 µs–10 ms/step`), no anytime-valid e-processes, no fixed-size abelian distributed conformal monoids, no Byzantine-safe multi-agent fusion.
   - `conformalForecast`, `AdaptiveConformal`, `avlm`, `evalinger`, `seqcomp`: fragmented R packages.
   - `kalibra`, `veridict`: offline batch JSONL regression scripts.
2. **Core Value Proposition:** Unify anytime-valid testing/racing, multi-rate online conformal + 640-byte abelian fleet monoids, Byzantine-safe multi-agent fusion (`SafeBOA` + `EchoPool`), and Horvitz-Thompson/VOI causal subsampling inside a single zero-allocation C11/AVX2 core with native Python C-API, single-header C/C++17, CLI, MCP 2026-07-28 server, and Agent Skills.

---

## 2. Final Pre-Registered Target Scorecard (`T1–T6` — 6/6 PASS)

| Target | Pre-Registered Bar | Final Verified Result (`[LIVE-CPU]`) | Status |
|---|---|---|---|
| **T1: Anytime-Valid Safety** | `eproc` FPR `<= 0.050` & naive peeking `>= 0.25` (`N=2,000, T=1,000`) | `eproc=3.90%` (`4.00%` Beta), naive=`39.20%` (`10.1×` reduction) | **PASS** |
| **T2: Eval Token Savings** | `>= 3.0×` stopping speedup / early elimination in `K=8` race | Direct stop `9.74×` (`88.7%` saved); `K=8` race `1.60×` med (`0.00%` wrong winner / `0.00%` null FWER) | **PASS** |
| **T3: Raw Kernel Speed** | `< 100 ns` E-process, `< 400 ns` 5-rate AgACI, `>= 200×` vs Python | E-process `27.5 ns` batch / `30.1 ns` C11; 5-rate AgACI `316.8 ns` (`273.0×`); LOO `267.8 ns` (`2,162×`); SiftVOI `11.0 µs` (`186.2×`) | **PASS** |
| **T4: Distributed Monoid** | `<= 1,024 B`, `0` integer drift, `< 500 µs` 1,024-agent merge | `640 B`, `0` drift, `74.9 µs` C11 / `0.51 ms` Py (`6,400×` smaller wire) | **PASS** |
| **T5: Multi-Agent Fusion** | `0%` poisoning blast, `0` echo double-counts, beats naive vote | Attack MSE `0.0220` (`-88.7%` vs anchor, naive `10.42B`); post-shift `0.0284` (`-85.0%`); `1,000` echoes dropped / `0` double-counted (`+0.407` vs `-0.484` Brier skill) | **PASS** |
| **T6: Causality & Hygiene** | `0.0` causality diff, `0` sanitizer/race issues, `< 20 ms` cold import | `42/42` Breaker PASS, `0` ASan/UBSan/FCO/TSan issues, `0` torn reads / 2M ops, `0.78 ms` cold import | **PASS** |

---

## 3. Cycle Execution History

- **Cycle 0:** Competitive landscape audit, mathematical design, frozen C11 ABI (`csrc/eproc.h`, `csrc/eproc_seqlock.h`), single-header bundler, CI workflow.
- **Cycle 1 (3 Parallel Builders):** Built zero-alloc C11/AVX2 core (`61/61` ABI symbols, `0` heap symbols), zero-dep Python/CLI/MCP 2026-07-28 package (`260/260` ABI offset checks), and fair head-to-head benchmark suite. Result: `5/6` targets PASS (`T3` failed at `896 ns` 5-rate AgACI and `~6 µs` ctypes scalar FFI overhead; `SiftVOI` took `1.6 ms`/select; `SafeFuser` lagged anchor post-regime-shift).
- **Cycle 2 (Independent Adversarial Breaker):** Executed 42 adversarial tests across 11,500+ nulls, strict causality, 1,000-shard monoid algebra, `±1e308`/`1e-310`/`NaN` torture, ASan+UBSan+FCO, TSan 2M seqlock stress, and MCP/CLI fuzzing. Confirmed core math & causality airtight; exposed 17 concrete edge-case witnesses (`W01–W17`).
- **Cycle 3 (Breaker Resolution + Optimization Ladder):**
  - Resolved **17/17** Breaker witnesses (`42/42` `tests/test_breaker.py` PASS, `25/25` `tests/test_python_api.py` PASS on both `_native` and `ctypes`, `dist/eproc_single.h` clean on GCC C11 + Clang C11 + G++ C++17).
  - Built zero-dependency CPython C-API `_native` (`59×` scalar speedup vs ctypes: `6,629 ns → 112 ns`).
  - Applied exact Gaussian Maximum-Entropy Branch-and-Bound to `SiftVOI` (`1,648 µs → 11.0 µs` = `149.8×` faster, 8,192-ch epoch `1,099 ms → 25.6 ms`, exact argmax preserved).
  - Vectorized `MultiRateConformal` with single-pass inline AVX2 shift + branchless bsearch (`896.5 ns → 316.8 ns` = `273×` vs pure Python).
  - Upgraded `SafeFuser` with anchor-centered tick-0 guard + Fixed-Share BOA (`eps=0.01`) + resettable Page-CUSUM `deg_e` (post-shift MSE `0.2983 → 0.0284`, `-85.0%` vs anchor).
  - Re-ran full benchmark suite: **6/6 pre-registered targets PASS**.
- **Cycle 4 (Release & Agent Discoverability):** Authored pure-Markdown `README.md`, `llms.txt`, `llms-full.txt`, updated `BREAKER_REPORT.md` with Cycle-3 certification, ran strict hygiene audit, and built interactive HTML benchmark & simulation lab.
