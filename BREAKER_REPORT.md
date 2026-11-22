# Adversarial Breaker Audit Report — `eproc` (Cycle 2 Red-Team + Cycle 3 Resolution)

**Auditor Role:** Independent Adversarial Breaker (Red-Team Scientist & Security Auditor)
**Target:** `eproc` repository root (Cycle 2 read-only audit followed by Cycle 3 resolution verification)
**Test Artifacts Created & Executed Live:**
- `tests/test_breaker.py` (42 `unittest` methods across Fronts A–F)
- `tests/breaker_c_torture.c` (C11 ABI, ASan+UBSan, `-fsanitize=float-cast-overflow`, TSan, and 2M-read concurrent seqlock torture)

> **Cycle 3 Final Status (`2026-11-22`):** **42 / 42 PASS (`0` FAIL witnesses remaining).** All 17 Cycle-2 witnesses (`W01–W17`) below were resolved at the root cause and re-verified live on both `_native` and `ctypes`, ASan+UBSan+FCO, TSan (2M seqlock reads), and GCC C11 / Clang C11 / G++ C++17 single-header builds. See Section 5 below for the resolution ledger.

---

## 1. Cycle 2 Audit Verdict (Historical Red-Team Baseline)

**CYCLE 2 VERDICT (SUPERSEDED BY CYCLE 3 PASS):** CORE STATISTICAL & CAUSAL MATH PASSED; 17 EDGE-CASE WITNESSES WERE EXPOSED AND SUBSEQUENTLY FIXED IN CYCLE 3.

### What Held Up Under Extreme Torture (Verified Strong)
- **Front A (Anytime-Valid Statistical Safety — 11,500+ null trajectories):** **100% mathematically sound.** Across Bernoulli(0.5), boundary Bernoulli(0.01/0.99), asymmetric 2-point `Bern(0.1)`, Markov adaptive non-stationary martingales, heteroskedastic martingales, Empirical-Bernstein CS (including bimodal `Beta(0.2, 0.2)`), 8-arm `AgentRace` global null, and `EDetector` ARL floor, **zero Ville/FWER/ARL violations** occurred. Naive fixed-horizon z-test peeking blew up to **33.0% FPR**, while every `eproc` test stayed strictly below `0.035` at nominal `alpha = 0.05`.
- **Front B (Strict Causality — 10/10 engines):** **0.0 exact IEEE-754 bitwise diff** on all prior outputs across every engine (`EProcess`, `ConfidenceSequence` EB/Normal, `EDetector`, `MultiRateConformal`, `DelayedConformal` with 100 out-of-order tickets, `BudgetConformal`, `SafeFuser`, `EchoPool`, `OnlineRLSSampler`, `VarOptSampler`) under future perturbations of `+1e15`, `-1e15`, `NaN`, `+Inf`, `-Inf`.
- **Front C (Integer Bin Abelian Group):** `bin_counts` & `count` had **0 drift across 1,000 random shards** for commutativity, associativity, LOO subtraction, and 640-byte single-bit-flip rejection across all 640 byte offsets.
- **Front D (Conformal Monotonicity, EchoPool, VarOpt, SIFT):** **0 quantile/interval crossings** over 2,000 ticks with 10× variance shocks & `1e15` spikes (with strong post-shock coverage recovery `[51.0%, 79.0%, 89.75%, 94.75%, 99.0%]`); **0 echo double-counts** under 50× spam; clean `EchoPool` 4096-slot saturation; **0.009% relative error** on 500-trial heavy-tailed Pareto VarOpt HT unbiasedness; quiet `+2.5` channel beats 3× loud `+4.0` distractor in `SiftVOISelector`.
- **Front E (C11 Memory Safety & Native Seqlock):** Zero ASan/UBSan/TSan issues on standard paths; **0 torn reads across 1,000,000 concurrent 64-expert SafeBOA seqlock reads** and **0 torn reads across 1,000,000 multi-writer raw seqlock reads**.
- **Front F (Cold Import & Dependencies):** Cold `import eproc` median **0.67 ms** (< 15 ms budget), zero eager `ctypes`/`typing`/`_ffi`/`numpy` imports, zero third-party runtime dependencies.

### What Broke (17 Reproducible FAIL Witnesses — Must Fix Before Release)
**Summary Scorecard:** `42` tests in `tests/test_breaker.py` → **25 PASS, 17 FAIL** (19 failing assertions counting `test_d2`'s 3 subtests).

| ID | Sev | Front | Component & Location | Defect Summary |
|---|---|---|---|---|
| **W01** | **P0** | D / E | `csrc/eproc_conformal.c:41,379-388` + `csrc/eproc_internal.h:20` | **Live C11 UB (`float-cast-overflow`) & permanent `NaN` corruption** in `MultiRateConformal` on finite `(-1e308, +1e308, +1e308)` |
| **W02** | **P0** | E | `python/eproc/fusion.py:78,130-137` | **Python `SafeFuser.snapshot()` thread race (11–125 torn reads)** — shared `self._weights_buf` defeats the lock-free C seqlock under concurrent Python readers |
| **W03** | **P0** | D | `csrc/eproc_fusion.c:101-127` | **Tick-0 Byzantine spread attack (`+2.0e12` catastrophic blast radius)** — 4 poisoned non-anchor experts `[1e12, 2e12, 3e12, 4e12]` capture median & MAD on tick 0 and bypass `max_dev` |
| **W04** | **P1** | C | `csrc/eproc_conformal.c:994-995,1011,1045` | **`ConformalMonoid` LOO `subtract(merge(A,B), A)` leaks removed shard `A`'s `min_val`/`max_val`** → `quantile(1.0)` returns `1e12` instead of `5.0` (12-order-of-magnitude error!) |
| **W05** | **P1** | D | `csrc/eproc_subsample.c:392-395` | **`OnlineRLSSampler` subnormal `1e-310` bug** — `1.0 / 1e-310` overflows to `+Inf`, `0.0 * +Inf = NaN`, emitting bogus `leverage = 1e300, prob = 1.0` for near-zero vector `[1e-310, 0, 0, 0]` |
| **W06** | **P1** | D | `csrc/eproc_internal.h:28-37` (`ConfidenceSequence`, `ConformalMonoid`, `VarOptSampler`) | **Neumaier Kahan overflow → `+Inf + (-Inf) = NaN`** on finite `1e308` inputs (`CS(normal).mean = nan`, `ConformalMonoid.mean = nan`, `VarOpt.estimate_sum() = nan` when `v=0`) |
| **W07** | **P1** | C | `csrc/eproc_conformal.c:800 vs 836` | **`ConformalMonoid.add` vs `add_batch` bin mismatch** — `score / scale` vs `score * (1.0 / scale)` puts `7.729435816691415` (`scale=7.0`) into bin 26 vs bin 25 |
| **W08** | **P1** | C | `csrc/eproc_conformal.c:861-922` | **`ConformalMonoid.merge` silently ignores mismatched `scale_hint`** (unlike `subtract()`), corrupting quantiles asymmetrically (`merge(A,B) != merge(B,A)`) |
| **W09** | **P1** | F | `python/eproc/cli.py:269,647` | **CLI crash (`ZeroDivisionError`)** in `eproc conformal` when JSONL contains `"scale": 0.0` (unhandled exception traceback instead of exit code 2) |
| **W10** | **P2** | C | `csrc/eproc_conformal.c:882-890` | **`ConformalMonoid` wire bytes fail commutativity & associativity** (`merge(A,B).to_bytes() != merge(B,A).to_bytes()` in 139/200 mixed-scale shards due to asymmetric Kahan merge) |
| **W11** | **P2** | C | `csrc/eproc_conformal.c:1107` + `python/eproc/_ffi.py:813` | **`ConformalMonoid.from_bytes` accepts trailing garbage (`len > 640`)** and raises `OverflowError` (instead of `ValueError`) on truncated buffers (`len < 640`) |
| **W12** | **P2** | A | `python/eproc/eprocess.py:442` | **`EDetector.update_batch([])` leaks stale pre-reset `stat`** (e.g. `1.77e14` while `log_stat=0.0`) after `reset_alarm()` |
| **W13** | **P2** | F | `dist/eproc_single.h:48,768` (`scripts/make_single_header.py`) | **Single-header fails C++17 compilation** (`error: '_Atomic' was not declared in this scope`) despite header advertising `In exactly ONE C/C++ translation unit` |
| **W14** | **P2** | F | `python/eproc/mcp_server.py:417,461,501,577,693,756` | **MCP `"reset": "false"` silently wipes live session** (`bool("false") == True`) |
| **W15** | **P2** | F | `python/eproc/mcp_server.py:882-885` | **MCP drops invalid JSON-RPC requests lacking `"id"`** instead of returning `ERR_INVALID_REQUEST (-32600)` with `"id": null` per JSON-RPC 2.0 §5 |
| **W16** | **P2** | F | `python/eproc/mcp_server.py:605-607,728,794` | **MCP silently coerces JSON booleans (`True`/`False`) to `1.0`/`0.0`** in `eproc_safe_fuse`, `eproc_echopool`, and `eproc_conformal` |
| **W17** | **P3** | Notes | Additional non-blocking audit findings | (1) `csrc/eproc_fusion.c:27` casts plain `uint64_t*` to `_Atomic uint64_t*` (strict C11 type-punning UB); (2) `MultiRateConformal` scalar vs batch `1e-16` drift from `eproc_exp4_small` caching; (3) `docs/MATHEMATICS.md:92` said "Brier variance reduction" while `eproc_subsample.c:470` computes Gauss-Hermite Mutual Information; (4) `COMPANY_STATE.md` contained internal working-draft references scrubbed before release. |

---

## 2. Detailed FAIL Witnesses & Root-Cause Diagnostics

### W01 [P0 — CRITICAL]: Live C11 Undefined Behavior (`float-cast-overflow`) & Permanent `NaN` Corruption in `MultiRateConformal`
- **Locations:** `csrc/eproc_conformal.c:41`, `csrc/eproc_conformal.c:379-388`, `csrc/eproc_internal.h:20-24`
- **Root Cause:**
  1. Feeding finite `-1e308`, `+1e308`, `+1e308` into `eproc_conformal_update` produces `rng = 1e308 - (-1e308) = +Inf`, `inv_scale_norm = 0.0`, `diff = score - q_exp[k] = +Inf`, `loss_exp[k] = +Inf`, `l_bar = +Inf`.
  2. Line 386 computes `(l_bar - loss_exp[k]) * inv_scale_norm = (+Inf - +Inf) * 0.0 = -NaN`.
  3. `eproc_clamp(NaN, -1.0, 1.0)` in `csrc/eproc_internal.h:20-24` has no `isnan(x)` guard and returns `-NaN`.
  4. `-NaN` permanently corrupts `exp_var`, `exp_log_w`, `wr`, `a_agg`, and `eff_alpha[0..4] = (nan, nan, nan, nan, nan)`.
  5. On line 462 → `eproc_sorted_quantile` (`csrc/eproc_conformal.c:41`), `pos = (1.0 - (-NaN)) * 4 = -NaN` bypasses both `pos <= 1.0` and `pos >= len` and executes `const uint32_t ipos = (uint32_t)pos;` — **C11 Undefined Behavior**.
- **Live UBSan Output:**
  ```
  csrc/eproc_conformal.c:41:5: runtime error: -nan is outside the range of representable values of type 'unsigned int'
      #0 in eproc_sorted_quantile csrc/eproc_conformal.c:41
      #1 in eproc_conformal_step_cached csrc/eproc_conformal.c:462
      #2 in eproc_conformal_update csrc/eproc_conformal.c:481
  ```
- **Repro:**
  ```bash
  PYTHONPATH=python python3 -m unittest \
    tests.test_breaker.TestFrontD_NumericalByzantineMonotonicity.test_d3_conformal_extreme_finite_nan_witness \
    tests.test_breaker.TestFrontE_ConcurrencyAndSanitizers.test_e3_conformal_ubsan_float_cast_overflow_witness -v
  ```
- **Fix:** Guard `isnan(x)` in `eproc_clamp` (or return `lo` / handle explicitly), compute `diff` scaled or clamp finite `diff`/`loss_exp` before subtraction, guard `!isfinite(rng)` using half-difference `0.5 * sorted[hi] - 0.5 * sorted[lo]`, and guard `!isfinite(alpha)` in `eproc_sorted_quantile`. Also note PAVA (`eproc_conformal.c:137`) overflows `blk_val * cnt` on `1e308`; use weighted-average form `a + (b - a) * (c2 / (c1 + c2))`.

---

### W02 [P0 — CRITICAL]: Python `SafeFuser.snapshot()` Concurrent Torn Reads (Shared Buffer Race)
- **Location:** `python/eproc/fusion.py:78,130-137`
- **Root Cause:** The C11 Boehm-2012 seqlock (`eproc_boa_snapshot_weights`) is 100% thread-safe in C (**0 torn reads across 1M C reads**). However, `SafeFuser` allocates ONE shared `self._weights_buf = (c_double * self._n_experts)()` in `__init__` and reuses it on every `snapshot()` / `weights()` call. Because `ctypes` releases the Python GIL during the C call, concurrent Python reader threads calling `fuser.snapshot()` overwrite each other's `self._weights_buf` mid-copy and mid-list-comprehension.
- **Measured Evidence:**
  - 8 concurrent Python reader threads + 1 writer on a 64-expert `SafeFuser`: **11 torn reads** in `test_e2` (and **125 torn reads** in our longer probe) where `fallback_active=True` returned non-anchor weights or `sum(weights) != 1.0`!
- **Repro:**
  ```bash
  PYTHONPATH=python python3 -m unittest \
    tests.test_breaker.TestFrontE_ConcurrencyAndSanitizers.test_e2_python_safe_fuser_snapshot_concurrent_thread_safety_witness -v
  ```
- **Fix:** Allocate a local `buf = (c_double * self._n_experts)()` (or `threading.local()`) inside `SafeFuser.snapshot()`, never a shared instance buffer.

---

### W03 [P0 — CRITICAL]: `SafeFuser` Tick-0 Byzantine Spread Attack (`+2.0e12` Catastrophic Blast Radius)
- **Location:** `csrc/eproc_fusion.c:101-127`
- **Root Cause:** On tick 0 (`boa->n_updates == 0ULL`), `eproc_boa_shadow_predict` sets `ref = anc`, but computes `med` and `mad` across ALL finite experts (`finite_vals`). When 100% of non-anchor experts (`4/4`) are Byzantine and emit spread values `[1e12, 2e12, 3e12, 4e12]` while the honest anchor predicts `0.0` (`y_true = 0.0`):
  - `med = 2e12` (captured by the Byzantine majority!)
  - `mad = 1e12`
  - `pred_scale = 2.0 * mad = 2e12`
  - `max_dev = 25.0 * 2e12 = 5e13`
  - All 4 Byzantine experts pass `fabs(p - ref) <= max_dev` and are averaged with equal weight `1/5`, yielding **`fused_pred = 2,000,000,000,000.0`** (`+2e12` error vs anchor `0.0`!).
- **Repro:**
  ```bash
  PYTHONPATH=python python3 -m unittest \
    tests.test_breaker.TestFrontD_NumericalByzantineMonotonicity.test_d6_safe_fuser_100pct_poisoned_steady_state_and_tick0_spread_witness -v
  ```
- **Fix:** On tick 0 (before `scale_ema` is grounded by any observed `y_true`), do NOT let an unauthenticated majority of non-anchor experts inflate `pred_scale` via unconstrained MAD! Either anchor tick-0 `pred_scale` purely to `fmax(1.0, fabs(ref) * 0.05)` (or start on tick 0 trusting anchor / conservative scale cap) so Byzantine non-anchors can never self-authorize a `5e13` clipping window.

---

### W04 [P1 — HIGH]: `ConformalMonoid` Leave-One-Out `subtract()` Leaks Removed Shard's `min_val` / `max_val`
- **Location:** `csrc/eproc_conformal.c:994-995`, `csrc/eproc_conformal.c:1008-1056`
- **Root Cause:** `eproc_cmonoid_subtract` copies `loo_out->min_val = global_m->min_val; loo_out->max_val = global_m->max_val;` verbatim. Because `min`/`max` is NOT a group operation (only a semilattice), subtracting shard `A` (containing outlier `1e12`) from `merge(A, B)` (where `B = [1, 2, 3, 4, 5]`) leaves `loo_b.max_val = 1e12`. Then `eproc_cmonoid_quantile` uses `min_val`/`max_val` directly on `q=0.0`/`q=1.0` and tail bins (`hit == 0` / `hit == 63`), returning **`loo_b.quantile(1.0) = 1e12`** instead of **`5.0`**!
- **Repro:**
  ```bash
  PYTHONPATH=python python3 -m unittest \
    tests.test_breaker.TestFrontC_ConformalMonoidAlgebra.test_c3_loo_subtract_quantile_outlier_leak_witness -v
  ```
- **Fix:** After `subtract`, clamp `loo_out->min_val` and `loo_out->max_val` against the first and last non-zero bins of `loo_out->bin_counts` (i.e. if `bin_counts[63] == 0`, `max_val` cannot exceed `EPROC_CMONOID_EDGES[last_nonzero_bin] * scale`; and similarly for `min_val`), or compute `q=0`/`q=1` from occupied bin bounds when `min_val`/`max_val` fall outside the occupied bin support (flagging `min_val`/`max_val` as stale after subtraction via `flags`).

---

### W05 [P1 — HIGH]: `OnlineRLSSampler` Subnormal `1e-310` Causes `1/scale = +Inf` → Bogus `leverage = 1e300, prob = 1.0`
- **Location:** `csrc/eproc_subsample.c:389-419`
- **Root Cause:** When `max_abs < 1e-100` (e.g. `x = [1e-310, 0.0, 0.0, 0.0]`), line 392 computes `inv_s = 1.0 / 1e-310 = +INFINITY`. Line 394 then computes `x_scaled[1] = 0.0 * (+Inf) = NaN` and `x_scaled[0] = +Inf`. This turns `lev_base` into `NaN`, which line 414 masks as `leverage = INFINITY`, returning **`RLSStepResult(keep=True, leverage=1e300, prob=1.0, ht_weight=1.0)`** for a subnormal near-zero vector whose true leverage is `1e-620 == 0.0` (expected `prob = min_prob = 0.02`)!
- **Repro:**
  ```bash
  PYTHONPATH=python python3 -m unittest \
    tests.test_breaker.TestFrontD_NumericalByzantineMonotonicity.test_d4_rls_subnormal_1e310_bogus_leverage_witness -v
  ```
- **Fix:** Replace `x[i] * (1.0 / scale)` with `x[i] / scale` (which yields `0.0 / 1e-310 = 0.0` and `1e-310 / 1e-310 = 1.0` with ZERO overflow even for `DBL_TRUE_MIN = 5e-324`!), and treat `max_abs < 1e-150` as zero-leverage (`leverage = 0.0`, `prob = min_prob`).

---

### W06 [P1 — HIGH]: Neumaier Kahan Overflow Produces `NaN` (`+Inf + (-Inf)`) on Finite `1e308` Inputs
- **Locations:** `csrc/eproc_internal.h:28-41`, `csrc/eproc_eprocess.c:558`, `csrc/eproc_conformal.c:804`, `csrc/eproc_subsample.c:124-252`
- **Root Cause:** When `sum` overflows to `+Inf` inside `eproc_kahan_add`, `(s - t) + x = (1e308 - Inf) + 1e308 = -Inf`, so `eproc_kahan_total(sum, comp) = (+Inf) + (-Inf) = NaN`. Additionally, in `VarOptSampler`, `tau_new` overflows to `+Inf` on three `(w=1e308, v=0.0)` pushes, causing `adj_weight * value = (+Inf) * 0.0 = NaN`.
- **Measured Evidence (3 failing subtests in `test_d2`):**
  - `ConfidenceSequence(mode='normal')` after 2× `1e308`: `mean = nan`
  - `ConformalMonoid` after 2× `1e308`: `mean = nan`
  - `VarOptSampler(k=2)` after 3× `(1e308, 0.0)`: `estimate_sum() = nan, tau = inf`
- **Repro:**
  ```bash
  PYTHONPATH=python python3 -m unittest \
    tests.test_breaker.TestFrontD_NumericalByzantineMonotonicity.test_d2_kahan_overflow_nan_witness -v
  ```
- **Fix:** In `eproc_kahan_add`, if `!isfinite(t)`, keep `*sum = t; *comp = 0.0; return;` (and clamp or rescale in `eproc_cs_step_normal`, `eproc_cmonoid_add`, and `eproc_varopt_push` when `sum_light` would overflow).

---

### W07 [P1 — HIGH]: `ConformalMonoid.add` vs `add_batch` Bin Mismatch
- **Location:** `csrc/eproc_conformal.c:800` vs `csrc/eproc_conformal.c:822,836`
- **Root Cause:** `eproc_cmonoid_add` computes `z = score / scale`, whereas `eproc_cmonoid_add_batch` precomputes `inv_scale = 1.0 / scale` and computes `s * inv_scale`. Because `x / 7.0 != x * (1.0 / 7.0)` in IEEE-754 binary64, the exact same input `7.729435816691415` at `scale_hint = 7.0` lands in **bin 26** via `add()` and **bin 25** via `add_batch()`! (Also reproduces at `scale_hint = 11.0`, `v = 325.9751053845587`: bin 61 vs bin 62.)
- **Repro:**
  ```bash
  PYTHONPATH=python python3 -m unittest \
    tests.test_breaker.TestFrontC_ConformalMonoidAlgebra.test_c4_add_vs_add_batch_bin_mismatch_witness -v
  ```
- **Fix:** Use the identical formula (`score / scale` or `score * (1.0 / scale)`) in BOTH `eproc_cmonoid_add` and `eproc_cmonoid_add_batch`.

---

### W08 [P1 — HIGH]: `ConformalMonoid.merge` Silently Ignores Incompatible `scale_hint`
- **Location:** `csrc/eproc_conformal.c:895` (`python/eproc/conformal.py:396-404`)
- **Root Cause:** `eproc_cmonoid_subtract` explicitly enforces matching `scale_hint` (`EPROC_ERR_MISMATCH`), but `eproc_cmonoid_merge` returns `void`, blindly copies `a->scale_hint`, and adds raw bin counts from two incompatible grids! `ma.merge(mb).quantile(0.5) = 1.052` while `mb.merge(ma).quantile(0.5) = 9.767`.
- **Repro:**
  ```bash
  PYTHONPATH=python python3 -m unittest \
    tests.test_breaker.TestFrontC_ConformalMonoidAlgebra.test_c5_merge_scale_hint_mismatch_witness -v
  ```
- **Fix:** Validate `scale_hint` compatibility in Python `ConformalMonoid.merge` (raising `ValueError` when both shards are non-empty and `scale_hint` differs beyond `1e-12 * max(1, |s|)`), and consider a checked C helper or flagging mismatch in `flags`.

---

### W09 [P1 — HIGH]: CLI Unhandled `ZeroDivisionError` Crash in `eproc conformal`
- **Location:** `python/eproc/cli.py:269,647`
- **Root Cause:** `_cmd_conformal` computes `score = abs(y_val - pred_val) / scale_val` without checking `scale_val <= 0.0` or `!math.isfinite(scale_val)`, and `main()` only catches `(ValueError, KeyError, json.JSONDecodeError, OSError)`. Passing `{"pred": 1.0, "y": 2.0, "scale": 0.0}` crashes the CLI with an unhandled traceback. (Negative `scale` also silently injects negative scores.)
- **Repro:**
  ```bash
  PYTHONPATH=python python3 -m unittest \
    tests.test_breaker.TestFrontF_PackagingHygieneAndMcpCli.test_f6_cli_conformal_zero_scale_crash_witness -v
  ```
- **Fix:** Validate `if not math.isfinite(scale_val) or scale_val <= 0.0: raise ValueError(...)` in `cli.py:_cmd_conformal` (matching `mcp_server.py:608`).

---

### W10 [P2 — MEDIUM]: `ConformalMonoid` Wire-Byte Commutativity & Associativity Failure
- **Location:** `csrc/eproc_conformal.c:882-890`
- **Root Cause:** Although `bin_counts` is 100% bit-exact commutative/associative, `merge(A, B)` folds `(b->sum, b->sum_comp)` sequentially into `(a->sum, a->sum_comp)` via Neumaier summation, which is order-dependent whenever `a->sum_comp != b->sum_comp`. Across 200 random mixed-scale shards, `merge(A, B).to_bytes() != merge(B, A).to_bytes()` in **139/200 trials (69.5%)**, and associativity failed byte-equality in **143/200 trials (71.5%)**.
- **Repro:**
  ```bash
  PYTHONPATH=python python3 -m unittest \
    tests.test_breaker.TestFrontC_ConformalMonoidAlgebra.test_c2_wire_bytes_commutativity_and_associativity_witness -v
  ```
- **Fix:** Compute symmetric TwoSum canonicalization (`s = a->sum + b->sum`, exact error term symmetric in `a` and `b`, `c = a->sum_comp + b->sum_comp + err`, then re-normalize `(s, c)` with FastTwoSum) so `merge(A, B)` is bit-exact commutative on ALL fields (and document clearly that exact associativity holds on `bin_counts` and `count` while IEEE-754 `sum`/`sum_sq` are compensated approximations).

---

### W11 [P2 — MEDIUM]: `ConformalMonoid.from_bytes` Accepts Trailing Garbage & Misclassifies Truncated Buffers
- **Location:** `csrc/eproc_conformal.c:1107`, `python/eproc/conformal.py:447-448`
- **Root Cause:** `eproc_cmonoid_deserialize` checks `if (len < 640U) return EPROC_ERR_CAPACITY;`. This (a) **silently accepts trailing garbage** (`len > 640`, e.g. `wire + b"TRAILING_GARBAGE"`), and (b) maps too-short buffers to `EPROC_ERR_CAPACITY` → Python `OverflowError("fixed internal capacity exceeded")` instead of `ValueError`.
- **Repro:**
  ```bash
  PYTHONPATH=python python3 -m unittest \
    tests.test_breaker.TestFrontC_ConformalMonoidAlgebra.test_c6_underflow_and_corrupted_wire_rejection -v
  ```
- **Fix:** Require `if (len != 640U) return EPROC_ERR_INVALID;` in `eproc_cmonoid_deserialize` (and validate `isfinite` / `min_val <= max_val` when `count > 0`).

---

### W12 [P2 — MEDIUM]: `EDetector.update_batch([])` Leaks Stale Pre-Reset `stat`
- **Location:** `python/eproc/eprocess.py:442`
- **Root Cause:** `EDetector.reset_alarm()` resets `self._state.log_cusum = 0.0` in C without resetting `self._state.n` or Python's cached `self._out.stat`. Calling `ed.update_batch([])` after `reset_alarm()` returns `stat = self._out.stat` (e.g. `1.775e14`) while `log_stat = 0.0` and `alarm = False`.
- **Repro:**
  ```bash
  PYTHONPATH=python python3 -m unittest \
    tests.test_breaker.TestFrontA_AnytimeValidTypeIError.test_a7_edetector_null_arl_and_reset_empty_batch_witness -v
  ```
- **Fix:** Compute `stat = math.exp(min(log_stat, 700.0))` directly from `self._state.log_cusum` (and reset `self._out.stat = 1.0` in `reset_alarm()`).

---

### W13 [P2 — MEDIUM]: `dist/eproc_single.h` Fails C++17 Compilation Despite Advertising C/C++ Support
- **Locations:** `dist/eproc_single.h:9-10,768`, `csrc/eproc_seqlock.h:23,36`
- **Root Cause:** `dist/eproc_single.h` instructs `In exactly ONE C/C++ translation unit, define: #define EPROC_IMPLEMENTATION`, but includes C11 `<stdatomic.h>` and `_Atomic uint64_t` unconditionally without `#ifdef __cplusplus` compatibility, failing `g++ -std=c++17` with `error: '_Atomic' was not declared in this scope`.
- **Repro:**
  ```bash
  PYTHONPATH=python python3 -m unittest \
    tests.test_breaker.TestFrontF_PackagingHygieneAndMcpCli.test_f3_single_header_c11_and_cpp17_compilation_witness -v
  ```
- **Fix:** Either add `#ifdef __cplusplus` `<atomic>` / `std::atomic<uint64_t>` + `static_assert` compatibility in `eproc_seqlock.h` / `eproc_conformal.c` / `eproc_fusion.c`, or explicitly restrict `EPROC_IMPLEMENTATION` in `make_single_header.py` to C11 translation units (`extern "C"` header only in C++).

---

### W14–W16 [P2 — MEDIUM]: Three MCP Server Protocol & Type-Validation Bugs (`python/eproc/mcp_server.py`)
1. **W14 (`test_f5a`):** `reset = bool(args.get("reset", False))` (lines 417, 461, 501, 577, 693, 756) evaluates `"reset": "false"` to `True`, **silently wiping an active session's state** (`n=1` instead of `3`). Fix: require `isinstance(val, bool)`.
2. **W15 (`test_f5b`):** Lines 882–885 silently return `None` when an invalid JSON-RPC request lacks `"id"` (e.g. `{"jsonrpc": "1.0", "method": "ping"}`). JSON-RPC 2.0 §5 mandates returning `{"jsonrpc": "2.0", "id": null, "error": {"code": -32600, ...}}`.
3. **W16 (`test_f5c`):** Lines 605–607 (`_tool_conformal`), 728 (`_tool_safe_fuse`), and 794 (`_tool_echopool`) call raw `float(...)` without rejecting `bool` (unlike `_require_finite_float` and `_as_float_list`), silently coercing `True` → `1.0` and strings like `"nan"`.
- **Repro:**
  ```bash
  PYTHONPATH=python python3 -m unittest \
    tests.test_breaker.TestFrontF_PackagingHygieneAndMcpCli.test_f5a_mcp_reset_string_false_witness \
    tests.test_breaker.TestFrontF_PackagingHygieneAndMcpCli.test_f5b_mcp_invalid_request_missing_id_witness \
    tests.test_breaker.TestFrontF_PackagingHygieneAndMcpCli.test_f5c_mcp_bool_coerced_to_float_witness -v
  ```

---

## 3. Complete Front-by-Front Live Verification Results

### Front A — Anytime-Valid Type-I Error Torture (11,500+ Null Trajectories, `alpha = 0.05`)

| Test Case | Horizon `T` | Trials `N` | Measured Rate (95% Wilson CI) | Bound | Status |
|---|---|---|---|---|---|
| `EProcess` Bern(0.5) `upper` | 500 | 2,000 | **0.0335** (`67/2000`, `[0.0265, 0.0423]`) | `<= 0.055` | **PASS** |
| `EProcess` Bern(0.5) `two_sided` | 500 | 2,000 | **0.0335** (`67/2000`, `[0.0265, 0.0423]`) | `<= 0.055` | **PASS** |
| Control: Naive fixed-horizon z-test peeking | 500 | 300 | **0.3300** (`99/300`, `[0.2792, 0.3851]`) | `>= 0.250` | **PASS (inflates as expected)** |
| `EProcess` Bern(0.01) `upper / lower / two_sided` | 500 | 1,000 × 3 | **0.0140 / 0.0180 / 0.0060** | `<= 0.055` | **PASS** |
| `EProcess` Bern(0.99) `upper / lower / two_sided` | 500 | 1,000 × 3 | **0.0220 / 0.0150 / 0.0060** | `<= 0.055` | **PASS** |
| `EProcess` Asymmetric 2-pt `Bern(0.1)` `u / l / 2s` | 500 | 1,000 × 3 | **0.0250 / 0.0300 / 0.0200** | `<= 0.055` | **PASS** |
| `EProcess` Non-stationary Markov martingale null (`0.50` if `X_{t-1}=1` else `0.30`, `mu0=0.50`) | 500 | 1,000 | **0.0210** (`21/1000`) | `<= 0.055` | **PASS** |
| `EProcess` Heteroskedastic martingale null (`E[X_t\|F_{t-1}]=0.50`, alternating Bern/Unif) | 500 | 1,000 | **0.0290** (`29/1000`) | `<= 0.055` | **PASS** |
| EB-CS simultaneous miscoverage: `Bern(0.5)` | 350 | 1,000 | **0.0170** (`17/1000`) | `<= 0.055` | **PASS** |
| EB-CS simultaneous miscoverage: `Bern(0.05)` | 350 | 1,000 | **0.0050** (`5/1000`) | `<= 0.055` | **PASS** |
| EB-CS simultaneous miscoverage: `Beta(0.2, 0.2)` | 350 | 1,000 | **0.0050** (`5/1000`) | `<= 0.055` | **PASS** |
| `AgentRace` 8-arm global null (`p=0.5`, LUCB1) | 320 | 500 | **0.0000** (`0/500` false winners; `0/500` eliminations) | `<= 0.055` | **PASS** |
| `EDetector` Bern(0.5) null (`alpha=0.05`, ARL floor = 20) | 1,000 | 500 | Alarmed `481/500 (96.2%)`; censored **ARL = 260.5** | `>= 20.0` | **PASS (ARL)** / **FAIL (W12 reset bug)** |

### Front B — Strict Causality (`assert_causal`, Future Perturbations: `+1e15, -1e15, NaN, +Inf, -Inf`)

| Engine | Test Method | Max Bitwise Diff on `t <= t0` | Status |
|---|---|---|---|
| `EProcess` (upper, lower, two_sided) | `test_b1` | `0.0` (0 bit diffs) | **PASS** |
| `ConfidenceSequence` (`eb`, `normal`) | `test_b2` | `0.0` (0 bit diffs) | **PASS** |
| `EDetector` | `test_b3` | `0.0` (0 bit diffs) | **PASS** |
| `MultiRateConformal` (5 rates) | `test_b4` | `0.0` (0 bit diffs) | **PASS** |
| `DelayedConformal` (100 tickets, out-of-order & late reveal) | `test_b5` | `0.0` (0 bit diffs) | **PASS** |
| `BudgetConformal` | `test_b6` | `0.0` (0 bit diffs) | **PASS** |
| `SafeFuser` | `test_b7` | `0.0` (0 bit diffs) | **PASS** |
| `EchoPool` (submits + resolutions) | `test_b8` | `0.0` (0 bit diffs) | **PASS** |
| `OnlineRLSSampler` & `VarOptSampler` | `test_b9` | `0.0` (0 bit diffs) | **PASS** |

### Front C — Exact Algebraic Monoid & Group Inverse (`ConformalMonoid`)

| Property | Measured Result | Status |
|---|---|---|
| 1,000 random shards (`[-1e12, 1e12, 1e-12, 0, NaN, ±Inf]`): `bin_counts` commutativity | `0 / 1,000` diffs | **PASS** |
| 1,000 random shards: `bin_counts` associativity `((A+B)+C) == (A+(B+C))` | `0 / 1,000` diffs | **PASS** |
| 1,000 random shards: LOO `subtract(merge(A,B), A).bin_counts == B.bin_counts` | `0 / 1,000` diffs | **PASS** |
| 1,000 random shards: wire roundtrip `from_bytes(to_bytes())` | `0 / 1,000` diffs (640 B exact) | **PASS** |
| Underflow protection (`subtract(A, larger_B)` & disjoint bin) | Raises `ValueError` cleanly (0 wraps) | **PASS** |
| Single-bit-flip sweep across ALL 640 wire bytes (`breaker_c_torture.c`) | `640 / 640` rejected | **PASS** |
| Wire-byte commutativity `merge(A,B).to_bytes() == merge(B,A).to_bytes()` | Fails in `139 / 200` mixed-scale shards | **FAIL (W10)** |
| LOO `subtract(merge(A,B), A).quantile(1.0) == B.quantile(1.0)` | Returns `1e12` instead of `5.0` | **FAIL (W04)** |
| Scalar `add(x)` vs `add_batch([x])` bin equivalence (`scale=7.0`) | Bin `26` vs bin `25` on `7.729435816691415` | **FAIL (W07)** |
| Incompatible `scale_hint` merge rejection | Silently merges `1.0` & `10.0` | **FAIL (W08)** |
| Trailing garbage (`len > 640`) wire rejection | Silently accepted | **FAIL (W11)** |

### Front D — Numerical, Byzantine & Monotonicity Torture

| Check | Measured Result | Status |
|---|---|---|
| `NaN, ±Inf, ±1e-310, ±1e15, ±0.0` across all 13 Python classes (`test_d1`) | `0` crashes, `0` NaN outputs | **PASS** |
| `MultiRateConformal` 2,000 ticks, 5 rates, 10× shock + `1e15` spikes (`test_d5`) | **0 crossings**; post-shock coverage `[51.0%, 79.0%, 89.75%, 94.75%, 99.0%]` | **PASS** |
| `EchoPool` 50× echoes, colliding claims, 100% liar, >4096 saturation (`test_d7`) | `0` echo double-counts; liar `rel=0.0`; clean `OverflowError` at 4097 | **PASS** |
| `VarOptSampler` 500 trials, Pareto `[1, 10000]` HT unbiasedness (`test_d8`) | Exact sum `12192.33`, MC mean `12193.37` (**rel_err = 0.00009**) | **PASS** |
| `SiftVOISelector` 3× loud distractor (`+4.0`) vs quiet (`+2.5`) (`test_d9`) | Quiet channel wins posterior & `select_next()` | **PASS** |
| `SafeFuser` post-warmup 100% poisoned non-anchors (`±1e12, NaN, Inf`) | `+0.00%` blast, `4/4` quarantined, Ville fallback trips | **PASS** |
| `SafeFuser` tick-0 100% poisoned spread `[0, 1e12, 2e12, 3e12, 4e12]` (`test_d6`) | **`fused_pred = 2.0e12`** (`+2e12` blast!) | **FAIL (W03)** |
| Neumaier Kahan overflow on finite `1e308` (`test_d2`) | `CS(normal)`, `ConformalMonoid`, `VarOpt` emit `NaN` | **FAIL (W06)** |
| `MultiRateConformal` finite `(-1e308, +1e308, +1e308)` (`test_d3`) | `eff_alphas = (nan, nan, nan, nan, nan)` | **FAIL (W01)** |
| `OnlineRLSSampler` subnormal `[1e-310, 0, 0, 0]` (`test_d4`) | Bogus `leverage = 1e300, prob = 1.0` | **FAIL (W05)** |

### Front E — Concurrency, Sanitizers & C11 Torture

| Check | Measured Result | Status |
|---|---|---|
| `breaker_c_torture.c` under **ASan + UBSan** (NULLs, OOB, 0-len, boundary `mu0=1e-12`/`1-1e-12`, 4096 saturation, 640-byte bit-flip) | `0` ASan/UBSan findings, `0` leaks | **PASS** |
| `breaker_c_torture.c` under **TSan**: 1M concurrent 64-expert SafeBOA seqlock reads vs 17,732–23,989 live updates | **0 torn reads**, **0 TSan data races** | **PASS** |
| `breaker_c_torture.c` under **TSan**: 1M concurrent multi-writer (2W+4R) raw seqlock reads | **0 torn reads**, **0 TSan data races** | **PASS** |
| `breaker_c_torture.c --ub-witness` under **`-fsanitize=address,undefined,float-cast-overflow`** (`test_e3`) | **Live UB caught** at `csrc/eproc_conformal.c:41` (`(uint32_t)(-nan)`) | **FAIL (W01)** |
| Python `SafeFuser.snapshot()` across 8 concurrent reader threads (`test_e2`) | **11–125 torn reads** on shared `self._weights_buf` | **FAIL (W02)** |

### Front F — Packaging, Hygiene, Single-Header, MCP & CLI

| Check | Measured Result | Status |
|---|---|---|
| Cold `import eproc` latency & eager-import audit (`test_f1`) | **0.67 ms** median (< 15 ms); `ctypes`, `typing`, `_ffi`, `numpy` NOT loaded | **PASS** |
| Zero third-party runtime dependencies (`test_f2`) | `0` non-stdlib imports across `python/eproc/*.py` | **PASS** |
| Private path/name leak scan in `csrc/`, `python/`, `docs/`, `dist/`, `skills/`, `examples/` (`test_f2`) | `0` leaks in distributable directories *(note: `COMPANY_STATE.md` has internal org/project names — see W17)* | **PASS (with note)** |
| `dist/eproc_single.h` C11 `-Wall -Wextra -Werror` build (`test_f3`) | Compiles cleanly (`rc=0`) | **PASS** |
| `dist/eproc_single.h` C++17 build as advertised in header (`test_f3`) | Fails (`_Atomic` undeclared) | **FAIL (W13)** |
| MCP 2026-07-28 handshake, `server/discover`, `tools/list`, malformed JSON/unknown tools/oversized 1 MB payload/LRU eviction (`test_f4`) | Clean protocol responses & LRU eviction | **PASS** |
| MCP `"reset": "false"` session wipe / missing-`id` invalid req / bool-as-float coercion (`test_f5a/b/c`) | All 3 validation defects reproduced | **FAIL (W14–W16)** |
| CLI `eproc conformal` with `"scale": 0.0` (`test_f6`) | Crashes with unhandled `ZeroDivisionError` | **FAIL (W09)** |

---

## 4. Exact Reproduction Commands

```bash
# 1. Full 6-front Python Adversarial Breaker Suite (42/42 PASS after Cycle 3)
PYTHONPATH=python python3 -m unittest tests/test_breaker.py -v

# 2. C11 Torture under ASan + UBSan (normal paths + float-cast-overflow UB witness)
gcc -std=c11 -ffp-contract=off -Wall -Wextra -Werror -O2 -g -march=native -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-cast-overflow -Icsrc \
  tests/breaker_c_torture.c csrc/eproc_eprocess.c csrc/eproc_conformal.c csrc/eproc_fusion.c csrc/eproc_subsample.c \
  -o /tmp/breaker_c_asan -pthread -lm
ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 /tmp/breaker_c_asan
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 /tmp/breaker_c_asan --ub-witness

# 3. C11 Torture under TSan (1M SafeBOA 64-expert seqlock reads + 1M multi-writer raw seqlock reads)
gcc -std=c11 -ffp-contract=off -Wall -Wextra -Werror -O2 -g -march=native -fno-omit-frame-pointer \
  -fsanitize=thread -Wno-tsan -Icsrc \
  tests/breaker_c_torture.c csrc/eproc_eprocess.c csrc/eproc_conformal.c csrc/eproc_fusion.c csrc/eproc_subsample.c \
  -o /tmp/breaker_c_tsan -pthread -lm
TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1 /tmp/breaker_c_tsan
```

---

## 5. Cycle 3 Resolution Certification (`42 / 42 PASS`)

Every witness (`W01–W17`) identified in Cycle 2 was fixed at the root cause in Cycle 3 and re-verified live on `2026-11-22`:

| Witness | Root-Cause Fix | Post-Fix Verification |
|---|---|---|
| **W01** | NaN-safe `eproc_clamp`, finite bounds before `(uint32_t)`, overflow-safe half-differences & PAVA weighted average (`csrc/eproc_conformal.c`, `csrc/eproc_internal.h`) | `test_d3`, `test_e3` (`-fsanitize=address,undefined,float-cast-overflow --ub-witness`) **PASS** |
| **W02** | Per-call stack/local snapshot buffer in `SafeFuser.snapshot()` / `_native` + `ctypes` (`python/eproc/fusion.py`, `csrc/eproc_pyext.c`) | `test_e2` (8 concurrent Python reader threads, **0 torn reads**) **PASS** |
| **W03** | Anchor-centered tick-0 `pred_scale` guard + Fixed-Share BOA (`eps=0.01`) + resettable Page-CUSUM `deg_e` (`csrc/eproc_fusion.c`) | `test_d6` (`0.00%` tick-0 blast, 4/4 immediate quarantine) & B4a post-shift **-85.0% vs anchor** **PASS** |
| **W04, W07, W08, W10, W11** | Abelian bin extrema tightening on LOO `subtract`, unified `score * inv_scale`, `scale_hint` mismatch rejection, canonical-order commutative Kahan merge, strict `len == 640` deserialize | `test_c1–c6` (0 drift, 0 LOO leak, bit-exact wire commutativity) **PASS** |
| **W05** | Subnormal norm guard (`max_abs < 1e-150` -> `leverage=0, prob=min_prob`) and overflow-free `x[i] / scale` (`csrc/eproc_subsample.c`) | `test_d4` **PASS** |
| **W06** | Overflow-safe `eproc_kahan_add`, normal CS finite midpoint guard, scaled `VarOpt` threshold arithmetic (`csrc/eproc_internal.h`, `csrc/eproc_eprocess.c`, `csrc/eproc_subsample.c`) | `test_d2` (all 3 subtests) **PASS** |
| **W09** | Strict finite `scale > 0` check rejecting booleans in `eproc conformal` CLI (`python/eproc/cli.py`) | `test_f6` (clean exit code `2`) **PASS** |
| **W12** | Fresh `stat = exp(min(log_cusum, 700))` on `EDetector.update_batch([])` & `reset_alarm()` cache clear (`python/eproc/eprocess.py`, `csrc/eproc_pyext.c`) | `test_a7` **PASS** |
| **W13, W17** | C++17 `<atomic>` + zero-UB `__atomic_*` plain-`uint64_t` seqlock helpers (`csrc/eproc_seqlock.h`, `csrc/eproc_fusion.c`); unified scalar/batch `exp4_neg50`; `docs/MATHEMATICS.md` MI + B&B proof; scrubbed `COMPANY_STATE.md` | `test_f3` (GCC C11 + Clang C11 + G++ C++17 `-Wall -Wextra -Werror`) **PASS** |
| **W14–W16** | Strict `_require_bool`, JSON-RPC 2.0 `-32600` (`id: null`) on missing-`id` invalid requests, strict `_require_number` rejecting booleans across all 7 MCP tools (`python/eproc/mcp_server.py`) | `test_f4`, `test_f5a`, `test_f5b`, `test_f5c` **PASS** |

**Final Re-Audit Command Result:** `PYTHONPATH=python python3 -m unittest tests/test_breaker.py -v` → **`Ran 42 tests — OK`** (`0` failures, `0` errors).

