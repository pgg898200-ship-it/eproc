/**
 * Comprehensive node:test verification suite for @eproc/wasm.
 *
 * Tests:
 *   1. Runtime loading (Node fs.readFile, sync init, browser Response fallback),
 *      WASM import minimalism, binary size < 80 KiB, allocator coalescing/zero leaks
 *   2. Pure-C erfc(double) accuracy (< 1e-15 max relative error vs glibc libm)
 *   3. Exact mathematical invariants for every class:
 *      EProcess, ConfidenceSequence, EDetector, AgentRace, MultiRateConformal,
 *      DelayedConformal, BudgetConformal, ConformalMonoid, SafeFuser, EchoPool,
 *      VarOptSampler, OnlineRLSSampler, SiftVOISelector
 *   4. Cross-language numerical & bit-exact wire parity against native C11 libeproc
 *
 * License: Apache-2.0
 */

import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  initEproc,
  initEprocSync,
  abiVersion,
  simdFlags,
  heapUsed,
  combineEValues,
  EProcess,
  ConfidenceSequence,
  EDetector,
  AgentRace,
  MultiRateConformal,
  DelayedConformal,
  BudgetConformal,
  ConformalMonoid,
  SafeFuser,
  EchoPool,
  VarOptSampler,
  OnlineRLSSampler,
  SiftVOISelector,
} from "./eproc.mjs";

const REPO_ROOT = fileURLToPath(new URL("..", import.meta.url));
const WASM_PATH = fileURLToPath(new URL("../dist/eproc.wasm", import.meta.url));
const PYTHON_DIR = fileURLToPath(new URL("../python", import.meta.url));

/** Run a short Python snippet against the native C11 eproc build and parse JSON output. */
function runNativeC11(pyCode) {
  const stdout = execFileSync("python3", ["-c", pyCode], {
    cwd: REPO_ROOT,
    env: { ...process.env, PYTHONPATH: PYTHON_DIR },
    encoding: "utf-8",
  });
  return JSON.parse(stdout);
}

/** Deterministic PRNG for reproducible test streams (xorshift32). */
function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296.0;
  };
}

function assertClose(actual, expected, tol = 1e-12, label = "") {
  const diff = Math.abs(actual - expected);
  const scale = Math.max(1.0, Math.abs(expected));
  assert.ok(
    diff <= tol * scale,
    `${label}: expected ${expected}, got ${actual} (diff=${diff}, tol=${tol * scale})`
  );
}

let wasmMod = null;

test("01. Runtime initialization, ABI, imports, binary size & allocator zero-leak", async () => {
  const wasmBytes = readFileSync(WASM_PATH);
  assert.ok(
    wasmBytes.byteLength < 80 * 1024,
    `dist/eproc.wasm must be < 80 KiB, got ${wasmBytes.byteLength} bytes`
  );

  // Verify zero WASI / Emscripten imports: only env.{log, log1p, exp[, tanh]}
  const compiled = new WebAssembly.Module(wasmBytes);
  const imports = WebAssembly.Module.imports(compiled);
  const allowedImports = new Set(["log", "log1p", "exp", "tanh"]);
  for (const imp of imports) {
    assert.equal(imp.module, "env", `Unexpected WASM import module: ${imp.module}.${imp.name}`);
    assert.ok(allowedImports.has(imp.name), `Unexpected WASM import: ${imp.module}.${imp.name}`);
  }

  const exportsList = WebAssembly.Module.exports(compiled);
  const fnExports = exportsList.filter((e) => e.kind === "function");
  assert.ok(fnExports.length >= 80, `Expected >= 80 exported functions, got ${fnExports.length}`);

  // Test browser-style Response initialization path
  const fakeResponse = new Response(wasmBytes, {
    headers: { "content-type": "application/wasm" },
  });
  const fromResp = await initEproc(fakeResponse);
  assert.equal(fromResp.abiVersion(), 1);

  // Test synchronous initialization from Uint8Array
  const fromSync = initEprocSync(wasmBytes);
  assert.equal(fromSync.abiVersion(), 1);

  // Test default Node.js fs.readFile initialization path
  wasmMod = await initEproc();
  assert.equal(abiVersion(), 1);
  assert.equal(simdFlags(), 0);

  // Allocator stress + memory.grow + coalescing zero-leak test
  const baseHeap = heapUsed();
  const objs = [];
  for (let i = 0; i < 12; ++i) {
    // SiftVOISelector is ~270 KiB -> forces multiple WASM memory.grow calls
    objs.push(new SiftVOISelector(64, 8, 1.5, 1.0));
    objs.push(new EchoPool(8, 16, 0.5));
    objs.push(new EProcess(0.5, 0.05));
  }
  assert.ok(heapUsed() > baseHeap + 3_000_000);
  for (const o of objs) {
    o.free();
    o.free(); // idempotent double-free must be safe
  }
  assert.equal(heapUsed(), baseHeap, "All freed structs must return heapUsed to exact baseline");

  // Failed constructor must not leak linear memory
  assert.throws(() => new EProcess({ mu0: 2.0, alpha: 0.05, lower: 0.0, upper: 1.0 }), RangeError);
  assert.equal(heapUsed(), baseHeap, "Failed constructor must free allocated struct");

  // Use-after-free must throw cleanly
  const ep = new EProcess(0.5);
  ep.free();
  assert.throws(() => ep.update(0.6), /freed/);
});

test("02. Pure-C erfc(double) matches glibc libm to < 1e-15 across [-30, +26.5]", () => {
  const erfc = wasmMod.wasmExports.eproc_wasm_erfc;
  assert.equal(typeof erfc, "function");

  // Special IEEE-754 values
  assert.ok(Number.isNaN(erfc(Number.NaN)));
  assert.equal(erfc(0.0), 1.0);
  assert.equal(erfc(-0.0), 1.0);
  assert.equal(erfc(Infinity), 0.0);
  assert.equal(erfc(-Infinity), 2.0);
  assert.equal(erfc(30.0), 0.0);
  assert.equal(erfc(-30.0), 2.0);

  const xs = [];
  for (let x = -10.0; x <= 26.5; x += 0.25) {
    xs.push(Number(x.toFixed(6)));
  }
  xs.push(0.001, 0.123456, 0.4999, 0.5, 0.5001, 1.0, 1.815, 2.5, 3.75, 5.0, 10.0, 15.777, 25.99, 26.4);

  const refs = runNativeC11(`
import json, math
xs = ${JSON.stringify(xs)}
print(json.dumps([math.erfc(x) for x in xs]))
`);

  let maxRel = 0.0;
  for (let i = 0; i < xs.length; ++i) {
    const got = erfc(xs[i]);
    const ref = refs[i];
    if (ref >= 2.2250738585072014e-308) {
      const rel = Math.abs(got - ref) / ref;
      if (rel > maxRel) maxRel = rel;
      assert.ok(rel < 1e-15, `erfc(${xs[i]}): rel=${rel} (got=${got}, ref=${ref})`);
    }
  }
  assert.ok(maxRel < 1e-15, `Max relative error ${maxRel} must be < 1e-15`);
});

test("03. EProcess: Ville rejection, null safety, NaN sanitization, combineEValues & C11 parity", () => {
  // Null stream: mu = 0.50, alternating around 0.50
  const nullEp = new EProcess({ mu0: 0.5, alpha: 0.05, lower: 0.0, upper: 1.0, side: "upper" });
  const rng = makeRng(1337);
  const nullStream = Array.from({ length: 200 }, () => (rng() < 0.5 ? 1.0 : 0.0));
  const nullRes = nullEp.updateBatch(nullStream);
  assert.equal(nullRes.n, 200);
  assert.equal(nullRes.rejected, false);
  assert.ok(nullRes.pValue > 0.05);
  nullEp.free();

  // Alternative stream (+0.25 uplift): must reject via Ville's inequality (eValue >= 1/alpha = 20)
  const rng2 = makeRng(4242);
  const altStream = Array.from({ length: 150 }, () => (rng2() < 0.75 ? 1.0 : 0.0));

  const epScalar = new EProcess(0.5, 0.05, 0.0, 1.0, "upper");
  let lastScalar = null;
  for (const x of altStream) {
    lastScalar = epScalar.update(x);
  }
  // Test NaN/Inf sanitization
  epScalar.update(Number.NaN);
  epScalar.update(Infinity);
  assert.equal(epScalar.n, 150);
  assert.equal(epScalar.nInvalid, 2);
  assert.equal(lastScalar.rejected, true);
  assert.ok(lastScalar.eValue >= 20.0, `Expected eValue >= 20, got ${lastScalar.eValue}`);
  assert.ok(lastScalar.pValue <= 0.05, `Expected pValue <= 0.05, got ${lastScalar.pValue}`);
  assert.ok(epScalar.rejectedAt > 0 && epScalar.rejectedAt <= 150);

  // Batch vs scalar parity
  const epBatch = new EProcess(0.5, 0.05, 0.0, 1.0, "upper");
  const lastBatch = epBatch.updateBatch(Float64Array.from(altStream));
  assert.equal(lastBatch.logE, lastScalar.logE);
  assert.equal(lastBatch.pValue, lastScalar.pValue);
  assert.equal(lastBatch.mean, lastScalar.mean);

  // Two-sided & lower-sided checks
  const epLower = new EProcess({ mu0: 0.7, alpha: 0.05, side: "lower" });
  const lowRes = epLower.updateBatch(Array(80).fill(0.2));
  assert.equal(lowRes.rejected, true);
  epLower.free();

  const epTwo = new EProcess({ mu0: 0.5, alpha: 0.05, side: "two_sided" });
  const twoRes = epTwo.updateBatch(Array(100).fill(0.8));
  assert.equal(twoRes.rejected, true);
  epTwo.free();

  // combineEValues (independent vs Vovk-Wang dependent)
  const ind = combineEValues([1.5, 2.5, -0.5], { independent: true });
  assertClose(ind, 3.5, 1e-14, "combineEValues independent");
  const dep = combineEValues([0.0, Math.log(3.0)], { independent: false });
  assertClose(dep, Math.log(2.0), 1e-14, "combineEValues dependent");

  // Cross-check exact numerical parity against native C11
  const nativeRes = runNativeC11(`
import json
from eproc import EProcess
ep = EProcess(mu0=0.5, alpha=0.05, lower=0.0, upper=1.0, side="upper")
res = ep.update_batch(${JSON.stringify(altStream)})
print(json.dumps({
    "n": res.n, "log_e": res.log_e, "max_log_e": res.max_log_e,
    "e_value": res.e_value, "p_value": res.p_value, "mean": res.mean,
    "rejected": res.rejected, "rejected_at": ep.rejected_at
}))
`);
  assert.equal(lastBatch.n, nativeRes.n);
  assert.equal(lastBatch.rejected, nativeRes.rejected);
  assert.equal(epBatch.rejectedAt, nativeRes.rejected_at);
  assertClose(lastBatch.logE, nativeRes.log_e, 1e-12, "EProcess.logE vs native C11");
  assertClose(lastBatch.maxLogE, nativeRes.max_log_e, 1e-12, "EProcess.maxLogE vs native C11");
  assertClose(lastBatch.eValue, nativeRes.e_value, 1e-12, "EProcess.eValue vs native C11");
  assertClose(lastBatch.pValue, nativeRes.p_value, 1e-12, "EProcess.pValue vs native C11");
  assertClose(lastBatch.mean, nativeRes.mean, 1e-14, "EProcess.mean vs native C11");

  epScalar.free();
  epBatch.free();
});

test("04. ConfidenceSequence: EB & Normal-Mixture monotonicity, coverage & C11 parity", () => {
  const rng = makeRng(2026);
  const data = Array.from({ length: 300 }, () => 0.65 + (rng() - 0.5) * 0.4);

  const cs = ConfidenceSequence.empiricalBernstein(0.05, 0.0, 1.0);
  let prevLow = -Infinity;
  let prevHigh = Infinity;
  let last = null;
  for (const x of data) {
    last = cs.update(x);
    assert.ok(last.low >= prevLow - 1e-15, "Running intersection low must be non-decreasing");
    assert.ok(last.high <= prevHigh + 1e-15, "Running intersection high must be non-increasing");
    assert.ok(last.low <= 0.65 && 0.65 <= last.high, "EB CS must cover true mean 0.65");
    prevLow = last.low;
    prevHigh = last.high;
  }
  assert.ok(last.width < 0.25, `Expected tight CS width < 0.25, got ${last.width}`);

  const csBatch = new ConfidenceSequence({ mode: "eb", alpha: 0.05, lower: 0.0, upper: 1.0 });
  const batchRes = csBatch.updateBatch(data);
  assert.equal(batchRes.low, last.low);
  assert.equal(batchRes.high, last.high);

  // Normal-mixture mode
  const csNorm = ConfidenceSequence.normalMixture(0.05, 0.25, 100.0);
  const normRes = csNorm.updateBatch(data);
  assert.ok(normRes.low <= 0.65 && 0.65 <= normRes.high);
  assert.ok(normRes.width < 0.3);

  // Native C11 parity
  const nativeRes = runNativeC11(`
import json
from eproc import ConfidenceSequence
data = ${JSON.stringify(data)}
eb = ConfidenceSequence(alpha=0.05, lower=0.0, upper=1.0, mode="eb").update_batch(data)
nm = ConfidenceSequence(alpha=0.05, mode="normal", sigma=0.25, v_opt=100.0).update_batch(data)
print(json.dumps({
    "eb": {"low": eb.low, "high": eb.high, "mean": eb.mean, "width": eb.width},
    "nm": {"low": nm.low, "high": nm.high, "mean": nm.mean, "width": nm.width},
}))
`);
  assertClose(last.low, nativeRes.eb.low, 1e-12, "CS EB low vs native C11");
  assertClose(last.high, nativeRes.eb.high, 1e-12, "CS EB high vs native C11");
  assertClose(normRes.low, nativeRes.nm.low, 1e-12, "CS Normal low vs native C11");
  assertClose(normRes.high, nativeRes.nm.high, 1e-12, "CS Normal high vs native C11");

  cs.free();
  csBatch.free();
  csNorm.free();
});

test("05. EDetector: drift changepoint detection, resetAlarm & C11 parity", () => {
  const ed = new EDetector({ mu0: 0.4, alpha: 0.05, lower: 0.0, upper: 1.0, side: "upper" });
  const preShift = Array.from({ length: 40 }, (_, i) => (i % 2 === 0 ? 0.35 : 0.4));
  const preRes = ed.updateBatch(preShift);
  assert.equal(preRes.alarm, false);

  const postShift = Array.from({ length: 60 }, () => 0.85);
  const postRes = ed.updateBatch(postShift);
  assert.equal(postRes.alarm, true);
  assert.ok(ed.nAlarms >= 1);
  assert.ok(ed.alarmAt > 40 && ed.alarmAt <= 100);

  const nativeRes = runNativeC11(`
import json
from eproc import EDetector
ed = EDetector(mu0=0.4, alpha=0.05, lower=0.0, upper=1.0, side="upper")
r = ed.update_batch(${JSON.stringify([...preShift, ...postShift])})
print(json.dumps({"n": r.n, "log_stat": r.log_stat, "alarm": r.alarm, "alarm_at": ed.alarm_at, "n_alarms": ed.n_alarms}))
`);
  assert.equal(postRes.n, nativeRes.n);
  assert.equal(postRes.alarm, nativeRes.alarm);
  assert.equal(ed.alarmAt, nativeRes.alarm_at);
  assert.equal(ed.nAlarms, nativeRes.n_alarms);
  assertClose(postRes.logStat, nativeRes.log_stat, 1e-12, "EDetector.logStat vs native C11");

  ed.resetAlarm();
  assert.equal(ed.logStat, 0.0);
  assert.equal(ed.alarmAt, 0);
  ed.free();
});

test("06. AgentRace: early elimination, LUCB1 suggestArm, named arms & C11 parity", () => {
  const trueMeans = [0.35, 0.5, 0.85, 0.4];
  const names = ["model-a", "model-b", "model-best", "model-d"];
  const race = new AgentRace({ nArms: 4, alpha: 0.05, lower: 0.0, upper: 1.0, names });
  const rng = makeRng(9999);
  const traceArms = [];
  const traceRewards = [];

  for (let step = 0; step < 600 && race.winner === null; ++step) {
    const arm = race.suggestArm();
    const reward = rng() < trueMeans[arm] ? 1.0 : 0.0;
    traceArms.push(arm);
    traceRewards.push(reward);
    race.update(arm, reward);
  }

  const st = race.status();
  assert.equal(st.winner, 2);
  assert.equal(st.winnerName, "model-best");
  assert.equal(race.winner, 2);
  assert.deepEqual(race.activeArms, [2]);

  const bestState = race.armState("model-best");
  assert.equal(bestState.active, true);
  assert.equal(bestState.name, "model-best");
  assert.ok(bestState.mean > 0.75);

  const elimState = race.armState(0);
  assert.equal(elimState.active, false);
  assert.ok(elimState.eliminatedAt > 0);

  // Batch replay & C11 parity
  const raceBatch = new AgentRace(4, 0.05, 0.0, 1.0, 0.0);
  const batchSt = raceBatch.updateBatch(traceArms, traceRewards);
  assert.equal(batchSt.winner, 2);
  assert.equal(batchSt.totalObs, traceArms.length);

  const nativeSt = runNativeC11(`
import json
from eproc import AgentRace
r = AgentRace(n_arms=4, alpha=0.05, lower=0.0, upper=1.0, margin=0.0)
st = r.update_batch(${JSON.stringify(traceArms)}, ${JSON.stringify(traceRewards)})
print(json.dumps({
    "winner": st.winner, "n_active": st.n_active, "total_obs": st.total_obs,
    "leader_low": st.leader_low, "leader_high": st.leader_high, "runner_up_high": st.runner_up_high
}))
`);
  assert.equal(batchSt.winner, nativeSt.winner);
  assert.equal(batchSt.nActive, nativeSt.n_active);
  assert.equal(batchSt.totalObs, nativeSt.total_obs);
  assertClose(batchSt.leaderLow, nativeSt.leader_low, 1e-12, "AgentRace.leaderLow vs native C11");
  assertClose(batchSt.leaderHigh, nativeSt.leader_high, 1e-12, "AgentRace.leaderHigh vs native C11");
  assertClose(batchSt.runnerUpHigh, nativeSt.runner_up_high, 1e-12, "AgentRace.runnerUpHigh vs native C11");

  race.free();
  raceBatch.free();
});

test("07. MultiRateConformal, DelayedConformal, BudgetConformal: zero-crossing & C11 parity", () => {
  const alphas = [0.5, 0.2, 0.1, 0.05, 0.01];
  const cp = new MultiRateConformal({ alphas, windowSize: 64, pidKi: 0.05 });
  const rng = makeRng(777);
  const scores = [];

  // Regime shift + adversarial outliers to stress PAVA isotonic non-crossing
  for (let t = 0; t < 300; ++t) {
    const scale = t < 150 ? 1.0 : 4.5;
    const s = Math.abs((rng() + rng() - 1.0) * scale);
    scores.push(s);
    const meta = cp.update(s);
    const pred = cp.predict(10.0, 1.5);
    for (let r = 1; r < alphas.length; ++r) {
      assert.ok(
        meta.quantiles[r] >= meta.quantiles[r - 1],
        `Quantile crossing at step ${t}, rate ${r}: ${meta.quantiles}`
      );
      assert.ok(
        pred.lows[r] <= pred.lows[r - 1] && pred.highs[r] >= pred.highs[r - 1],
        `Nested interval violation at step ${t}`
      );
    }
  }
  assert.equal(cp.nUpdates, 300);

  // Batch vs scalar parity AND exact native C11 parity (0-ULP scalar exp_neg50)
  const cpBatch = new MultiRateConformal({ alphas, windowSize: 64, pidKi: 0.05 });
  const batchMeta = cpBatch.updateBatch(scores);
  const finalPred = cp.predict(10.0, 2.0);
  const batchPred = cpBatch.predict(10.0, 2.0);
  assert.deepEqual(batchPred.quantiles, finalPred.quantiles);

  const nativeCp = runNativeC11(`
import json
from eproc import MultiRateConformal
cp = MultiRateConformal(alphas=(0.5, 0.2, 0.1, 0.05, 0.01), window_size=64, pid_ki=0.05)
m = cp.update_batch(${JSON.stringify(scores)})
p = cp.predict(10.0, scale=2.0)
print(json.dumps({
    "quantiles": list(m.quantiles),
    "eff_alphas": list(m.eff_alphas),
    "emp_coverage": list(m.emp_coverage),
    "lows": list(p.lows),
    "highs": list(p.highs),
}))
`);
  for (let r = 0; r < alphas.length; ++r) {
    assertClose(batchMeta.quantiles[r], nativeCp.quantiles[r], 1e-12, `conformal q[${r}]`);
    assertClose(batchMeta.effAlphas[r], nativeCp.eff_alphas[r], 1e-12, `conformal effAlpha[${r}]`);
    assertClose(batchMeta.empCoverage[r], nativeCp.emp_coverage[r], 1e-12, `conformal empCov[${r}]`);
    assertClose(batchPred.lows[r], nativeCp.lows[r], 1e-12, `conformal low[${r}]`);
    assertClose(batchPred.highs[r], nativeCp.highs[r], 1e-12, `conformal high[${r}]`);
  }

  // DelayedConformal: strict causality + out-of-order reveals
  const dc = new DelayedConformal({ alphas: [0.2, 0.1, 0.05], windowSize: 32, pidKi: 0.05 });
  const t1 = dc.issue(5.0, 1.0);
  const t2 = dc.issue(7.0, 1.0);
  assert.equal(dc.pendingCount, 2);
  assert.equal(dc.nUpdates, 0);
  // Reveal out-of-order (t2 before t1)
  dc.reveal(t2.ticketId, 8.5);
  assert.equal(dc.pendingCount, 1);
  assert.equal(dc.nUpdates, 1);
  const m1 = dc.reveal(t1.ticketId, 5.4);
  assert.equal(dc.pendingCount, 0);
  assert.equal(dc.nUpdates, 2);
  assert.ok(m1.quantiles[2] >= m1.quantiles[0]);

  // BudgetConformal: right-censored penalty inflates UPB
  const bc = new BudgetConformal({ alpha: 0.1, windowSize: 32, censorPenalty: 1.5 });
  for (let i = 0; i < 20; ++i) {
    bc.update(100.0, 110.0, false);
  }
  const boundBefore = bc.bound(100.0);
  for (let i = 0; i < 15; ++i) {
    bc.update(100.0, 120.0, true); // censored at 120 -> penalized to 180
  }
  const boundAfter = bc.bound(100.0);
  assert.equal(bc.nCompleted, 20);
  assert.equal(bc.nCensored, 15);
  assert.ok(boundAfter > boundBefore, `Censored updates should raise bound (${boundAfter} > ${boundBefore})`);

  cp.free();
  cpBatch.free();
  dc.free();
  bc.free();
});

test("08. ConformalMonoid: EXACT 640-byte wire parity with native C11 + abelian merge & LOO", () => {
  assert.equal(ConformalMonoid.wireSize, 640);

  const scoresA = [0.05, 0.12, 0.34, 0.88, 1.45, 2.7, 5.1, -0.5, 11.2, 0.0];
  const scoresB = [0.22, 0.67, 1.11, 3.33, 7.8, 14.2, -2.5, 0.09, 0.44, 22.0];
  const scoresC = [0.15, 0.99, 2.02, 4.44, 0.77, 1.88];

  const mA = new ConformalMonoid(1.0);
  mA.addBatch(scoresA);
  const mB = new ConformalMonoid(1.0);
  for (const s of scoresB) mB.add(s);
  const mC = new ConformalMonoid(1.0);
  mC.addBatch(scoresC);

  // Commutativity: merge(A, B) wire bytes === merge(B, A) wire bytes (BIT-EXACT!)
  const ab = mA.merge(mB);
  const ba = mB.merge(mA);
  const abBytes = ab.toBytes();
  const baBytes = ba.toBytes();
  assert.equal(abBytes.byteLength, 640);
  assert.deepEqual(abBytes, baBytes, "merge(A, B) and merge(B, A) must be bit-identical");

  // Associativity: merge(merge(A, B), C) === merge(A, merge(B, C)) (0 integer bin drift!)
  const abc1 = ab.merge(mC);
  const bc = mB.merge(mC);
  const abc2 = mA.merge(bc);
  assert.deepEqual(abc1.binCounts, abc2.binCounts);
  assert.equal(abc1.quantile(0.9), abc2.quantile(0.9));

  // Exact Leave-One-Out subtraction: subtract(merge(A, B), A) == B (0.0 quantile drift!)
  const looB = ab.subtract(mA);
  assert.deepEqual(looB.binCounts, mB.binCounts);
  assert.equal(looB.count, mB.count);
  assert.equal(looB.quantile(0.5), mB.quantile(0.5));
  assert.equal(looB.quantile(0.9), mB.quantile(0.9));

  // Wire round-trip & corruption rejection
  const rtAb = ConformalMonoid.fromBytes(abBytes);
  assert.deepEqual(rtAb.toBytes(), abBytes);
  assert.equal(rtAb.quantile(0.9), ab.quantile(0.9));

  const corrupted = abBytes.slice();
  corrupted[100] ^= 0x01;
  assert.throws(() => ConformalMonoid.fromBytes(corrupted), RangeError);
  assert.throws(() => ConformalMonoid.fromBytes(abBytes.subarray(0, 639)), RangeError);

  // EXACT 640-BYTE BIT-FOR-BIT WIRE PARITY WITH NATIVE C11!
  const nativeWire = runNativeC11(`
import json
from eproc import ConformalMonoid
a = ConformalMonoid(1.0); a.add_batch(${JSON.stringify(scoresA)})
b = ConformalMonoid(1.0); b.add_batch(${JSON.stringify(scoresB)})
ab = a.merge(b)
loo = ab.subtract(a)
print(json.dumps({
    "wire_hex": ab.to_bytes().hex(),
    "q50": ab.quantile(0.5),
    "q90": ab.quantile(0.9),
    "loo_q90": loo.quantile(0.9),
}))
`);
  const wasmHex = Buffer.from(abBytes).toString("hex");
  assert.equal(
    wasmHex,
    nativeWire.wire_hex,
    "WASM ConformalMonoid 640-byte wire payload MUST be 100% bit-identical to native C11!"
  );
  assert.equal(ab.quantile(0.5), nativeWire.q50);
  assert.equal(ab.quantile(0.9), nativeWire.q90);
  assert.equal(looB.quantile(0.9), nativeWire.loo_q90);

  // Cross-deserialize native C11 wire bytes directly inside WASM
  const fromNative = ConformalMonoid.fromBytes(Uint8Array.from(Buffer.from(nativeWire.wire_hex, "hex")));
  assert.equal(fromNative.count, scoresA.length + scoresB.length);
  assert.equal(fromNative.quantile(0.9), nativeWire.q90);

  for (const m of [mA, mB, mC, ab, ba, abc1, bc, abc2, looB, rtAb, fromNative]) {
    m.free();
  }
});

test("09. SafeFuser: 1e6 Byzantine quarantine, seqlock snapshot & C11 parity", () => {
  const fuser = new SafeFuser({ nExperts: 8, anchorIdx: 0, alphaDeg: 0.05 });
  const rng = makeRng(2025);
  const trace = [];
  let maxBlastErr = 0.0;
  let lastStep = null;

  for (let t = 0; t < 120; ++t) {
    const yTrue = Math.sin(t * 0.1) * 5.0;
    const preds = [
      yTrue + (rng() - 0.5) * 0.8, // 0: anchor (noisy)
      yTrue + (rng() - 0.5) * 0.1, // 1: strong honest expert
      yTrue + (rng() - 0.5) * 0.15, // 2: good honest expert
      yTrue + (rng() - 0.5) * 0.4,
      yTrue + (rng() - 0.5) * 0.5,
      yTrue + (rng() - 0.5) * 0.6,
      yTrue + (rng() - 0.5) * 0.7,
      t >= 20 ? 1e6 : yTrue + (rng() - 0.5) * 0.2, // 7: Byzantine 1e6 attacker starting at t=20!
    ];
    trace.push([preds, yTrue]);
    lastStep = fuser.update(preds, yTrue);
    if (t >= 20) {
      const err = Math.abs(lastStep.fusedPred - yTrue);
      if (err > maxBlastErr) maxBlastErr = err;
    }
  }

  // Expert 7 MUST be quarantined and blast radius MUST stay tiny (< 2.0 vs 1e6 attack!)
  assert.equal(fuser.quarantined[7], true, "Byzantine 1e6 expert must be quarantined");
  assert.ok(lastStep.nQuarantined >= 1);
  assert.ok(maxBlastErr < 2.0, `Byzantine blast error ${maxBlastErr} must be tiny (< 2.0)`);

  const snap = fuser.snapshot();
  assert.equal(snap.weights.length, 8);
  assert.equal(snap.weights[7], 0.0, "Quarantined expert must have exact 0.0 weight");
  const wSum = snap.weights.reduce((a, b) => a + b, 0.0);
  assertClose(wSum, 1.0, 1e-12, "Snapshot weights must sum to 1");

  // Native C11 parity
  const nativeBoa = runNativeC11(`
import json
from eproc import SafeFuser
f = SafeFuser(n_experts=8, anchor_idx=0, alpha_deg=0.05)
last = None
for preds, y in ${JSON.stringify(trace)}:
    last = f.update(preds, y)
w, fb = f.snapshot()
print(json.dumps({
    "fused_pred": last.fused_pred,
    "anchor_pred": last.anchor_pred,
    "n_quarantined": last.n_quarantined,
    "sup_log_e": last.sup_log_e,
    "deg_log_e": last.deg_log_e,
    "weights": w,
    "fallback_active": fb,
    "quarantined": f.quarantined,
}))
`);
  assert.equal(lastStep.nQuarantined, nativeBoa.n_quarantined);
  assert.equal(snap.fallbackActive, nativeBoa.fallback_active);
  assert.deepEqual(fuser.quarantined, nativeBoa.quarantined);
  assertClose(lastStep.fusedPred, nativeBoa.fused_pred, 1e-12, "SafeFuser.fusedPred vs C11");
  assertClose(lastStep.supLogE, nativeBoa.sup_log_e, 1e-12, "SafeFuser.supLogE vs C11");
  assertClose(lastStep.degLogE, nativeBoa.deg_log_e, 1e-12, "SafeFuser.degLogE vs C11");
  for (let i = 0; i < 8; ++i) {
    assertClose(snap.weights[i], nativeBoa.weights[i], 1e-12, `SafeFuser.weights[${i}] vs C11`);
  }

  fuser.free();
});

test("10. EchoPool: echo deduplication, string/bigint IDs, Beta-Bernoulli reliability & C11 parity", () => {
  const pool = new EchoPool({ nSources: 4, nClaims: 16, priorProb: 0.5 });

  // Submit novel evidence on claim 0, then 5 duplicate echoes from different sources
  const first = pool.submit(0, 0, 9001n, 1.5);
  assert.equal(first.accepted, true);
  assert.equal(first.uniqueCount, 1);
  assert.ok(first.posteriorProb > 0.8);

  for (let src = 1; src < 4; ++src) {
    const echo = pool.submit(src, 0, 9001n, 1.5);
    assert.equal(echo.accepted, false, "Duplicate (claimId, evidenceId) echo must be dropped");
    assert.equal(echo.uniqueCount, 1);
    assert.equal(echo.posteriorProb, first.posteriorProb);
  }
  assert.equal(pool.nAccepted, 1);
  assert.equal(pool.nEchoDropped, 3);

  // String & Uint8Array evidence IDs deduplicate deterministically
  const s1 = pool.submit(0, 1, "report-alpha-42", 0.8);
  const s2 = pool.submit(1, 1, "report-alpha-42", 0.8);
  const s3 = pool.submit(2, 1, new TextEncoder().encode("report-alpha-42"), 0.8);
  assert.equal(s1.accepted, true);
  assert.equal(s2.accepted, false);
  assert.equal(s3.accepted, false);

  // Train Beta-Bernoulli peer reliability: source 3 is a consistent Byzantine liar
  for (let c = 2; c < 12; ++c) {
    pool.submit(0, c, BigInt(100 + c), 2.0); // honest: predicts +
    pool.submit(3, c, BigInt(200 + c), -2.0); // liar: predicts -
    pool.resolve(c, true); // ground truth is +
  }
  assert.ok(pool.sourceReliability(0) > 0.9, "Honest source reliability must stay high");
  assert.equal(pool.sourceReliability(3), 0.0, "Consistent Byzantine liar reliability must shrink to 0");

  // Native C11 parity on numeric uint64 evidence IDs
  const p2 = new EchoPool(4, 16, 0.5);
  const rA = p2.submit(0, 0, 123456789n, 1.25);
  const rB = p2.submit(1, 0, 123456789n, 1.25);
  const rC = p2.submit(2, 0, 987654321n, -0.4);
  p2.resolve(0, true);
  const nativeEcho = runNativeC11(`
import json
from eproc import EchoPool
p = EchoPool(n_sources=4, n_claims=16, prior_prob=0.5)
a = p.submit(0, 0, 123456789, 1.25)
b = p.submit(1, 0, 123456789, 1.25)
c = p.submit(2, 0, 987654321, -0.4)
p.resolve(0, True)
print(json.dumps({
    "a_acc": a.accepted, "b_acc": b.accepted, "c_post": c.posterior_prob,
    "rel0": p.source_reliability(0), "rel2": p.source_reliability(2),
    "n_acc": p.n_accepted, "n_drop": p.n_echo_dropped
}))
`);
  assert.equal(rA.accepted, nativeEcho.a_acc);
  assert.equal(rB.accepted, nativeEcho.b_acc);
  assertClose(rC.posteriorProb, nativeEcho.c_post, 1e-12, "EchoPool posterior vs C11");
  assertClose(p2.sourceReliability(0), nativeEcho.rel0, 1e-14, "EchoPool rel0 vs C11");
  assertClose(p2.sourceReliability(2), nativeEcho.rel2, 1e-14, "EchoPool rel2 vs C11");
  assert.equal(p2.nAccepted, nativeEcho.n_acc);
  assert.equal(p2.nEchoDropped, nativeEcho.n_drop);

  pool.free();
  p2.free();
});

test("11. VarOptSampler & OnlineRLSSampler: unbiased HT estimates & exact C11 PRNG parity", () => {
  // VarOptSampler
  const vo = new VarOptSampler({ k: 32, seed: 42n });
  let trueWeightedSum = 0.0;
  let trueWeight = 0.0;
  const voItems = [];
  for (let i = 0; i < 400; ++i) {
    const w = i === 10 || i === 250 ? 250.0 : 1.0 + (i % 5) * 0.25;
    const v = 2.0 + (i % 7) * 0.5;
    vo.push(i, w, v);
    voItems.push([i, w, v]);
    trueWeightedSum += w * v;
    trueWeight += w;
  }
  assert.equal(vo.count, 32);
  assert.equal(vo.nSeen, 400);
  assert.ok(vo.nHeavy >= 2, "Heavy items (w=250) must stay deterministically in heavy partition");
  const estSum = vo.estimateSum();
  const estMean = vo.estimateMean();
  assert.ok(
    Math.abs(estSum - trueWeightedSum) / trueWeightedSum < 0.15,
    `VarOpt estimateSum ${estSum} should track true sum ${trueWeightedSum}`
  );
  assert.ok(
    Math.abs(estMean - trueWeightedSum / trueWeight) < 0.5,
    `VarOpt estimateMean ${estMean} should track true mean`
  );
  const retained = vo.items();
  assert.equal(retained.length, 32);
  assert.ok(retained.some((it) => it.itemId === 10n));
  assert.ok(retained.some((it) => it.itemId === 250n));

  // OnlineRLSSampler
  const rls = new OnlineRLSSampler({ d: 8, lambda: 1.0, oversample: 2.0, minProb: 0.05, seed: 77n });
  const rlsVecs = [];
  const rlsOuts = [];
  for (let i = 0; i < 60; ++i) {
    const v = new Array(8).fill(0.0);
    v[i % 8] = i < 8 ? 5.0 : 0.3;
    rlsVecs.push(v);
    const out = rls.step(v);
    rlsOuts.push(out);
    assert.ok(out.prob >= 0.05 && out.prob <= 1.0);
    if (out.keep) {
      assertClose(out.htWeight, 1.0 / out.prob, 1e-14, "RLS htWeight == 1/prob");
    } else {
      assert.equal(out.htWeight, 0.0);
    }
  }
  // First 8 orthogonal high-leverage directions must all be kept with prob=1
  for (let i = 0; i < 8; ++i) {
    assert.equal(rlsOuts[i].keep, true);
    assert.equal(rlsOuts[i].prob, 1.0);
  }

  // Exact C11 parity for both VarOpt and OnlineRLS (same deterministic PRNG!)
  const nativeSub = runNativeC11(`
import json
from eproc import VarOptSampler, OnlineRLSSampler
vo = VarOptSampler(k=32, seed=42)
for item_id, w, v in ${JSON.stringify(voItems)}:
    vo.push(item_id, w, v)
rls = OnlineRLSSampler(d=8, lambda_reg=1.0, oversample=2.0, min_prob=0.05, seed=77)
rls_res = [rls.step(x) for x in ${JSON.stringify(rlsVecs)}]
print(json.dumps({
    "vo_sum": vo.estimate_sum(),
    "vo_mean": vo.estimate_mean(),
    "vo_tau": vo.tau,
    "vo_heavy": vo.n_heavy,
    "rls_kept": rls.n_kept,
    "rls_last_lev": rls_res[-1].leverage,
    "rls_last_prob": rls_res[-1].prob,
    "rls_last_ht": rls_res[-1].ht_weight,
}))
`);
  assertClose(estSum, nativeSub.vo_sum, 1e-13, "VarOpt.estimateSum vs C11");
  assertClose(estMean, nativeSub.vo_mean, 1e-13, "VarOpt.estimateMean vs C11");
  assertClose(vo.tau, nativeSub.vo_tau, 1e-13, "VarOpt.tau vs C11");
  assert.equal(vo.nHeavy, nativeSub.vo_heavy);
  assert.equal(rls.nKept, nativeSub.rls_kept);
  assertClose(rlsOuts.at(-1).leverage, nativeSub.rls_last_lev, 1e-13, "RLS leverage vs C11");
  assertClose(rlsOuts.at(-1).prob, nativeSub.rls_last_prob, 1e-13, "RLS prob vs C11");
  assertClose(rlsOuts.at(-1).htWeight, nativeSub.rls_last_ht, 1e-13, "RLS htWeight vs C11");

  vo.free();
  rls.free();
});

test("12. SiftVOISelector: loud distractor avoidance, hot-site drilldown & C11 parity", () => {
  const sift = new SiftVOISelector({ nSites: 64, perSite: 8, signalMean: 1.5, noiseSd: 1.0 });
  assert.equal(sift.nChannels, 512);

  // Make site 0's channels noisy distractors (5x noise)
  for (let j = 0; j < 8; ++j) {
    sift.setNoiseScale(j, 5.0);
  }

  // Initial selection must NOT pick noisy site 0 (high variance -> low EVOI)
  const act0 = sift.selectNext({ costSite: 1.0, costChannel: 1.0, costTicks: 8.0, ticksPerRead: 8 });
  assert.equal(act0.actionName, "site");
  assert.notEqual(act0.targetIdx, 0, "VOI selector must skip loud distractor site 0");
  assert.ok(act0.evoiPerCost > 0.0);

  // Observe strong positive mean on site 17 -> posterior spikes and selector drills into site 17's channels!
  const pBefore = sift.siteProb(17);
  sift.observeSite(17, 1.2);
  const pAfter = sift.siteProb(17);
  assert.ok(pAfter > pBefore && pAfter > 0.5, `Hot site posterior should jump (${pBefore} -> ${pAfter})`);

  const act1 = sift.selectNext(1.0, 1.0, 8.0, 8);
  assert.ok(
    act1.actionType === 1 || act1.actionType === 2,
    `After hot site detection, selector should drill into channel/ticks, got ${act1.actionName}`
  );
  assert.equal(
    Math.floor(act1.targetIdx / 8),
    17,
    `Selected channel ${act1.targetIdx} must belong to hot site 17`
  );

  sift.observeChannel(act1.targetIdx, 1.8, 4);
  assert.ok(sift.channelProb(act1.targetIdx) > 0.9);
  const act2 = sift.selectNext(1.0, 1.0, 8.0, 8);

  // Native C11 parity
  const nativeSift = runNativeC11(`
import json
from eproc import SiftVOISelector
sf = SiftVOISelector(n_sites=64, per_site=8, signal_mean=1.5, noise_sd=1.0)
for j in range(8):
    sf.set_noise_scale(j, 5.0)
a0 = sf.select_next(1.0, 1.0, 8.0, 8)
sf.observe_site(17, 1.2)
a1 = sf.select_next(1.0, 1.0, 8.0, 8)
sf.observe_channel(a1.target_idx, 1.8, 4)
a2 = sf.select_next(1.0, 1.0, 8.0, 8)
print(json.dumps({
    "a0": [a0.action_type, a0.target_idx, a0.evoi_per_cost],
    "a1": [a1.action_type, a1.target_idx, a1.evoi_per_cost],
    "a2": [a2.action_type, a2.target_idx, a2.evoi_per_cost],
}))
`);
  assert.equal(act0.actionType, nativeSift.a0[0]);
  assert.equal(act0.targetIdx, nativeSift.a0[1]);
  assertClose(act0.evoiPerCost, nativeSift.a0[2], 1e-12, "Sift act0 EVOI vs C11");

  assert.equal(act1.actionType, nativeSift.a1[0]);
  assert.equal(act1.targetIdx, nativeSift.a1[1]);
  assertClose(act1.evoiPerCost, nativeSift.a1[2], 1e-12, "Sift act1 EVOI vs C11");

  assert.equal(act2.actionType, nativeSift.a2[0]);
  assert.equal(act2.targetIdx, nativeSift.a2[1]);
  assertClose(act2.evoiPerCost, nativeSift.a2[2], 1e-12, "Sift act2 EVOI vs C11");

  sift.free();
});

test("13. Large >16 KiB batches, browser fetch/MIME fallback, Symbol.dispose & error guards", async () => {
  // Large batch (> 16 KiB scratch buffer -> dynamic alloc/free spill path)
  const baseHeap = heapUsed();
  const nLarge = 5000;
  const bigArr = new Float64Array(nLarge);
  for (let i = 0; i < nLarge; ++i) {
    bigArr[i] = 0.65 + 0.1 * Math.sin(i * 0.05);
  }
  const ep = new EProcess({ mu0: 0.5, alpha: 0.05, lower: 0.0, upper: 1.0 });
  const epRes = ep.updateBatch(bigArr);
  assert.equal(epRes.n, nLarge);
  assert.equal(epRes.rejected, true);
  ep.free();

  const cm = new ConformalMonoid(1.0);
  cm.addBatch(bigArr);
  assert.equal(cm.count, nLarge);
  cm.free();
  assert.equal(heapUsed(), baseHeap, "Large spilled batches must leave 0 leaked bytes");

  // Strict validation & error guards (must not leak memory)
  assert.throws(() => new MultiRateConformal({ alphas: [0.5], windowSize: -5 }), RangeError);
  assert.throws(() => new AgentRace(-1), RangeError);
  assert.throws(() => new SafeFuser(0), RangeError);
  assert.throws(() => new EchoPool(0), RangeError);
  assert.throws(() => new VarOptSampler(0), RangeError);
  assert.throws(() => new OnlineRLSSampler(0), RangeError);
  assert.throws(() => new SiftVOISelector(0), RangeError);

  const m1 = new ConformalMonoid(1.0);
  const m2 = new ConformalMonoid(2.0);
  m1.add(0.5);
  m2.add(0.5);
  assert.throws(() => m1.merge(m2), RangeError);
  assert.throws(() => m1.subtract(m2), RangeError);
  m1.free();
  m2.free();

  const sf = new SafeFuser(4, 0, 0.05);
  assert.throws(() => sf.predict([1, 2, 3]), RangeError);
  assert.throws(() => sf.update([1, 2, 3], 1.0), RangeError);
  sf.free();

  const rls = new OnlineRLSSampler(4);
  assert.throws(() => rls.step([1, 2, 3]), RangeError);
  rls.free();

  const sift = new SiftVOISelector(4, 4);
  assert.throws(() => sift.observeChannel(0, 1.0, -1), RangeError);
  sift.free();

  // Symbol.dispose support
  if (typeof Symbol.dispose === "symbol") {
    const dEp = new EProcess(0.5);
    dEp[Symbol.dispose]();
    assert.throws(() => dEp.query(), /freed/);
  }
  assert.equal(heapUsed(), baseHeap, "Validation & dispose checks must leave 0 leaked bytes");

  // Browser fetch(url) path + fallback when server serves text/plain instead of application/wasm
  const wasmBytes = readFileSync(WASM_PATH);
  const origFetch = globalThis.fetch;
  let fetchCalled = 0;
  try {
    globalThis.fetch = async () => {
      fetchCalled++;
      return new Response(wasmBytes, {
        status: 200,
        headers: { "content-type": "application/octet-stream" },
      });
    };
    const fetchedMod = await initEproc(new URL("https://cdn.example.invalid/eproc.wasm"));
    assert.equal(fetchCalled, 1);
    assert.equal(fetchedMod.abiVersion(), 1);
  } finally {
    globalThis.fetch = origFetch;
  }
});

