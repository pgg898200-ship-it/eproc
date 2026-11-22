/**
 * @eproc/wasm — Zero-dependency ES module wrapper around the C11 WebAssembly core.
 *
 * Runs the real zero-allocation C11 statistical kernels (eproc_eprocess.c,
 * eproc_conformal.c, eproc_fusion.c, eproc_subsample.c) compiled directly to
 * dist/eproc.wasm with zero Emscripten, zero WASI, and zero runtime deps.
 *
 * Works identically in Node.js (fs.readFile) and browsers/edge runtimes (fetch).
 *
 * License: Apache-2.0
 */

export const EPROC_OK = 0;
export const EPROC_ERR_INVALID = -1;
export const EPROC_ERR_CAPACITY = -2;
export const EPROC_ERR_EMPTY = -3;
export const EPROC_ERR_MISMATCH = -4;

export const EPROC_SIDE_UPPER = 0;
export const EPROC_SIDE_LOWER = 1;
export const EPROC_SIDE_TWO_SIDED = 2;

const MAX_U64 = (1n << 64n) - 1n;
const SCRATCH_BYTES = 16384; /* 2048 f64s reusable without per-call alloc */
const EPROC_MAX_RATES = 5;
const EPROC_CMONOID_BINS = 64;

const SIDE_MAP = Object.freeze({
  upper: EPROC_SIDE_UPPER,
  greater: EPROC_SIDE_UPPER,
  lower: EPROC_SIDE_LOWER,
  less: EPROC_SIDE_LOWER,
  two_sided: EPROC_SIDE_TWO_SIDED,
  "two-sided": EPROC_SIDE_TWO_SIDED,
  both: EPROC_SIDE_TWO_SIDED,
});

const SIDE_NAMES = Object.freeze({
  [EPROC_SIDE_UPPER]: "upper",
  [EPROC_SIDE_LOWER]: "lower",
  [EPROC_SIDE_TWO_SIDED]: "two_sided",
});

const SIFT_ACTION_NAMES = Object.freeze({
  0: "site",
  1: "channel",
  2: "ticks",
});

const RC_MESSAGES = Object.freeze({
  [EPROC_ERR_INVALID]: "invalid argument",
  [EPROC_ERR_CAPACITY]: "capacity exceeded",
  [EPROC_ERR_EMPTY]: "empty state",
  [EPROC_ERR_MISMATCH]: "mismatch or checksum error",
});

const WASM_IMPORTS = {
  env: {
    log: Math.log,
    log1p: Math.log1p,
    exp: Math.exp,
    tanh: Math.tanh,
  },
};

const TEXT_ENCODER = typeof TextEncoder === "function" ? new TextEncoder() : null;

const FINALIZER =
  typeof FinalizationRegistry === "function"
    ? new FinalizationRegistry((token) => {
        if (token.ptr !== 0) {
          token.freeFn(token.ptr);
          token.ptr = 0;
        }
      })
    : null;

/** @type {WasmRuntime | null} */
let activeRuntime = null;
/** @type {Promise<any> | null} */
let defaultInitPromise = null;

class WasmRuntime {
  /** @param {WebAssembly.Instance} instance */
  constructor(instance) {
    this.instance = instance;
    this.exports = /** @type {Record<string, any>} */ (instance.exports);
    this.memory = /** @type {WebAssembly.Memory} */ (this.exports.memory);

    this._buf = null;
    this.u8 = null;
    this.i32 = null;
    this.u32 = null;
    this.f64 = null;
    this.dv = null;
    this.refreshViews();

    this.sizes = Object.freeze({
      bet: this.exports.eproc_wasm_sizeof_bet() >>> 0,
      betResult: this.exports.eproc_wasm_sizeof_bet_result() >>> 0,
      cs: this.exports.eproc_wasm_sizeof_cs() >>> 0,
      csResult: this.exports.eproc_wasm_sizeof_cs_result() >>> 0,
      edetect: this.exports.eproc_wasm_sizeof_edetect() >>> 0,
      edetectResult: this.exports.eproc_wasm_sizeof_edetect_result() >>> 0,
      race: this.exports.eproc_wasm_sizeof_race() >>> 0,
      raceStatus: this.exports.eproc_wasm_sizeof_race_status() >>> 0,
      conformal: this.exports.eproc_wasm_sizeof_conformal() >>> 0,
      conformalOut: this.exports.eproc_wasm_sizeof_conformal_out() >>> 0,
      delayedConformal: this.exports.eproc_wasm_sizeof_delayed_conformal() >>> 0,
      budgetConformal: this.exports.eproc_wasm_sizeof_budget_conformal() >>> 0,
      cmonoid: this.exports.eproc_wasm_sizeof_cmonoid() >>> 0,
      boa: this.exports.eproc_wasm_sizeof_boa() >>> 0,
      boaStep: this.exports.eproc_wasm_sizeof_boa_step() >>> 0,
      echopool: this.exports.eproc_wasm_sizeof_echopool() >>> 0,
      echoOut: this.exports.eproc_wasm_sizeof_echo_out() >>> 0,
      varopt: this.exports.eproc_wasm_sizeof_varopt() >>> 0,
      rls: this.exports.eproc_wasm_sizeof_rls() >>> 0,
      rlsOut: this.exports.eproc_wasm_sizeof_rls_out() >>> 0,
      sift: this.exports.eproc_wasm_sizeof_sift() >>> 0,
      siftAction: this.exports.eproc_wasm_sizeof_sift_action() >>> 0,
    });

    this.scratchCap = SCRATCH_BYTES;
    this.scratchPtr = this.alloc(SCRATCH_BYTES);
    this.outPtr = this.alloc(256);
    this._freeFn = (ptr) => this.exports.eproc_wasm_free(ptr);
  }

  refreshViews() {
    if (this._buf !== this.memory.buffer) {
      this._buf = this.memory.buffer;
      this.u8 = new Uint8Array(this._buf);
      this.i32 = new Int32Array(this._buf);
      this.u32 = new Uint32Array(this._buf);
      this.f64 = new Float64Array(this._buf);
      this.dv = new DataView(this._buf);
    }
  }

  /** @param {number} bytes */
  alloc(bytes) {
    const ptr = this.exports.eproc_wasm_alloc(bytes >>> 0) >>> 0;
    if (ptr === 0) {
      throw new RangeError(`eproc WASM allocation failed for ${bytes} bytes`);
    }
    this.refreshViews();
    return ptr;
  }

  /** @param {number} ptr */
  free(ptr) {
    if (ptr !== 0) {
      this.exports.eproc_wasm_free(ptr);
    }
  }

  /**
   * Run fn(ptr) with a guaranteed-aligned scratch buffer of at least `bytes`.
   * Reuses the preallocated scratch block for <= 16 KiB with zero alloc/free.
   * @template T
   * @param {number} bytes
   * @param {(ptr: number) => T} fn
   * @returns {T}
   */
  withScratch(bytes, fn) {
    if (bytes <= this.scratchCap) {
      this.refreshViews();
      return fn(this.scratchPtr);
    }
    const tmp = this.alloc(bytes);
    try {
      return fn(tmp);
    } finally {
      this.free(tmp);
    }
  }

  /** @param {number} ptr */
  readU64(ptr) {
    this.refreshViews();
    return Number(this.dv.getBigUint64(ptr, true));
  }

  /** @param {number} ptr */
  readU64Big(ptr) {
    this.refreshViews();
    return this.dv.getBigUint64(ptr, true);
  }

  /** @param {number} ptr */
  readF64(ptr) {
    this.refreshViews();
    return this.dv.getFloat64(ptr, true);
  }

  /** @param {number} ptr */
  readI32(ptr) {
    this.refreshViews();
    return this.dv.getInt32(ptr, true);
  }

  /** @param {number} ptr */
  readU32(ptr) {
    this.refreshViews();
    return this.dv.getUint32(ptr, true);
  }

  /**
   * Copy an ArrayLike<number> of doubles into WASM linear memory at `ptr`.
   * @param {number} ptr
   * @param {ArrayLike<number>} arr
   */
  writeF64Array(ptr, arr) {
    this.refreshViews();
    const base = ptr >>> 3;
    if (arr instanceof Float64Array) {
      this.f64.set(arr, base);
      return;
    }
    for (let i = 0; i < arr.length; ++i) {
      this.f64[base + i] = Number(arr[i]);
    }
  }

  /**
   * Copy an ArrayLike<number> of uint32s into WASM linear memory at `ptr`.
   * @param {number} ptr
   * @param {ArrayLike<number>} arr
   */
  writeU32Array(ptr, arr) {
    this.refreshViews();
    const base = ptr >>> 2;
    if (arr instanceof Uint32Array) {
      this.u32.set(arr, base);
      return;
    }
    for (let i = 0; i < arr.length; ++i) {
      this.u32[base + i] = Number(arr[i]) >>> 0;
    }
  }
}

function getRuntime() {
  if (activeRuntime === null) {
    throw new Error(
      "eproc WASM runtime not initialized. Call `await initEproc()` before using eproc."
    );
  }
  return activeRuntime;
}

function checkRc(rc, context) {
  if (rc !== EPROC_OK) {
    const detail = RC_MESSAGES[rc] ?? `error code ${rc}`;
    throw new RangeError(`${context}: ${detail} (rc=${rc})`);
  }
}

function parseSide(side) {
  if (typeof side === "number" && Number.isInteger(side)) {
    if (side === 0 || side === 1 || side === 2) {
      return side;
    }
    throw new RangeError(`Invalid side integer ${side}; expected 0, 1, or 2`);
  }
  if (typeof side === "string") {
    const key = side.trim().toLowerCase();
    if (Object.prototype.hasOwnProperty.call(SIDE_MAP, key)) {
      return SIDE_MAP[key];
    }
  }
  throw new RangeError(`Invalid side ${JSON.stringify(side)}; expected 'upper', 'lower', or 'two_sided'`);
}

function toU32(value, name) {
  const n = Number(value);
  if (!Number.isInteger(n) || n < 0 || n > 0xffffffff) {
    throw new RangeError(`${name} must be an integer in [0, 2^32 - 1], got ${value}`);
  }
  return n >>> 0;
}

function toU64BigInt(value, name) {
  if (typeof value === "bigint") {
    if (value < 0n || value > MAX_U64) {
      throw new RangeError(`${name} out of uint64 range: ${value}`);
    }
    return value;
  }
  if (typeof value === "number" && Number.isInteger(value) && value >= 0) {
    return BigInt(value);
  }
  throw new TypeError(`${name} must be a non-negative integer or bigint`);
}

function normalizeEvidenceId(evidenceId) {
  if (typeof evidenceId === "bigint") {
    return toU64BigInt(evidenceId, "evidenceId");
  }
  if (typeof evidenceId === "number") {
    return toU64BigInt(evidenceId, "evidenceId");
  }
  let bytes;
  if (typeof evidenceId === "string") {
    if (TEXT_ENCODER !== null) {
      bytes = TEXT_ENCODER.encode(evidenceId);
    } else {
      bytes = new Uint8Array(evidenceId.length);
      for (let i = 0; i < evidenceId.length; ++i) {
        bytes[i] = evidenceId.charCodeAt(i) & 0xff;
      }
    }
  } else if (evidenceId instanceof Uint8Array) {
    bytes = evidenceId;
  } else if (ArrayBuffer.isView(evidenceId)) {
    bytes = new Uint8Array(evidenceId.buffer, evidenceId.byteOffset, evidenceId.byteLength);
  } else {
    throw new TypeError("evidenceId must be a non-negative integer, bigint, string, or Uint8Array");
  }
  /* Deterministic 64-bit FNV-1a hash over raw UTF-8/byte payload */
  let h = 14695981039346656037n;
  const prime = 1099511628211n;
  for (let i = 0; i < bytes.length; ++i) {
    h ^= BigInt(bytes[i]);
    h = (h * prime) & MAX_U64;
  }
  return h;
}

function isPlainObject(v) {
  return v !== null && typeof v === "object" && !Array.isArray(v) && !ArrayBuffer.isView(v);
}

function isNodeRuntime() {
  return (
    typeof process !== "undefined" &&
    process !== null &&
    typeof process.versions === "object" &&
    typeof process.versions.node === "string"
  );
}

async function loadWasmBytesFromUrlOrPath(source) {
  const isUrlObj = source instanceof URL;
  const str = isUrlObj ? source.href : String(source);
  const isFileUrl = str.startsWith("file://");
  const isHttpUrl = /^https?:\/\//i.test(str) || str.startsWith("data:") || str.startsWith("blob:");

  if (isNodeRuntime() && (isFileUrl || !isHttpUrl)) {
    const { readFile } = await import("node:fs/promises");
    const target = isUrlObj ? source : isFileUrl ? new URL(str) : str;
    return await readFile(target);
  }

  const resp = await fetch(source);
  if (!resp.ok) {
    throw new Error(`Failed to fetch eproc.wasm from ${str}: HTTP ${resp.status}`);
  }
  return resp;
}

async function instantiateSource(source) {
  let src = source;
  if (src === undefined || src === null) {
    const defaultUrl = new URL("../dist/eproc.wasm", import.meta.url);
    src = await loadWasmBytesFromUrlOrPath(defaultUrl);
  } else if (typeof src === "string" || src instanceof URL) {
    src = await loadWasmBytesFromUrlOrPath(src);
  } else if (typeof src.then === "function") {
    src = await src;
  }

  if (src instanceof WebAssembly.Module) {
    return await WebAssembly.instantiate(src, WASM_IMPORTS);
  }
  if (typeof Response !== "undefined" && src instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming === "function") {
      try {
        const res = await WebAssembly.instantiateStreaming(src.clone(), WASM_IMPORTS);
        return res.instance;
      } catch {
        /* Fallback when server omits application/wasm Content-Type */
      }
    }
    const buf = await src.arrayBuffer();
    const res = await WebAssembly.instantiate(buf, WASM_IMPORTS);
    return res.instance;
  }
  if (src instanceof ArrayBuffer || ArrayBuffer.isView(src)) {
    const res = await WebAssembly.instantiate(src, WASM_IMPORTS);
    return res.instance;
  }
  throw new TypeError(
    "initEproc: unsupported wasmSource; pass a URL, path string, ArrayBuffer, Uint8Array, Response, or WebAssembly.Module"
  );
}

function buildModuleNamespace(rt) {
  return {
    EPROC_OK,
    EPROC_ERR_INVALID,
    EPROC_ERR_CAPACITY,
    EPROC_ERR_EMPTY,
    EPROC_ERR_MISMATCH,
    EPROC_SIDE_UPPER,
    EPROC_SIDE_LOWER,
    EPROC_SIDE_TWO_SIDED,
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
    wasmExports: rt.exports,
    memory: rt.memory,
  };
}

/**
 * Initialize the eproc WebAssembly runtime asynchronously.
 *
 * - Node.js: loads `../dist/eproc.wasm` via `node:fs/promises` (`readFile`).
 * - Browser / Edge: loads `../dist/eproc.wasm` via `fetch()` + `WebAssembly.instantiateStreaming`.
 * - Custom source: accepts URL, path string, `ArrayBuffer`, `Uint8Array`, `Response`, or `WebAssembly.Module`.
 *
 * @param {string | URL | ArrayBuffer | ArrayBufferView | Response | WebAssembly.Module} [wasmSource]
 */
export async function initEproc(wasmSource) {
  if (wasmSource === undefined && activeRuntime !== null) {
    return buildModuleNamespace(activeRuntime);
  }
  if (wasmSource === undefined && defaultInitPromise !== null) {
    return defaultInitPromise;
  }
  const promise = (async () => {
    const instance = await instantiateSource(wasmSource);
    activeRuntime = new WasmRuntime(instance);
    return buildModuleNamespace(activeRuntime);
  })();
  if (wasmSource === undefined) {
    defaultInitPromise = promise;
    promise.catch(() => {
      defaultInitPromise = null;
    });
  }
  return promise;
}

/**
 * Synchronously initialize the eproc WebAssembly runtime from raw bytes or a compiled WebAssembly.Module.
 * @param {ArrayBuffer | ArrayBufferView | WebAssembly.Module} wasmSource
 */
export function initEprocSync(wasmSource) {
  const mod =
    wasmSource instanceof WebAssembly.Module
      ? wasmSource
      : new WebAssembly.Module(wasmSource);
  const instance = new WebAssembly.Instance(mod, WASM_IMPORTS);
  activeRuntime = new WasmRuntime(instance);
  defaultInitPromise = Promise.resolve(buildModuleNamespace(activeRuntime));
  return buildModuleNamespace(activeRuntime);
}

/** Return the compiled C11 ABI version (`1`). */
export function abiVersion() {
  return getRuntime().exports.eproc_abi_version() >>> 0;
}

/** Return runtime SIMD capability flags (`0` on scalar WASM). */
export function simdFlags() {
  return getRuntime().exports.eproc_simd_flags() >>> 0;
}

/** Return currently live bytes allocated in the WASM linear-memory heap. */
export function heapUsed() {
  return getRuntime().exports.eproc_wasm_heap_used() >>> 0;
}

/**
 * Combine multiple log e-values into a single valid log e-value.
 * - `independent: true` -> sum of log e-values (product of independent e-values)
 * - `independent: false` (default) -> Vovk & Wang (2021) arithmetic mean `log((1/K) sum_k E_k)`
 *
 * @param {ArrayLike<number>} logEs
 * @param {{ independent?: boolean } | boolean} [options]
 * @returns {number}
 */
export function combineEValues(logEs, options = false) {
  const rt = getRuntime();
  const n = logEs.length >>> 0;
  if (n === 0) {
    throw new RangeError("combineEValues: logEs must not be empty");
  }
  const independent = typeof options === "boolean" ? options : Boolean(options?.independent);
  if (independent) {
    let acc = 0.0;
    for (let i = 0; i < n; ++i) {
      acc = rt.exports.eproc_combine_independent_log(acc, Number(logEs[i]));
    }
    return acc;
  }
  return rt.withScratch(n * 8, (bufPtr) => {
    rt.writeF64Array(bufPtr, logEs);
    return rt.exports.eproc_combine_dependent_log(bufPtr, n);
  });
}

/* ============================================================================
 * INTERNAL STRUCT RESULT DECODERS (LAYOUTS PROVEN BY _Static_assert IN SHIM)
 * ============================================================================ */

function readBetResult(rt, ptr) {
  rt.refreshViews();
  const dv = rt.dv;
  return {
    n: Number(dv.getBigUint64(ptr + 0, true)),
    eValue: dv.getFloat64(ptr + 8, true),
    logE: dv.getFloat64(ptr + 16, true),
    maxLogE: dv.getFloat64(ptr + 24, true),
    pValue: dv.getFloat64(ptr + 32, true),
    mean: dv.getFloat64(ptr + 40, true),
    rejected: dv.getInt32(ptr + 48, true) !== 0,
  };
}

function readCsResult(rt, ptr) {
  rt.refreshViews();
  const dv = rt.dv;
  return {
    n: Number(dv.getBigUint64(ptr + 0, true)),
    mean: dv.getFloat64(ptr + 8, true),
    low: dv.getFloat64(ptr + 16, true),
    high: dv.getFloat64(ptr + 24, true),
    width: dv.getFloat64(ptr + 32, true),
  };
}

function readEDetectResult(rt, ptr) {
  rt.refreshViews();
  const dv = rt.dv;
  return {
    n: Number(dv.getBigUint64(ptr + 0, true)),
    logStat: dv.getFloat64(ptr + 8, true),
    stat: dv.getFloat64(ptr + 16, true),
    alarm: dv.getInt32(ptr + 24, true) !== 0,
  };
}

function readConformalMeta(rt, ptr, alphas) {
  rt.refreshViews();
  const dv = rt.dv;
  const nRates = dv.getUint32(ptr + 0, true);
  const quantiles = new Array(nRates);
  const effAlphas = new Array(nRates);
  const empCoverage = new Array(nRates);
  for (let i = 0; i < nRates; ++i) {
    quantiles[i] = dv.getFloat64(ptr + 8 + i * 8, true);
    effAlphas[i] = dv.getFloat64(ptr + 48 + i * 8, true);
    empCoverage[i] = dv.getFloat64(ptr + 88 + i * 8, true);
  }
  return {
    nRates,
    alphas: alphas.slice(0, nRates),
    quantiles,
    effAlphas,
    empCoverage,
  };
}

/* ============================================================================
 * BASE LIFECYCLE HELPER FOR WASM-BACKED STRUCT INSTANCES
 * ============================================================================ */

class WasmStruct {
  /**
   * @param {WasmRuntime} rt
   * @param {number} size
   * @param {string} typeName
   */
  constructor(rt, size, typeName) {
    this._rt = rt;
    this._typeName = typeName;
    this._ptr = rt.alloc(size);
    this._token = { ptr: this._ptr, freeFn: rt._freeFn };
    if (FINALIZER !== null) {
      FINALIZER.register(this, this._token, this._token);
    }
  }

  _requirePtr() {
    if (this._ptr === 0) {
      throw new Error(`${this._typeName} instance has been freed`);
    }
    return this._ptr;
  }

  /** Deterministically release the underlying WASM linear-memory struct. */
  free() {
    if (this._ptr !== 0) {
      const ptr = this._ptr;
      this._ptr = 0;
      this._token.ptr = 0;
      if (FINALIZER !== null) {
        FINALIZER.unregister(this._token);
      }
      this._rt.free(ptr);
    }
  }
}

if (typeof Symbol.dispose === "symbol") {
  WasmStruct.prototype[Symbol.dispose] = function () {
    this.free();
  };
}

/* ============================================================================
 * 1. EProcess, ConfidenceSequence, EDetector, AgentRace
 * ============================================================================ */

export class EProcess extends WasmStruct {
  /**
   * Anytime-valid ONS betting supermartingale e-process for bounded streams X_t in [lower, upper].
   * Exact Ville inequality: P_{H0}(exists t >= 1: eValue >= 1/alpha) <= alpha.
   */
  constructor(optionsOrMu0, alpha = 0.05, lower = 0.0, upper = 1.0, side = "upper") {
    const opts = isPlainObject(optionsOrMu0)
      ? optionsOrMu0
      : { mu0: optionsOrMu0, alpha, lower, upper, side };
    const mu0Val = Number(opts.mu0);
    const alphaVal = opts.alpha === undefined ? 0.05 : Number(opts.alpha);
    const lowerVal = opts.lower === undefined ? 0.0 : Number(opts.lower);
    const upperVal = opts.upper === undefined ? 1.0 : Number(opts.upper);
    const sideCode = parseSide(opts.side === undefined ? "upper" : opts.side);

    const rt = getRuntime();
    super(rt, rt.sizes.bet, "EProcess");
    const rc = rt.exports.eproc_bet_init(
      this._ptr,
      mu0Val,
      alphaVal,
      lowerVal,
      upperVal,
      sideCode
    );
    if (rc !== EPROC_OK) {
      this.free();
      checkRc(rc, "EProcess");
    }
    this._mu0 = mu0Val;
    this._alpha = alphaVal;
    this._lower = lowerVal;
    this._upper = upperVal;
    this._side = SIDE_NAMES[sideCode];
  }

  /** Ingest one scalar observation and return the updated anytime-valid result. */
  update(x) {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const rc = rt.exports.eproc_bet_update(ptr, Number(x), rt.outPtr);
    checkRc(rc, "EProcess.update");
    return readBetResult(rt, rt.outPtr);
  }

  /** Ingest an array or Float64Array of observations in C/WASM. */
  updateBatch(xs) {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const n = xs.length >>> 0;
    if (n === 0) {
      return this.query();
    }
    return rt.withScratch(n * 8, (bufPtr) => {
      rt.writeF64Array(bufPtr, xs);
      const rc = rt.exports.eproc_bet_update_batch(ptr, bufPtr, n, rt.outPtr);
      checkRc(rc, "EProcess.updateBatch");
      return readBetResult(rt, rt.outPtr);
    });
  }

  /** Query the current anytime-valid e-value, p-value, mean, and verdict. */
  query() {
    const ptr = this._requirePtr();
    const rt = this._rt;
    rt.exports.eproc_bet_query(ptr, rt.outPtr);
    return readBetResult(rt, rt.outPtr);
  }

  result() {
    return this.query();
  }

  get n() { return this._rt.readU64(this._requirePtr() + 152); }
  get rejectedAt() { return this._rt.readU64(this._requirePtr() + 160); }
  get nInvalid() { return this._rt.readU64(this._requirePtr() + 168); }
  get eValue() { return this.query().eValue; }
  get logE() { return this.query().logE; }
  get maxLogE() { return this.query().maxLogE; }
  get pValue() { return this.query().pValue; }
  get mean() { return this.query().mean; }
  get rejected() { return this.query().rejected; }
  get mu0() { return this._mu0; }
  get alpha() { return this._alpha; }
  get lower() { return this._lower; }
  get upper() { return this._upper; }
  get side() { return this._side; }
}

export class ConfidenceSequence extends WasmStruct {
  /**
   * O(1)-time, O(1)-memory anytime-valid confidence sequence.
   * - mode="eb": Closed-form Empirical-Bernstein CS for X_t in [lower, upper]
   * - mode="normal": Sub-Gaussian Normal-Mixture CS parameterized by sigma & vOpt
   */
  constructor(optionsOrAlpha = {}, lower = 0.0, upper = 1.0, mode = "eb", sigma = 1.0, vOpt = 100.0) {
    const opts = isPlainObject(optionsOrAlpha)
      ? optionsOrAlpha
      : { alpha: optionsOrAlpha, lower, upper, mode, sigma, vOpt };
    const alphaVal = opts.alpha === undefined ? 0.05 : Number(opts.alpha);
    const lowerVal = opts.lower === undefined ? 0.0 : Number(opts.lower);
    const upperVal = opts.upper === undefined ? 1.0 : Number(opts.upper);
    const sigmaVal = opts.sigma === undefined ? 1.0 : Number(opts.sigma);
    const vOptVal = opts.vOpt === undefined ? 100.0 : Number(opts.vOpt);
    const rawMode = (opts.mode === undefined ? "eb" : String(opts.mode)).trim().toLowerCase();

    const rt = getRuntime();
    super(rt, rt.sizes.cs, "ConfidenceSequence");
    let rc;
    if (rawMode === "eb" || rawMode === "empirical_bernstein" || rawMode === "empirical-bernstein") {
      this._mode = "eb";
      rc = rt.exports.eproc_cs_init_eb(this._ptr, alphaVal, lowerVal, upperVal);
    } else if (
      rawMode === "normal" ||
      rawMode === "normal_mixture" ||
      rawMode === "normal-mixture" ||
      rawMode === "subgaussian"
    ) {
      this._mode = "normal";
      rc = rt.exports.eproc_cs_init_normal(this._ptr, alphaVal, sigmaVal, vOptVal);
    } else {
      this.free();
      throw new RangeError(`ConfidenceSequence: invalid mode ${JSON.stringify(opts.mode)}; expected 'eb' or 'normal'`);
    }
    if (rc !== EPROC_OK) {
      this.free();
      checkRc(rc, "ConfidenceSequence");
    }
    this._alpha = alphaVal;
  }

  static empiricalBernstein(alpha = 0.05, lower = 0.0, upper = 1.0) {
    return new ConfidenceSequence({ mode: "eb", alpha, lower, upper });
  }

  static normalMixture(alpha = 0.05, sigma = 1.0, vOpt = 100.0) {
    return new ConfidenceSequence({ mode: "normal", alpha, sigma, vOpt });
  }

  update(x) {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const rc = rt.exports.eproc_cs_update(ptr, Number(x), rt.outPtr);
    checkRc(rc, "ConfidenceSequence.update");
    return readCsResult(rt, rt.outPtr);
  }

  updateBatch(xs) {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const n = xs.length >>> 0;
    if (n === 0) {
      return this.query();
    }
    return rt.withScratch(n * 8, (bufPtr) => {
      rt.writeF64Array(bufPtr, xs);
      const rc = rt.exports.eproc_cs_update_batch(ptr, bufPtr, n, rt.outPtr);
      checkRc(rc, "ConfidenceSequence.updateBatch");
      return readCsResult(rt, rt.outPtr);
    });
  }

  query() {
    const ptr = this._requirePtr();
    const rt = this._rt;
    rt.exports.eproc_cs_query(ptr, rt.outPtr);
    return readCsResult(rt, rt.outPtr);
  }

  get n() { return this.query().n; }
  get mean() { return this.query().mean; }
  get low() { return this.query().low; }
  get high() { return this.query().high; }
  get width() { return this.query().width; }
  get bounds() {
    const r = this.query();
    return [r.low, r.high];
  }
  get alpha() { return this._alpha; }
  get mode() { return this._mode; }
}

export class EDetector extends WasmStruct {
  /**
   * Anytime-valid Page-CUSUM resettable e-detector for streaming drift detection.
   */
  constructor(optionsOrMu0, alpha = 0.05, lower = 0.0, upper = 1.0, side = "upper") {
    const opts = isPlainObject(optionsOrMu0)
      ? optionsOrMu0
      : { mu0: optionsOrMu0, alpha, lower, upper, side };
    const mu0Val = Number(opts.mu0);
    const alphaVal = opts.alpha === undefined ? 0.05 : Number(opts.alpha);
    const lowerVal = opts.lower === undefined ? 0.0 : Number(opts.lower);
    const upperVal = opts.upper === undefined ? 1.0 : Number(opts.upper);
    const sideCode = parseSide(opts.side === undefined ? "upper" : opts.side);

    const rt = getRuntime();
    super(rt, rt.sizes.edetect, "EDetector");
    const rc = rt.exports.eproc_edetect_init(
      this._ptr,
      mu0Val,
      alphaVal,
      lowerVal,
      upperVal,
      sideCode
    );
    if (rc !== EPROC_OK) {
      this.free();
      checkRc(rc, "EDetector");
    }
    this._mu0 = mu0Val;
    this._side = SIDE_NAMES[sideCode];
  }

  update(x) {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const rc = rt.exports.eproc_edetect_update(ptr, Number(x), rt.outPtr);
    checkRc(rc, "EDetector.update");
    return readEDetectResult(rt, rt.outPtr);
  }

  updateBatch(xs) {
    this._requirePtr();
    const n = xs.length >>> 0;
    if (n === 0) {
      const logStat = this.logStat;
      const logThresh = this._rt.readF64(this._ptr + 192);
      return {
        n: this.n,
        logStat,
        stat: Math.exp(Math.min(logStat, 700.0)),
        alarm: logStat >= logThresh,
      };
    }
    let last = null;
    for (let i = 0; i < n; ++i) {
      last = this.update(xs[i]);
    }
    return last;
  }

  resetAlarm() {
    const ptr = this._requirePtr();
    this._rt.exports.eproc_edetect_reset_alarm(ptr);
  }

  get logStat() { return this._rt.readF64(this._requirePtr() + 176); }
  get stat() { return Math.exp(Math.min(this.logStat, 700.0)); }
  get alarmAt() { return this._rt.readU64(this._requirePtr() + 200); }
  get nAlarms() { return this._rt.readU64(this._requirePtr() + 208); }
  get n() { return this._rt.readU64(this._requirePtr() + 216); }
  get mu0() { return this._mu0; }
  get side() { return this._side; }
}

export class AgentRace extends WasmStruct {
  /**
   * Anytime-valid multi-arm agent/model racer with automatic early elimination and exact FWER control.
   */
  constructor(optionsOrNArms, alpha = 0.05, lower = 0.0, upper = 1.0, margin = 0.0) {
    const opts = isPlainObject(optionsOrNArms)
      ? optionsOrNArms
      : { nArms: optionsOrNArms, alpha, lower, upper, margin };
    const nArmsVal = toU32(opts.nArms, "nArms");
    const alphaVal = opts.alpha === undefined ? 0.05 : Number(opts.alpha);
    const lowerVal = opts.lower === undefined ? 0.0 : Number(opts.lower);
    const upperVal = opts.upper === undefined ? 1.0 : Number(opts.upper);
    const marginVal = opts.margin === undefined ? 0.0 : Number(opts.margin);

    let names = null;
    let nameToIdx = null;
    if (opts.names !== undefined && opts.names !== null) {
      names = Array.from(opts.names, String);
      if (names.length !== nArmsVal) {
        throw new RangeError(`AgentRace: names.length (${names.length}) must equal nArms (${nArmsVal})`);
      }
      nameToIdx = new Map();
      for (let i = 0; i < names.length; ++i) {
        if (nameToIdx.has(names[i])) {
          throw new RangeError(`AgentRace: duplicate arm name ${JSON.stringify(names[i])}`);
        }
        nameToIdx.set(names[i], i);
      }
    }

    const rt = getRuntime();
    super(rt, rt.sizes.race, "AgentRace");
    const rc = rt.exports.eproc_race_init(
      this._ptr,
      nArmsVal,
      alphaVal,
      lowerVal,
      upperVal,
      marginVal
    );
    if (rc !== EPROC_OK) {
      this.free();
      checkRc(rc, "AgentRace");
    }
    this._nArms = nArmsVal;
    this._names = names ? Object.freeze(names) : null;
    this._nameToIdx = nameToIdx;
  }

  _resolveArm(arm) {
    if (typeof arm === "string") {
      if (this._nameToIdx === null || !this._nameToIdx.has(arm)) {
        throw new RangeError(`AgentRace: unknown arm name ${JSON.stringify(arm)}`);
      }
      return this._nameToIdx.get(arm);
    }
    const idx = Number(arm);
    if (!Number.isInteger(idx) || idx < 0 || idx >= this._nArms) {
      throw new RangeError(`AgentRace: arm index ${arm} out of range [0, ${this._nArms})`);
    }
    return idx;
  }

  _readStatus(ptr) {
    const rt = this._rt;
    rt.refreshViews();
    const dv = rt.dv;
    const rawWinner = dv.getInt32(ptr + 8, true);
    const rawLeader = dv.getInt32(ptr + 12, true);
    const winner = rawWinner < 0 ? null : rawWinner;
    const leader = rawLeader < 0 ? null : rawLeader;
    return {
      nArms: dv.getUint32(ptr + 0, true),
      nActive: dv.getUint32(ptr + 4, true),
      winner,
      leader,
      totalObs: Number(dv.getBigUint64(ptr + 16, true)),
      leaderLow: dv.getFloat64(ptr + 24, true),
      leaderHigh: dv.getFloat64(ptr + 32, true),
      runnerUpHigh: dv.getFloat64(ptr + 40, true),
      winnerName: this._names && winner !== null ? this._names[winner] : null,
      leaderName: this._names && leader !== null ? this._names[leader] : null,
    };
  }

  update(arm, reward) {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const idx = this._resolveArm(arm);
    const rc = rt.exports.eproc_race_update(ptr, idx >>> 0, Number(reward), rt.outPtr);
    checkRc(rc, "AgentRace.update");
    return this._readStatus(rt.outPtr);
  }

  updateBatch(arms, rewards) {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const n = arms.length >>> 0;
    if (n !== (rewards.length >>> 0)) {
      throw new RangeError(`AgentRace.updateBatch: arms length (${n}) !== rewards length (${rewards.length})`);
    }
    if (n === 0) {
      return this.status();
    }
    const armIds = new Uint32Array(n);
    for (let i = 0; i < n; ++i) {
      armIds[i] = this._resolveArm(arms[i]);
    }
    /* Align rewards to 8-byte boundary right after armIds */
    const armsBytes = ((n * 4 + 7) & ~7) >>> 0;
    const totalBytes = armsBytes + n * 8;
    return rt.withScratch(totalBytes, (bufPtr) => {
      const rewPtr = bufPtr + armsBytes;
      rt.writeU32Array(bufPtr, armIds);
      rt.writeF64Array(rewPtr, rewards);
      const rc = rt.exports.eproc_race_update_batch(ptr, bufPtr, rewPtr, n, rt.outPtr);
      checkRc(rc, "AgentRace.updateBatch");
      return this._readStatus(rt.outPtr);
    });
  }

  suggestArm() {
    const ptr = this._requirePtr();
    return this._rt.exports.eproc_race_suggest_arm(ptr);
  }

  status() {
    const ptr = this._requirePtr();
    const rt = this._rt;
    rt.exports.eproc_race_status(ptr, rt.outPtr);
    return this._readStatus(rt.outPtr);
  }

  armState(arm) {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const idx = this._resolveArm(arm);
    const activeOutPtr = rt.outPtr + 64;
    const rc = rt.exports.eproc_race_arm_state(ptr, idx >>> 0, rt.outPtr, activeOutPtr);
    checkRc(rc, "AgentRace.armState");
    const cs = readCsResult(rt, rt.outPtr);
    const active = rt.readI32(activeOutPtr) !== 0;
    const eliminatedAt = rt.readU64(ptr + 120 + idx * 8);
    return {
      arm: idx,
      name: this._names ? this._names[idx] : null,
      active,
      n: cs.n,
      mean: cs.mean,
      low: cs.low,
      high: cs.high,
      width: cs.width,
      eliminatedAt,
    };
  }

  get nArms() { return this._nArms; }
  get winner() {
    const w = this._rt.readI32(this._requirePtr() + 40);
    return w < 0 ? null : w;
  }
  get winnerName() {
    const w = this.winner;
    return this._names && w !== null ? this._names[w] : null;
  }
  get activeArms() {
    const ptr = this._requirePtr();
    const rt = this._rt;
    rt.refreshViews();
    const out = [];
    for (let i = 0; i < this._nArms; ++i) {
      if (rt.u8[ptr + 56 + i] !== 0) {
        out.push(i);
      }
    }
    return out;
  }
  get names() { return this._names; }
}

/* ============================================================================
 * 2. MultiRateConformal, DelayedConformal, BudgetConformal, ConformalMonoid
 * ============================================================================ */

function prepareAlphas(alphas) {
  const arr = Array.from(alphas, Number);
  if (arr.length < 1 || arr.length > EPROC_MAX_RATES) {
    throw new RangeError(`alphas length must be in [1, ${EPROC_MAX_RATES}], got ${arr.length}`);
  }
  for (let i = 0; i < arr.length; ++i) {
    if (!Number.isFinite(arr[i]) || !(arr[i] > 0 && arr[i] < 1)) {
      throw new RangeError(`each alpha must be finite in (0, 1), got ${arr[i]}`);
    }
  }
  arr.sort((a, b) => b - a);
  return arr;
}

export class MultiRateConformal extends WasmStruct {
  /**
   * Multi-rate crossing-free AgACI + Conformal PID calibrator (up to 5 rates).
   */
  constructor(options = {}) {
    const opts = isPlainObject(options) ? options : { alphas: options };
    const alphasInput = opts.alphas === undefined ? [0.5, 0.2, 0.1, 0.05, 0.01] : opts.alphas;
    const sortedAlphas = prepareAlphas(alphasInput);
    const windowSize = opts.windowSize === undefined ? 128 : toU32(opts.windowSize, "windowSize");
    const pidKi = opts.pidKi === undefined ? 0.05 : Number(opts.pidKi);

    const rt = getRuntime();
    super(rt, rt.sizes.conformal, "MultiRateConformal");
    const rc = rt.withScratch(sortedAlphas.length * 8, (aPtr) => {
      rt.writeF64Array(aPtr, sortedAlphas);
      return rt.exports.eproc_conformal_init(
        this._ptr,
        aPtr,
        sortedAlphas.length >>> 0,
        windowSize,
        pidKi
      );
    });
    if (rc !== EPROC_OK) {
      this.free();
      checkRc(rc, "MultiRateConformal");
    }
    this._alphas = Object.freeze(sortedAlphas);
  }

  predict(pointPred, scale = 1.0) {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const p = Number(pointPred);
    const s = Number(scale);
    const k = this._alphas.length;
    /* Layout inside scratch: lows[5] @ 0..40, highs[5] @ 40..80 */
    const lowsPtr = rt.scratchPtr;
    const highsPtr = rt.scratchPtr + 40;
    rt.exports.eproc_conformal_predict(ptr, p, s, lowsPtr, highsPtr, rt.outPtr);
    const meta = readConformalMeta(rt, rt.outPtr, this._alphas);
    const lows = new Array(k);
    const highs = new Array(k);
    for (let i = 0; i < k; ++i) {
      lows[i] = rt.dv.getFloat64(lowsPtr + i * 8, true);
      highs[i] = rt.dv.getFloat64(highsPtr + i * 8, true);
    }
    return {
      pointPred: p,
      scale: s,
      alphas: meta.alphas,
      lows,
      highs,
      quantiles: meta.quantiles,
      effAlphas: meta.effAlphas,
      empCoverage: meta.empCoverage,
    };
  }

  update(score) {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const rc = rt.exports.eproc_conformal_update(ptr, Number(score), rt.outPtr);
    checkRc(rc, "MultiRateConformal.update");
    return readConformalMeta(rt, rt.outPtr, this._alphas);
  }

  updateBatch(scores) {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const n = scores.length >>> 0;
    if (n === 0) {
      const p = this.predict(0.0, 1.0);
      return {
        nRates: this._alphas.length,
        alphas: p.alphas,
        quantiles: p.quantiles,
        effAlphas: p.effAlphas,
        empCoverage: p.empCoverage,
      };
    }
    return rt.withScratch(n * 8, (bufPtr) => {
      rt.writeF64Array(bufPtr, scores);
      const rc = rt.exports.eproc_conformal_update_batch(ptr, bufPtr, n, rt.outPtr);
      checkRc(rc, "MultiRateConformal.updateBatch");
      return readConformalMeta(rt, rt.outPtr, this._alphas);
    });
  }

  get alphas() { return this._alphas; }
  get nRates() { return this._alphas.length; }
  get nUpdates() { return this._rt.readU64(this._requirePtr() + 32); }
  get quantiles() {
    const ptr = this._requirePtr();
    const rt = this._rt;
    rt.refreshViews();
    const out = new Array(this._alphas.length);
    for (let i = 0; i < this._alphas.length; ++i) {
      out[i] = rt.dv.getFloat64(ptr + 672 + i * 8, true);
    }
    return out;
  }
}

export class DelayedConformal extends WasmStruct {
  /**
   * Strictly causal delayed-feedback multi-rate conformal wrapper.
   */
  constructor(options = {}) {
    const opts = isPlainObject(options) ? options : { alphas: options };
    const alphasInput = opts.alphas === undefined ? [0.2, 0.1, 0.05] : opts.alphas;
    const sortedAlphas = prepareAlphas(alphasInput);
    const windowSize = opts.windowSize === undefined ? 128 : toU32(opts.windowSize, "windowSize");
    const pidKi = opts.pidKi === undefined ? 0.05 : Number(opts.pidKi);

    const rt = getRuntime();
    super(rt, rt.sizes.delayedConformal, "DelayedConformal");
    const rc = rt.withScratch(sortedAlphas.length * 8, (aPtr) => {
      rt.writeF64Array(aPtr, sortedAlphas);
      return rt.exports.eproc_delayed_conformal_init(
        this._ptr,
        aPtr,
        sortedAlphas.length >>> 0,
        windowSize,
        pidKi
      );
    });
    if (rc !== EPROC_OK) {
      this.free();
      checkRc(rc, "DelayedConformal");
    }
    this._alphas = Object.freeze(sortedAlphas);
  }

  issue(pred, scale = 1.0) {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const k = this._alphas.length;
    const ticketPtr = rt.scratchPtr;
    const lowsPtr = rt.scratchPtr + 16;
    const highsPtr = rt.scratchPtr + 64;
    const rc = rt.exports.eproc_delayed_conformal_issue(
      ptr,
      Number(pred),
      Number(scale),
      ticketPtr,
      lowsPtr,
      highsPtr
    );
    checkRc(rc, "DelayedConformal.issue");
    rt.refreshViews();
    const ticketId = rt.dv.getBigUint64(ticketPtr, true);
    const lows = new Array(k);
    const highs = new Array(k);
    const intervals = new Array(k);
    for (let i = 0; i < k; ++i) {
      const lo = rt.dv.getFloat64(lowsPtr + i * 8, true);
      const hi = rt.dv.getFloat64(highsPtr + i * 8, true);
      lows[i] = lo;
      highs[i] = hi;
      intervals[i] = [lo, hi];
    }
    return { ticketId, lows, highs, intervals };
  }

  reveal(ticketId, yTrue) {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const tid = toU64BigInt(ticketId, "ticketId");
    const rc = rt.exports.eproc_delayed_conformal_reveal(ptr, tid, Number(yTrue), rt.outPtr);
    checkRc(rc, "DelayedConformal.reveal");
    return readConformalMeta(rt, rt.outPtr, this._alphas);
  }

  get pendingCount() {
    return this._rt.readU32(this._requirePtr() + this._rt.sizes.conformal);
  }
  get alphas() { return this._alphas; }
  get nRates() { return this._alphas.length; }
  get nUpdates() { return this._rt.readU64(this._requirePtr() + 32); }
}

export class BudgetConformal extends WasmStruct {
  /**
   * Right-censored agent budget conformal calibrator (PROBE-style UPB).
   */
  constructor(options = {}) {
    const opts = isPlainObject(options) ? options : { alpha: options };
    const alphaVal = opts.alpha === undefined ? 0.1 : Number(opts.alpha);
    const windowSize = opts.windowSize === undefined ? 128 : toU32(opts.windowSize, "windowSize");
    const censorPenalty = opts.censorPenalty === undefined ? 1.25 : Number(opts.censorPenalty);

    const rt = getRuntime();
    super(rt, rt.sizes.budgetConformal, "BudgetConformal");
    const rc = rt.exports.eproc_budget_conformal_init(
      this._ptr,
      alphaVal,
      windowSize,
      censorPenalty
    );
    if (rc !== EPROC_OK) {
      this.free();
      checkRc(rc, "BudgetConformal");
    }
  }

  bound(baseEstimate) {
    return this._rt.exports.eproc_budget_conformal_bound(
      this._requirePtr(),
      Number(baseEstimate)
    );
  }

  update(baseEstimate, observedCost, censored = false) {
    const rc = this._rt.exports.eproc_budget_conformal_update(
      this._requirePtr(),
      Number(baseEstimate),
      Number(observedCost),
      censored ? 1 : 0
    );
    checkRc(rc, "BudgetConformal.update");
  }

  get nCompleted() {
    return this._rt.readU64(this._requirePtr() + this._rt.sizes.conformal + 8);
  }
  get nCensored() {
    return this._rt.readU64(this._requirePtr() + this._rt.sizes.conformal + 16);
  }
}

export class ConformalMonoid extends WasmStruct {
  /**
   * Fixed 640-byte associative, commutative, exact leave-one-out distributed conformal monoid.
   */
  constructor(scaleHint = 1.0, _skipInit = false) {
    const rt = getRuntime();
    super(rt, rt.sizes.cmonoid, "ConformalMonoid");
    if (!_skipInit) {
      rt.exports.eproc_cmonoid_init(this._ptr, Number(scaleHint));
    }
  }

  static get wireSize() {
    return getRuntime().exports.eproc_cmonoid_wire_size() >>> 0;
  }

  add(score) {
    this._rt.exports.eproc_cmonoid_add(this._requirePtr(), Number(score));
  }

  addBatch(scores) {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const n = scores.length >>> 0;
    if (n === 0) {
      return;
    }
    rt.withScratch(n * 8, (bufPtr) => {
      rt.writeF64Array(bufPtr, scores);
      rt.exports.eproc_cmonoid_add_batch(ptr, bufPtr, n);
    });
  }

  merge(other) {
    if (!(other instanceof ConformalMonoid)) {
      throw new TypeError("ConformalMonoid.merge: expected a ConformalMonoid instance");
    }
    const ptrA = this._requirePtr();
    const ptrB = other._requirePtr();
    if (this.count > 0 && other.count > 0) {
      const s1 = this.scaleHint;
      const s2 = other.scaleHint;
      const tol = 1e-12 * Math.max(1.0, Math.abs(s1), Math.abs(s2));
      if (Math.abs(s1 - s2) > tol) {
        throw new RangeError(`ConformalMonoid.merge: incompatible scaleHint (${s1} vs ${s2})`);
      }
    }
    const out = new ConformalMonoid(1.0, true);
    this._rt.exports.eproc_cmonoid_merge(ptrA, ptrB, out._ptr);
    return out;
  }

  subtract(local) {
    if (!(local instanceof ConformalMonoid)) {
      throw new TypeError("ConformalMonoid.subtract: expected a ConformalMonoid instance");
    }
    const ptrG = this._requirePtr();
    const ptrL = local._requirePtr();
    const out = new ConformalMonoid(1.0, true);
    const rc = this._rt.exports.eproc_cmonoid_subtract(ptrG, ptrL, out._ptr);
    if (rc !== EPROC_OK) {
      out.free();
      checkRc(rc, "ConformalMonoid.subtract");
    }
    return out;
  }

  quantile(q) {
    const qf = Number(q);
    if (!Number.isFinite(qf) || qf < 0.0 || qf > 1.0) {
      throw new RangeError(`ConformalMonoid.quantile: q must be in [0, 1], got ${q}`);
    }
    return this._rt.exports.eproc_cmonoid_quantile(this._requirePtr(), qf);
  }

  /** Serialize to the deterministic 640-byte wire payload (exact C11 little-endian + FNV-1a). */
  toBytes() {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const wireSz = rt.exports.eproc_cmonoid_wire_size() >>> 0;
    return rt.withScratch(wireSz, (bufPtr) => {
      const rc = rt.exports.eproc_cmonoid_serialize(ptr, bufPtr, wireSz);
      checkRc(rc, "ConformalMonoid.toBytes");
      rt.refreshViews();
      return rt.u8.slice(bufPtr, bufPtr + wireSz);
    });
  }

  /** Deserialize from an exact 640-byte wire payload. */
  static fromBytes(data) {
    const rt = getRuntime();
    let bytes;
    if (data instanceof Uint8Array) {
      bytes = data;
    } else if (data instanceof ArrayBuffer) {
      bytes = new Uint8Array(data);
    } else if (ArrayBuffer.isView(data)) {
      bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    } else {
      throw new TypeError("ConformalMonoid.fromBytes: expected Uint8Array, ArrayBufferView, or ArrayBuffer");
    }
    const wireSz = rt.exports.eproc_cmonoid_wire_size() >>> 0;
    if (bytes.byteLength !== wireSz) {
      throw new RangeError(
        `ConformalMonoid.fromBytes: expected exactly ${wireSz} bytes, got ${bytes.byteLength}`
      );
    }
    const out = new ConformalMonoid(1.0, true);
    const rc = rt.withScratch(wireSz, (bufPtr) => {
      rt.refreshViews();
      rt.u8.set(bytes, bufPtr);
      return rt.exports.eproc_cmonoid_deserialize(out._ptr, bufPtr, wireSz);
    });
    if (rc !== EPROC_OK) {
      out.free();
      checkRc(rc, "ConformalMonoid.fromBytes");
    }
    return out;
  }

  get count() { return this._rt.readU64(this._requirePtr() + 0); }
  get mean() {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const c = rt.readU64(ptr + 0);
    if (c === 0) return 0.0;
    const s = rt.readF64(ptr + 520);
    if (!Number.isFinite(s)) return s;
    const comp = rt.readF64(ptr + 528);
    return (s + comp) / c;
  }
  get minVal() { return this._rt.readF64(this._requirePtr() + 552); }
  get maxVal() { return this._rt.readF64(this._requirePtr() + 560); }
  get scaleHint() { return this._rt.readF64(this._requirePtr() + 576); }
  get binCounts() {
    const ptr = this._requirePtr();
    const rt = this._rt;
    rt.refreshViews();
    const out = new Array(EPROC_CMONOID_BINS);
    for (let i = 0; i < EPROC_CMONOID_BINS; ++i) {
      out[i] = Number(rt.dv.getBigUint64(ptr + 8 + i * 8, true));
    }
    return out;
  }
}

/* ============================================================================
 * 3. SafeFuser (BOA + 3-Gate Ville + Seqlock) & EchoPool
 * ============================================================================ */

export class SafeFuser extends WasmStruct {
  /**
   * Scale-Free Bernstein Online Aggregation (BOA) with 3-Gate Ville E-Process
   * Non-Inferiority Fallback, Byzantine quarantine, and Boehm-2012 seqlock snapshots.
   */
  constructor(optionsOrNExperts, anchorIdx = 0, alphaDeg = 0.05) {
    const opts = isPlainObject(optionsOrNExperts)
      ? optionsOrNExperts
      : { nExperts: optionsOrNExperts, anchorIdx, alphaDeg };
    const nExpertsVal = toU32(opts.nExperts, "nExperts");
    const anchorVal = opts.anchorIdx === undefined ? 0 : toU32(opts.anchorIdx, "anchorIdx");
    const alphaDegVal = opts.alphaDeg === undefined ? 0.05 : Number(opts.alphaDeg);

    const rt = getRuntime();
    super(rt, rt.sizes.boa, "SafeFuser");
    const rc = rt.exports.eproc_boa_init(
      this._ptr,
      nExpertsVal,
      anchorVal,
      alphaDegVal
    );
    if (rc !== EPROC_OK) {
      this.free();
      checkRc(rc, "SafeFuser");
    }
    this._nExperts = nExpertsVal;
    this._anchorIdx = anchorVal;
  }

  _checkPreds(expertPreds, method) {
    if (expertPreds.length !== this._nExperts) {
      throw new RangeError(
        `SafeFuser.${method}: expected ${this._nExperts} expert predictions, got ${expertPreds.length}`
      );
    }
  }

  predict(expertPreds) {
    const ptr = this._requirePtr();
    this._checkPreds(expertPreds, "predict");
    const rt = this._rt;
    return rt.withScratch(this._nExperts * 8, (bufPtr) => {
      rt.writeF64Array(bufPtr, expertPreds);
      return rt.exports.eproc_boa_predict(ptr, bufPtr);
    });
  }

  update(expertPreds, yTrue) {
    const ptr = this._requirePtr();
    this._checkPreds(expertPreds, "update");
    const rt = this._rt;
    return rt.withScratch(this._nExperts * 8, (bufPtr) => {
      rt.writeF64Array(bufPtr, expertPreds);
      const rc = rt.exports.eproc_boa_update(ptr, bufPtr, Number(yTrue), rt.outPtr);
      checkRc(rc, "SafeFuser.update");
      rt.refreshViews();
      const dv = rt.dv;
      return {
        fusedPred: dv.getFloat64(rt.outPtr + 0, true),
        anchorPred: dv.getFloat64(rt.outPtr + 8, true),
        fallbackActive: dv.getInt32(rt.outPtr + 16, true) !== 0,
        nQuarantined: dv.getUint32(rt.outPtr + 20, true),
        supLogE: dv.getFloat64(rt.outPtr + 24, true),
        degLogE: dv.getFloat64(rt.outPtr + 32, true),
        nUpdates: Number(dv.getBigUint64(rt.outPtr + 40, true)),
      };
    });
  }

  snapshot() {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const k = this._nExperts;
    const wPtr = rt.scratchPtr;
    const fbPtr = rt.outPtr;
    const rc = rt.exports.eproc_boa_snapshot_weights(ptr, wPtr, k >>> 0, fbPtr);
    checkRc(rc, "SafeFuser.snapshot");
    rt.refreshViews();
    const weights = new Array(k);
    for (let i = 0; i < k; ++i) {
      weights[i] = rt.dv.getFloat64(wPtr + i * 8, true);
    }
    const fallbackActive = rt.dv.getInt32(fbPtr, true) !== 0;
    return { weights, fallbackActive };
  }

  weights() {
    return this.snapshot().weights;
  }

  get nExperts() { return this._nExperts; }
  get anchorIdx() { return this._anchorIdx; }
  get fallbackActive() { return this.snapshot().fallbackActive; }
  get nUpdates() { return this._rt.readU64(this._requirePtr() + 32); }
  get nPoisonClipped() { return this._rt.readU64(this._requirePtr() + 40); }
  get nFallbackTrips() { return this._rt.readU64(this._requirePtr() + 48); }
  get quarantined() {
    const ptr = this._requirePtr();
    const rt = this._rt;
    rt.refreshViews();
    const out = new Array(this._nExperts);
    for (let i = 0; i < this._nExperts; ++i) {
      out[i] = rt.u8[ptr + 2104 + i] !== 0;
    }
    return out;
  }
}

export class EchoPool extends WasmStruct {
  /**
   * Provenance-deduplicated multi-agent evidence pool with Beta-Bernoulli peer reliability.
   */
  constructor(optionsOrNSources, nClaims = 64, priorProb = 0.5) {
    const opts = isPlainObject(optionsOrNSources)
      ? optionsOrNSources
      : { nSources: optionsOrNSources, nClaims, priorProb };
    const nSourcesVal = toU32(opts.nSources, "nSources");
    const nClaimsVal = opts.nClaims === undefined ? 64 : toU32(opts.nClaims, "nClaims");
    const priorProbVal = opts.priorProb === undefined ? 0.5 : Number(opts.priorProb);

    const rt = getRuntime();
    super(rt, rt.sizes.echopool, "EchoPool");
    const rc = rt.exports.eproc_echopool_init(
      this._ptr,
      nSourcesVal,
      nClaimsVal,
      priorProbVal
    );
    if (rc !== EPROC_OK) {
      this.free();
      checkRc(rc, "EchoPool");
    }
    this._nSources = nSourcesVal;
    this._nClaims = nClaimsVal;
  }

  submit(sourceId, claimId, evidenceId, llr) {
    const ptr = this._requirePtr();
    const rt = this._rt;
    const src = Number(sourceId);
    const clm = Number(claimId);
    if (!Number.isInteger(src) || src < 0 || src >= this._nSources) {
      throw new RangeError(`EchoPool.submit: sourceId ${sourceId} out of range [0, ${this._nSources})`);
    }
    if (!Number.isInteger(clm) || clm < 0 || clm >= this._nClaims) {
      throw new RangeError(`EchoPool.submit: claimId ${claimId} out of range [0, ${this._nClaims})`);
    }
    const evU64 = normalizeEvidenceId(evidenceId);
    const rc = rt.exports.eproc_echopool_submit(
      ptr,
      src >>> 0,
      clm >>> 0,
      evU64,
      Number(llr),
      rt.outPtr
    );
    checkRc(rc, "EchoPool.submit");
    rt.refreshViews();
    const dv = rt.dv;
    return {
      accepted: dv.getInt32(rt.outPtr + 0, true) !== 0,
      posteriorProb: dv.getFloat64(rt.outPtr + 8, true),
      logOdds: dv.getFloat64(rt.outPtr + 16, true),
      sourceReliability: dv.getFloat64(rt.outPtr + 24, true),
      uniqueCount: Number(dv.getBigUint64(rt.outPtr + 32, true)),
    };
  }

  resolve(claimId, outcomePositive) {
    const ptr = this._requirePtr();
    const clm = Number(claimId);
    if (!Number.isInteger(clm) || clm < 0 || clm >= this._nClaims) {
      throw new RangeError(`EchoPool.resolve: claimId ${claimId} out of range [0, ${this._nClaims})`);
    }
    const rc = this._rt.exports.eproc_echopool_resolve(ptr, clm >>> 0, outcomePositive ? 1 : 0);
    checkRc(rc, "EchoPool.resolve");
  }

  posterior(claimId) {
    return this._rt.exports.eproc_echopool_posterior(this._requirePtr(), toU32(claimId, "claimId"));
  }

  sourceReliability(sourceId) {
    return this._rt.exports.eproc_echopool_source_reliability(
      this._requirePtr(),
      toU32(sourceId, "sourceId")
    );
  }

  get nSources() { return this._nSources; }
  get nClaims() { return this._nClaims; }
  get nAccepted() { return this._rt.readU64(this._requirePtr() + 8); }
  get nEchoDropped() { return this._rt.readU64(this._requirePtr() + 16); }
}

/* ============================================================================
 * 4. VarOptSampler, OnlineRLSSampler, SiftVOISelector
 * ============================================================================ */

export class VarOptSampler extends WasmStruct {
  /**
   * Variance-Optimal weighted reservoir sampler (Cohen et al. 2011) with exact HT weights.
   */
  constructor(optionsOrK, seed = 42n) {
    const opts = isPlainObject(optionsOrK) ? optionsOrK : { k: optionsOrK, seed };
    const kVal = toU32(opts.k, "k");
    const seedU64 = toU64BigInt(opts.seed === undefined ? 42n : opts.seed, "seed");

    const rt = getRuntime();
    super(rt, rt.sizes.varopt, "VarOptSampler");
    const rc = rt.exports.eproc_varopt_init(this._ptr, kVal, seedU64);
    if (rc !== EPROC_OK) {
      this.free();
      checkRc(rc, "VarOptSampler");
    }
    this._k = kVal;
  }

  push(itemId, weight, value) {
    const ptr = this._requirePtr();
    const idU64 = toU64BigInt(itemId, "itemId");
    const rc = this._rt.exports.eproc_varopt_push(ptr, idU64, Number(weight), Number(value));
    checkRc(rc, "VarOptSampler.push");
  }

  estimateSum() {
    return this._rt.exports.eproc_varopt_estimate_sum(this._requirePtr());
  }

  estimateMean() {
    return this._rt.exports.eproc_varopt_estimate_mean(this._requirePtr());
  }

  items() {
    const ptr = this._requirePtr();
    const rt = this._rt;
    rt.refreshViews();
    const dv = rt.dv;
    const count = dv.getUint32(ptr + 4, true);
    const out = new Array(count);
    for (let i = 0; i < count; ++i) {
      const base = ptr + 64 + i * 40;
      out[i] = {
        itemId: dv.getBigUint64(base + 0, true),
        origWeight: dv.getFloat64(base + 8, true),
        adjWeight: dv.getFloat64(base + 16, true),
        value: dv.getFloat64(base + 24, true),
      };
    }
    return out;
  }

  get k() { return this._k; }
  get count() { return this._rt.readU32(this._requirePtr() + 4); }
  get nHeavy() { return this._rt.readU32(this._requirePtr() + 8); }
  get tau() { return this._rt.readF64(this._requirePtr() + 16); }
  get totalWeight() {
    const ptr = this._requirePtr();
    return this._rt.readF64(ptr + 24) + this._rt.readF64(ptr + 32);
  }
  get nSeen() { return this._rt.readU64(this._requirePtr() + 40); }
}

export class OnlineRLSSampler extends WasmStruct {
  /**
   * Online Ridge Leverage Score (RLS) causal subsampler (Sherman-Morrison rank-1 updates).
   */
  constructor(optionsOrD, lambdaReg = 1.0, oversample = 2.0, minProb = 0.02, seed = 42n) {
    const opts = isPlainObject(optionsOrD)
      ? optionsOrD
      : { d: optionsOrD, lambdaReg, oversample, minProb, seed };
    const dVal = toU32(opts.d, "d");
    const lamVal =
      opts.lambda !== undefined
        ? Number(opts.lambda)
        : opts.lambdaReg !== undefined
          ? Number(opts.lambdaReg)
          : 1.0;
    const overVal = opts.oversample === undefined ? 2.0 : Number(opts.oversample);
    const minPVal = opts.minProb === undefined ? 0.02 : Number(opts.minProb);
    const seedU64 = toU64BigInt(opts.seed === undefined ? 42n : opts.seed, "seed");

    const rt = getRuntime();
    super(rt, rt.sizes.rls, "OnlineRLSSampler");
    const rc = rt.exports.eproc_rls_init(
      this._ptr,
      dVal,
      lamVal,
      overVal,
      minPVal,
      seedU64
    );
    if (rc !== EPROC_OK) {
      this.free();
      checkRc(rc, "OnlineRLSSampler");
    }
    this._d = dVal;
  }

  step(x) {
    const ptr = this._requirePtr();
    if (x.length !== this._d) {
      throw new RangeError(`OnlineRLSSampler.step: expected vector of length ${this._d}, got ${x.length}`);
    }
    const rt = this._rt;
    return rt.withScratch(this._d * 8, (bufPtr) => {
      rt.writeF64Array(bufPtr, x);
      const rc = rt.exports.eproc_rls_step(ptr, bufPtr, rt.outPtr);
      checkRc(rc, "OnlineRLSSampler.step");
      rt.refreshViews();
      const dv = rt.dv;
      return {
        keep: dv.getInt32(rt.outPtr + 0, true) !== 0,
        leverage: dv.getFloat64(rt.outPtr + 8, true),
        prob: dv.getFloat64(rt.outPtr + 16, true),
        htWeight: dv.getFloat64(rt.outPtr + 24, true),
      };
    });
  }

  get d() { return this._d; }
  get nSeen() { return this._rt.readU64(this._requirePtr() + 32); }
  get nKept() { return this._rt.readU64(this._requirePtr() + 40); }
  get keepRate() {
    const seen = this.nSeen;
    return seen > 0 ? this.nKept / seen : 0.0;
  }
}

export class SiftVOISelector extends WasmStruct {
  /**
   * Hierarchical multi-resolution Bayes Value-of-Information (VOI) selector
   * with exact Gaussian maximum-entropy branch-and-bound.
   */
  constructor(optionsOrNSites, perSite = 8, signalMean = 1.5, noiseSd = 1.0) {
    const opts = isPlainObject(optionsOrNSites)
      ? optionsOrNSites
      : { nSites: optionsOrNSites, perSite, signalMean, noiseSd };
    const nSitesVal = toU32(opts.nSites, "nSites");
    const perSiteVal = opts.perSite === undefined ? 8 : toU32(opts.perSite, "perSite");
    const signalMeanVal = opts.signalMean === undefined ? 1.5 : Number(opts.signalMean);
    const noiseSdVal = opts.noiseSd === undefined ? 1.0 : Number(opts.noiseSd);

    const rt = getRuntime();
    super(rt, rt.sizes.sift, "SiftVOISelector");
    const rc = rt.exports.eproc_sift_init(
      this._ptr,
      nSitesVal,
      perSiteVal,
      signalMeanVal,
      noiseSdVal
    );
    if (rc !== EPROC_OK) {
      this.free();
      checkRc(rc, "SiftVOISelector");
    }
    this._nSites = nSitesVal;
    this._perSite = perSiteVal;
  }

  setNoiseScale(channelIdx, noiseScale) {
    const idx = Number(channelIdx);
    if (!Number.isInteger(idx) || idx < 0 || idx >= this.nChannels) {
      throw new RangeError(`SiftVOISelector.setNoiseScale: channelIdx ${channelIdx} out of range [0, ${this.nChannels})`);
    }
    const rc = this._rt.exports.eproc_sift_set_noise_scale(
      this._requirePtr(),
      idx >>> 0,
      Number(noiseScale)
    );
    checkRc(rc, "SiftVOISelector.setNoiseScale");
  }

  observeChannel(channelIdx, obsMean, nTicks = 1) {
    const idx = Number(channelIdx);
    if (!Number.isInteger(idx) || idx < 0 || idx >= this.nChannels) {
      throw new RangeError(`SiftVOISelector.observeChannel: channelIdx ${channelIdx} out of range [0, ${this.nChannels})`);
    }
    const ticks = toU32(nTicks, "nTicks");
    const rc = this._rt.exports.eproc_sift_observe_channel(
      this._requirePtr(),
      idx >>> 0,
      Number(obsMean),
      ticks
    );
    checkRc(rc, "SiftVOISelector.observeChannel");
  }

  observeSite(siteIdx, siteMean) {
    const idx = Number(siteIdx);
    if (!Number.isInteger(idx) || idx < 0 || idx >= this._nSites) {
      throw new RangeError(`SiftVOISelector.observeSite: siteIdx ${siteIdx} out of range [0, ${this._nSites})`);
    }
    const rc = this._rt.exports.eproc_sift_observe_site(
      this._requirePtr(),
      idx >>> 0,
      Number(siteMean)
    );
    checkRc(rc, "SiftVOISelector.observeSite");
  }

  selectNext(optionsOrCostSite = 1.0, costChannel = 1.0, costTicks = 8.0, ticksPerRead = 8) {
    const opts = isPlainObject(optionsOrCostSite)
      ? optionsOrCostSite
      : { costSite: optionsOrCostSite, costChannel, costTicks, ticksPerRead };
    const cSite = opts.costSite === undefined ? 1.0 : Number(opts.costSite);
    const cChan = opts.costChannel === undefined ? 1.0 : Number(opts.costChannel);
    const cTicks = opts.costTicks === undefined ? 8.0 : Number(opts.costTicks);
    const ticks = opts.ticksPerRead === undefined ? 8 : toU32(opts.ticksPerRead, "ticksPerRead");

    const ptr = this._requirePtr();
    const rt = this._rt;
    const rc = rt.exports.eproc_sift_select_next(
      ptr,
      cSite,
      cChan,
      cTicks,
      ticks,
      rt.outPtr
    );
    checkRc(rc, "SiftVOISelector.selectNext");
    rt.refreshViews();
    const dv = rt.dv;
    const actionType = dv.getInt32(rt.outPtr + 0, true);
    return {
      actionType,
      actionName: SIFT_ACTION_NAMES[actionType] ?? `unknown(${actionType})`,
      targetIdx: dv.getUint32(rt.outPtr + 4, true),
      evoiPerCost: dv.getFloat64(rt.outPtr + 8, true),
    };
  }

  siteProb(siteIdx) {
    const idx = Number(siteIdx);
    if (!Number.isInteger(idx) || idx < 0 || idx >= this._nSites) {
      throw new RangeError(`SiftVOISelector.siteProb: siteIdx ${siteIdx} out of range`);
    }
    return Math.exp(this._rt.readF64(this._requirePtr() + 48 + idx * 8));
  }

  channelProb(channelIdx) {
    const idx = Number(channelIdx);
    if (!Number.isInteger(idx) || idx < 0 || idx >= this.nChannels) {
      throw new RangeError(`SiftVOISelector.channelProb: channelIdx ${channelIdx} out of range`);
    }
    return this._rt.readF64(this._requirePtr() + 8240 + idx * 8);
  }

  get nSites() { return this._nSites; }
  get perSite() { return this._perSite; }
  get nChannels() { return this._nSites * this._perSite; }
}
