/**
 * TypeScript definitions for @eproc/wasm — zero-dependency C11 WebAssembly runtime
 * for anytime-valid statistics, multi-rate online conformal calibration,
 * Byzantine-safe multi-agent fusion, and causal subsampling.
 *
 * License: Apache-2.0
 */

export declare const EPROC_OK: 0;
export declare const EPROC_ERR_INVALID: -1;
export declare const EPROC_ERR_CAPACITY: -2;
export declare const EPROC_ERR_EMPTY: -3;
export declare const EPROC_ERR_MISMATCH: -4;

export declare const EPROC_SIDE_UPPER: 0;
export declare const EPROC_SIDE_LOWER: 1;
export declare const EPROC_SIDE_TWO_SIDED: 2;

export type TestSide =
  | "upper"
  | "greater"
  | "lower"
  | "less"
  | "two_sided"
  | "two-sided"
  | "both"
  | 0
  | 1
  | 2;

export type CanonicalSide = "upper" | "lower" | "two_sided";

export type WasmSource =
  | string
  | URL
  | ArrayBuffer
  | ArrayBufferView
  | Response
  | Promise<Response>
  | WebAssembly.Module;

export interface EprocModule {
  readonly EPROC_OK: typeof EPROC_OK;
  readonly EPROC_ERR_INVALID: typeof EPROC_ERR_INVALID;
  readonly EPROC_ERR_CAPACITY: typeof EPROC_ERR_CAPACITY;
  readonly EPROC_ERR_EMPTY: typeof EPROC_ERR_EMPTY;
  readonly EPROC_ERR_MISMATCH: typeof EPROC_ERR_MISMATCH;
  readonly EPROC_SIDE_UPPER: typeof EPROC_SIDE_UPPER;
  readonly EPROC_SIDE_LOWER: typeof EPROC_SIDE_LOWER;
  readonly EPROC_SIDE_TWO_SIDED: typeof EPROC_SIDE_TWO_SIDED;
  readonly abiVersion: typeof abiVersion;
  readonly simdFlags: typeof simdFlags;
  readonly heapUsed: typeof heapUsed;
  readonly combineEValues: typeof combineEValues;
  readonly EProcess: typeof EProcess;
  readonly ConfidenceSequence: typeof ConfidenceSequence;
  readonly EDetector: typeof EDetector;
  readonly AgentRace: typeof AgentRace;
  readonly MultiRateConformal: typeof MultiRateConformal;
  readonly DelayedConformal: typeof DelayedConformal;
  readonly BudgetConformal: typeof BudgetConformal;
  readonly ConformalMonoid: typeof ConformalMonoid;
  readonly SafeFuser: typeof SafeFuser;
  readonly EchoPool: typeof EchoPool;
  readonly VarOptSampler: typeof VarOptSampler;
  readonly OnlineRLSSampler: typeof OnlineRLSSampler;
  readonly SiftVOISelector: typeof SiftVOISelector;
  readonly wasmExports: Record<string, WebAssembly.ExportValue>;
  readonly memory: WebAssembly.Memory;
}

/**
 * Initialize the eproc WebAssembly runtime asynchronously.
 * Works in Node.js (via `node:fs/promises`) and browsers/edge runtimes (via `fetch`).
 */
export declare function initEproc(wasmSource?: WasmSource): Promise<EprocModule>;

/**
 * Synchronously initialize the eproc WebAssembly runtime from raw bytes or a compiled WebAssembly.Module.
 */
export declare function initEprocSync(
  wasmSource: ArrayBuffer | ArrayBufferView | WebAssembly.Module
): EprocModule;

/** Return the compiled C11 ABI version (`1`). */
export declare function abiVersion(): number;

/** Return runtime SIMD capability flags (`0` on scalar WebAssembly). */
export declare function simdFlags(): number;

/** Return currently live bytes allocated in the WASM linear-memory heap. */
export declare function heapUsed(): number;

/**
 * Combine multiple log e-values into a single valid log e-value.
 * - `independent: true`: sum of log e-values (product of independent e-values)
 * - `independent: false` (default): Vovk & Wang (2021) arithmetic mean `log((1/K) * sum_k E_k)`
 */
export declare function combineEValues(
  logEs: ArrayLike<number>,
  options?: boolean | { independent?: boolean }
): number;

/* ============================================================================
 * 1. EProcess, ConfidenceSequence, EDetector, AgentRace
 * ============================================================================ */

export interface EProcessOptions {
  mu0: number;
  alpha?: number;
  lower?: number;
  upper?: number;
  side?: TestSide;
}

export interface EProcessResult {
  readonly n: number;
  readonly eValue: number;
  readonly logE: number;
  readonly maxLogE: number;
  readonly pValue: number;
  readonly mean: number;
  readonly rejected: boolean;
}

export declare class EProcess {
  constructor(options: EProcessOptions);
  constructor(
    mu0: number,
    alpha?: number,
    lower?: number,
    upper?: number,
    side?: TestSide
  );

  update(x: number): EProcessResult;
  updateBatch(xs: ArrayLike<number>): EProcessResult;
  query(): EProcessResult;
  result(): EProcessResult;
  free(): void;
  [Symbol.dispose](): void;

  readonly n: number;
  readonly rejectedAt: number;
  readonly nInvalid: number;
  readonly eValue: number;
  readonly logE: number;
  readonly maxLogE: number;
  readonly pValue: number;
  readonly mean: number;
  readonly rejected: boolean;
  readonly mu0: number;
  readonly alpha: number;
  readonly lower: number;
  readonly upper: number;
  readonly side: CanonicalSide;
}

export type ConfidenceSequenceMode =
  | "eb"
  | "empirical_bernstein"
  | "empirical-bernstein"
  | "normal"
  | "normal_mixture"
  | "normal-mixture"
  | "subgaussian";

export interface ConfidenceSequenceOptions {
  alpha?: number;
  lower?: number;
  upper?: number;
  mode?: ConfidenceSequenceMode;
  sigma?: number;
  vOpt?: number;
}

export interface ConfidenceSequenceResult {
  readonly n: number;
  readonly mean: number;
  readonly low: number;
  readonly high: number;
  readonly width: number;
}

export declare class ConfidenceSequence {
  constructor(options?: ConfidenceSequenceOptions);
  constructor(
    alpha: number,
    lower?: number,
    upper?: number,
    mode?: ConfidenceSequenceMode,
    sigma?: number,
    vOpt?: number
  );

  static empiricalBernstein(
    alpha?: number,
    lower?: number,
    upper?: number
  ): ConfidenceSequence;

  static normalMixture(
    alpha?: number,
    sigma?: number,
    vOpt?: number
  ): ConfidenceSequence;

  update(x: number): ConfidenceSequenceResult;
  updateBatch(xs: ArrayLike<number>): ConfidenceSequenceResult;
  query(): ConfidenceSequenceResult;
  free(): void;
  [Symbol.dispose](): void;

  readonly n: number;
  readonly mean: number;
  readonly low: number;
  readonly high: number;
  readonly width: number;
  readonly bounds: [number, number];
  readonly alpha: number;
  readonly mode: "eb" | "normal";
}

export interface EDetectorOptions {
  mu0: number;
  alpha?: number;
  lower?: number;
  upper?: number;
  side?: TestSide;
}

export interface EDetectorResult {
  readonly n: number;
  readonly logStat: number;
  readonly stat: number;
  readonly alarm: boolean;
}

export declare class EDetector {
  constructor(options: EDetectorOptions);
  constructor(
    mu0: number,
    alpha?: number,
    lower?: number,
    upper?: number,
    side?: TestSide
  );

  update(x: number): EDetectorResult;
  updateBatch(xs: ArrayLike<number>): EDetectorResult;
  resetAlarm(): void;
  free(): void;
  [Symbol.dispose](): void;

  readonly n: number;
  readonly logStat: number;
  readonly stat: number;
  readonly alarmAt: number;
  readonly nAlarms: number;
  readonly mu0: number;
  readonly side: CanonicalSide;
}

export interface AgentRaceOptions {
  nArms: number;
  alpha?: number;
  lower?: number;
  upper?: number;
  margin?: number;
  names?: readonly string[];
}

export interface RaceStatus {
  readonly nArms: number;
  readonly nActive: number;
  readonly winner: number | null;
  readonly leader: number | null;
  readonly totalObs: number;
  readonly leaderLow: number;
  readonly leaderHigh: number;
  readonly runnerUpHigh: number;
  readonly winnerName: string | null;
  readonly leaderName: string | null;
}

export interface RaceArmState {
  readonly arm: number;
  readonly name: string | null;
  readonly active: boolean;
  readonly n: number;
  readonly mean: number;
  readonly low: number;
  readonly high: number;
  readonly width: number;
  readonly eliminatedAt: number;
}

export declare class AgentRace {
  constructor(options: AgentRaceOptions);
  constructor(
    nArms: number,
    alpha?: number,
    lower?: number,
    upper?: number,
    margin?: number
  );

  update(arm: number | string, reward: number): RaceStatus;
  updateBatch(
    arms: ArrayLike<number | string>,
    rewards: ArrayLike<number>
  ): RaceStatus;
  suggestArm(): number;
  status(): RaceStatus;
  armState(arm: number | string): RaceArmState;
  free(): void;
  [Symbol.dispose](): void;

  readonly nArms: number;
  readonly winner: number | null;
  readonly winnerName: string | null;
  readonly activeArms: number[];
  readonly names: readonly string[] | null;
}

/* ============================================================================
 * 2. MultiRateConformal, DelayedConformal, BudgetConformal, ConformalMonoid
 * ============================================================================ */

export interface MultiRateConformalOptions {
  alphas?: ArrayLike<number>;
  windowSize?: number;
  pidKi?: number;
}

export interface ConformalMeta {
  readonly nRates: number;
  readonly alphas: number[];
  readonly quantiles: number[];
  readonly effAlphas: number[];
  readonly empCoverage: number[];
}

export interface ConformalPrediction {
  readonly pointPred: number;
  readonly scale: number;
  readonly alphas: number[];
  readonly lows: number[];
  readonly highs: number[];
  readonly quantiles: number[];
  readonly effAlphas: number[];
  readonly empCoverage: number[];
}

export declare class MultiRateConformal {
  constructor(options?: MultiRateConformalOptions | ArrayLike<number>);

  predict(pointPred: number, scale?: number): ConformalPrediction;
  update(score: number): ConformalMeta;
  updateBatch(scores: ArrayLike<number>): ConformalMeta;
  free(): void;
  [Symbol.dispose](): void;

  readonly alphas: readonly number[];
  readonly nRates: number;
  readonly nUpdates: number;
  readonly quantiles: number[];
}

export interface DelayedIssueResult {
  readonly ticketId: bigint;
  readonly lows: number[];
  readonly highs: number[];
  readonly intervals: [number, number][];
}

export declare class DelayedConformal {
  constructor(options?: MultiRateConformalOptions | ArrayLike<number>);

  issue(pred: number, scale?: number): DelayedIssueResult;
  reveal(ticketId: bigint | number, yTrue: number): ConformalMeta;
  free(): void;
  [Symbol.dispose](): void;

  readonly pendingCount: number;
  readonly alphas: readonly number[];
  readonly nRates: number;
  readonly nUpdates: number;
}

export interface BudgetConformalOptions {
  alpha?: number;
  windowSize?: number;
  censorPenalty?: number;
}

export declare class BudgetConformal {
  constructor(options?: BudgetConformalOptions | number);

  bound(baseEstimate: number): number;
  update(baseEstimate: number, observedCost: number, censored?: boolean): void;
  free(): void;
  [Symbol.dispose](): void;

  readonly nCompleted: number;
  readonly nCensored: number;
}

export declare class ConformalMonoid {
  constructor(scaleHint?: number);

  static readonly wireSize: number;
  static fromBytes(data: Uint8Array | ArrayBufferView | ArrayBuffer): ConformalMonoid;

  add(score: number): void;
  addBatch(scores: ArrayLike<number>): void;
  merge(other: ConformalMonoid): ConformalMonoid;
  subtract(local: ConformalMonoid): ConformalMonoid;
  quantile(q: number): number;
  toBytes(): Uint8Array;
  free(): void;
  [Symbol.dispose](): void;

  readonly count: number;
  readonly mean: number;
  readonly minVal: number;
  readonly maxVal: number;
  readonly scaleHint: number;
  readonly binCounts: number[];
}

/* ============================================================================
 * 3. SafeFuser & EchoPool
 * ============================================================================ */

export interface SafeFuserOptions {
  nExperts: number;
  anchorIdx?: number;
  alphaDeg?: number;
}

export interface SafeFuserStep {
  readonly fusedPred: number;
  readonly anchorPred: number;
  readonly fallbackActive: boolean;
  readonly nQuarantined: number;
  readonly supLogE: number;
  readonly degLogE: number;
  readonly nUpdates: number;
}

export interface SafeFuserSnapshot {
  readonly weights: number[];
  readonly fallbackActive: boolean;
}

export declare class SafeFuser {
  constructor(options: SafeFuserOptions);
  constructor(nExperts: number, anchorIdx?: number, alphaDeg?: number);

  predict(expertPreds: ArrayLike<number>): number;
  update(expertPreds: ArrayLike<number>, yTrue: number): SafeFuserStep;
  snapshot(): SafeFuserSnapshot;
  weights(): number[];
  free(): void;
  [Symbol.dispose](): void;

  readonly nExperts: number;
  readonly anchorIdx: number;
  readonly fallbackActive: boolean;
  readonly nUpdates: number;
  readonly nPoisonClipped: number;
  readonly nFallbackTrips: number;
  readonly quarantined: boolean[];
}

export interface EchoPoolOptions {
  nSources: number;
  nClaims?: number;
  priorProb?: number;
}

export interface EchoSubmitResult {
  readonly accepted: boolean;
  readonly posteriorProb: number;
  readonly logOdds: number;
  readonly sourceReliability: number;
  readonly uniqueCount: number;
}

export declare class EchoPool {
  constructor(options: EchoPoolOptions);
  constructor(nSources: number, nClaims?: number, priorProb?: number);

  submit(
    sourceId: number,
    claimId: number,
    evidenceId: number | bigint | string | Uint8Array,
    llr: number
  ): EchoSubmitResult;
  resolve(claimId: number, outcomePositive: boolean): void;
  posterior(claimId: number): number;
  sourceReliability(sourceId: number): number;
  free(): void;
  [Symbol.dispose](): void;

  readonly nSources: number;
  readonly nClaims: number;
  readonly nAccepted: number;
  readonly nEchoDropped: number;
}

/* ============================================================================
 * 4. VarOptSampler, OnlineRLSSampler, SiftVOISelector
 * ============================================================================ */

export interface VarOptOptions {
  k: number;
  seed?: number | bigint;
}

export interface VarOptItem {
  readonly itemId: bigint;
  readonly origWeight: number;
  readonly adjWeight: number;
  readonly value: number;
}

export declare class VarOptSampler {
  constructor(options: VarOptOptions);
  constructor(k: number, seed?: number | bigint);

  push(itemId: number | bigint, weight: number, value: number): void;
  estimateSum(): number;
  estimateMean(): number;
  items(): VarOptItem[];
  free(): void;
  [Symbol.dispose](): void;

  readonly k: number;
  readonly count: number;
  readonly nHeavy: number;
  readonly tau: number;
  readonly totalWeight: number;
  readonly nSeen: number;
}

export interface OnlineRLSOptions {
  d: number;
  lambda?: number;
  lambdaReg?: number;
  oversample?: number;
  minProb?: number;
  seed?: number | bigint;
}

export interface RLSStepResult {
  readonly keep: boolean;
  readonly leverage: number;
  readonly prob: number;
  readonly htWeight: number;
}

export declare class OnlineRLSSampler {
  constructor(options: OnlineRLSOptions);
  constructor(
    d: number,
    lambdaReg?: number,
    oversample?: number,
    minProb?: number,
    seed?: number | bigint
  );

  step(x: ArrayLike<number>): RLSStepResult;
  free(): void;
  [Symbol.dispose](): void;

  readonly d: number;
  readonly nSeen: number;
  readonly nKept: number;
  readonly keepRate: number;
}

export interface SiftVOIOptions {
  nSites: number;
  perSite?: number;
  signalMean?: number;
  noiseSd?: number;
}

export interface SiftSelectOptions {
  costSite?: number;
  costChannel?: number;
  costTicks?: number;
  ticksPerRead?: number;
}

export interface SiftAction {
  readonly actionType: number;
  readonly actionName: "site" | "channel" | "ticks" | string;
  readonly targetIdx: number;
  readonly evoiPerCost: number;
}

export declare class SiftVOISelector {
  constructor(options: SiftVOIOptions);
  constructor(
    nSites: number,
    perSite?: number,
    signalMean?: number,
    noiseSd?: number
  );

  setNoiseScale(channelIdx: number, noiseScale: number): void;
  observeChannel(channelIdx: number, obsMean: number, nTicks?: number): void;
  observeSite(siteIdx: number, siteMean: number): void;
  selectNext(options?: SiftSelectOptions): SiftAction;
  selectNext(
    costSite?: number,
    costChannel?: number,
    costTicks?: number,
    ticksPerRead?: number
  ): SiftAction;
  siteProb(siteIdx: number): number;
  channelProb(channelIdx: number): number;
  free(): void;
  [Symbol.dispose](): void;

  readonly nSites: number;
  readonly perSite: number;
  readonly nChannels: number;
}
