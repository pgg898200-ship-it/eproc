// Package eproc provides zero-dependency Go CGo bindings for eproc's
// zero-allocation C11 statistical inference core (ABI v1).
package eproc

/*
#cgo CFLAGS: -std=c11 -O3 -I${SRCDIR}/../../csrc
#cgo LDFLAGS: -lm
#include "eproc.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"unsafe"
)

const (
	// WireSizeConformalMonoid is the fixed checksummed wire size of ConformalMonoid (640 bytes).
	WireSizeConformalMonoid = 640

	// SIMDAVX2 indicates AVX2 kernels are active (compile-time or runtime CPUID dispatch).
	SIMDAVX2 uint32 = 1 << 0
	// SIMDFMA indicates compile-time FMA is enabled.
	SIMDFMA uint32 = 1 << 1
	// SIMDNEON indicates ARM NEON is enabled.
	SIMDNEON uint32 = 1 << 2
)

var (
	// ErrInvalid indicates an invalid parameter or out-of-range index.
	ErrInvalid = errors.New("eproc: invalid argument or state")
	// ErrCapacity indicates a fixed zero-allocation capacity limit was exceeded.
	ErrCapacity = errors.New("eproc: capacity exceeded")
	// ErrEmpty indicates an empty stream or table.
	ErrEmpty = errors.New("eproc: empty")
	// ErrMismatch indicates a wire checksum or format mismatch.
	ErrMismatch = errors.New("eproc: checksum or format mismatch")
)

func checkRC(rc C.int) error {
	switch rc {
	case C.EPROC_OK:
		return nil
	case C.EPROC_ERR_INVALID:
		return ErrInvalid
	case C.EPROC_ERR_CAPACITY:
		return ErrCapacity
	case C.EPROC_ERR_EMPTY:
		return ErrEmpty
	case C.EPROC_ERR_MISMATCH:
		return ErrMismatch
	default:
		return fmt.Errorf("eproc: unknown status code %d", int(rc))
	}
}

// Side specifies the alternative hypothesis direction for e-processes and e-detectors.
type Side int32

const (
	SideUpper    Side = C.EPROC_SIDE_UPPER
	SideLower    Side = C.EPROC_SIDE_LOWER
	SideTwoSided Side = C.EPROC_SIDE_TWO_SIDED
)

// ABIVersion returns the compiled C11 ABI version (1).
func ABIVersion() uint32 {
	return uint32(C.eproc_abi_version())
}

// SIMDFlags returns active runtime SIMD capability flags (bit 0: AVX2, bit 1: FMA, bit 2: NEON).
func SIMDFlags() uint32 {
	return uint32(C.eproc_simd_flags())
}

// CombineIndependentLog combines two independent e-processes in log-space.
func CombineIndependentLog(logEA, logEB float64) float64 {
	return float64(C.eproc_combine_independent_log(C.double(logEA), C.double(logEB)))
}

// CombineDependentLog combines arbitrarily dependent e-processes via arithmetic averaging.
func CombineDependentLog(logEs []float64) float64 {
	if len(logEs) == 0 {
		return 0.0
	}
	return float64(C.eproc_combine_dependent_log(
		(*C.double)(unsafe.Pointer(&logEs[0])),
		C.size_t(len(logEs)),
	))
}

// ============================================================================
// 1. EProcess, ConfidenceSequence, EDetector, AgentRace
// ============================================================================

// EProcessResult holds the anytime-valid summary returned by EProcess.
type EProcessResult struct {
	N        uint64
	EValue   float64
	LogE     float64
	MaxLogE  float64
	PValue   float64
	Mean     float64
	Rejected bool
}

func betResultFromC(r C.eproc_bet_result_t) EProcessResult {
	return EProcessResult{
		N:        uint64(r.n),
		EValue:   float64(r.e_value),
		LogE:     float64(r.log_e),
		MaxLogE:  float64(r.max_log_e),
		PValue:   float64(r.p_value),
		Mean:     float64(r.mean),
		Rejected: r.rejected != 0,
	}
}

// EProcess is an anytime-valid ONS betting supermartingale e-process.
type EProcess struct {
	state C.eproc_bet_t
}

// NewEProcess initializes an anytime-valid e-process on [lower, upper].
func NewEProcess(mu0, alpha, lower, upper float64, side Side) (*EProcess, error) {
	ep := &EProcess{}
	if err := checkRC(C.eproc_bet_init(
		&ep.state, C.double(mu0), C.double(alpha), C.double(lower), C.double(upper), C.int32_t(side),
	)); err != nil {
		return nil, err
	}
	return ep, nil
}

// Update ingests a single observation x.
func (ep *EProcess) Update(x float64) (EProcessResult, error) {
	var out C.eproc_bet_result_t
	if err := checkRC(C.eproc_bet_update(&ep.state, C.double(x), &out)); err != nil {
		return EProcessResult{}, err
	}
	return betResultFromC(out), nil
}

// UpdateBatch ingests a slice of observations bit-identically to Update.
func (ep *EProcess) UpdateBatch(xs []float64) (EProcessResult, error) {
	if len(xs) == 0 {
		return ep.Query(), nil
	}
	var out C.eproc_bet_result_t
	if err := checkRC(C.eproc_bet_update_batch(
		&ep.state, (*C.double)(unsafe.Pointer(&xs[0])), C.size_t(len(xs)), &out,
	)); err != nil {
		return EProcessResult{}, err
	}
	return betResultFromC(out), nil
}

// Query returns the current anytime-valid test summary without mutating state.
func (ep *EProcess) Query() EProcessResult {
	var out C.eproc_bet_result_t
	C.eproc_bet_query(&ep.state, &out)
	return betResultFromC(out)
}

// N returns the number of finite observations ingested.
func (ep *EProcess) N() uint64 { return uint64(ep.state.n) }

// NInvalid returns the number of sanitized NaN/Inf inputs.
func (ep *EProcess) NInvalid() uint64 { return uint64(ep.state.n_invalid) }

// ConfidenceSequenceResult holds the anytime-valid interval bounds.
type ConfidenceSequenceResult struct {
	N     uint64
	Mean  float64
	Low   float64
	High  float64
	Width float64
}

func csResultFromC(r C.eproc_cs_result_t) ConfidenceSequenceResult {
	return ConfidenceSequenceResult{
		N:     uint64(r.n),
		Mean:  float64(r.mean),
		Low:   float64(r.low),
		High:  float64(r.high),
		Width: float64(r.width),
	}
}

// ConfidenceSequence tracks an anytime-valid Empirical-Bernstein or Normal-Mixture confidence sequence.
type ConfidenceSequence struct {
	state C.eproc_cs_t
}

// NewEmpiricalBernsteinCS initializes a bounded Empirical-Bernstein confidence sequence.
func NewEmpiricalBernsteinCS(alpha, lower, upper float64) (*ConfidenceSequence, error) {
	cs := &ConfidenceSequence{}
	if err := checkRC(C.eproc_cs_init_eb(&cs.state, C.double(alpha), C.double(lower), C.double(upper))); err != nil {
		return nil, err
	}
	return cs, nil
}

// NewNormalMixtureCS initializes a sub-Gaussian Normal-Mixture confidence sequence.
func NewNormalMixtureCS(alpha, sigma, vOpt float64) (*ConfidenceSequence, error) {
	cs := &ConfidenceSequence{}
	if err := checkRC(C.eproc_cs_init_normal(&cs.state, C.double(alpha), C.double(sigma), C.double(vOpt))); err != nil {
		return nil, err
	}
	return cs, nil
}

// Update ingests a single observation x.
func (cs *ConfidenceSequence) Update(x float64) (ConfidenceSequenceResult, error) {
	var out C.eproc_cs_result_t
	if err := checkRC(C.eproc_cs_update(&cs.state, C.double(x), &out)); err != nil {
		return ConfidenceSequenceResult{}, err
	}
	return csResultFromC(out), nil
}

// UpdateBatch ingests a slice of observations.
func (cs *ConfidenceSequence) UpdateBatch(xs []float64) (ConfidenceSequenceResult, error) {
	if len(xs) == 0 {
		return cs.Query(), nil
	}
	var out C.eproc_cs_result_t
	if err := checkRC(C.eproc_cs_update_batch(
		&cs.state, (*C.double)(unsafe.Pointer(&xs[0])), C.size_t(len(xs)), &out,
	)); err != nil {
		return ConfidenceSequenceResult{}, err
	}
	return csResultFromC(out), nil
}

// Query returns the current confidence sequence bounds.
func (cs *ConfidenceSequence) Query() ConfidenceSequenceResult {
	var out C.eproc_cs_result_t
	C.eproc_cs_query(&cs.state, &out)
	return csResultFromC(out)
}

// EDetectorResult holds a Page-CUSUM e-detector step output.
type EDetectorResult struct {
	N       uint64
	LogStat float64
	Stat    float64
	Alarm   bool
}

// EDetector implements an anytime-valid Page-CUSUM resettable drift detector.
type EDetector struct {
	state C.eproc_edetect_t
}

// NewEDetector initializes a Page-CUSUM e-detector.
func NewEDetector(mu0, alpha, lower, upper float64, side Side) (*EDetector, error) {
	ed := &EDetector{}
	if err := checkRC(C.eproc_edetect_init(
		&ed.state, C.double(mu0), C.double(alpha), C.double(lower), C.double(upper), C.int32_t(side),
	)); err != nil {
		return nil, err
	}
	return ed, nil
}

// Update ingests a single observation into the e-detector.
func (ed *EDetector) Update(x float64) (EDetectorResult, error) {
	var out C.eproc_edetect_result_t
	if err := checkRC(C.eproc_edetect_update(&ed.state, C.double(x), &out)); err != nil {
		return EDetectorResult{}, err
	}
	return EDetectorResult{
		N:       uint64(out.n),
		LogStat: float64(out.log_stat),
		Stat:    float64(out.stat),
		Alarm:   out.alarm != 0,
	}, nil
}

// ResetAlarm re-arms the e-detector after handling an alarm.
func (ed *EDetector) ResetAlarm() {
	C.eproc_edetect_reset_alarm(&ed.state)
}

// RaceStatus summarizes an AgentRace best-arm identification session.
type RaceStatus struct {
	NArms        int
	NActive      int
	Winner       int
	Leader       int
	TotalObs     uint64
	LeaderLow    float64
	LeaderHigh   float64
	RunnerUpHigh float64
}

func raceStatusFromC(s C.eproc_race_status_t) RaceStatus {
	return RaceStatus{
		NArms:        int(s.n_arms),
		NActive:      int(s.n_active),
		Winner:       int(s.winner),
		Leader:       int(s.leader),
		TotalObs:     uint64(s.total_obs),
		LeaderLow:    float64(s.leader_low),
		LeaderHigh:   float64(s.leader_high),
		RunnerUpHigh: float64(s.runner_up_high),
	}
}

// AgentRace performs anytime-valid multi-arm best-arm identification with early elimination.
type AgentRace struct {
	state C.eproc_race_t
}

// NewAgentRace initializes an anytime-valid multi-arm racer with FWER control.
func NewAgentRace(nArms int, alpha, lower, upper, margin float64) (*AgentRace, error) {
	if nArms <= 0 {
		return nil, ErrInvalid
	}
	r := &AgentRace{}
	if err := checkRC(C.eproc_race_init(
		&r.state, C.uint32_t(nArms), C.double(alpha), C.double(lower), C.double(upper), C.double(margin),
	)); err != nil {
		return nil, err
	}
	return r, nil
}

// Update records a single reward observation for arm.
func (r *AgentRace) Update(arm int, reward float64) (RaceStatus, error) {
	if arm < 0 {
		return RaceStatus{}, ErrInvalid
	}
	var st C.eproc_race_status_t
	if err := checkRC(C.eproc_race_update(&r.state, C.uint32_t(arm), C.double(reward), &st)); err != nil {
		return RaceStatus{}, err
	}
	return raceStatusFromC(st), nil
}

// UpdateBatch records a batch of (arm, reward) pairs.
func (r *AgentRace) UpdateBatch(arms []uint32, rewards []float64) (RaceStatus, error) {
	if len(arms) != len(rewards) {
		return RaceStatus{}, ErrInvalid
	}
	if len(arms) == 0 {
		return r.Status(), nil
	}
	var st C.eproc_race_status_t
	if err := checkRC(C.eproc_race_update_batch(
		&r.state,
		(*C.uint32_t)(unsafe.Pointer(&arms[0])),
		(*C.double)(unsafe.Pointer(&rewards[0])),
		C.size_t(len(arms)),
		&st,
	)); err != nil {
		return RaceStatus{}, err
	}
	return raceStatusFromC(st), nil
}

// Status returns the current race status snapshot.
func (r *AgentRace) Status() RaceStatus {
	var st C.eproc_race_status_t
	C.eproc_race_status(&r.state, &st)
	return raceStatusFromC(st)
}

// ArmState returns the confidence sequence and active flag for arm.
func (r *AgentRace) ArmState(arm int) (ConfidenceSequenceResult, bool, error) {
	if arm < 0 {
		return ConfidenceSequenceResult{}, false, ErrInvalid
	}
	var out C.eproc_cs_result_t
	var active C.int32_t
	if err := checkRC(C.eproc_race_arm_state(&r.state, C.uint32_t(arm), &out, &active)); err != nil {
		return ConfidenceSequenceResult{}, false, err
	}
	return csResultFromC(out), active != 0, nil
}

// SuggestArm recommends the next arm index to sample under the LUCB1 rule (-1 if none).
func (r *AgentRace) SuggestArm() int {
	return int(C.eproc_race_suggest_arm(&r.state))
}

// ============================================================================
// 2. MultiRateConformal & ConformalMonoid
// ============================================================================

// ConformalOutput holds multi-rate quantile radii, effective alphas, and empirical coverage.
type ConformalOutput struct {
	NRates      int
	Quantiles   []float64
	EffAlphas   []float64
	EmpCoverage []float64
}

func conformalOutFromC(o C.eproc_conformal_out_t) ConformalOutput {
	n := int(o.n_rates)
	if n > C.EPROC_MAX_RATES {
		n = C.EPROC_MAX_RATES
	}
	q := make([]float64, n)
	ea := make([]float64, n)
	ec := make([]float64, n)
	for i := 0; i < n; i++ {
		q[i] = float64(o.quantiles[i])
		ea[i] = float64(o.eff_alphas[i])
		ec[i] = float64(o.emp_coverage[i])
	}
	return ConformalOutput{
		NRates:      n,
		Quantiles:   q,
		EffAlphas:   ea,
		EmpCoverage: ec,
	}
}

// ConformalIntervals holds nested prediction intervals across all calibrated rates.
type ConformalIntervals struct {
	Lows  []float64
	Highs []float64
	Meta  ConformalOutput
}

// MultiRateConformal calibrates up to 5 crossing-free online conformal levels simultaneously.
type MultiRateConformal struct {
	state C.eproc_conformal_t
}

// NewMultiRateConformal initializes a multi-rate crossing-free AgACI + Conformal PID calibrator.
func NewMultiRateConformal(alphas []float64, windowSize int, pidKi float64) (*MultiRateConformal, error) {
	if len(alphas) == 0 || windowSize <= 0 {
		return nil, ErrInvalid
	}
	cp := &MultiRateConformal{}
	if err := checkRC(C.eproc_conformal_init(
		&cp.state,
		(*C.double)(unsafe.Pointer(&alphas[0])),
		C.uint32_t(len(alphas)),
		C.uint32_t(windowSize),
		C.double(pidKi),
	)); err != nil {
		return nil, err
	}
	return cp, nil
}

// NRates returns the number of calibrated coverage levels.
func (cp *MultiRateConformal) NRates() int { return int(cp.state.n_rates) }

// TargetAlphas returns the sorted descending target miscoverage levels.
func (cp *MultiRateConformal) TargetAlphas() []float64 {
	n := cp.NRates()
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = float64(cp.state.target_alphas[i])
	}
	return out
}

// Predict returns nested prediction intervals around pointPred scaled by scale.
func (cp *MultiRateConformal) Predict(pointPred, scale float64) ConformalIntervals {
	n := cp.NRates()
	var lowsC [C.EPROC_MAX_RATES]C.double
	var highsC [C.EPROC_MAX_RATES]C.double
	var metaC C.eproc_conformal_out_t
	C.eproc_conformal_predict(
		&cp.state, C.double(pointPred), C.double(scale), &lowsC[0], &highsC[0], &metaC,
	)
	lows := make([]float64, n)
	highs := make([]float64, n)
	for i := 0; i < n; i++ {
		lows[i] = float64(lowsC[i])
		highs[i] = float64(highsC[i])
	}
	return ConformalIntervals{
		Lows:  lows,
		Highs: highs,
		Meta:  conformalOutFromC(metaC),
	}
}

// Update ingests a single nonconformity score.
func (cp *MultiRateConformal) Update(nonconformityScore float64) (ConformalOutput, error) {
	var metaC C.eproc_conformal_out_t
	if err := checkRC(C.eproc_conformal_update(&cp.state, C.double(nonconformityScore), &metaC)); err != nil {
		return ConformalOutput{}, err
	}
	return conformalOutFromC(metaC), nil
}

// UpdateBatch ingests a slice of nonconformity scores.
func (cp *MultiRateConformal) UpdateBatch(scores []float64) (ConformalOutput, error) {
	if len(scores) == 0 {
		return cp.Predict(0, 1).Meta, nil
	}
	var metaC C.eproc_conformal_out_t
	if err := checkRC(C.eproc_conformal_update_batch(
		&cp.state, (*C.double)(unsafe.Pointer(&scores[0])), C.size_t(len(scores)), &metaC,
	)); err != nil {
		return ConformalOutput{}, err
	}
	return conformalOutFromC(metaC), nil
}

// ConformalMonoid is an exact associative and commutative 640-byte distributed conformal monoid.
type ConformalMonoid struct {
	state C.eproc_cmonoid_t
}

// NewConformalMonoid initializes an empty distributed conformal monoid with reference scaleHint.
func NewConformalMonoid(scaleHint float64) *ConformalMonoid {
	m := &ConformalMonoid{}
	C.eproc_cmonoid_init(&m.state, C.double(scaleHint))
	return m
}

// ConformalMonoidWireSize returns the wire frame size in bytes reported by C11 core (640).
func ConformalMonoidWireSize() int {
	return int(C.eproc_cmonoid_wire_size())
}

// Add inserts a single score into the monoid.
func (m *ConformalMonoid) Add(score float64) {
	C.eproc_cmonoid_add(&m.state, C.double(score))
}

// AddBatch inserts a slice of scores into the monoid.
func (m *ConformalMonoid) AddBatch(scores []float64) {
	if len(scores) == 0 {
		return
	}
	C.eproc_cmonoid_add_batch(&m.state, (*C.double)(unsafe.Pointer(&scores[0])), C.size_t(len(scores)))
}

// Merge returns the bit-exact associative and commutative sum m + other.
func (m *ConformalMonoid) Merge(other *ConformalMonoid) *ConformalMonoid {
	out := &ConformalMonoid{}
	C.eproc_cmonoid_merge(&m.state, &other.state, &out.state)
	return out
}

// Subtract computes the exact leave-one-out abelian difference m - local.
func (m *ConformalMonoid) Subtract(local *ConformalMonoid) (*ConformalMonoid, error) {
	out := &ConformalMonoid{}
	if err := checkRC(C.eproc_cmonoid_subtract(&m.state, &local.state, &out.state)); err != nil {
		return nil, err
	}
	return out, nil
}

// Quantile returns the estimated q-quantile (q in [0, 1]).
func (m *ConformalMonoid) Quantile(q float64) float64 {
	return float64(C.eproc_cmonoid_quantile(&m.state, C.double(q)))
}

// Count returns the total finite observations accumulated.
func (m *ConformalMonoid) Count() uint64 {
	return uint64(m.state.count)
}

// BinCounts returns a copy of the 64 exact uint64 histogram bin counts.
func (m *ConformalMonoid) BinCounts() [64]uint64 {
	var bins [64]uint64
	for i := 0; i < 64; i++ {
		bins[i] = uint64(m.state.bin_counts[i])
	}
	return bins
}

// MarshalWire serializes the monoid into a checksummed little-endian [640]byte wire frame.
func (m *ConformalMonoid) MarshalWire() [WireSizeConformalMonoid]byte {
	var buf [WireSizeConformalMonoid]byte
	_ = C.eproc_cmonoid_serialize(
		&m.state, (*C.uint8_t)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)),
	)
	return buf
}

// UnmarshalConformalMonoid decodes and FNV-1a checksum-verifies a [640]byte wire frame.
func UnmarshalConformalMonoid(wire [WireSizeConformalMonoid]byte) (*ConformalMonoid, error) {
	return UnmarshalConformalMonoidBytes(wire[:])
}

// UnmarshalConformalMonoidBytes decodes and verifies a raw byte slice (must be exactly 640 bytes).
func UnmarshalConformalMonoidBytes(buf []byte) (*ConformalMonoid, error) {
	if len(buf) == 0 {
		return nil, ErrInvalid
	}
	m := &ConformalMonoid{}
	if err := checkRC(C.eproc_cmonoid_deserialize(
		&m.state, (*C.uint8_t)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)),
	)); err != nil {
		return nil, err
	}
	return m, nil
}

// ============================================================================
// 3. SafeFuser (Boehm-2012 Atomic Seqlock) & EchoPool
// ============================================================================

// BoaStep holds the output of a single SafeFuser.Update call.
type BoaStep struct {
	FusedPred      float64
	AnchorPred     float64
	FallbackActive bool
	NQuarantined   uint32
	SupLogE        float64
	DegLogE        float64
	NUpdates       uint64
}

// BoaSnapshot holds a tear-free Boehm-2012 atomic seqlock weight snapshot.
type BoaSnapshot struct {
	Weights        []float64
	FallbackActive bool
}

// SafeFuser implements Byzantine-safe BOA fusion with 3-gate Ville fallback and lock-free seqlock snapshots.
type SafeFuser struct {
	nExperts  int
	anchorIdx int
	state     C.eproc_boa_t
}

// NewSafeFuser initializes a SafeFuser over nExperts with baseline anchorIdx and degradation level alphaDeg.
func NewSafeFuser(nExperts, anchorIdx int, alphaDeg float64) (*SafeFuser, error) {
	if nExperts <= 0 || anchorIdx < 0 {
		return nil, ErrInvalid
	}
	f := &SafeFuser{nExperts: nExperts, anchorIdx: anchorIdx}
	if err := checkRC(C.eproc_boa_init(
		&f.state, C.uint32_t(nExperts), C.uint32_t(anchorIdx), C.double(alphaDeg),
	)); err != nil {
		return nil, err
	}
	return f, nil
}

// NExperts returns the configured number of experts.
func (f *SafeFuser) NExperts() int { return f.nExperts }

// AnchorIdx returns the anchor baseline expert index.
func (f *SafeFuser) AnchorIdx() int { return f.anchorIdx }

// Predict returns the Byzantine-safe fused prediction without mutating state.
func (f *SafeFuser) Predict(expertPreds []float64) (float64, error) {
	if len(expertPreds) != f.nExperts {
		return 0, ErrInvalid
	}
	return float64(C.eproc_boa_predict(&f.state, (*C.double)(unsafe.Pointer(&expertPreds[0])))), nil
}

// Update ingests expert predictions and ground truth yTrue, updating weights and publishing the seqlock.
func (f *SafeFuser) Update(expertPreds []float64, yTrue float64) (BoaStep, error) {
	if len(expertPreds) != f.nExperts {
		return BoaStep{}, ErrInvalid
	}
	var out C.eproc_boa_step_t
	if err := checkRC(C.eproc_boa_update(
		&f.state, (*C.double)(unsafe.Pointer(&expertPreds[0])), C.double(yTrue), &out,
	)); err != nil {
		return BoaStep{}, err
	}
	return BoaStep{
		FusedPred:      float64(out.fused_pred),
		AnchorPred:     float64(out.anchor_pred),
		FallbackActive: out.fallback_active != 0,
		NQuarantined:   uint32(out.n_quarantined),
		SupLogE:        float64(out.sup_log_e),
		DegLogE:        float64(out.deg_log_e),
		NUpdates:       uint64(out.n_updates),
	}, nil
}

// Snapshot reads a tear-free weight snapshot via the lock-free Boehm-2012 atomic seqlock.
// Safe to call concurrently with a single goroutine invoking Update.
func (f *SafeFuser) Snapshot() BoaSnapshot {
	weights := make([]float64, f.nExperts)
	var fb C.int32_t
	_ = C.eproc_boa_snapshot_weights(
		&f.state, (*C.double)(unsafe.Pointer(&weights[0])), C.uint32_t(f.nExperts), &fb,
	)
	return BoaSnapshot{
		Weights:        weights,
		FallbackActive: fb != 0,
	}
}

// FallbackActive reports whether the Ville degradation fallback gate is currently tripped.
func (f *SafeFuser) FallbackActive() bool { return f.state.fallback_active != 0 }

// IsQuarantined reports whether expert has been isolated by poisoning detection.
func (f *SafeFuser) IsQuarantined(expert int) bool {
	if expert < 0 || expert >= f.nExperts {
		return false
	}
	return f.state.quarantined[expert] != 0
}

// Weights returns a copy of the current expert weights.
func (f *SafeFuser) Weights() []float64 {
	w := make([]float64, f.nExperts)
	for i := 0; i < f.nExperts; i++ {
		w[i] = float64(f.state.weights[i])
	}
	return w
}

// EchoOutcome holds the result of an EchoPool.Submit call.
type EchoOutcome struct {
	Accepted          bool
	PosteriorProb     float64
	LogOdds           float64
	SourceReliability float64
	UniqueCount       uint64
}

// EchoPool is a provenance-deduplicated multi-agent evidence pool with Beta-Bernoulli peer reliability.
type EchoPool struct {
	state C.eproc_echopool_t
}

// NewEchoPool initializes an EchoPool with nSources peers, nClaims claims, and uniform priorProb.
func NewEchoPool(nSources, nClaims int, priorProb float64) (*EchoPool, error) {
	if nSources <= 0 || nClaims <= 0 {
		return nil, ErrInvalid
	}
	p := &EchoPool{}
	if err := checkRC(C.eproc_echopool_init(
		&p.state, C.uint32_t(nSources), C.uint32_t(nClaims), C.double(priorProb),
	)); err != nil {
		return nil, err
	}
	return p, nil
}

// Submit inserts an evidence item (claimID, evidenceID) with log-likelihood ratio llr from sourceID.
func (p *EchoPool) Submit(sourceID, claimID int, evidenceID uint64, llr float64) (EchoOutcome, error) {
	if sourceID < 0 || claimID < 0 {
		return EchoOutcome{}, ErrInvalid
	}
	var out C.eproc_echo_out_t
	if err := checkRC(C.eproc_echopool_submit(
		&p.state, C.uint32_t(sourceID), C.uint32_t(claimID), C.uint64_t(evidenceID), C.double(llr), &out,
	)); err != nil {
		return EchoOutcome{}, err
	}
	return EchoOutcome{
		Accepted:          out.accepted != 0,
		PosteriorProb:     float64(out.posterior_prob),
		LogOdds:           float64(out.log_odds),
		SourceReliability: float64(out.source_reliability),
		UniqueCount:       uint64(out.unique_count),
	}, nil
}

// Resolve records ground truth for claimID (idempotent) and updates peer reliability posteriors.
func (p *EchoPool) Resolve(claimID int, outcomePositive bool) error {
	if claimID < 0 {
		return ErrInvalid
	}
	pos := C.int32_t(0)
	if outcomePositive {
		pos = 1
	}
	return checkRC(C.eproc_echopool_resolve(&p.state, C.uint32_t(claimID), pos))
}

// Posterior returns the current posterior probability for claimID.
func (p *EchoPool) Posterior(claimID int) float64 {
	if claimID < 0 {
		return math.NaN()
	}
	return float64(C.eproc_echopool_posterior(&p.state, C.uint32_t(claimID)))
}

// SourceReliability returns the Beta-Bernoulli reliability weight in [0, 1] for sourceID.
func (p *EchoPool) SourceReliability(sourceID int) float64 {
	if sourceID < 0 {
		return math.NaN()
	}
	return float64(C.eproc_echopool_source_reliability(&p.state, C.uint32_t(sourceID)))
}

// NAccepted returns the number of unique evidence items accepted.
func (p *EchoPool) NAccepted() uint64 { return uint64(p.state.n_accepted) }

// NEchoDropped returns the number of duplicate echo submissions dropped.
func (p *EchoPool) NEchoDropped() uint64 { return uint64(p.state.n_echo_dropped) }

// ============================================================================
// 4. VarOptSampler, OnlineRLSSampler, SiftVOISelector
// ============================================================================

// VarOptSampler implements Cohen-2011 variance-optimal weighted reservoir sampling.
type VarOptSampler struct {
	state C.eproc_varopt_t
}

// NewVarOptSampler initializes a VarOpt sampler with reservoir size k and deterministic PRNG seed.
func NewVarOptSampler(k int, seed uint64) (*VarOptSampler, error) {
	if k <= 0 {
		return nil, ErrInvalid
	}
	v := &VarOptSampler{}
	if err := checkRC(C.eproc_varopt_init(&v.state, C.uint32_t(k), C.uint64_t(seed))); err != nil {
		return nil, err
	}
	return v, nil
}

// Push streams a weighted item into the reservoir.
func (v *VarOptSampler) Push(itemID uint64, weight, value float64) error {
	return checkRC(C.eproc_varopt_push(&v.state, C.uint64_t(itemID), C.double(weight), C.double(value)))
}

// EstimateSum returns the unbiased Horvitz-Thompson weighted sum estimate.
func (v *VarOptSampler) EstimateSum() float64 {
	return float64(C.eproc_varopt_estimate_sum(&v.state))
}

// EstimateMean returns the Horvitz-Thompson weighted mean estimate.
func (v *VarOptSampler) EstimateMean() float64 {
	return float64(C.eproc_varopt_estimate_mean(&v.state))
}

// Len returns the current number of items stored in the reservoir.
func (v *VarOptSampler) Len() int { return int(v.state.count) }

// AdjustedWeightSum returns the sum of Horvitz-Thompson adjusted weights in the reservoir.
func (v *VarOptSampler) AdjustedWeightSum() float64 {
	sum := 0.0
	for i := 0; i < int(v.state.count); i++ {
		sum += float64(v.state.items[i].adj_weight)
	}
	return sum
}

// RLSOutcome holds the sampling decision of OnlineRLSSampler.Step.
type RLSOutcome struct {
	Keep     bool
	Leverage float64
	Prob     float64
	HTWeight float64
}

// OnlineRLSSampler implements online ridge leverage score causal subsampling.
type OnlineRLSSampler struct {
	dim   int
	state C.eproc_rls_t
}

// NewOnlineRLSSampler initializes a d-dimensional online RLS subsampler.
func NewOnlineRLSSampler(d int, lambda, oversample, minProb float64, seed uint64) (*OnlineRLSSampler, error) {
	if d <= 0 {
		return nil, ErrInvalid
	}
	r := &OnlineRLSSampler{dim: d}
	if err := checkRC(C.eproc_rls_init(
		&r.state, C.uint32_t(d), C.double(lambda), C.double(oversample), C.double(minProb), C.uint64_t(seed),
	)); err != nil {
		return nil, err
	}
	return r, nil
}

// Step processes feature vector x and returns its leverage score and inclusion decision.
func (r *OnlineRLSSampler) Step(x []float64) (RLSOutcome, error) {
	if len(x) != r.dim {
		return RLSOutcome{}, ErrInvalid
	}
	var out C.eproc_rls_out_t
	if err := checkRC(C.eproc_rls_step(&r.state, (*C.double)(unsafe.Pointer(&x[0])), &out)); err != nil {
		return RLSOutcome{}, err
	}
	return RLSOutcome{
		Keep:     out.keep != 0,
		Leverage: float64(out.leverage),
		Prob:     float64(out.prob),
		HTWeight: float64(out.ht_weight),
	}, nil
}

// SiftActionType denotes the sensing resolution chosen by SiftVOISelector.
type SiftActionType int32

const (
	SiftReadSiteMean    SiftActionType = 0
	SiftReadChannelMean SiftActionType = 1
	SiftReadRawTicks    SiftActionType = 2
)

// SiftAction represents the optimal next Value-of-Information action.
type SiftAction struct {
	ActionType  SiftActionType
	TargetIdx   int
	EVOIPerCost float64
}

// SiftVOISelector implements hierarchical multi-resolution Bayes Value-of-Information selection.
type SiftVOISelector struct {
	state C.eproc_sift_t
}

// NewSiftVOISelector initializes a SiftVOISelector over nSites x perSite channels.
func NewSiftVOISelector(nSites, perSite int, signalMean, noiseSD float64) (*SiftVOISelector, error) {
	if nSites <= 0 || perSite <= 0 {
		return nil, ErrInvalid
	}
	sf := &SiftVOISelector{}
	if err := checkRC(C.eproc_sift_init(
		&sf.state, C.uint32_t(nSites), C.uint32_t(perSite), C.double(signalMean), C.double(noiseSD),
	)); err != nil {
		return nil, err
	}
	return sf, nil
}

// SetNoiseScale sets a per-channel heteroskedastic noise multiplier.
func (sf *SiftVOISelector) SetNoiseScale(channelIdx int, noiseScale float64) error {
	if channelIdx < 0 {
		return ErrInvalid
	}
	return checkRC(C.eproc_sift_set_noise_scale(&sf.state, C.uint32_t(channelIdx), C.double(noiseScale)))
}

// ObserveSite incorporates a coarse site-mean observation.
func (sf *SiftVOISelector) ObserveSite(siteIdx int, siteMean float64) error {
	if siteIdx < 0 {
		return ErrInvalid
	}
	return checkRC(C.eproc_sift_observe_site(&sf.state, C.uint32_t(siteIdx), C.double(siteMean)))
}

// ObserveChannel incorporates a fine channel observation averaged over nTicks.
func (sf *SiftVOISelector) ObserveChannel(channelIdx int, obsMean float64, nTicks uint32) error {
	if channelIdx < 0 {
		return ErrInvalid
	}
	return checkRC(C.eproc_sift_observe_channel(
		&sf.state, C.uint32_t(channelIdx), C.double(obsMean), C.uint32_t(nTicks),
	))
}

// SelectNext selects the action maximizing expected Value-of-Information per unit cost.
func (sf *SiftVOISelector) SelectNext(costSite, costChannel, costTicks float64, ticksPerRead uint32) (SiftAction, error) {
	var act C.eproc_sift_action_t
	if err := checkRC(C.eproc_sift_select_next(
		&sf.state,
		C.double(costSite),
		C.double(costChannel),
		C.double(costTicks),
		C.uint32_t(ticksPerRead),
		&act,
	)); err != nil {
		return SiftAction{}, err
	}
	return SiftAction{
		ActionType:  SiftActionType(act.action_type),
		TargetIdx:   int(act.target_idx),
		EVOIPerCost: float64(act.evoi_per_cost),
	}, nil
}

// SiteProb returns the posterior probability that siteIdx is hot.
func (sf *SiftVOISelector) SiteProb(siteIdx int) float64 {
	if siteIdx < 0 || siteIdx >= int(sf.state.n_sites) {
		return math.NaN()
	}
	return math.Exp(float64(sf.state.site_log_prob[siteIdx]))
}

// ChannelProb returns the posterior probability that flat channelIdx is active.
func (sf *SiftVOISelector) ChannelProb(channelIdx int) float64 {
	total := int(sf.state.n_sites) * int(sf.state.per_site)
	if channelIdx < 0 || channelIdx >= total {
		return math.NaN()
	}
	return float64(sf.state.chan_prob[channelIdx])
}
