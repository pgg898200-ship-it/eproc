package eproc

import (
	"errors"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

type splitMix64 struct {
	state uint64
}

func (r *splitMix64) next() uint64 {
	r.state += 0x9e3779b97f4a7c15
	z := r.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func (r *splitMix64) u01() float64 {
	return float64(r.next()>>11) * (1.0 / 9007199254740992.0)
}

func (r *splitMix64) norm() float64 {
	u1 := r.u01()
	if u1 < 1e-15 {
		u1 = 1e-15
	}
	u2 := r.u01()
	return math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
}

func TestABIVersionAndSIMDFlags(t *testing.T) {
	if got := ABIVersion(); got != 1 {
		t.Fatalf("ABIVersion() = %d, want 1", got)
	}
	if got := ConformalMonoidWireSize(); got != WireSizeConformalMonoid {
		t.Fatalf("ConformalMonoidWireSize() = %d, want %d", got, WireSizeConformalMonoid)
	}
	flags := SIMDFlags()
	if flags&^(SIMDAVX2|SIMDFMA|SIMDNEON) != 0 {
		t.Fatalf("unexpected unknown SIMD flag bits: 0x%x", flags)
	}
	t.Logf("ABI v%d | GOARCH=%s | SIMD flags=0x%x (AVX2=%v)", ABIVersion(), runtime.GOARCH, flags, (flags&SIMDAVX2) != 0)
	if runtime.GOARCH == "amd64" && (flags&SIMDAVX2) == 0 {
		t.Logf("note: AVX2 not active on this amd64 CPU (scalar fallback active)")
	}
}

func TestEProcessVilleAndCombiners(t *testing.T) {
	rng := &splitMix64{state: 12345}
	falseRejects := 0
	const trials = 300
	for tr := 0; tr < trials; tr++ {
		ep, err := NewEProcess(0.5, 0.05, 0.0, 1.0, SideUpper)
		if err != nil {
			t.Fatalf("NewEProcess failed: %v", err)
		}
		var res EProcessResult
		for step := 0; step < 300; step++ {
			x := 0.0
			if rng.u01() < 0.5 {
				x = 1.0
			}
			res, err = ep.Update(x)
			if err != nil {
				t.Fatalf("Update failed: %v", err)
			}
		}
		if res.Rejected {
			falseRejects++
		}
	}
	fpr := float64(falseRejects) / float64(trials)
	if fpr > 0.06 {
		t.Fatalf("Type-I FPR = %.4f exceeds anytime-valid bound", fpr)
	}

	// Power under H1 + NaN/Inf sanitization + scalar/batch bit-identity.
	xs := make([]float64, 260)
	for i := 0; i < 250; i++ {
		if rng.u01() < 0.72 {
			xs[i] = 1.0
		}
	}
	xs[250] = math.NaN()
	xs[251] = math.Inf(1)
	xs[252] = math.Inf(-1)
	xs[253] = 1e-315
	xs[254] = 1e15
	xs[255] = -1e15
	for i := 256; i < 260; i++ {
		xs[i] = 0.8
	}

	epS, err := NewEProcess(0.5, 0.01, 0.0, 1.0, SideUpper)
	if err != nil {
		t.Fatal(err)
	}
	epB, err := NewEProcess(0.5, 0.01, 0.0, 1.0, SideUpper)
	if err != nil {
		t.Fatal(err)
	}
	var rs EProcessResult
	for _, x := range xs {
		rs, err = epS.Update(x)
		if err != nil {
			t.Fatal(err)
		}
	}
	rb, err := epB.UpdateBatch(xs)
	if err != nil {
		t.Fatal(err)
	}
	if !rs.Rejected || rs.PValue >= 0.01 {
		t.Fatalf("expected H1 rejection with p < 0.01, got rejected=%v p=%v", rs.Rejected, rs.PValue)
	}
	if epS.N() != 257 || epS.NInvalid() != 3 {
		t.Fatalf("got N=%d NInvalid=%d, want 257 and 3", epS.N(), epS.NInvalid())
	}
	if math.Float64bits(rs.LogE) != math.Float64bits(rb.LogE) {
		t.Fatalf("batch LogE mismatch: scalar=%v batch=%v", rs.LogE, rb.LogE)
	}

	if math.Abs(CombineIndependentLog(2.5, 3.5)-6.0) > 1e-12 {
		t.Fatal("CombineIndependentLog mismatch")
	}
	dep := CombineDependentLog([]float64{0.0, math.Log(3.0), math.Log(5.0)})
	if math.Abs(dep-math.Log(3.0)) > 1e-12 {
		t.Fatalf("CombineDependentLog = %v, want %v", dep, math.Log(3.0))
	}
}

func TestConfidenceSequenceAndEDetector(t *testing.T) {
	rng := &splitMix64{state: 54321}
	eb, err := NewEmpiricalBernsteinCS(0.05, 0.0, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	nm, err := NewNormalMixtureCS(0.05, 1.0, 100.0)
	if err != nil {
		t.Fatal(err)
	}
	prevW := math.Inf(1)
	for step := 0; step < 300; step++ {
		x := 0.0
		if rng.u01() < 0.35 {
			x = 1.0
		}
		r, err := eb.Update(x)
		if err != nil {
			t.Fatal(err)
		}
		if r.Width > prevW+1e-15 {
			t.Fatalf("EB CS width expanded: %v > %v", r.Width, prevW)
		}
		prevW = r.Width
		if _, err := nm.Update(2.5 + rng.norm()); err != nil {
			t.Fatal(err)
		}
	}
	reb := eb.Query()
	rnm := nm.Query()
	if !(reb.Low <= 0.35 && reb.High >= 0.35) {
		t.Fatalf("EB CS [%v, %v] missed true mean 0.35", reb.Low, reb.High)
	}
	if !(rnm.Low <= 2.5 && rnm.High >= 2.5) {
		t.Fatalf("NormalMix CS [%v, %v] missed true mean 2.5", rnm.Low, rnm.High)
	}

	ed, err := NewEDetector(0.5, 0.01, 0.0, 1.0, SideUpper)
	if err != nil {
		t.Fatal(err)
	}
	for step := 0; step < 200; step++ {
		x := 0.0
		if rng.u01() < 0.5 {
			x = 1.0
		}
		r, err := ed.Update(x)
		if err != nil {
			t.Fatal(err)
		}
		if r.Alarm {
			t.Fatalf("unexpected H0 false alarm at step %d", step)
		}
	}
	detectedAt := -1
	for step := 0; step < 200; step++ {
		x := 0.0
		if rng.u01() < 0.85 {
			x = 1.0
		}
		r, err := ed.Update(x)
		if err != nil {
			t.Fatal(err)
		}
		if r.Alarm && detectedAt < 0 {
			detectedAt = step
		}
	}
	if detectedAt < 0 || detectedAt >= 80 {
		t.Fatalf("drift detection too slow: detectedAt=%d", detectedAt)
	}
	ed.ResetAlarm()
}

func TestAgentRace(t *testing.T) {
	rng := &splitMix64{state: 99999}
	means := []float64{0.25, 0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.88}
	race, err := NewAgentRace(len(means), 0.05, 0.0, 1.0, 0.0)
	if err != nil {
		t.Fatal(err)
	}
	var st RaceStatus
	for step := 0; step < 5000; step++ {
		arm := race.SuggestArm()
		if arm < 0 || arm >= len(means) {
			t.Fatalf("invalid suggested arm: %d", arm)
		}
		rew := 0.0
		if rng.u01() < means[arm] {
			rew = 1.0
		}
		st, err = race.Update(arm, rew)
		if err != nil {
			t.Fatal(err)
		}
		if st.Winner >= 0 {
			break
		}
	}
	if st.Winner != 7 || st.NActive != 1 {
		t.Fatalf("got winner=%d active=%d, want winner=7 active=1", st.Winner, st.NActive)
	}
	arm7, active7, err := race.ArmState(7)
	if err != nil || !active7 || arm7.Mean <= 0.75 {
		t.Fatalf("unexpected arm 7 state: %+v active=%v err=%v", arm7, active7, err)
	}
}

func TestMultiRateConformalCrossingFree(t *testing.T) {
	rng := &splitMix64{state: 77777}
	unsorted := []float64{0.05, 0.50, 0.01, 0.20, 0.10}
	cp, err := NewMultiRateConformal(unsorted, 128, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	alphas := cp.TargetAlphas()
	if len(alphas) != 5 || alphas[0] != 0.50 || alphas[4] != 0.01 {
		t.Fatalf("unexpected sorted target alphas: %v", alphas)
	}

	var last ConformalOutput
	for step := 0; step < 800; step++ {
		scale := 1.0
		if step >= 400 {
			scale = 3.0
		}
		s := math.Abs(rng.norm()) * scale
		if step == 123 {
			s = math.NaN()
		} else if step == 456 {
			s = math.Inf(1)
		}
		last, err = cp.Update(s)
		if err != nil {
			t.Fatal(err)
		}
		for r := 1; r < 5; r++ {
			if last.Quantiles[r] < last.Quantiles[r-1] {
				t.Fatalf("quantile crossing at step %d: %v", step, last.Quantiles)
			}
		}
		iv := cp.Predict(10.0, 1.0)
		for r := 1; r < 5; r++ {
			if iv.Lows[r] > iv.Lows[r-1] || iv.Highs[r] < iv.Highs[r-1] {
				t.Fatalf("non-nested intervals at step %d", step)
			}
		}
	}
	if last.EmpCoverage[2] <= 0.75 || last.EmpCoverage[2] >= 0.98 {
		t.Fatalf("90%% empirical coverage out of expected range: %.3f", last.EmpCoverage[2])
	}
}

func TestConformalMonoidAbelianLOOAndWire(t *testing.T) {
	rng := &splitMix64{state: 88888}
	a := NewConformalMonoid(1.0)
	b := NewConformalMonoid(1.0)
	c := NewConformalMonoid(1.0)

	for i := 0; i < 500; i++ {
		a.Add(rng.u01() * 10.0)
		b.Add(rng.u01() * 10.0)
		c.Add(rng.u01()*20.0 - 5.0)
	}
	a.Add(math.NaN())
	a.Add(math.Inf(1))
	if a.Count() != 500 {
		t.Fatalf("expected NaN/Inf skipped, got count=%d", a.Count())
	}

	// Associativity: (A + B) + C == A + (B + C)
	ab := a.Merge(b)
	abc1 := ab.Merge(c)
	abc2 := a.Merge(b.Merge(c))
	if abc1.Count() != abc2.Count() || abc1.BinCounts() != abc2.BinCounts() {
		t.Fatal("associativity drift detected")
	}

	// Commutativity: A + B == B + A
	ba := b.Merge(a)
	if ab.BinCounts() != ba.BinCounts() {
		t.Fatal("commutativity drift detected")
	}

	// Exact abelian leave-one-out subtraction: (A + B) - A == B
	loo, err := ab.Subtract(a)
	if err != nil {
		t.Fatalf("Subtract failed: %v", err)
	}
	if loo.Count() != b.Count() || loo.BinCounts() != b.BinCounts() {
		t.Fatal("leave-one-out subtraction not bit-exact")
	}

	// Underflow protection
	if _, err := a.Subtract(ab); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected ErrInvalid on underflow, got %v", err)
	}

	// Quantile sanity on Uniform(0, 10)
	q50 := ab.Quantile(0.50)
	q90 := ab.Quantile(0.90)
	if math.Abs(q50-5.0) >= 0.7 || math.Abs(q90-9.0) >= 0.9 {
		t.Fatalf("unexpected quantiles q50=%.3f q90=%.3f", q50, q90)
	}

	// 640-byte wire serialization roundtrip + bit-flip detection
	wire := abc1.MarshalWire()
	decoded, err := UnmarshalConformalMonoid(wire)
	if err != nil {
		t.Fatalf("UnmarshalConformalMonoid failed: %v", err)
	}
	if decoded.Count() != abc1.Count() || decoded.BinCounts() != abc1.BinCounts() {
		t.Fatal("wire roundtrip mismatch")
	}

	wire[100] ^= 0x01
	if _, err := UnmarshalConformalMonoid(wire); !errors.Is(err, ErrMismatch) {
		t.Fatalf("expected ErrMismatch on corrupted wire frame, got %v", err)
	}
	if _, err := UnmarshalConformalMonoidBytes(wire[:639]); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected ErrInvalid on truncated wire frame, got %v", err)
	}
}

func TestSafeFuserByzantineQuarantineAndSeqlock(t *testing.T) {
	rng := &splitMix64{state: 314159}
	fuser, err := NewSafeFuser(8, 0, 0.05)
	if err != nil {
		t.Fatal(err)
	}

	preds := make([]float64, 8)
	for step := 0; step < 100; step++ {
		y := 10.0 + math.Sin(float64(step)*0.1)
		preds[0] = y + 0.5*rng.norm()
		preds[1] = y + 0.4*rng.norm()
		preds[2] = y + 0.3*rng.norm()
		preds[3] = y + 0.05*rng.norm() // best expert
		preds[4] = y + 0.8*rng.norm()
		if step >= 70 && step <= 74 {
			preds[5] = y + 50.0
		} else {
			preds[5] = y + 0.5*rng.norm()
		}
		if step == 60 {
			preds[6] = math.NaN()
		} else {
			preds[6] = y + 0.5*rng.norm()
		}
		if step == 50 {
			preds[7] = 1e6
		} else {
			preds[7] = y + 0.5*rng.norm()
		}

		pPred, err := fuser.Predict(preds)
		if err != nil {
			t.Fatal(err)
		}
		if step == 50 && math.Abs(pPred-y) >= 5.0 {
			t.Fatalf("1e6 Byzantine attack had blast radius %.3f", math.Abs(pPred-y))
		}
		st, err := fuser.Update(preds, y)
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(st.FusedPred) || math.IsInf(st.FusedPred, 0) {
			t.Fatal("non-finite fused prediction")
		}
	}
	if !fuser.IsQuarantined(7) || !fuser.IsQuarantined(6) || !fuser.IsQuarantined(5) {
		t.Fatal("expected Byzantine/NaN/3-strike experts to be quarantined")
	}
	if fuser.Weights()[3] <= 0.35 {
		t.Fatalf("expected BOA to concentrate on best expert 3, got w=%.3f", fuser.Weights()[3])
	}

	// Concurrent Boehm-2012 atomic seqlock stress test (4 reader goroutines + 1 writer goroutine).
	live, err := NewSafeFuser(8, 0, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	var stop atomic.Bool
	var torn atomic.Uint64
	var wg sync.WaitGroup

	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var localTorn uint64
			for it := 0; it < 50000; it++ {
				snap := live.Snapshot()
				sum := 0.0
				bad := false
				for i, w := range snap.Weights {
					if math.IsNaN(w) || math.IsInf(w, 0) || w < 0.0 || w > 1.0+1e-12 {
						bad = true
						break
					}
					if snap.FallbackActive {
						expW := 0.0
						if i == 0 {
							expW = 1.0
						}
						if w != expW {
							bad = true
							break
						}
					}
					sum += w
				}
				if bad || math.Abs(sum-1.0) > 1e-9 {
					localTorn++
				}
			}
			torn.Add(localTorn)
		}()
	}

	writerDone := make(chan struct{})
	go func() {
		defer close(writerDone)
		wrng := &splitMix64{state: 0xCAFEBABE}
		p := make([]float64, 8)
		for n := uint64(0); !stop.Load(); n++ {
			y := math.Sin(float64(n)*0.01) * 5.0
			for i := 0; i < 8; i++ {
				p[i] = y + (0.2+0.3*float64(i))*wrng.norm()
			}
			_, _ = live.Update(p, y)
		}
	}()

	wg.Wait()
	stop.Store(true)
	<-writerDone
	if torn.Load() != 0 {
		t.Fatalf("observed %d torn seqlock reads", torn.Load())
	}
}

func TestEchoPoolDedupAndReliability(t *testing.T) {
	pool, err := NewEchoPool(16, 32, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	o1, err := pool.Submit(0, 0, 1001, 2.0)
	if err != nil || !o1.Accepted || o1.UniqueCount != 1 {
		t.Fatalf("first submit failed: %+v err=%v", o1, err)
	}
	for s := 1; s < 10; s++ {
		dup, err := pool.Submit(s, 0, 1001, 2.0)
		if err != nil {
			t.Fatal(err)
		}
		if dup.Accepted {
			t.Fatalf("duplicate echo from source %d was accepted", s)
		}
		if math.Float64bits(dup.PosteriorProb) != math.Float64bits(o1.PosteriorProb) {
			t.Fatal("posterior mutated by duplicate echo")
		}
	}
	if pool.NEchoDropped() != 9 {
		t.Fatalf("got NEchoDropped=%d, want 9", pool.NEchoDropped())
	}

	for c := 1; c <= 10; c++ {
		if _, err := pool.Submit(0, c, uint64(2000+c), 2.0); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Submit(1, c, uint64(3000+c), -2.0); err != nil {
			t.Fatal(err)
		}
		if err := pool.Resolve(c, true); err != nil {
			t.Fatal(err)
		}
		if err := pool.Resolve(c, true); err != nil {
			t.Fatal(err)
		}
	}
	if pool.SourceReliability(0) <= 0.99 {
		t.Fatalf("honest peer reliability dropped: %.3f", pool.SourceReliability(0))
	}
	if pool.SourceReliability(1) != 0.0 {
		t.Fatalf("adversarial peer reliability = %.3f, want 0.0", pool.SourceReliability(1))
	}
	if pool.Posterior(0) <= 0.5 {
		t.Fatalf("claim 0 posterior = %.3f, want > 0.5", pool.Posterior(0))
	}
}

func TestVarOptRLSAndSiftVOI(t *testing.T) {
	rng := &splitMix64{state: 2026}
	vo, err := NewVarOptSampler(32, 42)
	if err != nil {
		t.Fatal(err)
	}
	trueW := 0.0
	for i := 0; i < 200; i++ {
		w := 0.5 + 3.0*rng.u01()
		if i%20 == 0 {
			w = 50.0
		}
		trueW += w
		if err := vo.Push(uint64(i), w, 1.0+0.1*float64(i)); err != nil {
			t.Fatal(err)
		}
	}
	if vo.Len() != 32 {
		t.Fatalf("reservoir len = %d, want 32", vo.Len())
	}
	if math.Abs(vo.AdjustedWeightSum()-trueW) >= 1e-8*trueW {
		t.Fatalf("VarOpt weight conservation drift: got %.6f want %.6f", vo.AdjustedWeightSum(), trueW)
	}

	rls, err := NewOnlineRLSSampler(16, 1.0, 2.0, 0.05, 1337)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float64, 16)
	for step := 0; step < 100; step++ {
		for i := range x {
			x[i] = rng.norm()
		}
		ro, err := rls.Step(x)
		if err != nil {
			t.Fatal(err)
		}
		if ro.Leverage < 0.0 || ro.Prob < 0.05 || ro.Prob > 1.0 {
			t.Fatalf("invalid RLS output: %+v", ro)
		}
	}

	sf, err := NewSiftVOISelector(16, 8, 2.0, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	act0, err := sf.SelectNext(1.0, 1.0, 4.0, 16)
	if err != nil || act0.ActionType != SiftReadSiteMean {
		t.Fatalf("expected coarse site read initially, got %+v err=%v", act0, err)
	}
	loudCh := 5*8 + 0
	cleanCh := 5*8 + 1
	if err := sf.SetNoiseScale(loudCh, 3.0); err != nil {
		t.Fatal(err)
	}
	if err := sf.ObserveSite(5, 1.6); err != nil {
		t.Fatal(err)
	}
	if sf.SiteProb(5) <= 0.5 {
		t.Fatalf("expected hot site 5 posterior > 0.5, got %.3f", sf.SiteProb(5))
	}
	act1, err := sf.SelectNext(1.0, 1.0, 4.0, 16)
	if err != nil || act1.ActionType != SiftReadChannelMean || act1.TargetIdx != cleanCh {
		t.Fatalf("expected zoom into clean channel %d, got %+v err=%v", cleanCh, act1, err)
	}
	if err := sf.ObserveChannel(cleanCh, 2.5, 4); err != nil {
		t.Fatal(err)
	}
	if sf.ChannelProb(cleanCh) <= 0.95 {
		t.Fatalf("expected active channel posterior > 0.95, got %.3f", sf.ChannelProb(cleanCh))
	}
}
