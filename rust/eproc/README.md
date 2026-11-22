# `eproc` for Rust

Zero-dependency Rust bindings for `eproc`, compiling directly against the canonical zero-allocation C11 core (`../../csrc/`) via `build.rs` with zero external crates (`[dependencies]` and `[build-dependencies]` are empty).

## Capabilities

- **Anytime-valid inference:** `EProcess`, `ConfidenceSequence`, `EDetector`, `AgentRace`, `combine_independent_log`, `combine_dependent_log`
- **Multi-rate online conformal:** `MultiRateConformal` (up to 5 crossing-free rates), `DelayedConformal`, `BudgetConformal`, and `ConformalMonoid` (640-byte checksummed wire format with exact abelian `merge` and leave-one-out `subtract`)
- **Byzantine-safe multi-agent fusion:** `SafeFuser` (`Send + Sync` with wait-free Boehm-2012 atomic seqlock `SafeFuserReader::snapshot`) and `EchoPool`
- **Causal subsampling & VOI:** `VarOptSampler`, `OnlineRLSSampler`, `SiftVOISelector`
- **Runtime introspection:** `abi_version()`, `simd_flags()`

## Quickstart

Add `eproc` as a path dependency (or from the repository workspace):

```toml
[dependencies]
eproc = { path = "rust/eproc" }
```

```rust
use eproc::{ConformalMonoid, EProcess, MultiRateConformal, SafeFuser, Side};

fn main() -> eproc::Result<()> {
    // 1. Anytime-valid ONS betting e-process (exact Ville guarantee)
    let mut ep = EProcess::new(0.5, 0.05, 0.0, 1.0, Side::Upper)?;
    for x in [1.0, 1.0, 0.8, 1.0, 0.9] {
        let res = ep.update(x)?;
        if res.rejected {
            println!("Rejected H0 at n={} (p={:.4})", res.n, res.p_value);
            break;
        }
    }

    // 2. 5-rate crossing-free conformal calibrator
    let mut cp = MultiRateConformal::new(&[0.50, 0.20, 0.10, 0.05, 0.01], 128, 0.05)?;
    cp.update(0.42)?;
    let intervals = cp.predict(10.0, 1.0);
    assert_eq!(intervals.lows().len(), 5);

    // 3. 640-byte abelian conformal monoid
    let mut m = ConformalMonoid::new(1.0);
    m.add_batch(&[0.2, 0.5, 1.1]);
    let wire: [u8; 640] = m.to_bytes();
    let roundtrip = ConformalMonoid::from_bytes(&wire)?;
    assert_eq!(roundtrip.count(), 3);

    // 4. Byzantine-safe fuser with concurrent lock-free seqlock snapshot
    let mut fuser = SafeFuser::new(4, 0, 0.05)?;
    let reader = fuser.reader();
    let _step = fuser.update(&[10.0, 10.1, 1e6, 9.9], 10.0)?;
    let snap = reader.snapshot();
    assert_eq!(snap.weights().len(), 4);
    Ok(())
}
```

## Running Tests

```sh
cargo test --manifest-path rust/eproc/Cargo.toml --release
```
