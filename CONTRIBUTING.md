# Contributing to `eproc`

Thank you for your interest in contributing to `eproc`. Because `eproc` provides statistical safety guarantees (Ville's inequality, crossing-free conformal calibration, strict causality, and Byzantine-safe fusion) in a zero-allocation C11 engine, all contributions must pass the deterministic verification bar below.

## Core Design Invariants

1. **Zero heap allocations in C11 (`csrc/`):** `libeproc.so` and `libeproc_portable.so` must have `0` references to `malloc`, `calloc`, `realloc`, or `free` (enforced by `make check-no-alloc`).
2. **0-ULP portability:** AVX2/FMA (`-march=native`) and portable scalar builds must produce identical 64-bit state fingerprints under `-std=c11 -ffp-contract=off`.
3. **Zero third-party Python dependencies (`python/eproc/`):** Only Python 3.10+ standard library modules and internal `eproc` modules may be imported (enforced by `scripts/verify_release_hygiene.py`).
4. **Cold import budget:** In-process `import eproc` must remain `< 20 ms` (PEP 562 lazy exports).
5. **Strict causality & adversarial resilience:** No future lookahead, no `NaN`/`Inf`/`±1e308` corruption, no torn reads across concurrent `SafeFuser.snapshot()` readers.
6. **Zero workstation paths or personal identifiers:** Never commit local filesystem paths or private identifiers anywhere in code, tests, or documentation.

## Local Verification Bar (Required Before Opening a PR)

Run the full C11, sanitizer, Python (native + ctypes fallback), breaker, single-header, and hygiene checks:

```bash
# 1. Build native + portable C11 libraries, verify 0-malloc & 0-ULP bit-identity
make test

# 2. AddressSanitizer + UBSan + float-cast-overflow
make test-asan

# 3. ThreadSanitizer + Boehm-2012 lock-free seqlock verification
make test-tsan

# 4. Build native CPython extension (if csrc/ or eproc_pyext.c changed)
python3 scripts/build_pyext.py

# 5. Full Python unit, adapter, and 42-test Adversarial Breaker suites (native C-API)
PYTHONPATH=python python3 -m unittest discover -s tests -v

# 6. Full Python suites on pure ctypes fallback (verifies native/ctypes parity)
EPROC_DISABLE_NATIVE=1 PYTHONPATH=python python3 -m unittest discover -s tests -v

# 7. Regenerate and compile-check single-header C11 & C++17 (if csrc/ changed)
python3 scripts/make_single_header.py

# 8. Deterministic release hygiene & zero-dependency gate
python3 scripts/verify_release_hygiene.py --strict
```

### Adversarial C11 Torture Harness

If your change touches `csrc/` or `csrc/eproc_seqlock.h`, also run the standalone C11 torture harness under ASan/UBSan and TSan:

```bash
cc -std=c11 -O2 -g -ffp-contract=off -Wall -Wextra -Werror -Icsrc \
   -fsanitize=address,undefined,float-cast-overflow \
   tests/breaker_c_torture.c csrc/eproc_eprocess.c csrc/eproc_conformal.c \
   csrc/eproc_fusion.c csrc/eproc_subsample.c -pthread -lm -o /tmp/breaker_asan && /tmp/breaker_asan

cc -std=c11 -O2 -g -ffp-contract=off -Wall -Wextra -Werror -Icsrc \
   -fsanitize=thread -Wno-tsan \
   tests/breaker_c_torture.c csrc/eproc_eprocess.c csrc/eproc_conformal.c \
   csrc/eproc_fusion.c csrc/eproc_subsample.c -pthread -lm -o /tmp/breaker_tsan && /tmp/breaker_tsan
```

### Reproducible Benchmarks (When Changing Hot Paths)

Any PR claiming a performance improvement or modifying mathematical kernels should include live output from:

```bash
PYTHONPATH=python python3 benchmarks/bench_suite.py
PYTHONPATH=python python3 benchmarks/compare_incumbents.py
```

## Pull Request Guidelines

- Keep PRs focused and include targeted regression tests in `tests/` (unit tests and, where relevant, adversarial witnesses in `tests/test_breaker.py`).
- Preserve existing public C11 ABI struct layouts unless bumping `EPROC_ABI_VERSION` and updating both `csrc/eproc.h` and `python/eproc/_ffi.py` in lockstep.
- Update `SHA256SUMS` if you intentionally modify any tracked release artifact (`dist/eproc_single.h`, `csrc/eproc.h`, `csrc/eproc_seqlock.h`, `benchmarks/receipts.json`, `LICENSE`, `NOTICE`).

## Licensing

By contributing to `eproc`, you agree that your contributions will be licensed under the Apache License, Version 2.0 (see [`LICENSE`](LICENSE)).
