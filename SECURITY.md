# Security Policy

## Supported Versions

Security fixes and hardening updates are provided for the current release series:

| Version | Supported |
|---|---|
| `0.1.x` | Yes |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for suspected security vulnerabilities, memory-safety bugs (buffer overflows, undefined behavior, data races in C11 or the Python C-API extension), or MCP JSON-RPC parsing issues.

Instead, report privately via **GitHub Security Advisories**:

- **Private Advisory Form:** `https://github.com/pgg898200-ship-it/eproc/security/advisories/new`

Please include:

1. Affected component (`csrc/`, `dist/eproc_single.h`, `python/eproc/`, or `eproc-mcp`) and version / commit SHA.
2. Minimal deterministic reproduction script or C test case (including compiler flags or Python version).
3. Observed impact (e.g., ASan/UBSan/TSan diagnostic, crash, invalid statistical guarantee, or JSON-RPC DoS).

We aim to acknowledge reports within **72 hours** and provide an initial triage assessment or fix timeline within **7 days**.

## Supply-Chain & Cryptographic Integrity Verification (`SHA256SUMS`)

Canonical headers, benchmark receipts, and license files are hashed in [`SHA256SUMS`](SHA256SUMS). Tagged GitHub Releases also publish `SHA256SUMS.txt` covering wheels, source distributions, and `eproc_single.h`.

### Verify Repository Files (Linux)

```bash
sha256sum -c SHA256SUMS
```

### Verify Repository Files (macOS / BSD)

```bash
shasum -a 256 -c SHA256SUMS
```

### Verify a Downloaded Release Artifact

```bash
# Download release assets and SHA256SUMS.txt from:
# https://github.com/pgg898200-ship-it/eproc/releases
sha256sum -c --ignore-missing SHA256SUMS.txt
```

### Verify Single-Header Integrity Before Embedding

```bash
sha256sum dist/eproc_single.h
grep 'dist/eproc_single.h' SHA256SUMS
```

## Built-In Hardening & Memory-Safety Guarantees

Every release of `eproc` is verified against the following mechanical invariants:

- **Zero heap allocations in C11 core:** `make check-no-alloc` verifies `0` undefined references to `malloc`, `calloc`, `realloc`, or `free` in `libeproc.so` and `libeproc_portable.so`.
- **Zero third-party runtime dependencies:** `python3 scripts/verify_release_hygiene.py --strict` AST-checks `python/eproc/` to guarantee stdlib-only imports.
- **Sanitizer verification:** C11 kernels and ABI torture harnesses are tested under AddressSanitizer (`-fsanitize=address`), UndefinedBehaviorSanitizer (`-fsanitize=undefined,float-cast-overflow`), and ThreadSanitizer (`-fsanitize=thread`).
- **Strict input sanitization:** Non-finite (`NaN`, `±Inf`), subnormal (`1e-310`), and extreme finite (`±1e308`) inputs are handled deterministically without undefined behavior or state corruption; `ConformalMonoid.from_bytes` verifies exact 640-byte wire framing and checksums.
- **Bounded MCP session memory:** `eproc-mcp` enforces strict JSON-RPC 2.0 type checking (rejecting boolean-as-number coercion and malformed requests) and LRU-bounded session storage.
