# `eproc` Organic Distribution & Release Checklist

This checklist follows a transparent, engineering-first release model: zero astroturfing, zero spam, reproducible benchmarks, supply-chain checksums (`SHA256SUMS`), and frictionless discovery for both humans and coding agents (`llms.txt`, `llms-full.txt`, MCP 2026-07-28, Agent Skill).

---

## 1. Repository Setup

1. Target public repository: `https://github.com/pgg898200-ship-it/eproc`.
2. Push `main` using a repository-scoped SSH Deploy Key (or authenticated CLI):
   ```bash
   git remote add origin git@github.com:pgg898200-ship-it/eproc.git
   git push -u origin main
   ```
3. Set GitHub repository metadata for organic discovery:
   - **Description:** `Sub-microsecond anytime-valid statistics, multi-rate online conformal calibration, and Byzantine-safe multi-agent fusion (zero-dep C11/AVX2 + Python + C++17 + MCP).`
   - **Website:** `https://pgg898200-ship-it.github.io/eproc/`
   - **Topics:** `e-values`, `anytime-valid`, `conformal-prediction`, `sequential-testing`, `multi-agent`, `byzantine-fault-tolerance`, `mcp-server`, `c11`, `avx2`, `statistics`, `llm-evaluation`
4. Enable GitHub Pages: **Settings → Pages → Source: GitHub Actions**.
   - Pushing to `main` runs `.github/workflows/pages.yml`, automatically publishing:
     - Interactive browser lab: `https://pgg898200-ship-it.github.io/eproc/`
     - Agent-optimized Markdown: `https://pgg898200-ship-it.github.io/eproc/llms.txt` and `/llms-full.txt`
     - Single-header direct download: `https://pgg898200-ship-it.github.io/eproc/eproc_single.h`
     - Integrity checksums: `https://pgg898200-ship-it.github.io/eproc/SHA256SUMS`

---

## 2. Zero-Secret PyPI Release (OIDC Trusted Publishing)

1. Create a PyPI account (with 2FA/passkey) and register a **Pending Publisher** at `https://pypi.org/manage/account/publishing/`:
   - **PyPI Project Name:** `eproc`
   - **Owner:** `pgg898200-ship-it`
   - **Repository:** `eproc`
   - **Workflow name:** `release.yml`
   - **Environment name:** `pypi`
2. In GitHub repo **Settings → Environments**, create an environment named `pypi`.
3. Tag and push `v0.1.0`:
   ```bash
   git tag -a v0.1.0 -m "eproc v0.1.0"
   git push origin v0.1.0
   ```
   `.github/workflows/release.yml` automatically:
   - Verifies C11/C++17 tests, single-header build, and `verify_release_hygiene.py --strict`
   - Builds portable wheels via `cibuildwheel` (Linux x86_64/aarch64, macOS arm64/x86_64, Python 3.10–3.13) + `sdist`
   - Publishes to PyPI via OIDC (zero stored tokens)
   - Creates the GitHub Release with wheels, `eproc_single.h`, `eproc.h`, and `SHA256SUMS.txt`

---

## 3. Agent & Coding-Tool Discovery (Ramp `llms.txt` + MCP + Skills)

- **Pure-Markdown Agent Indexing:** `README.md`, `llms.txt`, `llms-full.txt`, and `docs/MATHEMATICS.md` contain concise, citation-ready benchmark claims backed by `benchmarks/receipts.json`.
- **MCP Registry Submission:** Submit `eproc mcp` (`eproc-mcp` console script, protocol `2025-11-25` & `2026-07-28`, 7 tools) to:
  - Official MCP Registry (`registry.modelcontextprotocol.io`)
  - `modelcontextprotocol/servers` community list (factual 1-line PR)
- **Agent Skill Manifests:** Already bundled in-repo:
  - Claude Code: `.claude-plugin/plugin.json` + `skills/eproc/SKILL.md`
  - Gemini CLI: `gemini-extension.json` + `skills/eproc/SKILL.md`

---

## 4. Curated Ecosystem Lists & Technical Showcases (Organic Only)

Submit concise, factual entries with reproducible commands — never promotional hype:

1. **Curated GitHub Awesome Lists (1-line factual PRs):**
   - `awesome-conformal-prediction` (online multi-rate crossing-free conformal + 640-byte abelian fleet monoid)
   - `awesome-mcp-servers` (zero-dep anytime-valid statistics & multi-agent eval gating MCP server)
   - `single_file_libs` (`dist/eproc_single.h` stb-style C11/C++17 single header, zero `malloc`)
2. **Show HN / Technical Communities (`r/MachineLearning`, `r/LocalLLaMA`, `r/C_Programming`):**
   - Lead with the verifiable falsifiable claim and 1-command reproduction:
     - *Title:* `Show HN: eproc – Sub-microsecond anytime-valid statistics, online conformal & Byzantine-safe agent fusion (zero-dep C11/Python/MCP)`
     - *Core hook:* Continuous peeking breaks 5% t-tests (`39.2%` FPR); `eproc` enforces Ville's inequality (`3.9%` FPR), cuts eval tokens `88.7%` (`9.74×` faster stop), calibrates 5 crossing-free conformal rates in `316 ns` (`273×` vs pure Python), and merges 1,024-agent calibration state in `640 bytes/agent` (`2,162×` faster LOO).
     - Link directly to the interactive browser lab (`https://pgg898200-ship-it.github.io/eproc/`), `benchmarks/BENCHMARKS.md`, and `BREAKER_REPORT.md` (42/42 adversarial tests + ASan/UBSan/TSan receipts).
3. **Post-Launch Maintainer Cadence:**
   - Respond rapidly to issues/PRs with deterministic tests and updated receipts.
   - Keep `SHA256SUMS` and `benchmarks/receipts.json` reproducible via `python3 benchmarks/bench_suite.py` on every release.
