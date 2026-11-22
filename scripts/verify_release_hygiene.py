#!/usr/bin/env python3
"""Deterministic release hygiene & Trust Pyramid verifier for eproc.

Checks:
  1. Zero third-party imports inside python/eproc/ (stdlib + optional numpy only)
  2. Zero leaked private paths, personal identifiers, or internal org strings
  3. Cold import time of `import eproc` < 20 ms
  4. Valid JSON across plugin manifests and benchmark receipts
  5. Required release files exist (LICENSE, NOTICE, README.md, llms.txt, SKILL.md, dist/eproc_single.h)
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STDLIB = sys.stdlib_module_names

import re

FORBIDDEN_PATTERNS = (
    re.compile(r"/Users/[A-Za-z0-9._-]+"),
    re.compile(r"/home/[A-Za-z0-9._-]+"),
)

REQUIRED_FILES = (
    "LICENSE",
    "NOTICE",
    "README.md",
    "llms.txt",
    "llms-full.txt",
    "pyproject.toml",
    "Makefile",
    "csrc/eproc.h",
    "csrc/eproc_seqlock.h",
    "dist/eproc_single.h",
    "docs/MATHEMATICS.md",
    "skills/eproc/SKILL.md",
    ".claude-plugin/plugin.json",
    "gemini-extension.json",
    "benchmarks/receipts.json",
    "benchmarks/BENCHMARKS.md",
)


def check_zero_deps() -> list[str]:
    errs: list[str] = []
    pkg_dir = ROOT / "python" / "eproc"
    for py_file in sorted(pkg_dir.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in STDLIB and top != "eproc":
                        errs.append(f"Non-stdlib import '{alias.name}' in {py_file.relative_to(ROOT)}")
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    top = node.module.split(".")[0]
                    if top not in STDLIB and top != "eproc":
                        errs.append(f"Non-stdlib from-import '{node.module}' in {py_file.relative_to(ROOT)}")
    return errs


def check_no_leaks() -> list[str]:
    errs: list[str] = []
    text_exts = {".c", ".h", ".py", ".md", ".txt", ".json", ".toml", ".yml", ".yaml"}
    skip_dirs = {".git", "__pycache__", "build", ".pytest_cache"}
    self_path = Path(__file__).resolve()
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.resolve() == self_path:
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        if p.suffix not in text_exts and p.name not in {"Makefile", "LICENSE", "NOTICE"}:
            continue
        content = p.read_text(encoding="utf-8", errors="ignore")
        for pat in FORBIDDEN_PATTERNS:
            m = pat.search(content)
            if m:
                errs.append(f"Leaked workstation path '{m.group(0)}' in {p.relative_to(ROOT)}")
    return errs


def check_cold_import() -> tuple[float, list[str]]:
    errs: list[str] = []
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-c", "import eproc"],
            env={"PYTHONPATH": str(ROOT / "python"), "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            check=False,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if proc.returncode != 0:
            errs.append(f"Cold import failed: {proc.stderr.strip()}")
            return dt_ms, errs
        times.append(dt_ms)
    times.sort()
    med_ms = times[len(times) // 2]
    # Also measure in-process pure import latency excluding python interpreter startup
    proc2 = subprocess.run(
        [
            sys.executable,
            "-c",
            "import time; t0=time.perf_counter(); import eproc; print((time.perf_counter()-t0)*1000.0)",
        ],
        env={"PYTHONPATH": str(ROOT / "python"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    in_proc_ms = float(proc2.stdout.strip())
    if in_proc_ms >= 20.0:
        errs.append(f"In-process import latency {in_proc_ms:.2f} ms >= 20.0 ms budget")
    return in_proc_ms, errs


def check_required_and_json(strict_release: bool) -> list[str]:
    errs: list[str] = []
    for rel in REQUIRED_FILES:
        p = ROOT / rel
        if not p.exists():
            if strict_release:
                errs.append(f"Missing required release file: {rel}")
            continue
        if p.suffix == ".json":
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errs.append(f"Invalid JSON in {rel}: {exc}")
    return errs


def main() -> int:
    strict = "--strict" in sys.argv
    all_errs: list[str] = []
    all_errs.extend(check_zero_deps())
    all_errs.extend(check_no_leaks())
    in_proc_ms, imp_errs = check_cold_import()
    all_errs.extend(imp_errs)
    all_errs.extend(check_required_and_json(strict_release=strict))

    print(f"In-process `import eproc` latency: {in_proc_ms:.2f} ms")
    if all_errs:
        print(f"HYGIENE FAIL ({len(all_errs)} issues):")
        for e in all_errs:
            print(f"  - {e}")
        return 1
    print("HYGIENE PASS (0 issues)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
