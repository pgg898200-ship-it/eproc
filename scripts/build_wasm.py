#!/usr/bin/env python3
"""Deterministic zero-dependency WebAssembly build script for eproc.

Compiles the real zero-allocation C11 core (eproc_eprocess.c, eproc_conformal.c,
eproc_fusion.c, eproc_subsample.c) plus csrc/eproc_wasm_shim.c into a standalone
dist/eproc.wasm binary using clang --target=wasm32 + wasm-ld (no Emscripten,
no WASI, no libc).

License: Apache-2.0
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSRC = ROOT / "csrc"
WASM_INC = CSRC / "wasm_inc"
DIST_DIR = ROOT / "dist"
OUT_WASM = DIST_DIR / "eproc.wasm"

CORE_SOURCES = (
    "eproc_eprocess.c",
    "eproc_conformal.c",
    "eproc_fusion.c",
    "eproc_subsample.c",
    "eproc_wasm_shim.c",
)

CORE_ABI_EXPORTS = (
    # 1. Anytime-valid betting e-processes, confidence sequences, e-detector, agent racing
    "eproc_bet_init",
    "eproc_bet_update",
    "eproc_bet_update_batch",
    "eproc_bet_query",
    "eproc_combine_independent_log",
    "eproc_combine_dependent_log",
    "eproc_cs_init_eb",
    "eproc_cs_init_normal",
    "eproc_cs_update",
    "eproc_cs_update_batch",
    "eproc_cs_query",
    "eproc_edetect_init",
    "eproc_edetect_update",
    "eproc_edetect_reset_alarm",
    "eproc_race_init",
    "eproc_race_update",
    "eproc_race_update_batch",
    "eproc_race_status",
    "eproc_race_arm_state",
    "eproc_race_suggest_arm",
    # 2. Multi-rate conformal, delayed feedback, budget bounds, 640-byte monoid
    "eproc_conformal_init",
    "eproc_conformal_predict",
    "eproc_conformal_update",
    "eproc_conformal_update_batch",
    "eproc_delayed_conformal_init",
    "eproc_delayed_conformal_issue",
    "eproc_delayed_conformal_reveal",
    "eproc_budget_conformal_init",
    "eproc_budget_conformal_bound",
    "eproc_budget_conformal_update",
    "eproc_cmonoid_init",
    "eproc_cmonoid_add",
    "eproc_cmonoid_add_batch",
    "eproc_cmonoid_merge",
    "eproc_cmonoid_subtract",
    "eproc_cmonoid_quantile",
    "eproc_cmonoid_wire_size",
    "eproc_cmonoid_serialize",
    "eproc_cmonoid_deserialize",
    # 3. Byzantine-safe multi-agent fusion (BOA + Ville + Seqlock) & EchoPool
    "eproc_boa_init",
    "eproc_boa_predict",
    "eproc_boa_update",
    "eproc_boa_snapshot_weights",
    "eproc_echopool_init",
    "eproc_echopool_submit",
    "eproc_echopool_resolve",
    "eproc_echopool_posterior",
    "eproc_echopool_source_reliability",
    # 4. Causal subsampling (VarOpt, Online RLS) & SiftVOI + ABI info
    "eproc_varopt_init",
    "eproc_varopt_push",
    "eproc_varopt_estimate_sum",
    "eproc_varopt_estimate_mean",
    "eproc_rls_init",
    "eproc_rls_step",
    "eproc_sift_init",
    "eproc_sift_set_noise_scale",
    "eproc_sift_observe_channel",
    "eproc_sift_observe_site",
    "eproc_sift_select_next",
    "eproc_abi_version",
    "eproc_simd_flags",
)

SHIM_EXPORTS = (
    "eproc_wasm_alloc",
    "eproc_wasm_free",
    "eproc_wasm_reset",
    "eproc_wasm_heap_used",
    "eproc_wasm_erfc",
    "eproc_wasm_sizeof_bet",
    "eproc_wasm_sizeof_bet_result",
    "eproc_wasm_sizeof_cs",
    "eproc_wasm_sizeof_cs_result",
    "eproc_wasm_sizeof_edetect",
    "eproc_wasm_sizeof_edetect_result",
    "eproc_wasm_sizeof_race",
    "eproc_wasm_sizeof_race_status",
    "eproc_wasm_sizeof_conformal",
    "eproc_wasm_sizeof_conformal_out",
    "eproc_wasm_sizeof_delayed_conformal",
    "eproc_wasm_sizeof_budget_conformal",
    "eproc_wasm_sizeof_cmonoid",
    "eproc_wasm_sizeof_boa",
    "eproc_wasm_sizeof_boa_step",
    "eproc_wasm_sizeof_echopool",
    "eproc_wasm_sizeof_echo_out",
    "eproc_wasm_sizeof_varopt",
    "eproc_wasm_sizeof_rls",
    "eproc_wasm_sizeof_rls_out",
    "eproc_wasm_sizeof_sift",
    "eproc_wasm_sizeof_sift_action",
)

ALL_EXPORTS = CORE_ABI_EXPORTS + SHIM_EXPORTS

MAX_WASM_BYTES = 80 * 1024


def _find_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required toolchain binary '{name}' not found on PATH")
    return path


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n")
        if proc.stdout:
            sys.stderr.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    if proc.stderr:
        sys.stderr.write(proc.stderr)


def build_wasm() -> Path:
    clang = _find_tool("clang")
    wasm_ld = _find_tool("wasm-ld")

    DIST_DIR.mkdir(parents=True, exist_ok=True)

    cflags = [
        "--target=wasm32",
        "-std=c11",
        "-O3",
        "-fno-unroll-loops",
        "-ffp-contract=off",
        "-DEPROC_FORCE_SCALAR",
        "-nostdlib",
        "-nostdinc",
        f"-I{WASM_INC}",
        f"-I{CSRC}",
        "-ffunction-sections",
        "-fdata-sections",
        "-Wall",
        "-Wextra",
        "-Werror",
    ]

    with tempfile.TemporaryDirectory(prefix="eproc_wasm_") as tmpdir:
        tmp = Path(tmpdir)
        objs: list[str] = []
        for src_name in CORE_SOURCES:
            src_path = CSRC / src_name
            obj_path = tmp / (src_path.stem + ".o")
            _run([clang, *cflags, "-c", str(src_path), "-o", str(obj_path)])
            objs.append(str(obj_path))

        allow_undef = tmp / "js_math_imports.syms"
        allow_undef.write_text("log\nlog1p\nexp\ntanh\n", encoding="utf-8")

        ld_cmd = [
            wasm_ld,
            "--no-entry",
            "--gc-sections",
            "--strip-all",
            "--export-memory",
            f"--allow-undefined-file={allow_undef}",
            "--stack-first",
            "-z",
            "stack-size=65536",
            "--initial-memory=262144",
        ]
        for sym in ALL_EXPORTS:
            ld_cmd.append(f"--export={sym}")
        ld_cmd.extend(objs)
        ld_cmd.extend(["-o", str(OUT_WASM)])
        _run(ld_cmd)

    data = OUT_WASM.read_bytes()
    size_bytes = len(data)
    sha256 = hashlib.sha256(data).hexdigest()

    print(
        f"Built dist/eproc.wasm: {size_bytes:,} bytes ({size_bytes / 1024.0:.2f} KiB) | "
        f"{len(ALL_EXPORTS)} exported functions ({len(CORE_ABI_EXPORTS)} core ABI + {len(SHIM_EXPORTS)} shim)"
    )
    print(f"SHA-256: {sha256}")

    if size_bytes >= MAX_WASM_BYTES:
        sys.stderr.write(
            f"ERROR: dist/eproc.wasm size {size_bytes} bytes exceeds {MAX_WASM_BYTES} byte budget\n"
        )
        raise SystemExit(1)

    return OUT_WASM


if __name__ == "__main__":
    build_wasm()
