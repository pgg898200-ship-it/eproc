#!/usr/bin/env python3
"""Zero-dependency builder for the ``eproc._native`` CPython C-API extension.

Compiles ``csrc/eproc_pyext.c`` into ``python/eproc/_native<EXT_SUFFIX>`` using
the system C11 compiler and Python's stdlib ``sysconfig`` — no setuptools,
Cython, pybind11, or third-party build packages required.

Links dynamically against ``libeproc`` with ``-Wl,-rpath,$ORIGIN/../..`` so
updates to ``libeproc.so`` at the repository root take effect immediately.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


def _find_compiler(preferred: str | None = None) -> str:
    for cand in (preferred, os.environ.get("CC"), "gcc", "clang", "cc"):
        if cand and shutil.which(cand):
            return cand
    raise RuntimeError("No C11 compiler found (checked CC, gcc, clang, cc)")


def build_extension(
    repo_root: Path | None = None,
    compiler: str | None = None,
    verbose: bool = False,
) -> Path:
    root = (Path(__file__).resolve().parents[1] if repo_root is None else repo_root.resolve())
    csrc = root / "csrc"
    pyext_c = csrc / "eproc_pyext.c"
    if not pyext_c.is_file():
        raise FileNotFoundError(f"Missing C extension source: {pyext_c}")

    py_inc = sysconfig.get_path("include")
    if not py_inc or not (Path(py_inc) / "Python.h").is_file():
        raise RuntimeError(f"Python.h not found in {py_inc!r}")

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out_dir = root / "python" / "eproc"
    out_path = out_dir / f"_native{ext_suffix}"
    tmp_path = out_dir / f"_native.tmp.{os.getpid()}{ext_suffix}"

    cc = _find_compiler(compiler)

    if sys.platform == "darwin":
        lib_path = root / "libeproc.dylib"
        if not lib_path.is_file():
            raise FileNotFoundError(f"Build {lib_path} before building _native")
        cmd = [
            cc,
            "-std=c11",
            "-O3",
            "-fPIC",
            "-dynamiclib",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-parameter",
            f"-I{csrc}",
            f"-I{py_inc}",
            str(pyext_c),
            f"-L{root}",
            "-leproc",
            "-Wl,-rpath,@loader_path/../..",
            "-Wl,-rpath,@loader_path",
            "-undefined",
            "dynamic_lookup",
            "-lm",
            "-o",
            str(tmp_path),
        ]
    else:
        lib_path = root / "libeproc.so"
        if not lib_path.is_file():
            raise FileNotFoundError(f"Build {lib_path} before building _native")
        cmd = [
            cc,
            "-std=c11",
            "-O3",
            "-fPIC",
            "-shared",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-parameter",
            f"-I{csrc}",
            f"-I{py_inc}",
            str(pyext_c),
            f"-L{root}",
            "-leproc",
            "-Wl,-rpath,$ORIGIN/../..",
            "-Wl,-rpath,$ORIGIN",
            "-lm",
            "-o",
            str(tmp_path),
        ]

    if verbose:
        print("Running:", " ".join(cmd))

    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            f"Failed to compile eproc._native (rc={res.returncode}):\n{res.stderr}"
        )

    os.replace(tmp_path, out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build eproc._native CPython C-API accelerator.")
    parser.add_argument("--cc", type=str, default=None, help="C11 compiler override.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print compiler command.")
    args = parser.parse_args(argv)
    out = build_extension(compiler=args.cc, verbose=args.verbose)
    print(f"Built {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
