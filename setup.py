"""Setuptools build configuration for eproc's self-contained C11 extension.

Compiles the zero-allocation C11 core directly into ``eproc._native`` so
``pip install eproc`` produces a single self-contained wheel with zero runtime
dependencies and full C11 ABI visibility (supporting both the fast C-API path
and ``EPROC_DISABLE_NATIVE=1`` ctypes fallback).

Optional environment flags when compiling from source:
  - ``EPROC_NATIVE_ARCH=1``: pass ``-march=native`` for host AVX2/FMA/NEON.
  - ``EPROC_AVX2=1``: pass ``-mavx2`` on x86_64 hosts.
"""

from __future__ import annotations

import os
import platform
import sys
from setuptools import Extension, setup

CORE_SOURCES = [
    "csrc/eproc_pyext.c",
    "csrc/eproc_eprocess.c",
    "csrc/eproc_conformal.c",
    "csrc/eproc_fusion.c",
    "csrc/eproc_subsample.c",
]

CORE_HEADERS = [
    "csrc/eproc.h",
    "csrc/eproc_seqlock.h",
    "csrc/eproc_internal.h",
]


def _build_extensions() -> list[Extension]:
    if sys.platform == "win32":
        extra_compile_args = ["/O2", "/fp:precise"]
        libraries: list[str] = []
    else:
        extra_compile_args = [
            "-std=c11",
            "-O3",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
        ]
        if os.environ.get("EPROC_NATIVE_ARCH") == "1":
            extra_compile_args.append("-march=native")
        elif (
            os.environ.get("EPROC_AVX2") == "1"
            and platform.machine().lower() in ("x86_64", "amd64")
        ):
            extra_compile_args.append("-mavx2")
        libraries = ["m"]

    return [
        Extension(
            name="eproc._native",
            sources=CORE_SOURCES,
            include_dirs=["csrc"],
            depends=CORE_HEADERS,
            extra_compile_args=extra_compile_args,
            libraries=libraries,
        )
    ]


setup(ext_modules=_build_extensions())
