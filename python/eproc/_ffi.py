"""Low-level C11 ABI bindings for ``libeproc`` using only Python stdlib ``ctypes``.

Every ``ctypes.Structure`` layout, padding field, array bound, and function
prototype in this module mirrors ``csrc/eproc.h`` byte-for-byte.
"""

from __future__ import annotations

import array
import ctypes
from ctypes import (
    CDLL,
    POINTER,
    Structure,
    c_double,
    c_int,
    c_int32,
    c_size_t,
    c_uint8,
    c_uint32,
    c_uint64,
)
import ctypes.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# ABI & Capacity Constants (must match csrc/eproc.h exactly)
# ---------------------------------------------------------------------------

EPROC_ABI_VERSION = 1

# Status codes
EPROC_OK = 0
EPROC_ERR_INVALID = -1
EPROC_ERR_CAPACITY = -2
EPROC_ERR_EMPTY = -3
EPROC_ERR_MISMATCH = -4

# Verdict codes
EPROC_VERDICT_CONTINUE = 0
EPROC_VERDICT_REJECT = 1
EPROC_VERDICT_PROMOTE = 1
EPROC_VERDICT_ROLLBACK = -1

# Test sides
EPROC_SIDE_UPPER = 0
EPROC_SIDE_LOWER = 1
EPROC_SIDE_TWO_SIDED = 2

# Confidence sequence modes
EPROC_CS_EMPIRICAL_BERNSTEIN = 0
EPROC_CS_NORMAL_MIXTURE = 1

# Hard capacity bounds for zero-allocation hot paths
EPROC_MAX_ARMS = 64
EPROC_MAX_RATES = 5
EPROC_AGACI_EXPERTS = 4
EPROC_CONFORMAL_MAX_WIN = 256
EPROC_CMONOID_BINS = 64
EPROC_DELAY_MAX_PENDING = 4096
EPROC_BOA_MAX_EXPERTS = 64
EPROC_ECHO_MAX_SOURCES = 256
EPROC_ECHO_MAX_CLAIMS = 256
EPROC_ECHO_TABLE_CAP = 4096
EPROC_VAROPT_MAX_K = 4096
EPROC_RLS_MAX_D = 32
EPROC_SIFT_MAX_SITES = 1024
EPROC_SIFT_MAX_PER_SITE = 16

# SIMD capability bitmask
EPROC_SIMD_AVX2 = 1 << 0
EPROC_SIMD_FMA = 1 << 1
EPROC_SIMD_NEON = 1 << 2

_NATIVE_F64_TYPESTRS = ("<f8", "=f8") if sys.byteorder == "little" else (">f8", "=f8")
_NATIVE_U32_TYPESTRS = ("<u4", "=u4") if sys.byteorder == "little" else (">u4", "=u4")
_ARRAY_U32_CODE = "I" if array.array("I").itemsize == 4 else "L"


# ---------------------------------------------------------------------------
# C11 Struct Definitions (byte-for-byte mirror of csrc/eproc.h)
# ---------------------------------------------------------------------------


class eproc_bet_t(Structure):
    _fields_ = [
        ("lower", c_double),
        ("upper", c_double),
        ("inv_span", c_double),
        ("mu0_norm", c_double),
        ("alpha", c_double),
        ("log_thresh", c_double),
        ("c_max", c_double),
        ("side", c_int32),
        ("_pad0", c_uint32),
        ("lam_pos", c_double),
        ("a_pos", c_double),
        ("log_e_pos", c_double),
        ("lam_neg", c_double),
        ("a_neg", c_double),
        ("log_e_neg", c_double),
        ("log_e", c_double),
        ("max_log_e", c_double),
        ("sum_x", c_double),
        ("sum_x_comp", c_double),
        ("sum_sq", c_double),
        ("n", c_uint64),
        ("rejected_at", c_uint64),
        ("n_invalid", c_uint64),
    ]


class eproc_bet_result_t(Structure):
    _fields_ = [
        ("n", c_uint64),
        ("e_value", c_double),
        ("log_e", c_double),
        ("max_log_e", c_double),
        ("p_value", c_double),
        ("mean", c_double),
        ("rejected", c_int32),
        ("_pad0", c_uint32),
    ]


class eproc_cs_t(Structure):
    _fields_ = [
        ("lower", c_double),
        ("upper", c_double),
        ("span", c_double),
        ("inv_span", c_double),
        ("alpha", c_double),
        ("log_2_over_alpha", c_double),
        ("sigma", c_double),
        ("rho", c_double),
        ("mode", c_int32),
        ("_pad0", c_uint32),
        ("sum_y", c_double),
        ("sum_y_comp", c_double),
        ("sum_sq_dev", c_double),
        ("sum_lam", c_double),
        ("sum_lam_y", c_double),
        ("sum_psi_v", c_double),
        ("cs_low", c_double),
        ("cs_high", c_double),
        ("sum_x", c_double),
        ("sum_x_comp", c_double),
        ("n", c_uint64),
        ("n_invalid", c_uint64),
    ]


class eproc_cs_result_t(Structure):
    _fields_ = [
        ("n", c_uint64),
        ("mean", c_double),
        ("low", c_double),
        ("high", c_double),
        ("width", c_double),
    ]


class eproc_edetect_t(Structure):
    _fields_ = [
        ("inner", eproc_bet_t),
        ("log_cusum", c_double),
        ("max_log_cusum", c_double),
        ("log_thresh", c_double),
        ("alarm_at", c_uint64),
        ("n_alarms", c_uint64),
        ("n", c_uint64),
    ]


class eproc_edetect_result_t(Structure):
    _fields_ = [
        ("n", c_uint64),
        ("log_stat", c_double),
        ("stat", c_double),
        ("alarm", c_int32),
        ("_pad0", c_uint32),
    ]


class eproc_race_t(Structure):
    _fields_ = [
        ("n_arms", c_uint32),
        ("n_active", c_uint32),
        ("alpha", c_double),
        ("lower", c_double),
        ("upper", c_double),
        ("margin", c_double),
        ("winner", c_int32),
        ("_pad0", c_uint32),
        ("total_obs", c_uint64),
        ("active", c_uint8 * EPROC_MAX_ARMS),
        ("eliminated_at", c_uint64 * EPROC_MAX_ARMS),
        ("arms", eproc_cs_t * EPROC_MAX_ARMS),
    ]


class eproc_race_status_t(Structure):
    _fields_ = [
        ("n_arms", c_uint32),
        ("n_active", c_uint32),
        ("winner", c_int32),
        ("leader", c_int32),
        ("total_obs", c_uint64),
        ("leader_low", c_double),
        ("leader_high", c_double),
        ("runner_up_high", c_double),
    ]


class eproc_conformal_t(Structure):
    _fields_ = [
        ("n_rates", c_uint32),
        ("win_cap", c_uint32),
        ("win_len", c_uint32),
        ("ring_pos", c_uint32),
        ("pid_ki", c_double),
        ("pid_csat", c_double),
        ("n_updates", c_uint64),
        ("target_alphas", c_double * EPROC_MAX_RATES),
        ("gammas", c_double * EPROC_AGACI_EXPERTS),
        ("alpha_t", (c_double * EPROC_AGACI_EXPERTS) * EPROC_MAX_RATES),
        ("exp_log_w", (c_double * EPROC_AGACI_EXPERTS) * EPROC_MAX_RATES),
        ("exp_var", (c_double * EPROC_AGACI_EXPERTS) * EPROC_MAX_RATES),
        ("pid_err_sum", c_double * EPROC_MAX_RATES),
        ("eff_alpha", c_double * EPROC_MAX_RATES),
        ("current_q", c_double * EPROC_MAX_RATES),
        ("emp_cov_ema", c_double * EPROC_MAX_RATES),
        ("hit_counts", c_uint64 * EPROC_MAX_RATES),
        ("ring", c_double * EPROC_CONFORMAL_MAX_WIN),
        ("sorted", c_double * EPROC_CONFORMAL_MAX_WIN),
    ]


class eproc_conformal_out_t(Structure):
    _fields_ = [
        ("n_rates", c_uint32),
        ("_pad0", c_uint32),
        ("quantiles", c_double * EPROC_MAX_RATES),
        ("eff_alphas", c_double * EPROC_MAX_RATES),
        ("emp_coverage", c_double * EPROC_MAX_RATES),
    ]


class eproc_delay_slot_t(Structure):
    _fields_ = [
        ("ticket_id", c_uint64),
        ("pred", c_double),
        ("scale", c_double),
        ("occupied", c_uint8),
        ("_pad", c_uint8 * 7),
    ]


class eproc_delayed_conformal_t(Structure):
    _fields_ = [
        ("core", eproc_conformal_t),
        ("pending_count", c_uint32),
        ("_pad0", c_uint32),
        ("next_ticket", c_uint64),
        ("slots", eproc_delay_slot_t * EPROC_DELAY_MAX_PENDING),
    ]


class eproc_budget_conformal_t(Structure):
    _fields_ = [
        ("core", eproc_conformal_t),
        ("censor_penalty", c_double),
        ("n_completed", c_uint64),
        ("n_censored", c_uint64),
    ]


class eproc_cmonoid_t(Structure):
    _fields_ = [
        ("count", c_uint64),
        ("bin_counts", c_uint64 * EPROC_CMONOID_BINS),
        ("sum", c_double),
        ("sum_comp", c_double),
        ("sum_sq", c_double),
        ("sum_sq_comp", c_double),
        ("min_val", c_double),
        ("max_val", c_double),
        ("log_wealth", c_double),
        ("scale_hint", c_double),
        ("flags", c_uint64),
        ("_reserved", c_uint64 * 6),
    ]


class eproc_boa_t(Structure):
    _fields_ = [
        ("n_experts", c_uint32),
        ("anchor_idx", c_uint32),
        ("fallback_active", c_int32),
        ("_pad0", c_uint32),
        ("scale_ema", c_double),
        ("clip_mult", c_double),
        ("n_updates", c_uint64),
        ("n_poison_clipped", c_uint64),
        ("n_fallback_trips", c_uint64),
        ("weights", c_double * EPROC_BOA_MAX_EXPERTS),
        ("log_w", c_double * EPROC_BOA_MAX_EXPERTS),
        ("grad_sq_sum", c_double * EPROC_BOA_MAX_EXPERTS),
        ("cum_loss", c_double * EPROC_BOA_MAX_EXPERTS),
        ("quarantined", c_uint8 * EPROC_BOA_MAX_EXPERTS),
        ("poison_strikes", c_uint32 * EPROC_BOA_MAX_EXPERTS),
        ("sup_e", eproc_bet_t),
        ("deg_e", eproc_bet_t),
        ("rec_e", eproc_bet_t),
        ("seqlock_seq", c_uint64),
        ("seqlock_words", c_uint64 * (EPROC_BOA_MAX_EXPERTS + 2)),
    ]


class eproc_boa_step_t(Structure):
    _fields_ = [
        ("fused_pred", c_double),
        ("anchor_pred", c_double),
        ("fallback_active", c_int32),
        ("n_quarantined", c_uint32),
        ("sup_log_e", c_double),
        ("deg_log_e", c_double),
        ("n_updates", c_uint64),
    ]


class eproc_echo_slot_t(Structure):
    _fields_ = [
        ("key_hash", c_uint64),
        ("evidence_id", c_uint64),
        ("claim_id", c_uint32),
        ("source_id", c_uint32),
        ("weighted_llr", c_double),
        ("used", c_uint8),
        ("_pad", c_uint8 * 7),
    ]


class eproc_echopool_t(Structure):
    _fields_ = [
        ("n_sources", c_uint32),
        ("n_claims", c_uint32),
        ("n_accepted", c_uint64),
        ("n_echo_dropped", c_uint64),
        ("n_table_full", c_uint64),
        ("src_alpha", c_double * EPROC_ECHO_MAX_SOURCES),
        ("src_beta", c_double * EPROC_ECHO_MAX_SOURCES),
        ("src_reliability", c_double * EPROC_ECHO_MAX_SOURCES),
        ("claim_prior_lo", c_double * EPROC_ECHO_MAX_CLAIMS),
        ("claim_lo", c_double * EPROC_ECHO_MAX_CLAIMS),
        ("claim_lo_comp", c_double * EPROC_ECHO_MAX_CLAIMS),
        ("claim_unique", c_uint64 * EPROC_ECHO_MAX_CLAIMS),
        ("table", eproc_echo_slot_t * EPROC_ECHO_TABLE_CAP),
    ]


class eproc_echo_out_t(Structure):
    _fields_ = [
        ("accepted", c_int32),
        ("_pad0", c_uint32),
        ("posterior_prob", c_double),
        ("log_odds", c_double),
        ("source_reliability", c_double),
        ("unique_count", c_uint64),
    ]


class eproc_varopt_item_t(Structure):
    _fields_ = [
        ("item_id", c_uint64),
        ("orig_weight", c_double),
        ("adj_weight", c_double),
        ("value", c_double),
        ("rand_u", c_double),
    ]


class eproc_varopt_t(Structure):
    _fields_ = [
        ("k", c_uint32),
        ("count", c_uint32),
        ("n_heavy", c_uint32),
        ("_pad0", c_uint32),
        ("tau", c_double),
        ("total_weight", c_double),
        ("total_weight_comp", c_double),
        ("n_seen", c_uint64),
        ("rng_state", c_uint64 * 2),
        ("items", eproc_varopt_item_t * EPROC_VAROPT_MAX_K),
    ]


class eproc_rls_t(Structure):
    _fields_ = [
        ("d", c_uint32),
        ("_pad0", c_uint32),
        ("lambda_", c_double),
        ("oversample", c_double),
        ("min_prob", c_double),
        ("n_seen", c_uint64),
        ("n_kept", c_uint64),
        ("rng_state", c_uint64 * 2),
        ("inv_a", c_double * (EPROC_RLS_MAX_D * EPROC_RLS_MAX_D)),
    ]


class eproc_rls_out_t(Structure):
    _fields_ = [
        ("keep", c_int32),
        ("_pad0", c_uint32),
        ("leverage", c_double),
        ("prob", c_double),
        ("ht_weight", c_double),
    ]


class eproc_sift_t(Structure):
    _fields_ = [
        ("n_sites", c_uint32),
        ("per_site", c_uint32),
        ("prior_hot_site", c_double),
        ("prior_active_cold", c_double),
        ("prior_active_hot", c_double),
        ("signal_mean", c_double),
        ("noise_sd", c_double),
        ("site_log_prob", c_double * EPROC_SIFT_MAX_SITES),
        ("chan_prob", c_double * (EPROC_SIFT_MAX_SITES * EPROC_SIFT_MAX_PER_SITE)),
        ("chan_noise_scale", c_double * (EPROC_SIFT_MAX_SITES * EPROC_SIFT_MAX_PER_SITE)),
    ]


class eproc_sift_action_t(Structure):
    _fields_ = [
        ("action_type", c_int32),
        ("target_idx", c_uint32),
        ("evoi_per_cost", c_double),
    ]


# Compile-time sanity check on fixed wire monoid size (640 bytes in eproc.h)
assert ctypes.sizeof(eproc_cmonoid_t) == 640, (
    f"eproc_cmonoid_t size mismatch: expected 640, got {ctypes.sizeof(eproc_cmonoid_t)}"
)


# ---------------------------------------------------------------------------
# Library Discovery, On-Demand Compilation & Prototype Binding
# ---------------------------------------------------------------------------

_LIB_LOCK = threading.Lock()
_LIB: CDLL | None = None
_LIB_PATH: str | None = None


def _shared_lib_name() -> str:
    if sys.platform == "darwin":
        return "libeproc.dylib"
    if sys.platform == "win32":
        return "eproc.dll"
    return "libeproc.so"


def _candidate_lib_paths() -> list[Path]:
    libname = _shared_lib_name()
    pkg_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []

    env_path = os.environ.get("EPROC_LIB_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(pkg_dir / libname)
    # Editable / source checkout layout: <repo>/python/eproc/_ffi.py -> <repo>/libeproc.so
    if len(pkg_dir.parents) >= 2:
        repo_root = pkg_dir.parents[1]
        candidates.append(repo_root / libname)
        candidates.append(repo_root / "build" / libname)
        candidates.append(repo_root / "csrc" / libname)
    return candidates


def _find_csrc_dir() -> Path | None:
    pkg_dir = Path(__file__).resolve().parent
    for base in (pkg_dir, *pkg_dir.parents[:3]):
        csrc = base / "csrc"
        if (csrc / "eproc.h").is_file():
            return csrc
    return None


def _compile_on_demand() -> Path | None:
    """Compile ``libeproc`` on-demand from ``csrc/*.c`` if a C11 compiler exists."""
    csrc = _find_csrc_dir()
    if csrc is None:
        return None

    c_files = sorted(
        p for p in csrc.glob("*.c")
        if not p.name.startswith(("test_", "bench_", "main"))
        and p.name != "eproc_pyext.c"
    )
    if not c_files:
        return None

    compiler = None
    for candidate in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if candidate and shutil.which(candidate):
            compiler = candidate
            break
    if compiler is None:
        return None

    libname = _shared_lib_name()
    repo_root = csrc.parent
    out_path = repo_root / libname
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        probe_writable = os.access(out_path.parent, os.W_OK)
    except OSError:
        probe_writable = False

    if not probe_writable:
        cache_dir = Path.home() / ".cache" / "eproc"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / libname

    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    cmd = [
        compiler,
        "-std=c11",
        "-O3",
        "-fPIC",
        "-shared",
        "-ffp-contract=off",
        f"-I{csrc}",
        *[str(f) for f in c_files],
        "-o",
        str(tmp_path),
        "-lm",
    ]
    if sys.platform == "darwin":
        cmd = [
            compiler,
            "-std=c11",
            "-O3",
            "-fPIC",
            "-dynamiclib",
            "-ffp-contract=off",
            f"-I{csrc}",
            *[str(f) for f in c_files],
            "-o",
            str(tmp_path),
            "-lm",
        ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            f"Failed to compile {libname} via {' '.join(cmd)}:\n{res.stderr}"
        )

    os.replace(tmp_path, out_path)
    return out_path


def _bind_prototypes(lib: CDLL) -> None:
    """Attach exact ``argtypes`` and ``restype`` for all 61 symbols in ``eproc.h``."""

    def _sig(name: str, argtypes: list[Any], restype: Any) -> None:
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype

    # 1. Betting E-Process
    _sig("eproc_bet_init", [POINTER(eproc_bet_t), c_double, c_double, c_double, c_double, c_int32], c_int)
    _sig("eproc_bet_update", [POINTER(eproc_bet_t), c_double, POINTER(eproc_bet_result_t)], c_int)
    _sig("eproc_bet_update_batch", [POINTER(eproc_bet_t), POINTER(c_double), c_size_t, POINTER(eproc_bet_result_t)], c_int)
    _sig("eproc_bet_query", [POINTER(eproc_bet_t), POINTER(eproc_bet_result_t)], None)
    _sig("eproc_combine_independent_log", [c_double, c_double], c_double)
    _sig("eproc_combine_dependent_log", [POINTER(c_double), c_size_t], c_double)

    # Confidence Sequences
    _sig("eproc_cs_init_eb", [POINTER(eproc_cs_t), c_double, c_double, c_double], c_int)
    _sig("eproc_cs_init_normal", [POINTER(eproc_cs_t), c_double, c_double, c_double], c_int)
    _sig("eproc_cs_update", [POINTER(eproc_cs_t), c_double, POINTER(eproc_cs_result_t)], c_int)
    _sig("eproc_cs_update_batch", [POINTER(eproc_cs_t), POINTER(c_double), c_size_t, POINTER(eproc_cs_result_t)], c_int)
    _sig("eproc_cs_query", [POINTER(eproc_cs_t), POINTER(eproc_cs_result_t)], None)

    # E-Detector
    _sig("eproc_edetect_init", [POINTER(eproc_edetect_t), c_double, c_double, c_double, c_double, c_int32], c_int)
    _sig("eproc_edetect_update", [POINTER(eproc_edetect_t), c_double, POINTER(eproc_edetect_result_t)], c_int)
    _sig("eproc_edetect_reset_alarm", [POINTER(eproc_edetect_t)], None)

    # Agent Racer
    _sig("eproc_race_init", [POINTER(eproc_race_t), c_uint32, c_double, c_double, c_double, c_double], c_int)
    _sig("eproc_race_update", [POINTER(eproc_race_t), c_uint32, c_double, POINTER(eproc_race_status_t)], c_int)
    _sig(
        "eproc_race_update_batch",
        [POINTER(eproc_race_t), POINTER(c_uint32), POINTER(c_double), c_size_t, POINTER(eproc_race_status_t)],
        c_int,
    )
    _sig("eproc_race_status", [POINTER(eproc_race_t), POINTER(eproc_race_status_t)], None)
    _sig(
        "eproc_race_arm_state",
        [POINTER(eproc_race_t), c_uint32, POINTER(eproc_cs_result_t), POINTER(c_int32)],
        c_int,
    )
    _sig("eproc_race_suggest_arm", [POINTER(eproc_race_t)], c_int32)

    # 2. Conformal Calibration, Delayed, Budget & Monoid
    _sig("eproc_conformal_init", [POINTER(eproc_conformal_t), POINTER(c_double), c_uint32, c_uint32, c_double], c_int)
    _sig(
        "eproc_conformal_predict",
        [
            POINTER(eproc_conformal_t),
            c_double,
            c_double,
            POINTER(c_double),
            POINTER(c_double),
            POINTER(eproc_conformal_out_t),
        ],
        None,
    )
    _sig("eproc_conformal_update", [POINTER(eproc_conformal_t), c_double, POINTER(eproc_conformal_out_t)], c_int)
    _sig(
        "eproc_conformal_update_batch",
        [POINTER(eproc_conformal_t), POINTER(c_double), c_size_t, POINTER(eproc_conformal_out_t)],
        c_int,
    )

    _sig(
        "eproc_delayed_conformal_init",
        [POINTER(eproc_delayed_conformal_t), POINTER(c_double), c_uint32, c_uint32, c_double],
        c_int,
    )
    _sig(
        "eproc_delayed_conformal_issue",
        [
            POINTER(eproc_delayed_conformal_t),
            c_double,
            c_double,
            POINTER(c_uint64),
            POINTER(c_double),
            POINTER(c_double),
        ],
        c_int,
    )
    _sig(
        "eproc_delayed_conformal_reveal",
        [POINTER(eproc_delayed_conformal_t), c_uint64, c_double, POINTER(eproc_conformal_out_t)],
        c_int,
    )

    _sig("eproc_budget_conformal_init", [POINTER(eproc_budget_conformal_t), c_double, c_uint32, c_double], c_int)
    _sig("eproc_budget_conformal_bound", [POINTER(eproc_budget_conformal_t), c_double], c_double)
    _sig("eproc_budget_conformal_update", [POINTER(eproc_budget_conformal_t), c_double, c_double, c_int32], c_int)

    _sig("eproc_cmonoid_init", [POINTER(eproc_cmonoid_t), c_double], None)
    _sig("eproc_cmonoid_add", [POINTER(eproc_cmonoid_t), c_double], None)
    _sig("eproc_cmonoid_add_batch", [POINTER(eproc_cmonoid_t), POINTER(c_double), c_size_t], None)
    _sig("eproc_cmonoid_merge", [POINTER(eproc_cmonoid_t), POINTER(eproc_cmonoid_t), POINTER(eproc_cmonoid_t)], None)
    _sig(
        "eproc_cmonoid_subtract",
        [POINTER(eproc_cmonoid_t), POINTER(eproc_cmonoid_t), POINTER(eproc_cmonoid_t)],
        c_int,
    )
    _sig("eproc_cmonoid_quantile", [POINTER(eproc_cmonoid_t), c_double], c_double)
    _sig("eproc_cmonoid_wire_size", [], c_size_t)
    _sig("eproc_cmonoid_serialize", [POINTER(eproc_cmonoid_t), POINTER(c_uint8), c_size_t], c_int)
    _sig("eproc_cmonoid_deserialize", [POINTER(eproc_cmonoid_t), POINTER(c_uint8), c_size_t], c_int)

    # 3. Byzantine-Safe Fusion (BOA & EchoPool)
    _sig("eproc_boa_init", [POINTER(eproc_boa_t), c_uint32, c_uint32, c_double], c_int)
    _sig("eproc_boa_predict", [POINTER(eproc_boa_t), POINTER(c_double)], c_double)
    _sig("eproc_boa_update", [POINTER(eproc_boa_t), POINTER(c_double), c_double, POINTER(eproc_boa_step_t)], c_int)
    _sig(
        "eproc_boa_snapshot_weights",
        [POINTER(eproc_boa_t), POINTER(c_double), c_uint32, POINTER(c_int32)],
        c_int,
    )

    _sig("eproc_echopool_init", [POINTER(eproc_echopool_t), c_uint32, c_uint32, c_double], c_int)
    _sig(
        "eproc_echopool_submit",
        [POINTER(eproc_echopool_t), c_uint32, c_uint32, c_uint64, c_double, POINTER(eproc_echo_out_t)],
        c_int,
    )
    _sig("eproc_echopool_resolve", [POINTER(eproc_echopool_t), c_uint32, c_int32], c_int)
    _sig("eproc_echopool_posterior", [POINTER(eproc_echopool_t), c_uint32], c_double)
    _sig("eproc_echopool_source_reliability", [POINTER(eproc_echopool_t), c_uint32], c_double)

    # 4. Causal Subsampling & Hierarchical VOI
    _sig("eproc_varopt_init", [POINTER(eproc_varopt_t), c_uint32, c_uint64], c_int)
    _sig("eproc_varopt_push", [POINTER(eproc_varopt_t), c_uint64, c_double, c_double], c_int)
    _sig("eproc_varopt_estimate_sum", [POINTER(eproc_varopt_t)], c_double)
    _sig("eproc_varopt_estimate_mean", [POINTER(eproc_varopt_t)], c_double)

    _sig("eproc_rls_init", [POINTER(eproc_rls_t), c_uint32, c_double, c_double, c_double, c_uint64], c_int)
    _sig("eproc_rls_step", [POINTER(eproc_rls_t), POINTER(c_double), POINTER(eproc_rls_out_t)], c_int)

    _sig("eproc_sift_init", [POINTER(eproc_sift_t), c_uint32, c_uint32, c_double, c_double], c_int)
    _sig("eproc_sift_set_noise_scale", [POINTER(eproc_sift_t), c_uint32, c_double], c_int)
    _sig("eproc_sift_observe_channel", [POINTER(eproc_sift_t), c_uint32, c_double, c_uint32], c_int)
    _sig("eproc_sift_observe_site", [POINTER(eproc_sift_t), c_uint32, c_double], c_int)
    _sig(
        "eproc_sift_select_next",
        [POINTER(eproc_sift_t), c_double, c_double, c_double, c_uint32, POINTER(eproc_sift_action_t)],
        c_int,
    )

    # Runtime capability queries
    _sig("eproc_abi_version", [], c_uint32)
    _sig("eproc_simd_flags", [], c_uint32)


def get_lib() -> CDLL:
    """Load (or compile on-demand) ``libeproc`` and return the bound ``CDLL``."""
    global _LIB, _LIB_PATH
    if _LIB is not None:
        return _LIB

    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB

        chosen: str | None = None
        for path in _candidate_lib_paths():
            if path.is_file():
                chosen = str(path)
                break

        if chosen is None:
            sys_lib = ctypes.util.find_library("eproc")
            if sys_lib:
                chosen = sys_lib

        if chosen is None:
            compiled = _compile_on_demand()
            if compiled is not None:
                chosen = str(compiled)

        if chosen is None:
            raise OSError(
                "Could not find or compile libeproc shared library. "
                "Set EPROC_LIB_PATH or build libeproc.so from csrc/."
            )

        lib = CDLL(chosen)
        _bind_prototypes(lib)

        abi = int(lib.eproc_abi_version())
        if abi != EPROC_ABI_VERSION:
            raise RuntimeError(
                f"libeproc ABI mismatch at {chosen}: expected {EPROC_ABI_VERSION}, got {abi}"
            )

        _LIB = lib
        _LIB_PATH = chosen
        return _LIB


def get_lib_path() -> str:
    """Return the resolved path of the loaded ``libeproc`` shared library."""
    get_lib()
    assert _LIB_PATH is not None
    return _LIB_PATH


def abi_version() -> int:
    """Return the C11 ABI version reported by ``libeproc``."""
    return int(get_lib().eproc_abi_version())


def simd_flags() -> int:
    """Return the SIMD capability bitmask reported by ``libeproc``."""
    return int(get_lib().eproc_simd_flags())


def simd_flag_names(flags: int | None = None) -> list[str]:
    """Decode the SIMD bitmask into human-readable instruction set names."""
    mask = simd_flags() if flags is None else int(flags)
    names: list[str] = []
    if mask & EPROC_SIMD_AVX2:
        names.append("AVX2")
    if mask & EPROC_SIMD_FMA:
        names.append("FMA")
    if mask & EPROC_SIMD_NEON:
        names.append("NEON")
    return names


def check_rc(rc: int, context: str = "eproc") -> None:
    """Translate a C status code into an idiomatic Python exception."""
    if rc == EPROC_OK:
        return
    if rc == EPROC_ERR_INVALID:
        raise ValueError(f"{context}: invalid argument or parameter out of range")
    if rc == EPROC_ERR_CAPACITY:
        raise OverflowError(f"{context}: fixed internal capacity exceeded")
    if rc == EPROC_ERR_EMPTY:
        raise ValueError(f"{context}: operation requires non-empty state")
    if rc == EPROC_ERR_MISMATCH:
        raise ValueError(f"{context}: incompatible operand or leave-one-out mismatch")
    raise RuntimeError(f"{context}: unexpected error code {rc}")


# ---------------------------------------------------------------------------
# Zero-Copy Buffer Helpers (zero NumPy import overhead!)
# ---------------------------------------------------------------------------

_EMPTY_DOUBLE_PTR = POINTER(c_double)()
_EMPTY_UINT32_PTR = POINTER(c_uint32)()


def _as_double_ptr(values: Any) -> tuple[Any, int, Any]:
    """Convert a sequence or buffer to ``(POINTER(c_double), length, owner)``.

    Zero-copy fast paths (in order):
      1. ``array.array('d')``
      2. ``ctypes`` ``c_double`` array
      3. Optional ``numpy.ndarray`` (or any object implementing ``__array_interface__``)
         that is C-contiguous native IEEE-754 float64 — detected without importing NumPy!
      4. Contiguous writable ``memoryview`` with format ``'d'``
    Fallback:
      - C-speed ``array.array('d', values)`` for any Python list/tuple/iterable or
        non-contiguous/non-float64 array.
    """
    # 1. stdlib array('d') — zero copy
    if isinstance(values, array.array) and values.typecode == "d":
        addr, n = values.buffer_info()
        if n == 0:
            return _EMPTY_DOUBLE_PTR, 0, values
        return ctypes.cast(addr, POINTER(c_double)), n, values

    # 2. ctypes array of c_double — zero copy
    if isinstance(values, ctypes.Array) and getattr(values, "_type_", None) is c_double:
        n = len(values)
        if n == 0:
            return _EMPTY_DOUBLE_PTR, 0, values
        return ctypes.cast(values, POINTER(c_double)), n, values

    # 3. Optional NumPy / __array_interface__ — zero copy without importing numpy!
    ai = getattr(values, "__array_interface__", None)
    if isinstance(ai, dict):
        typestr = ai.get("typestr", "")
        strides = ai.get("strides")
        shape = ai.get("shape", ())
        data = ai.get("data")
        if (
            typestr in _NATIVE_F64_TYPESTRS
            and isinstance(data, tuple)
            and len(data) >= 1
            and isinstance(data[0], int)
            and len(shape) >= 1
            and (strides is None or (len(shape) == 1 and strides == (8,)))
        ):
            n = int(math.prod(shape))
            if n == 0:
                return _EMPTY_DOUBLE_PTR, 0, values
            ptr_addr = data[0]
            if ptr_addr != 0:
                return ctypes.cast(ptr_addr, POINTER(c_double)), n, values

    # 4. memoryview with native double format
    if isinstance(values, memoryview):
        if values.format == "d" and values.c_contiguous:
            n = values.nbytes // 8
            if n == 0:
                return _EMPTY_DOUBLE_PTR, 0, values
            if not values.readonly:
                flat = values.cast("B") if values.ndim > 1 else values
                arr_ct = (c_double * n).from_buffer(flat)
                return ctypes.cast(arr_ct, POINTER(c_double)), n, (values, arr_ct)
            # Single C memcpy for readonly memoryview
            arr_ct = (c_double * n).from_buffer_copy(values.tobytes())
            return ctypes.cast(arr_ct, POINTER(c_double)), n, arr_ct

    # 5. General Python sequence / iterable fallback (C-speed array('d'))
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("Expected a numeric sequence or buffer, not str/bytes")

    buf = array.array("d", values)
    addr, n = buf.buffer_info()
    if n == 0:
        return _EMPTY_DOUBLE_PTR, 0, buf
    return ctypes.cast(addr, POINTER(c_double)), n, buf


def _as_uint32_ptr(values: Any) -> tuple[Any, int, Any]:
    """Convert a sequence or buffer to ``(POINTER(c_uint32), length, owner)``."""
    # 1. stdlib array('I') (4-byte unsigned int) — zero copy
    if isinstance(values, array.array) and values.typecode == _ARRAY_U32_CODE and values.itemsize == 4:
        addr, n = values.buffer_info()
        if n == 0:
            return _EMPTY_UINT32_PTR, 0, values
        return ctypes.cast(addr, POINTER(c_uint32)), n, values

    # 2. ctypes array of c_uint32 — zero copy
    if isinstance(values, ctypes.Array) and getattr(values, "_type_", None) is c_uint32:
        n = len(values)
        if n == 0:
            return _EMPTY_UINT32_PTR, 0, values
        return ctypes.cast(values, POINTER(c_uint32)), n, values

    # 3. Optional NumPy / __array_interface__ uint32 — zero copy
    ai = getattr(values, "__array_interface__", None)
    if isinstance(ai, dict):
        typestr = ai.get("typestr", "")
        strides = ai.get("strides")
        shape = ai.get("shape", ())
        data = ai.get("data")
        if (
            typestr in _NATIVE_U32_TYPESTRS
            and isinstance(data, tuple)
            and len(data) >= 1
            and isinstance(data[0], int)
            and len(shape) >= 1
            and (strides is None or (len(shape) == 1 and strides == (4,)))
        ):
            n = int(math.prod(shape))
            if n == 0:
                return _EMPTY_UINT32_PTR, 0, values
            ptr_addr = data[0]
            if ptr_addr != 0:
                return ctypes.cast(ptr_addr, POINTER(c_uint32)), n, values

    # 4. General Python sequence fallback
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("Expected a sequence of non-negative integers, not str/bytes")

    try:
        buf = array.array(_ARRAY_U32_CODE, values)
    except OverflowError as exc:
        raise ValueError("Unsigned 32-bit integer out of range [0, 2^32-1]") from exc
    addr, n = buf.buffer_info()
    if n == 0:
        return _EMPTY_UINT32_PTR, 0, buf
    return ctypes.cast(addr, POINTER(c_uint32)), n, buf


def _check_len(seq: Sequence[Any] | Iterable[Any], expected: int, name: str) -> tuple[Any, int, Any]:
    ptr, n, owner = _as_double_ptr(seq)
    if n != expected:
        raise ValueError(f"{name}: expected {expected} values, got {n}")
    return ptr, n, owner
