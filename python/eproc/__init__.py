"""``eproc`` — Sub-microsecond anytime-valid online statistical inference engine.

Uses PEP 562 lazy imports with zero eager stdlib imports (not even ``typing`` or
``ctypes``) so ``import eproc`` completes in ~1 ms without loading ``libeproc``
until a symbol is accessed.
"""

from __future__ import annotations

__version__ = "0.1.0"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Anytime-valid e-processes, confidence sequences, e-detectors & racing
    "EProcess": (".eprocess", "EProcess"),
    "EProcessResult": (".eprocess", "EProcessResult"),
    "ConfidenceSequence": (".eprocess", "ConfidenceSequence"),
    "ConfidenceSequenceResult": (".eprocess", "ConfidenceSequenceResult"),
    "EDetector": (".eprocess", "EDetector"),
    "EDetectorResult": (".eprocess", "EDetectorResult"),
    "AgentRace": (".eprocess", "AgentRace"),
    "RaceStatus": (".eprocess", "RaceStatus"),
    "RaceArmState": (".eprocess", "RaceArmState"),
    "combine_evalues": (".eprocess", "combine_evalues"),
    # Online conformal calibration, delayed, censored budgets & mergeable monoid
    "MultiRateConformal": (".conformal", "MultiRateConformal"),
    "ConformalPrediction": (".conformal", "ConformalPrediction"),
    "ConformalMeta": (".conformal", "ConformalMeta"),
    "DelayedConformal": (".conformal", "DelayedConformal"),
    "BudgetConformal": (".conformal", "BudgetConformal"),
    "ConformalMonoid": (".conformal", "ConformalMonoid"),
    # Byzantine-safe fusion & provenance-deduplicated evidence pool
    "SafeFuser": (".fusion", "SafeFuser"),
    "SafeFuserStep": (".fusion", "SafeFuserStep"),
    "EchoPool": (".fusion", "EchoPool"),
    "EchoSubmitResult": (".fusion", "EchoSubmitResult"),
    # Causal online subsampling & multi-resolution VOI
    "VarOptSampler": (".subsample", "VarOptSampler"),
    "VarOptItem": (".subsample", "VarOptItem"),
    "OnlineRLSSampler": (".subsample", "OnlineRLSSampler"),
    "RLSStepResult": (".subsample", "RLSStepResult"),
    "SiftVOISelector": (".subsample", "SiftVOISelector"),
    "SiftAction": (".subsample", "SiftAction"),
    # Drop-in eval/CI/sklearn/fleet adapters
    "EvalGate": (".adapters", "EvalGate"),
    "EvalGateSummary": (".adapters", "EvalGateSummary"),
    "PairedEvalGate": (".adapters", "PairedEvalGate"),
    "ConformalWrapper": (".adapters", "ConformalWrapper"),
    "IntervalResult": (".adapters", "IntervalResult"),
    "FleetCalibrationResult": (".adapters", "FleetCalibrationResult"),
    "assert_eprocess_uplift": (".adapters", "assert_eprocess_uplift"),
    "assert_no_regression": (".adapters", "assert_no_regression"),
    "fleet_calibrate": (".adapters", "fleet_calibrate"),
    # Runtime inspection helpers
    "abi_version": ("._ffi", "abi_version"),
    "simd_flags": ("._ffi", "simd_flags"),
    "simd_flag_names": ("._ffi", "simd_flag_names"),
}

__all__ = ["__version__", *_LAZY_EXPORTS.keys()]


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    mod_name, attr_name = target
    mod = import_module(mod_name, __name__)
    val = getattr(mod, attr_name)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
