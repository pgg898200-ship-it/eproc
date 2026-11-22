"""Type stub for lazy top-level ``eproc`` exports."""

from ._ffi import abi_version, simd_flag_names, simd_flags
from .conformal import (
    BudgetConformal,
    ConformalMeta,
    ConformalMonoid,
    ConformalPrediction,
    DelayedConformal,
    MultiRateConformal,
)
from .eprocess import (
    AgentRace,
    ConfidenceSequence,
    ConfidenceSequenceResult,
    EDetector,
    EDetectorResult,
    EProcess,
    EProcessResult,
    RaceArmState,
    RaceStatus,
    combine_evalues,
)
from .fusion import EchoPool, EchoSubmitResult, SafeFuser, SafeFuserStep
from .subsample import (
    OnlineRLSSampler,
    RLSStepResult,
    SiftAction,
    SiftVOISelector,
    VarOptItem,
    VarOptSampler,
)

__version__: str

__all__ = [
    "__version__",
    "EProcess",
    "EProcessResult",
    "ConfidenceSequence",
    "ConfidenceSequenceResult",
    "EDetector",
    "EDetectorResult",
    "AgentRace",
    "RaceStatus",
    "RaceArmState",
    "combine_evalues",
    "MultiRateConformal",
    "ConformalPrediction",
    "ConformalMeta",
    "DelayedConformal",
    "BudgetConformal",
    "ConformalMonoid",
    "SafeFuser",
    "SafeFuserStep",
    "EchoPool",
    "EchoSubmitResult",
    "VarOptSampler",
    "VarOptItem",
    "OnlineRLSSampler",
    "RLSStepResult",
    "SiftVOISelector",
    "SiftAction",
    "abi_version",
    "simd_flags",
    "simd_flag_names",
]
