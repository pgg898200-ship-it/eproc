"""Anytime-valid betting e-processes, confidence sequences, e-detectors, and agent racing.

All computations execute inside ``libeproc`` (C11, zero hot-path allocations,
strict Ville inequality safety under continuous every-tick peeking).
"""

from __future__ import annotations

from ctypes import byref, c_double, c_int32, c_size_t, c_uint32
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Sequence

from ._ffi import (
    EPROC_SIDE_LOWER,
    EPROC_SIDE_TWO_SIDED,
    EPROC_SIDE_UPPER,
    _as_double_ptr,
    _as_uint32_ptr,
    check_rc,
    eproc_bet_result_t,
    eproc_bet_t,
    eproc_cs_result_t,
    eproc_cs_t,
    eproc_edetect_result_t,
    eproc_edetect_t,
    eproc_race_status_t,
    eproc_race_t,
    get_lib,
)

_SIDE_MAP: dict[str, int] = {
    "upper": EPROC_SIDE_UPPER,
    "greater": EPROC_SIDE_UPPER,
    "lower": EPROC_SIDE_LOWER,
    "less": EPROC_SIDE_LOWER,
    "two_sided": EPROC_SIDE_TWO_SIDED,
    "two-sided": EPROC_SIDE_TWO_SIDED,
    "both": EPROC_SIDE_TWO_SIDED,
}

_SIDE_NAMES: dict[int, str] = {
    EPROC_SIDE_UPPER: "upper",
    EPROC_SIDE_LOWER: "lower",
    EPROC_SIDE_TWO_SIDED: "two_sided",
}


def _parse_side(side: str | int) -> int:
    if isinstance(side, int) and not isinstance(side, bool):
        if side in _SIDE_NAMES:
            return side
        raise ValueError(f"Invalid side integer {side}; expected 0, 1, or 2")
    if isinstance(side, str):
        key = side.strip().lower()
        if key in _SIDE_MAP:
            return _SIDE_MAP[key]
    raise ValueError(f"Invalid side {side!r}; expected 'upper', 'lower', or 'two_sided'")


@dataclass(slots=True, frozen=True)
class EProcessResult:
    """Snapshot returned by :class:`EProcess` updates and queries."""

    n: int
    e_value: float
    log_e: float
    max_log_e: float
    p_value: float
    mean: float
    rejected: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class ConfidenceSequenceResult:
    """Anytime-valid confidence interval returned by :class:`ConfidenceSequence`."""

    n: int
    mean: float
    low: float
    high: float
    width: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class EDetectorResult:
    """Streaming changepoint / drift detection snapshot returned by :class:`EDetector`."""

    n: int
    log_stat: float
    stat: float
    alarm: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class RaceStatus:
    """Global status of an :class:`AgentRace` multi-arm elimination tournament."""

    n_arms: int
    n_active: int
    winner: int | None
    leader: int | None
    total_obs: int
    leader_low: float
    leader_high: float
    runner_up_high: float
    winner_name: str | None = None
    leader_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class RaceArmState:
    """Per-arm confidence sequence and elimination state inside :class:`AgentRace`."""

    arm: int
    name: str | None
    active: bool
    n: int
    mean: float
    low: float
    high: float
    width: float
    eliminated_at: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bet_result_from_c(raw: eproc_bet_result_t) -> EProcessResult:
    return EProcessResult(
        n=int(raw.n),
        e_value=float(raw.e_value),
        log_e=float(raw.log_e),
        max_log_e=float(raw.max_log_e),
        p_value=float(raw.p_value),
        mean=float(raw.mean),
        rejected=bool(raw.rejected),
    )


def _cs_result_from_c(raw: eproc_cs_result_t) -> ConfidenceSequenceResult:
    return ConfidenceSequenceResult(
        n=int(raw.n),
        mean=float(raw.mean),
        low=float(raw.low),
        high=float(raw.high),
        width=float(raw.width),
    )


class EProcess:
    """Anytime-valid ONS betting supermartingale e-process for bounded streams.

    Tests ``H0: E[X] <= mu0`` (``side='upper'``), ``H0: E[X] >= mu0`` (``side='lower'``),
    or ``H0: E[X] == mu0`` (``side='two_sided'``) for observations ``X_t in [lower, upper]``.
    Ville's inequality guarantees ``P_{H0}(exists t >= 1: E_t >= 1/alpha) <= alpha``
    even when peeking and stopping at every single observation.
    """

    __slots__ = ("_lib", "_state", "_out", "_mu0", "_side")

    def __init__(
        self,
        mu0: float,
        alpha: float = 0.05,
        lower: float = 0.0,
        upper: float = 1.0,
        side: str | int = "upper",
    ) -> None:
        self._lib = get_lib()
        self._state = eproc_bet_t()
        self._out = eproc_bet_result_t()
        side_code = _parse_side(side)
        self._mu0 = float(mu0)
        self._side = _SIDE_NAMES[side_code]
        rc = self._lib.eproc_bet_init(
            byref(self._state),
            c_double(self._mu0),
            c_double(float(alpha)),
            c_double(float(lower)),
            c_double(float(upper)),
            c_int32(side_code),
        )
        check_rc(rc, "EProcess.__init__")

    def update(self, x: float) -> EProcessResult:
        """Ingest one scalar observation and return the updated anytime-valid result."""
        rc = self._lib.eproc_bet_update(
            byref(self._state), c_double(float(x)), byref(self._out)
        )
        check_rc(rc, "EProcess.update")
        return _bet_result_from_c(self._out)

    def update_batch(self, xs: Any) -> EProcessResult:
        """Ingest a contiguous buffer or Python sequence of observations in C."""
        ptr, n, _owner = _as_double_ptr(xs)
        if n == 0:
            return self.result()
        rc = self._lib.eproc_bet_update_batch(
            byref(self._state), ptr, c_size_t(n), byref(self._out)
        )
        check_rc(rc, "EProcess.update_batch")
        return _bet_result_from_c(self._out)

    def result(self) -> EProcessResult:
        """Query the current anytime-valid e-value, p-value, mean, and verdict."""
        self._lib.eproc_bet_query(byref(self._state), byref(self._out))
        return _bet_result_from_c(self._out)

    @property
    def n(self) -> int:
        return int(self._state.n)

    @property
    def e_value(self) -> float:
        return self.result().e_value

    @property
    def log_e(self) -> float:
        return float(self._state.log_e)

    @property
    def max_log_e(self) -> float:
        return float(self._state.max_log_e)

    @property
    def p_value(self) -> float:
        return self.result().p_value

    @property
    def rejected(self) -> bool:
        return self.result().rejected

    @property
    def mean(self) -> float:
        return self.result().mean

    @property
    def rejected_at(self) -> int:
        return int(self._state.rejected_at)

    @property
    def n_invalid(self) -> int:
        return int(self._state.n_invalid)

    @property
    def alpha(self) -> float:
        return float(self._state.alpha)

    @property
    def mu0(self) -> float:
        return self._mu0

    @property
    def lower(self) -> float:
        return float(self._state.lower)

    @property
    def upper(self) -> float:
        return float(self._state.upper)

    @property
    def side(self) -> str:
        return self._side

    def __repr__(self) -> str:
        res = self.result()
        return (
            f"EProcess(mu0={self._mu0}, alpha={self.alpha}, side={self._side!r}, "
            f"n={res.n}, mean={res.mean:.4g}, p_value={res.p_value:.4g}, rejected={res.rejected})"
        )


class ConfidenceSequence:
    """Anytime-valid confidence sequence with O(1) time and O(1) memory per update.

    Supports:
      - ``mode='eb'``: Closed-form Empirical-Bernstein CS (Waudby-Smith & Ramdas 2024)
        for bounded streams ``X_t in [lower, upper]``.
      - ``mode='normal'``: Sub-Gaussian Normal-Mixture CS (Howard et al. 2021)
        parameterized by ``sigma`` and ``v_opt``.
    """

    __slots__ = ("_lib", "_state", "_out", "_mode")

    def __init__(
        self,
        alpha: float = 0.05,
        lower: float = 0.0,
        upper: float = 1.0,
        mode: str = "eb",
        sigma: float = 1.0,
        v_opt: float = 100.0,
    ) -> None:
        self._lib = get_lib()
        self._state = eproc_cs_t()
        self._out = eproc_cs_result_t()
        norm_mode = mode.strip().lower()
        if norm_mode in ("eb", "empirical_bernstein", "empirical-bernstein"):
            self._mode = "eb"
            rc = self._lib.eproc_cs_init_eb(
                byref(self._state),
                c_double(float(alpha)),
                c_double(float(lower)),
                c_double(float(upper)),
            )
        elif norm_mode in ("normal", "normal_mixture", "normal-mixture", "subgaussian"):
            self._mode = "normal"
            rc = self._lib.eproc_cs_init_normal(
                byref(self._state),
                c_double(float(alpha)),
                c_double(float(sigma)),
                c_double(float(v_opt)),
            )
        else:
            raise ValueError(f"Invalid mode {mode!r}; expected 'eb' or 'normal'")
        check_rc(rc, "ConfidenceSequence.__init__")

    def update(self, x: float) -> ConfidenceSequenceResult:
        """Ingest one observation and return the running-intersection confidence interval."""
        rc = self._lib.eproc_cs_update(
            byref(self._state), c_double(float(x)), byref(self._out)
        )
        check_rc(rc, "ConfidenceSequence.update")
        return _cs_result_from_c(self._out)

    def update_batch(self, xs: Any) -> ConfidenceSequenceResult:
        """Ingest a batch of observations in C and return the updated interval."""
        ptr, n, _owner = _as_double_ptr(xs)
        if n == 0:
            return self.query()
        rc = self._lib.eproc_cs_update_batch(
            byref(self._state), ptr, c_size_t(n), byref(self._out)
        )
        check_rc(rc, "ConfidenceSequence.update_batch")
        return _cs_result_from_c(self._out)

    def query(self) -> ConfidenceSequenceResult:
        """Query the current anytime-valid confidence interval."""
        self._lib.eproc_cs_query(byref(self._state), byref(self._out))
        return _cs_result_from_c(self._out)

    @property
    def bounds(self) -> tuple[float, float]:
        res = self.query()
        return (res.low, res.high)

    @property
    def width(self) -> float:
        return self.query().width

    @property
    def mean(self) -> float:
        return self.query().mean

    @property
    def n(self) -> int:
        return int(self._state.n)

    @property
    def alpha(self) -> float:
        return float(self._state.alpha)

    @property
    def mode(self) -> str:
        return self._mode

    def __repr__(self) -> str:
        res = self.query()
        return (
            f"ConfidenceSequence(mode={self._mode!r}, alpha={self.alpha}, "
            f"n={res.n}, mean={res.mean:.4g}, bounds=({res.low:.4g}, {res.high:.4g}))"
        )


class EDetector:
    """Anytime-valid Page / Shiryaev-Roberts resettable e-detector for drift monitoring.

    Maintains a resettable CUSUM-style e-statistic with average run length to
    false alarm ``ARL >= 1 / alpha`` under ``H0``.
    """

    __slots__ = ("_lib", "_state", "_out", "_mu0", "_side")

    def __init__(
        self,
        mu0: float,
        alpha: float = 0.05,
        lower: float = 0.0,
        upper: float = 1.0,
        side: str | int = "upper",
    ) -> None:
        self._lib = get_lib()
        self._state = eproc_edetect_t()
        self._out = eproc_edetect_result_t()
        side_code = _parse_side(side)
        self._mu0 = float(mu0)
        self._side = _SIDE_NAMES[side_code]
        rc = self._lib.eproc_edetect_init(
            byref(self._state),
            c_double(self._mu0),
            c_double(float(alpha)),
            c_double(float(lower)),
            c_double(float(upper)),
            c_int32(side_code),
        )
        check_rc(rc, "EDetector.__init__")

    def update(self, x: float) -> EDetectorResult:
        """Ingest one observation and return the e-detector alarm state."""
        rc = self._lib.eproc_edetect_update(
            byref(self._state), c_double(float(x)), byref(self._out)
        )
        check_rc(rc, "EDetector.update")
        return EDetectorResult(
            n=int(self._out.n),
            log_stat=float(self._out.log_stat),
            stat=float(self._out.stat),
            alarm=bool(self._out.alarm),
        )

    def update_batch(self, xs: Iterable[float]) -> EDetectorResult:
        """Ingest an iterable of observations sequentially."""
        res: EDetectorResult | None = None
        for val in xs:
            res = self.update(val)
        if res is None:
            log_stat = float(self._state.log_cusum)
            stat = math.exp(min(log_stat, 700.0))
            return EDetectorResult(
                n=int(self._state.n),
                log_stat=log_stat,
                stat=stat,
                alarm=bool(log_stat >= float(self._state.log_thresh)),
            )
        return res

    def reset_alarm(self) -> None:
        """Reset the CUSUM e-detector after handling an alarm."""
        self._lib.eproc_edetect_reset_alarm(byref(self._state))
        self._out.log_stat = 0.0
        self._out.stat = 1.0
        self._out.alarm = 0

    @property
    def n(self) -> int:
        return int(self._state.n)

    @property
    def log_stat(self) -> float:
        return float(self._state.log_cusum)

    @property
    def stat(self) -> float:
        return math.exp(min(float(self._state.log_cusum), 700.0))

    @property
    def alarm_at(self) -> int:
        return int(self._state.alarm_at)

    @property
    def n_alarms(self) -> int:
        return int(self._state.n_alarms)

    def __repr__(self) -> str:
        return (
            f"EDetector(mu0={self._mu0}, side={self._side!r}, n={self.n}, "
            f"log_stat={self.log_stat:.4g}, n_alarms={self.n_alarms})"
        )


class AgentRace:
    """Anytime-valid multi-arm agent/model racer with automatic early elimination.

    Maintains a Bonferroni-adjusted Empirical-Bernstein confidence sequence per
    arm, eliminating provably suboptimal arms as soon as their upper confidence
    bound falls below another arm's lower confidence bound minus ``margin``.
    """

    __slots__ = ("_lib", "_state", "_status_out", "_cs_out", "_names", "_name_to_idx")

    def __init__(
        self,
        n_arms: int,
        alpha: float = 0.05,
        lower: float = 0.0,
        upper: float = 1.0,
        margin: float = 0.0,
        names: Sequence[str] | None = None,
    ) -> None:
        self._lib = get_lib()
        self._state = eproc_race_t()
        self._status_out = eproc_race_status_t()
        self._cs_out = eproc_cs_result_t()
        if names is not None:
            name_tuple = tuple(str(name) for name in names)
            if len(name_tuple) != int(n_arms):
                raise ValueError(
                    f"len(names) ({len(name_tuple)}) must equal n_arms ({n_arms})"
                )
            if len(set(name_tuple)) != len(name_tuple):
                raise ValueError("Arm names must be unique")
            self._names: tuple[str, ...] | None = name_tuple
            self._name_to_idx: dict[str, int] | None = {
                name: idx for idx, name in enumerate(name_tuple)
            }
        else:
            self._names = None
            self._name_to_idx = None

        rc = self._lib.eproc_race_init(
            byref(self._state),
            c_uint32(int(n_arms)),
            c_double(float(alpha)),
            c_double(float(lower)),
            c_double(float(upper)),
            c_double(float(margin)),
        )
        check_rc(rc, "AgentRace.__init__")

    def _resolve_arm(self, arm: int | str) -> int:
        if isinstance(arm, str):
            if self._name_to_idx is None or arm not in self._name_to_idx:
                raise KeyError(f"Unknown arm name {arm!r}")
            return self._name_to_idx[arm]
        idx = int(arm)
        if idx < 0 or idx >= int(self._state.n_arms):
            raise ValueError(f"Arm index {idx} out of range [0, {self._state.n_arms})")
        return idx

    def _status_from_c(self, raw: eproc_race_status_t) -> RaceStatus:
        winner = None if raw.winner < 0 else int(raw.winner)
        leader = None if raw.leader < 0 else int(raw.leader)
        winner_name = self._names[winner] if (self._names is not None and winner is not None) else None
        leader_name = self._names[leader] if (self._names is not None and leader is not None) else None
        return RaceStatus(
            n_arms=int(raw.n_arms),
            n_active=int(raw.n_active),
            winner=winner,
            leader=leader,
            total_obs=int(raw.total_obs),
            leader_low=float(raw.leader_low),
            leader_high=float(raw.leader_high),
            runner_up_high=float(raw.runner_up_high),
            winner_name=winner_name,
            leader_name=leader_name,
        )

    def update(self, arm: int | str, reward: float) -> RaceStatus:
        """Record a reward for ``arm`` (index or name) and update eliminations."""
        idx = self._resolve_arm(arm)
        rc = self._lib.eproc_race_update(
            byref(self._state),
            c_uint32(idx),
            c_double(float(reward)),
            byref(self._status_out),
        )
        check_rc(rc, "AgentRace.update")
        return self._status_from_c(self._status_out)

    def update_batch(self, arms: Any, rewards: Any) -> RaceStatus:
        """Record a batch of ``(arms, rewards)`` pairs in C."""
        if (
            self._name_to_idx is not None
            and isinstance(arms, Sequence)
            and not isinstance(arms, (bytes, bytearray, str))
            and len(arms) > 0
            and isinstance(arms[0], str)
        ):
            arms = [self._resolve_arm(a) for a in arms]

        arm_ptr, n_arms, _arm_owner = _as_uint32_ptr(arms)
        rew_ptr, n_rew, _rew_owner = _as_double_ptr(rewards)
        if n_arms != n_rew:
            raise ValueError(
                f"arms length ({n_arms}) must match rewards length ({n_rew})"
            )
        if n_arms == 0:
            return self.status()
        rc = self._lib.eproc_race_update_batch(
            byref(self._state),
            arm_ptr,
            rew_ptr,
            c_size_t(n_arms),
            byref(self._status_out),
        )
        check_rc(rc, "AgentRace.update_batch")
        return self._status_from_c(self._status_out)

    def suggest_arm(self) -> int:
        """Recommend the next active arm to evaluate (LCB/UCB gap maximizer)."""
        return int(self._lib.eproc_race_suggest_arm(byref(self._state)))

    def status(self) -> RaceStatus:
        """Return the current race status and certified winner (if any)."""
        self._lib.eproc_race_status(byref(self._state), byref(self._status_out))
        return self._status_from_c(self._status_out)

    def arm_state(self, arm: int | str) -> RaceArmState:
        """Inspect the confidence sequence and active status for a single arm."""
        idx = self._resolve_arm(arm)
        is_active = c_int32(0)
        rc = self._lib.eproc_race_arm_state(
            byref(self._state), c_uint32(idx), byref(self._cs_out), byref(is_active)
        )
        check_rc(rc, "AgentRace.arm_state")
        name = self._names[idx] if self._names is not None else None
        return RaceArmState(
            arm=idx,
            name=name,
            active=bool(is_active.value),
            n=int(self._cs_out.n),
            mean=float(self._cs_out.mean),
            low=float(self._cs_out.low),
            high=float(self._cs_out.high),
            width=float(self._cs_out.width),
            eliminated_at=int(self._state.eliminated_at[idx]),
        )

    @property
    def n_arms(self) -> int:
        return int(self._state.n_arms)

    @property
    def winner(self) -> int | None:
        w = int(self._state.winner)
        return None if w < 0 else w

    @property
    def winner_name(self) -> str | None:
        w = self.winner
        if w is None or self._names is None:
            return None
        return self._names[w]

    @property
    def active_arms(self) -> list[int]:
        return [i for i in range(int(self._state.n_arms)) if self._state.active[i] != 0]

    @property
    def names(self) -> tuple[str, ...] | None:
        return self._names

    def __repr__(self) -> str:
        st = self.status()
        return (
            f"AgentRace(n_arms={st.n_arms}, n_active={st.n_active}, "
            f"winner={st.winner}, total_obs={st.total_obs})"
        )


def combine_evalues(log_es: Any, independent: bool = False) -> float:
    """Combine multiple log e-values into a single valid log e-value.

    - ``independent=True``: product of independent e-values (sum of log e-values).
    - ``independent=False`` (default): Vovk & Wang (2021) arithmetic mean
      ``log((1/K) * sum_k exp(log_e_k))``, valid under arbitrary dependence.
    """
    lib = get_lib()
    ptr, n, _owner = _as_double_ptr(log_es)
    if n == 0:
        raise ValueError("combine_evalues: log_es must not be empty")
    if independent:
        acc = 0.0
        for i in range(n):
            acc = float(lib.eproc_combine_independent_log(c_double(acc), c_double(ptr[i])))
        return acc
    return float(lib.eproc_combine_dependent_log(ptr, c_size_t(n)))


import os as _os  # noqa: E402

if not _os.environ.get("EPROC_DISABLE_NATIVE"):
    try:
        from . import _native as _nat  # noqa: E402
    except ImportError:
        _nat = None
    if _nat is not None:
        EProcess = _nat.EProcess  # type: ignore[misc,assignment]
        EProcessResult = _nat.EProcessResult  # type: ignore[misc,assignment]
        ConfidenceSequence = _nat.ConfidenceSequence  # type: ignore[misc,assignment]
        ConfidenceSequenceResult = _nat.ConfidenceSequenceResult  # type: ignore[misc,assignment]
        EDetector = _nat.EDetector  # type: ignore[misc,assignment]
        EDetectorResult = _nat.EDetectorResult  # type: ignore[misc,assignment]
        AgentRace = _nat.AgentRace  # type: ignore[misc,assignment]
        RaceStatus = _nat.RaceStatus  # type: ignore[misc,assignment]
        RaceArmState = _nat.RaceArmState  # type: ignore[misc,assignment]
        combine_evalues = _nat.combine_evalues  # type: ignore[assignment]

