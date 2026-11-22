"""Multi-rate online conformal calibration, delayed feedback, censored budgets, and mergeable monoids.

All algorithms run in C11 inside ``libeproc`` with crossing-free isotonic
projection (PAVA), strict causality under delayed label revelation, and exact
640-byte abelian group monoids for distributed leave-one-out calibration.
"""

from __future__ import annotations

from ctypes import byref, c_double, c_int32, c_size_t, c_uint8, c_uint32, c_uint64
from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

from ._ffi import (
    EPROC_CMONOID_BINS,
    EPROC_MAX_RATES,
    _as_double_ptr,
    check_rc,
    eproc_budget_conformal_t,
    eproc_cmonoid_t,
    eproc_conformal_out_t,
    eproc_conformal_t,
    eproc_delayed_conformal_t,
    get_lib,
)


def _prepare_alphas(alphas: Sequence[float]) -> list[float]:
    vals = [float(a) for a in alphas]
    if not (1 <= len(vals) <= EPROC_MAX_RATES):
        raise ValueError(
            f"Number of alpha levels must be in [1, {EPROC_MAX_RATES}], got {len(vals)}"
        )
    for a in vals:
        if not math.isfinite(a) or not (0.0 < a < 1.0):
            raise ValueError(f"Each alpha must be finite in (0, 1), got {a!r}")
    # Descending alpha corresponds to ascending nominal coverage (e.g. 50%..99%)
    vals.sort(reverse=True)
    return vals


@dataclass(slots=True, frozen=True)
class ConformalMeta:
    """Calibration state snapshot returned by conformal update operations."""

    n_rates: int
    alphas: tuple[float, ...]
    quantiles: tuple[float, ...]
    eff_alphas: tuple[float, ...]
    emp_coverage: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class ConformalPrediction:
    """Multi-rate crossing-free prediction intervals for a point forecast."""

    point_pred: float
    scale: float
    alphas: tuple[float, ...]
    lows: tuple[float, ...]
    highs: tuple[float, ...]
    quantiles: tuple[float, ...]
    eff_alphas: tuple[float, ...]
    emp_coverage: tuple[float, ...]

    def intervals(self) -> list[tuple[float, float]]:
        return list(zip(self.lows, self.highs))

    def by_alpha(self) -> dict[float, tuple[float, float]]:
        return {a: (lo, hi) for a, lo, hi in zip(self.alphas, self.lows, self.highs)}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["intervals"] = [
            {"alpha": a, "coverage": round(1.0 - a, 12), "low": lo, "high": hi}
            for a, lo, hi in zip(self.alphas, self.lows, self.highs)
        ]
        return d


def _meta_from_c(raw: eproc_conformal_out_t, alphas: tuple[float, ...]) -> ConformalMeta:
    k = int(raw.n_rates)
    return ConformalMeta(
        n_rates=k,
        alphas=alphas[:k],
        quantiles=tuple(float(raw.quantiles[i]) for i in range(k)),
        eff_alphas=tuple(float(raw.eff_alphas[i]) for i in range(k)),
        emp_coverage=tuple(float(raw.emp_coverage[i]) for i in range(k)),
    )


class MultiRateConformal:
    """Multi-rate crossing-free AgACI + Conformal PID calibrator.

    Simultaneously tracks up to 5 coverage levels (default 50/80/90/95/99%) with
    guaranteed non-crossing quantiles via isotonic projection and 4-expert BOA
    step-size aggregation per rate.
    """

    __slots__ = ("_lib", "_state", "_meta_out", "_lows_buf", "_highs_buf", "_alphas")

    def __init__(
        self,
        alphas: Sequence[float] = (0.50, 0.20, 0.10, 0.05, 0.01),
        window_size: int = 128,
        pid_ki: float = 0.05,
    ) -> None:
        self._lib = get_lib()
        self._state = eproc_conformal_t()
        self._meta_out = eproc_conformal_out_t()
        self._lows_buf = (c_double * EPROC_MAX_RATES)()
        self._highs_buf = (c_double * EPROC_MAX_RATES)()

        sorted_alphas = _prepare_alphas(alphas)
        alpha_arr = (c_double * len(sorted_alphas))(*sorted_alphas)
        rc = self._lib.eproc_conformal_init(
            byref(self._state),
            alpha_arr,
            c_uint32(len(sorted_alphas)),
            c_uint32(int(window_size)),
            c_double(float(pid_ki)),
        )
        check_rc(rc, "MultiRateConformal.__init__")
        k = int(self._state.n_rates)
        self._alphas = tuple(float(self._state.target_alphas[i]) for i in range(k))

    def predict(self, pred: float, scale: float = 1.0) -> ConformalPrediction:
        """Compute crossing-free prediction intervals around ``pred``."""
        p = float(pred)
        s = float(scale)
        self._lib.eproc_conformal_predict(
            byref(self._state),
            c_double(p),
            c_double(s),
            self._lows_buf,
            self._highs_buf,
            byref(self._meta_out),
        )
        k = int(self._meta_out.n_rates)
        return ConformalPrediction(
            point_pred=p,
            scale=s,
            alphas=self._alphas[:k],
            lows=tuple(float(self._lows_buf[i]) for i in range(k)),
            highs=tuple(float(self._highs_buf[i]) for i in range(k)),
            quantiles=tuple(float(self._meta_out.quantiles[i]) for i in range(k)),
            eff_alphas=tuple(float(self._meta_out.eff_alphas[i]) for i in range(k)),
            emp_coverage=tuple(float(self._meta_out.emp_coverage[i]) for i in range(k)),
        )

    def update(self, score: float) -> ConformalMeta:
        """Update calibration state with one nonconformity score (e.g. ``|y - pred| / scale``)."""
        rc = self._lib.eproc_conformal_update(
            byref(self._state), c_double(float(score)), byref(self._meta_out)
        )
        check_rc(rc, "MultiRateConformal.update")
        return _meta_from_c(self._meta_out, self._alphas)

    def update_batch(self, scores: Any) -> ConformalMeta:
        """Update calibration state with a contiguous buffer or sequence of scores."""
        ptr, n, _owner = _as_double_ptr(scores)
        if n == 0:
            pred = self.predict(0.0, 1.0)
            return ConformalMeta(
                n_rates=len(self._alphas),
                alphas=self._alphas,
                quantiles=pred.quantiles,
                eff_alphas=pred.eff_alphas,
                emp_coverage=pred.emp_coverage,
            )
        rc = self._lib.eproc_conformal_update_batch(
            byref(self._state), ptr, c_size_t(n), byref(self._meta_out)
        )
        check_rc(rc, "MultiRateConformal.update_batch")
        return _meta_from_c(self._meta_out, self._alphas)

    @property
    def alphas(self) -> tuple[float, ...]:
        return self._alphas

    @property
    def n_rates(self) -> int:
        return len(self._alphas)

    @property
    def n_updates(self) -> int:
        return int(self._state.n_updates)

    @property
    def quantiles(self) -> tuple[float, ...]:
        k = len(self._alphas)
        return tuple(float(self._state.current_q[i]) for i in range(k))

    def __repr__(self) -> str:
        return (
            f"MultiRateConformal(alphas={self._alphas}, n_updates={self.n_updates}, "
            f"quantiles={tuple(round(q, 4) for q in self.quantiles)})"
        )


class DelayedConformal:
    """Strictly causal delayed-feedback multi-rate conformal wrapper.

    Predictions issued via :meth:`issue` depend strictly on ground-truth labels
    revealed prior to issuance. Labels may arrive out-of-order via :meth:`reveal`.
    """

    __slots__ = ("_lib", "_state", "_meta_out", "_lows_buf", "_highs_buf", "_alphas")

    def __init__(
        self,
        alphas: Sequence[float] = (0.20, 0.10, 0.05),
        window_size: int = 128,
        pid_ki: float = 0.05,
    ) -> None:
        self._lib = get_lib()
        self._state = eproc_delayed_conformal_t()
        self._meta_out = eproc_conformal_out_t()
        self._lows_buf = (c_double * EPROC_MAX_RATES)()
        self._highs_buf = (c_double * EPROC_MAX_RATES)()

        sorted_alphas = _prepare_alphas(alphas)
        alpha_arr = (c_double * len(sorted_alphas))(*sorted_alphas)
        rc = self._lib.eproc_delayed_conformal_init(
            byref(self._state),
            alpha_arr,
            c_uint32(len(sorted_alphas)),
            c_uint32(int(window_size)),
            c_double(float(pid_ki)),
        )
        check_rc(rc, "DelayedConformal.__init__")
        k = int(self._state.core.n_rates)
        self._alphas = tuple(float(self._state.core.target_alphas[i]) for i in range(k))

    def issue(self, pred: float, scale: float = 1.0) -> tuple[int, list[tuple[float, float]]]:
        """Issue causal intervals for ``pred`` and return ``(ticket_id, intervals)``."""
        ticket = c_uint64(0)
        rc = self._lib.eproc_delayed_conformal_issue(
            byref(self._state),
            c_double(float(pred)),
            c_double(float(scale)),
            byref(ticket),
            self._lows_buf,
            self._highs_buf,
        )
        check_rc(rc, "DelayedConformal.issue")
        k = len(self._alphas)
        intervals = [(float(self._lows_buf[i]), float(self._highs_buf[i])) for i in range(k)]
        return int(ticket.value), intervals

    def reveal(self, ticket_id: int, y_true: float) -> ConformalMeta:
        """Reveal ground truth ``y_true`` for a previously issued ``ticket_id``."""
        if int(ticket_id) < 0:
            raise ValueError(f"ticket_id must be non-negative, got {ticket_id}")
        rc = self._lib.eproc_delayed_conformal_reveal(
            byref(self._state),
            c_uint64(int(ticket_id)),
            c_double(float(y_true)),
            byref(self._meta_out),
        )
        check_rc(rc, "DelayedConformal.reveal")
        return _meta_from_c(self._meta_out, self._alphas)

    @property
    def pending_count(self) -> int:
        return int(self._state.pending_count)

    @property
    def alphas(self) -> tuple[float, ...]:
        return self._alphas

    @property
    def n_rates(self) -> int:
        return len(self._alphas)

    @property
    def n_updates(self) -> int:
        return int(self._state.core.n_updates)

    def __repr__(self) -> str:
        return (
            f"DelayedConformal(alphas={self._alphas}, pending={self.pending_count}, "
            f"n_updates={self.n_updates})"
        )


class BudgetConformal:
    """Right-censored agent budget conformal calibrator (PROBE-style UPB).

    Produces conservative upper predictive bounds on agent token/step/latency
    cost even when timed-out trajectories are right-censored at their budget cap.
    """

    __slots__ = ("_lib", "_state")

    def __init__(
        self,
        alpha: float = 0.10,
        window_size: int = 128,
        censor_penalty: float = 1.25,
    ) -> None:
        self._lib = get_lib()
        self._state = eproc_budget_conformal_t()
        rc = self._lib.eproc_budget_conformal_init(
            byref(self._state),
            c_double(float(alpha)),
            c_uint32(int(window_size)),
            c_double(float(censor_penalty)),
        )
        check_rc(rc, "BudgetConformal.__init__")

    def bound(self, base_estimate: float) -> float:
        """Return the calibrated upper predictive budget bound for ``base_estimate``."""
        return float(
            self._lib.eproc_budget_conformal_bound(
                byref(self._state), c_double(float(base_estimate))
            )
        )

    def update(
        self,
        base_estimate: float,
        observed_cost: float,
        censored: bool = False,
    ) -> None:
        """Update with an observed (or right-censored) trajectory cost."""
        rc = self._lib.eproc_budget_conformal_update(
            byref(self._state),
            c_double(float(base_estimate)),
            c_double(float(observed_cost)),
            c_int32(1 if censored else 0),
        )
        check_rc(rc, "BudgetConformal.update")

    @property
    def alpha(self) -> float:
        return float(self._state.core.target_alphas[0])

    @property
    def censor_penalty(self) -> float:
        return float(self._state.censor_penalty)

    @property
    def n_completed(self) -> int:
        return int(self._state.n_completed)

    @property
    def n_censored(self) -> int:
        return int(self._state.n_censored)

    def __repr__(self) -> str:
        return (
            f"BudgetConformal(alpha={self.alpha}, completed={self.n_completed}, "
            f"censored={self.n_censored})"
        )


class _WireSizeDescriptor:
    def __get__(self, obj: Any, owner: Any = None) -> int:
        return int(get_lib().eproc_cmonoid_wire_size())


class ConformalMonoid:
    """Fixed 640-byte associative, commutative, and exact-LOO conformal monoid.

    Uses exact ``uint64_t`` bin counts across 64 signed log-spaced buckets so
    distributed ``merge`` and leave-one-out ``subtract`` have zero integer drift
    across arbitrary reduction trees.
    """

    __slots__ = ("_lib", "_state")

    wire_size: int = _WireSizeDescriptor()  # type: ignore[assignment]

    def __init__(self, scale_hint: float = 1.0, *, _empty: bool = False) -> None:
        self._lib = get_lib()
        self._state = eproc_cmonoid_t()
        if not _empty:
            self._lib.eproc_cmonoid_init(byref(self._state), c_double(float(scale_hint)))

    def add(self, score: float) -> None:
        """Insert a single nonconformity score into the monoid."""
        self._lib.eproc_cmonoid_add(byref(self._state), c_double(float(score)))

    def add_batch(self, scores: Any) -> None:
        """Insert a contiguous buffer or sequence of scores into the monoid."""
        ptr, n, _owner = _as_double_ptr(scores)
        if n == 0:
            return
        self._lib.eproc_cmonoid_add_batch(byref(self._state), ptr, c_size_t(n))

    def merge(self, other: ConformalMonoid) -> ConformalMonoid:
        """Return a new :class:`ConformalMonoid` equal to ``self + other``."""
        if not isinstance(other, ConformalMonoid):
            raise TypeError(f"Expected ConformalMonoid, got {type(other).__name__}")
        if self.count > 0 and other.count > 0:
            s1 = self.scale_hint
            s2 = other.scale_hint
            tol = 1e-12 * max(1.0, abs(s1), abs(s2))
            if abs(s1 - s2) > tol:
                raise ValueError(
                    f"ConformalMonoid.merge: incompatible scale_hint ({s1} vs {s2})"
                )
        out = ConformalMonoid(_empty=True)
        self._lib.eproc_cmonoid_merge(
            byref(self._state), byref(other._state), byref(out._state)
        )
        return out

    def subtract(self, local: ConformalMonoid) -> ConformalMonoid:
        """Return exact leave-one-out monoid ``self - local``."""
        if not isinstance(local, ConformalMonoid):
            raise TypeError(f"Expected ConformalMonoid, got {type(local).__name__}")
        out = ConformalMonoid(_empty=True)
        rc = self._lib.eproc_cmonoid_subtract(
            byref(self._state), byref(local._state), byref(out._state)
        )
        check_rc(rc, "ConformalMonoid.subtract")
        return out

    def __add__(self, other: ConformalMonoid) -> ConformalMonoid:
        return self.merge(other)

    def __sub__(self, other: ConformalMonoid) -> ConformalMonoid:
        return self.subtract(other)

    def quantile(self, q: float) -> float:
        """Compute the calibrated quantile at probability ``q in [0, 1]``."""
        qf = float(q)
        if not math.isfinite(qf) or qf < 0.0 or qf > 1.0:
            raise ValueError(f"Quantile probability q must be in [0, 1], got {q!r}")
        return float(self._lib.eproc_cmonoid_quantile(byref(self._state), c_double(qf)))

    def to_bytes(self) -> bytes:
        """Serialize the monoid to its deterministic wire representation (640 bytes)."""
        sz = int(self._lib.eproc_cmonoid_wire_size())
        buf = (c_uint8 * sz)()
        rc = self._lib.eproc_cmonoid_serialize(byref(self._state), buf, c_size_t(sz))
        check_rc(rc, "ConformalMonoid.to_bytes")
        return bytes(buf)

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> ConformalMonoid:
        """Deserialize a monoid from its exact 640-byte wire representation."""
        raw = memoryview(data).tobytes()
        n = len(raw)
        expected = int(get_lib().eproc_cmonoid_wire_size())
        if n != expected:
            raise ValueError(
                f"ConformalMonoid.from_bytes: expected exactly {expected} bytes, got {n}"
            )
        buf = (c_uint8 * n).from_buffer_copy(raw)
        out = cls(_empty=True)
        rc = out._lib.eproc_cmonoid_deserialize(byref(out._state), buf, c_size_t(n))
        check_rc(rc, "ConformalMonoid.from_bytes")
        return out

    @property
    def count(self) -> int:
        return int(self._state.count)

    @property
    def mean(self) -> float:
        c = int(self._state.count)
        if c == 0:
            return 0.0
        s = float(self._state.sum)
        if not math.isfinite(s):
            return s
        return float((s + float(self._state.sum_comp)) / c)

    @property
    def min_val(self) -> float:
        return float(self._state.min_val)

    @property
    def max_val(self) -> float:
        return float(self._state.max_val)

    @property
    def scale_hint(self) -> float:
        return float(self._state.scale_hint)

    @property
    def bin_counts(self) -> tuple[int, ...]:
        return tuple(int(self._state.bin_counts[i]) for i in range(EPROC_CMONOID_BINS))

    def __repr__(self) -> str:
        return (
            f"ConformalMonoid(count={self.count}, scale_hint={self.scale_hint}, "
            f"wire_size={self.wire_size})"
        )


import os as _os  # noqa: E402

if not _os.environ.get("EPROC_DISABLE_NATIVE"):
    try:
        from . import _native as _nat  # noqa: E402
    except ImportError:
        _nat = None
    if _nat is not None:
        MultiRateConformal = _nat.MultiRateConformal  # type: ignore[misc,assignment]
        ConformalPrediction = _nat.ConformalPrediction  # type: ignore[misc,assignment]
        ConformalMeta = _nat.ConformalMeta  # type: ignore[misc,assignment]
        DelayedConformal = _nat.DelayedConformal  # type: ignore[misc,assignment]
        BudgetConformal = _nat.BudgetConformal  # type: ignore[misc,assignment]
        ConformalMonoid = _nat.ConformalMonoid  # type: ignore[misc,assignment]

