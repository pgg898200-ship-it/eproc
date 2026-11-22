"""Zero-dependency drop-in adapters for ML, LLM evaluation, CI gates, and fleet calibration.

Built exclusively on the Python standard library and :mod:`eproc` (zero third-party
imports). Designed for 1-line adoption in existing evaluation harnesses, ``pytest``
suites, ``scikit-learn`` / ``MAPIE`` regression pipelines, and distributed fleets:

- :class:`EvalGate` & :class:`PairedEvalGate`: anytime-valid sequential eval gates
  with early stopping and fixed-horizon token/sample savings reporting.
- :class:`ConformalWrapper` & :class:`IntervalResult`: drop-in ``sklearn``/``MAPIE``-
  style multi-rate crossing-free conformal wrapper around any ``.predict(X)`` model
  or callable ``fn(x)``.
- :func:`assert_eprocess_uplift` & :func:`assert_no_regression`: ``pytest`` / CI
  anytime-valid assertion helpers with rich diagnostic messages.
- :func:`fleet_calibrate` & :class:`FleetCalibrationResult`: 1-line distributed
  640-byte abelian monoid merge + exact leave-one-out peer quantiles per shard.
"""

from __future__ import annotations

import array
from dataclasses import asdict, dataclass
import math
import numbers
from typing import Any, Callable, Iterable, Iterator, Sequence

from .conformal import ConformalMeta, ConformalMonoid, MultiRateConformal
from .eprocess import EProcess, EProcessResult

__all__ = [
    "EvalGateSummary",
    "EvalGate",
    "PairedEvalGate",
    "IntervalResult",
    "ConformalWrapper",
    "assert_eprocess_uplift",
    "assert_no_regression",
    "FleetCalibrationResult",
    "fleet_calibrate",
]


def _is_real_scalar(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _validate_fixed_horizon(fixed_horizon_n: int | None) -> int | None:
    if fixed_horizon_n is None:
        return None
    if isinstance(fixed_horizon_n, bool) or not isinstance(fixed_horizon_n, int):
        raise TypeError(
            f"fixed_horizon_n must be a positive int or None, got {type(fixed_horizon_n).__name__}"
        )
    if fixed_horizon_n < 1:
        raise ValueError(f"fixed_horizon_n must be >= 1, got {fixed_horizon_n}")
    return fixed_horizon_n


@dataclass(slots=True, frozen=True)
class EvalGateSummary:
    """Structured anytime-valid summary returned by :meth:`EvalGate.summary`."""

    verdict: str
    rejected: bool
    n: int
    e_value: float
    log_e: float
    p_value: float
    mean: float
    mu0: float
    alpha: float
    side: str
    rejected_at: int
    fixed_horizon_n: int | None = None
    samples_saved_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        out = asdict(self)
        if self.fixed_horizon_n is None:
            out.pop("fixed_horizon_n", None)
            out.pop("samples_saved_pct", None)
        return out

    def __getitem__(self, key: str) -> Any:
        if key == "fixed_horizon_n" and self.fixed_horizon_n is None:
            raise KeyError(key)
        if key == "samples_saved_pct" and self.samples_saved_pct is None:
            raise KeyError(key)
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in ("fixed_horizon_n", "samples_saved_pct"):
            return self.fixed_horizon_n is not None
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class EvalGate:
    """Drop-in anytime-valid sequential gate wrapping :class:`eproc.EProcess`.

    Safe to inspect :attr:`should_stop` after every single observation without
    inflating Type-I error (Ville's inequality guarantees false-positive rate
    ``<= alpha`` under continuous peeking).
    """

    __slots__ = ("_ep", "_fixed_horizon_n")

    def __init__(
        self,
        mu0: float,
        alpha: float = 0.05,
        lower: float = 0.0,
        upper: float = 1.0,
        side: str | int = "upper",
        fixed_horizon_n: int | None = None,
    ) -> None:
        self._fixed_horizon_n = _validate_fixed_horizon(fixed_horizon_n)
        self._ep = EProcess(
            mu0=mu0,
            alpha=alpha,
            lower=lower,
            upper=upper,
            side=side,
        )

    def observe(self, score: float) -> EProcessResult:
        """Ingest one scalar observation and return the updated :class:`EProcessResult`."""
        return self._ep.update(score)

    def observe_batch(self, scores: Any) -> EProcessResult:
        """Ingest a batch/iterable/buffer of observations in C."""
        return self._ep.update_batch(scores)

    def result(self) -> EProcessResult:
        """Return the current :class:`EProcessResult` snapshot without mutating state."""
        return self._ep.result()

    @property
    def should_stop(self) -> bool:
        """Whether the anytime-valid e-process has crossed ``1 / alpha`` and rejected ``H0``."""
        return self._ep.rejected

    @property
    def rejected(self) -> bool:
        return self._ep.rejected

    @property
    def budget_reached(self) -> bool:
        """Whether ``n`` has reached ``fixed_horizon_n`` (always ``False`` if unset)."""
        return self._fixed_horizon_n is not None and self._ep.n >= self._fixed_horizon_n

    @property
    def n(self) -> int:
        return self._ep.n

    @property
    def e_value(self) -> float:
        return self._ep.e_value

    @property
    def p_value(self) -> float:
        return self._ep.p_value

    @property
    def mean(self) -> float:
        return self._ep.mean

    @property
    def rejected_at(self) -> int:
        return self._ep.rejected_at

    @property
    def fixed_horizon_n(self) -> int | None:
        return self._fixed_horizon_n

    @property
    def eprocess(self) -> EProcess:
        """Access the underlying :class:`eproc.EProcess` instance."""
        return self._ep

    def summary(self) -> EvalGateSummary:
        """Return a structured summary with verdict, e/p-values, mean, and sample savings."""
        res = self._ep.result()
        rej_at = self._ep.rejected_at
        saved_pct: float | None = None
        if self._fixed_horizon_n is not None:
            if res.rejected:
                stop_n = rej_at if rej_at > 0 else res.n
                saved_pct = max(0.0, 100.0 * (1.0 - (stop_n / self._fixed_horizon_n)))
            else:
                saved_pct = 0.0
        return EvalGateSummary(
            verdict="REJECT" if res.rejected else "CONTINUE",
            rejected=res.rejected,
            n=res.n,
            e_value=res.e_value,
            log_e=res.log_e,
            p_value=res.p_value,
            mean=res.mean,
            mu0=self._ep.mu0,
            alpha=self._ep.alpha,
            side=self._ep.side,
            rejected_at=rej_at,
            fixed_horizon_n=self._fixed_horizon_n,
            samples_saved_pct=saved_pct,
        )

    def __repr__(self) -> str:
        s = self.summary()
        extra = f", saved={s.samples_saved_pct:.1f}%" if s.samples_saved_pct is not None else ""
        return (
            f"EvalGate(verdict={s.verdict!r}, n={s.n}, mean={s.mean:.4g}, "
            f"e_value={s.e_value:.4g}, p_value={s.p_value:.4g}{extra})"
        )


class PairedEvalGate:
    """Anytime-valid paired A/B evaluation gate for candidate vs. baseline scores.

    Tests paired score differences ``d_t = candidate_t - baseline_t`` bounded in
    ``[-span, +span]`` against ``H0: E[d_t] <= mu0`` (default ``mu0 = 0.0``,
    ``side = 'upper'``) via ``EProcess(mu0=0.0, lower=-span, upper=span)``.
    """

    __slots__ = (
        "_gate",
        "_span",
        "_cand_sum",
        "_base_sum",
        "_pair_n",
        "_wins",
        "_losses",
        "_ties",
    )

    def __init__(
        self,
        alpha: float = 0.05,
        span: float = 1.0,
        side: str | int = "upper",
        mu0: float = 0.0,
        fixed_horizon_n: int | None = None,
    ) -> None:
        span_f = float(span)
        if not math.isfinite(span_f) or span_f <= 0.0:
            raise ValueError(f"span must be finite and > 0, got {span!r}")
        self._span = span_f
        self._gate = EvalGate(
            mu0=mu0,
            alpha=alpha,
            lower=-span_f,
            upper=span_f,
            side=side,
            fixed_horizon_n=fixed_horizon_n,
        )
        self._cand_sum = 0.0
        self._base_sum = 0.0
        self._pair_n = 0
        self._wins = 0
        self._losses = 0
        self._ties = 0

    def observe_pair(self, candidate_score: float, baseline_score: float) -> EProcessResult:
        """Observe one paired ``(candidate_score, baseline_score)`` evaluation."""
        c = float(candidate_score)
        b = float(baseline_score)
        if math.isfinite(c) and math.isfinite(b):
            self._cand_sum += c
            self._base_sum += b
            self._pair_n += 1
            if c > b:
                self._wins += 1
            elif c < b:
                self._losses += 1
            else:
                self._ties += 1
        return self._gate.observe(c - b)

    def observe_pairs(self, pairs: Iterable[tuple[float, float]]) -> EProcessResult:
        """Observe an iterable of ``(candidate_score, baseline_score)`` tuples."""
        diffs = array.array("d")
        for item in pairs:
            if len(item) != 2:
                raise ValueError(f"Expected 2-tuple (candidate, baseline), got length {len(item)}")
            c = float(item[0])
            b = float(item[1])
            if math.isfinite(c) and math.isfinite(b):
                self._cand_sum += c
                self._base_sum += b
                self._pair_n += 1
                if c > b:
                    self._wins += 1
                elif c < b:
                    self._losses += 1
                else:
                    self._ties += 1
            diffs.append(c - b)
        return self._gate.observe_batch(diffs)

    def observe(self, candidate_or_diff: float, baseline_score: float | None = None) -> EProcessResult:
        """Observe either ``(candidate_score, baseline_score)`` or a pre-computed ``diff``."""
        if baseline_score is not None:
            return self.observe_pair(candidate_or_diff, baseline_score)
        return self._gate.observe(candidate_or_diff)

    def observe_batch(
        self,
        candidates_or_diffs: Any,
        baseline_scores: Any | None = None,
    ) -> EProcessResult:
        """Observe paired sequences ``(candidates, baselines)``, pair tuples, or raw diffs."""
        if baseline_scores is not None:
            c_list = [float(x) for x in candidates_or_diffs]
            b_list = [float(x) for x in baseline_scores]
            if len(c_list) != len(b_list):
                raise ValueError(
                    f"Candidate length ({len(c_list)}) must match baseline length ({len(b_list)})"
                )
            return self.observe_pairs(zip(c_list, b_list))

        if isinstance(candidates_or_diffs, (bytes, bytearray, memoryview, array.array)):
            return self._gate.observe_batch(candidates_or_diffs)

        items = list(candidates_or_diffs)
        if not items:
            return self._gate.result()
        first = items[0]
        if isinstance(first, (tuple, list)) and len(first) == 2:
            return self.observe_pairs(items)
        return self._gate.observe_batch(items)

    def result(self) -> EProcessResult:
        return self._gate.result()

    def summary(self) -> EvalGateSummary:
        return self._gate.summary()

    @property
    def should_stop(self) -> bool:
        return self._gate.should_stop

    @property
    def rejected(self) -> bool:
        return self._gate.rejected

    @property
    def budget_reached(self) -> bool:
        return self._gate.budget_reached

    @property
    def n(self) -> int:
        return self._gate.n

    @property
    def e_value(self) -> float:
        return self._gate.e_value

    @property
    def p_value(self) -> float:
        return self._gate.p_value

    @property
    def mean(self) -> float:
        """Mean paired difference ``E[candidate - baseline]``."""
        return self._gate.mean

    @property
    def rejected_at(self) -> int:
        return self._gate.rejected_at

    @property
    def span(self) -> float:
        return self._span

    @property
    def candidate_mean(self) -> float:
        return self._cand_sum / self._pair_n if self._pair_n > 0 else 0.0

    @property
    def baseline_mean(self) -> float:
        return self._base_sum / self._pair_n if self._pair_n > 0 else 0.0

    @property
    def win_rate(self) -> float:
        """Fraction of paired evaluations where ``candidate > baseline``."""
        return self._wins / self._pair_n if self._pair_n > 0 else 0.0

    @property
    def wins(self) -> int:
        return self._wins

    @property
    def losses(self) -> int:
        return self._losses

    @property
    def ties(self) -> int:
        return self._ties

    @property
    def eprocess(self) -> EProcess:
        return self._gate.eprocess

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"PairedEvalGate(verdict={s.verdict!r}, n={s.n}, mean_diff={s.mean:+.4g}, "
            f"e_value={s.e_value:.4g}, p_value={s.p_value:.4g})"
        )


@dataclass(slots=True, frozen=True)
class IntervalResult:
    """Multi-rate crossing-free prediction intervals returned by :meth:`ConformalWrapper.predict_interval`.

    Supports both structured attribute access (``res.y_hat``, ``res.lows_by_rate[0.05]``,
    ``res.highs_by_rate[0.05]``) and 3-tuple unpacking::

        y_hat, lows_by_rate, highs_by_rate = wrapper.predict_interval(X)
    """

    y_hat: tuple[float, ...]
    lows_by_rate: dict[float, tuple[float, ...]]
    highs_by_rate: dict[float, tuple[float, ...]]
    alphas: tuple[float, ...]
    quantiles: tuple[float, ...]
    eff_alphas: tuple[float, ...]
    emp_coverage: tuple[float, ...]

    def __iter__(self) -> Iterator[Any]:
        yield self.y_hat
        yield self.lows_by_rate
        yield self.highs_by_rate

    def __len__(self) -> int:
        return len(self.y_hat)

    def interval_for(self, alpha: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return ``(lows, highs)`` tuples for a configured ``alpha`` level."""
        key = float(alpha)
        if key in self.lows_by_rate:
            return self.lows_by_rate[key], self.highs_by_rate[key]
        for a in self.alphas:
            if abs(a - key) <= 1e-12:
                return self.lows_by_rate[a], self.highs_by_rate[a]
        raise KeyError(f"Alpha {alpha!r} not in calibrated alphas {self.alphas}")

    def has_crossings(self, tol: float = 1e-12) -> bool:
        """Return ``True`` if any sample violates strict interval nesting across rates."""
        k = len(self.alphas)
        if k <= 1:
            return False
        n = len(self.y_hat)
        for r in range(k - 1):
            a_narrow = self.alphas[r]
            a_wide = self.alphas[r + 1]
            lo_narrow = self.lows_by_rate[a_narrow]
            lo_wide = self.lows_by_rate[a_wide]
            hi_narrow = self.highs_by_rate[a_narrow]
            hi_wide = self.highs_by_rate[a_wide]
            for i in range(n):
                if lo_wide[i] > lo_narrow[i] + tol or hi_wide[i] < hi_narrow[i] - tol:
                    return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "y_hat": list(self.y_hat),
            "alphas": list(self.alphas),
            "lows_by_rate": {str(a): list(self.lows_by_rate[a]) for a in self.alphas},
            "highs_by_rate": {str(a): list(self.highs_by_rate[a]) for a in self.alphas},
            "quantiles": list(self.quantiles),
            "eff_alphas": list(self.eff_alphas),
            "emp_coverage": list(self.emp_coverage),
        }


class ConformalWrapper:
    """Drop-in ``sklearn``/``MAPIE``-style multi-rate online conformal regressor wrapper.

    Wraps any predictor exposing ``.predict(X)`` or any callable ``fn(x)`` with a
    :class:`eproc.MultiRateConformal` calibrator guaranteeing zero quantile crossing
    across simultaneous coverage rates (default 50/80/90/95/99%).
    """

    __slots__ = (
        "_estimator",
        "_has_predict",
        "_batch_callable",
        "_init_alphas",
        "_window_size",
        "_pid_ki",
        "_scale_fn",
        "_calibrator",
    )

    def __init__(
        self,
        estimator: Any,
        *,
        alphas: Sequence[float] = (0.50, 0.20, 0.10, 0.05, 0.01),
        window_size: int = 128,
        pid_ki: float = 0.05,
        scale_fn: Callable[[Any], Any] | None = None,
        batch_callable: bool | None = None,
    ) -> None:
        has_predict = hasattr(estimator, "predict") and callable(estimator.predict)
        if not has_predict and not callable(estimator):
            raise TypeError(
                "estimator must implement a callable .predict(X) method or be callable"
            )
        if scale_fn is not None and not callable(scale_fn):
            raise TypeError("scale_fn must be callable or None")
        self._estimator = estimator
        self._has_predict = has_predict
        self._batch_callable = batch_callable
        self._init_alphas = tuple(float(a) for a in alphas)
        self._window_size = int(window_size)
        self._pid_ki = float(pid_ki)
        self._scale_fn = scale_fn
        self._calibrator = MultiRateConformal(
            alphas=self._init_alphas,
            window_size=self._window_size,
            pid_ki=self._pid_ki,
        )

    def _predict_raw(self, X: Any) -> list[float]:
        if self._has_predict:
            raw = self._estimator.predict(X)
            if _is_real_scalar(raw):
                return [float(raw)]
            return [float(v) for v in raw]

        if self._batch_callable is True:
            raw = self._estimator(X)
            if _is_real_scalar(raw):
                return [float(raw)]
            return [float(v) for v in raw]

        if self._batch_callable is False:
            return [float(self._estimator(row)) for row in X]

        # Auto-detect per-row callable fn(x) vs batch callable fn(X)
        if _is_real_scalar(X):
            return [float(self._estimator(X))]
        rows = list(X)
        if not rows:
            return []
        try:
            first = self._estimator(rows[0])
        except Exception:
            raw = self._estimator(rows)
            if _is_real_scalar(raw):
                return [float(raw)]
            return [float(v) for v in raw]

        if _is_real_scalar(first):
            out = [float(first)]
            for row in rows[1:]:
                out.append(float(self._estimator(row)))
            return out

        raw = self._estimator(rows)
        if _is_real_scalar(raw):
            return [float(raw)]
        return [float(v) for v in raw]

    def _resolve_scales(
        self,
        X: Any,
        n: int,
        scale: float | Sequence[float] | None,
    ) -> list[float]:
        spec = scale
        if spec is None and self._scale_fn is not None:
            spec = self._scale_fn(X)
        if spec is None:
            return [1.0] * n
        if _is_real_scalar(spec):
            s = float(spec)
            if not math.isfinite(s) or s <= 0.0:
                raise ValueError(f"scale must be finite and > 0, got {spec!r}")
            return [s] * n
        scales = [float(s) for s in spec]
        if len(scales) != n:
            raise ValueError(f"scale length ({len(scales)}) must match X length ({n})")
        for idx, s in enumerate(scales):
            if not math.isfinite(s) or s <= 0.0:
                raise ValueError(f"scale[{idx}] must be finite and > 0, got {s!r}")
        return scales

    def _compute_scores(
        self,
        X: Any,
        y: Any,
        scale: float | Sequence[float] | None,
    ) -> array.array:
        y_hat = self._predict_raw(X)
        if _is_real_scalar(y):
            y_vals = [float(y)]
        else:
            y_vals = [float(v) for v in y]
        if len(y_hat) != len(y_vals):
            raise ValueError(
                f"Prediction length ({len(y_hat)}) must match target length ({len(y_vals)})"
            )
        scales = self._resolve_scales(X, len(y_vals), scale)
        scores = array.array("d", (abs(yi - pi) / si for yi, pi, si in zip(y_vals, y_hat, scales)))
        return scores

    def fit(
        self,
        X_cal: Any,
        y_cal: Any,
        *,
        scale: float | Sequence[float] | None = None,
        reset: bool = True,
    ) -> ConformalWrapper:
        """Calibrate multi-rate conformal quantiles on ``(X_cal, y_cal)`` and return ``self``."""
        if reset:
            self._calibrator = MultiRateConformal(
                alphas=self._init_alphas,
                window_size=self._window_size,
                pid_ki=self._pid_ki,
            )
        scores = self._compute_scores(X_cal, y_cal, scale)
        if len(scores) > 0:
            self._calibrator.update_batch(scores)
        return self

    def partial_fit(
        self,
        X: Any,
        y: Any,
        *,
        scale: float | Sequence[float] | None = None,
    ) -> ConformalMeta:
        """Perform an online AgACI + Conformal PID update without resetting state."""
        scores = self._compute_scores(X, y, scale)
        return self._calibrator.update_batch(scores)

    def predict(self, X: Any) -> list[float]:
        """Return point predictions from the wrapped estimator."""
        return self._predict_raw(X)

    def predict_interval(
        self,
        X: Any,
        scale: float | Sequence[float] | None = None,
    ) -> IntervalResult:
        """Return multi-rate crossing-free intervals for ``X``.

        Returns an :class:`IntervalResult` (unpackable as ``(y_hat, lows_by_rate, highs_by_rate)``).
        """
        y_hat = self._predict_raw(X)
        n = len(y_hat)
        alphas = self._calibrator.alphas
        k = len(alphas)
        if n == 0:
            meta = self._calibrator.predict(0.0, scale=1.0)
            return IntervalResult(
                y_hat=(),
                lows_by_rate={a: () for a in alphas},
                highs_by_rate={a: () for a in alphas},
                alphas=alphas,
                quantiles=meta.quantiles,
                eff_alphas=meta.eff_alphas,
                emp_coverage=meta.emp_coverage,
            )

        scales = self._resolve_scales(X, n, scale)
        lows_cols: list[list[float]] = [[] for _ in range(k)]
        highs_cols: list[list[float]] = [[] for _ in range(k)]
        last_pred = None
        for p, s in zip(y_hat, scales):
            last_pred = self._calibrator.predict(p, scale=s)
            for r in range(k):
                lows_cols[r].append(last_pred.lows[r])
                highs_cols[r].append(last_pred.highs[r])

        assert last_pred is not None
        lows_by_rate = {alphas[r]: tuple(lows_cols[r]) for r in range(k)}
        highs_by_rate = {alphas[r]: tuple(highs_cols[r]) for r in range(k)}
        return IntervalResult(
            y_hat=tuple(y_hat),
            lows_by_rate=lows_by_rate,
            highs_by_rate=highs_by_rate,
            alphas=alphas,
            quantiles=last_pred.quantiles,
            eff_alphas=last_pred.eff_alphas,
            emp_coverage=last_pred.emp_coverage,
        )

    @property
    def estimator(self) -> Any:
        return self._estimator

    @property
    def calibrator(self) -> MultiRateConformal:
        return self._calibrator

    @property
    def alphas(self) -> tuple[float, ...]:
        return self._calibrator.alphas

    @property
    def quantiles(self) -> tuple[float, ...]:
        return self._calibrator.quantiles

    @property
    def n_updates(self) -> int:
        return self._calibrator.n_updates

    def __repr__(self) -> str:
        return (
            f"ConformalWrapper(estimator={self._estimator!r}, alphas={self.alphas}, "
            f"n_updates={self.n_updates})"
        )


def assert_eprocess_uplift(
    scores: Any,
    *,
    mu0: float,
    alpha: float = 0.05,
    lower: float = 0.0,
    upper: float = 1.0,
    side: str | int = "upper",
) -> EProcessResult:
    """Assert statistically significant uplift via an anytime-valid :class:`EProcess`.

    Intended for ``pytest`` / CI evaluation gates. Raises :class:`AssertionError`
    with diagnostic e-value, anytime p-value, sample count, and mean if ``H0`` is
    not rejected at level ``alpha``.
    """
    ep = EProcess(mu0=mu0, alpha=alpha, lower=lower, upper=upper, side=side)
    res = ep.update_batch(scores)
    if not res.rejected:
        thresh = 1.0 / ep.alpha
        raise AssertionError(
            f"eproc uplift gate FAILED (H0 not rejected at alpha={ep.alpha:g}, side={ep.side!r}): "
            f"n={res.n}, mean={res.mean:.6g}, mu0={ep.mu0:.6g}, "
            f"e_value={res.e_value:.4g} (threshold >= {thresh:.4g}), p_value={res.p_value:.6g}"
        )
    return res


def assert_no_regression(
    diffs: Any,
    *,
    alpha: float = 0.05,
    lower: float = -1.0,
    upper: float = 1.0,
    mu0: float = 0.0,
) -> EProcessResult:
    """Assert no statistically significant regression on bounded score differences.

    Runs a lower-sided :class:`EProcess` testing ``H0: E[diff] >= mu0`` against
    ``H1: E[diff] < mu0``. Raises :class:`AssertionError` with diagnostic statistics
    if a significant regression is detected.
    """
    ep = EProcess(mu0=mu0, alpha=alpha, lower=lower, upper=upper, side="lower")
    res = ep.update_batch(diffs)
    if res.rejected:
        raise AssertionError(
            f"eproc regression gate FAILED (significant negative shift detected at alpha={ep.alpha:g}): "
            f"n={res.n}, mean_diff={res.mean:.6g} < mu0={ep.mu0:.6g}, "
            f"e_value={res.e_value:.4g}, p_value={res.p_value:.6g}, rejected_at={ep.rejected_at}"
        )
    return res


@dataclass(slots=True, frozen=True)
class FleetCalibrationResult:
    """Global and exact leave-one-out peer calibration returned by :func:`fleet_calibrate`.

    Supports both attribute access and 3-tuple unpacking::

        global_monoid, global_quantiles, peer_quantiles = fleet_calibrate(shards)
    """

    global_monoid: ConformalMonoid
    global_quantiles: dict[float, float]
    peer_quantiles: tuple[dict[float, float], ...]
    peer_monoids: tuple[ConformalMonoid, ...]
    shard_counts: tuple[int, ...]
    total_count: int
    wire_size: int

    def __iter__(self) -> Iterator[Any]:
        yield self.global_monoid
        yield self.global_quantiles
        yield self.peer_quantiles

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_count": self.total_count,
            "wire_size": self.wire_size,
            "shard_counts": list(self.shard_counts),
            "global_quantiles": {str(q): v for q, v in self.global_quantiles.items()},
            "peer_quantiles": [
                {str(q): v for q, v in pq.items()} for pq in self.peer_quantiles
            ],
        }


def _shard_to_monoid(
    shard: Sequence[float] | bytes | bytearray | memoryview | ConformalMonoid,
    scale_hint: float,
) -> ConformalMonoid:
    if isinstance(shard, ConformalMonoid):
        return shard
    if isinstance(shard, (bytes, bytearray, memoryview)):
        return ConformalMonoid.from_bytes(shard)
    monoid = ConformalMonoid(scale_hint=scale_hint)
    monoid.add_batch(shard)
    return monoid


def _tree_merge_monoids(monoids: list[ConformalMonoid]) -> ConformalMonoid:
    level = list(monoids)
    while len(level) > 1:
        nxt: list[ConformalMonoid] = []
        i = 0
        n = len(level)
        while i + 1 < n:
            nxt.append(level[i].merge(level[i + 1]))
            i += 2
        if i < n:
            nxt.append(level[i])
        level = nxt
    return level[0]


def fleet_calibrate(
    shards: Sequence[Sequence[float] | bytes | bytearray | memoryview | ConformalMonoid],
    *,
    quantiles: Sequence[float] = (0.50, 0.90, 0.95),
    scale_hint: float = 1.0,
) -> FleetCalibrationResult:
    """Merge calibration shards into a 640-byte :class:`ConformalMonoid` with exact LOO peer quantiles.

    Each element of ``shards`` may be a sequence of float nonconformity scores,
    a serialized 640-byte wire payload (``bytes``/``bytearray``/``memoryview``),
    or an existing :class:`ConformalMonoid`.

    Returns a :class:`FleetCalibrationResult` (unpackable as
    ``(global_monoid, global_quantiles, peer_quantiles)``).
    """
    q_levels = tuple(float(q) for q in quantiles)
    for q in q_levels:
        if not math.isfinite(q) or q < 0.0 or q > 1.0:
            raise ValueError(f"Each quantile must be finite in [0, 1], got {q!r}")

    shard_monoids = [_shard_to_monoid(sh, scale_hint) for sh in shards]
    if not shard_monoids:
        raise ValueError("shards must contain at least one shard")
    global_monoid = _tree_merge_monoids(shard_monoids)
    global_q = {q: global_monoid.quantile(q) for q in q_levels}

    peer_monoids_list: list[ConformalMonoid] = []
    peer_q_list: list[dict[float, float]] = []
    for local in shard_monoids:
        peer_m = global_monoid.subtract(local)
        peer_monoids_list.append(peer_m)
        peer_q_list.append({q: peer_m.quantile(q) for q in q_levels})

    return FleetCalibrationResult(
        global_monoid=global_monoid,
        global_quantiles=global_q,
        peer_quantiles=tuple(peer_q_list),
        peer_monoids=tuple(peer_monoids_list),
        shard_counts=tuple(m.count for m in shard_monoids),
        total_count=global_monoid.count,
        wire_size=global_monoid.wire_size,
    )
