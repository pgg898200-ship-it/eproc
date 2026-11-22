"""Byzantine-safe multi-agent prediction fusion (BOA + Ville gates) and provenance-deduplicated EchoPool."""

from __future__ import annotations

from ctypes import byref, c_double, c_int32, c_uint32, c_uint64
from dataclasses import asdict, dataclass
import hashlib
from typing import Any

from ._ffi import (
    _check_len,
    check_rc,
    eproc_boa_step_t,
    eproc_boa_t,
    eproc_echo_out_t,
    eproc_echopool_t,
    get_lib,
)


@dataclass(slots=True, frozen=True)
class SafeFuserStep:
    """Per-step diagnostics returned by :meth:`SafeFuser.update`."""

    fused_pred: float
    anchor_pred: float
    fallback_active: bool
    n_quarantined: int
    sup_log_e: float
    deg_log_e: float
    n_updates: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class EchoSubmitResult:
    """Deduplication and posterior update returned by :meth:`EchoPool.submit`."""

    accepted: bool
    posterior_prob: float
    log_odds: float
    source_reliability: float
    unique_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_evidence_id(evidence_id: int | str | bytes) -> int:
    if isinstance(evidence_id, bool):
        raise TypeError("evidence_id must be an int, str, or bytes, not bool")
    if isinstance(evidence_id, int):
        if evidence_id < 0 or evidence_id >= (1 << 64):
            raise ValueError(f"evidence_id integer out of uint64 range: {evidence_id}")
        return evidence_id
    if isinstance(evidence_id, str):
        raw = evidence_id.encode("utf-8")
    elif isinstance(evidence_id, (bytes, bytearray)):
        raw = bytes(evidence_id)
    else:
        raise TypeError(
            f"evidence_id must be int, str, or bytes, got {type(evidence_id).__name__}"
        )
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


class SafeFuser:
    """Scale-free Bernstein Online Aggregation (BOA) with 3-gate Ville e-process fallback.

    Protects multi-agent ensembles from poisoned or degraded experts via
    loss clipping, strike-based quarantine, non-inferiority fallback to a
    designated anchor expert, and lock-free Boehm-2012 seqlocked weight reads.
    """

    __slots__ = ("_lib", "_state", "_step_out", "_n_experts")

    def __init__(
        self,
        n_experts: int,
        anchor_idx: int = 0,
        alpha_deg: float = 0.05,
    ) -> None:
        self._lib = get_lib()
        self._state = eproc_boa_t()
        self._step_out = eproc_boa_step_t()
        n = int(n_experts)
        if n <= 0:
            raise ValueError(f"n_experts must be positive, got {n}")
        rc = self._lib.eproc_boa_init(
            byref(self._state),
            c_uint32(n),
            c_uint32(int(anchor_idx)),
            c_double(float(alpha_deg)),
        )
        check_rc(rc, "SafeFuser.__init__")
        self._n_experts = int(self._state.n_experts)

    def predict(self, preds: Any) -> float:
        """Fuse a vector of ``n_experts`` predictions without mutating state."""
        ptr, _n, _owner = _check_len(preds, self._n_experts, "SafeFuser.predict")
        return float(self._lib.eproc_boa_predict(byref(self._state), ptr))

    def update(self, preds: Any, y_true: float) -> SafeFuserStep:
        """Observe expert predictions and ground truth ``y_true``, updating BOA & Ville gates."""
        ptr, _n, _owner = _check_len(preds, self._n_experts, "SafeFuser.update")
        rc = self._lib.eproc_boa_update(
            byref(self._state),
            ptr,
            c_double(float(y_true)),
            byref(self._step_out),
        )
        check_rc(rc, "SafeFuser.update")
        return SafeFuserStep(
            fused_pred=float(self._step_out.fused_pred),
            anchor_pred=float(self._step_out.anchor_pred),
            fallback_active=bool(self._step_out.fallback_active),
            n_quarantined=int(self._step_out.n_quarantined),
            sup_log_e=float(self._step_out.sup_log_e),
            deg_log_e=float(self._step_out.deg_log_e),
            n_updates=int(self._step_out.n_updates),
        )

    def snapshot(self) -> tuple[list[float], bool]:
        """Read a lock-free Boehm-2012 seqlocked snapshot of ``(weights, fallback_active)``."""
        buf = (c_double * self._n_experts)()
        fb = c_int32(0)
        rc = self._lib.eproc_boa_snapshot_weights(
            byref(self._state),
            buf,
            c_uint32(self._n_experts),
            byref(fb),
        )
        check_rc(rc, "SafeFuser.snapshot")
        return [float(buf[i]) for i in range(self._n_experts)], bool(fb.value)

    def weights(self) -> list[float]:
        """Return the active expert weight vector via a lock-free seqlock snapshot."""
        w, _fb = self.snapshot()
        return w

    @property
    def n_experts(self) -> int:
        return self._n_experts

    @property
    def anchor_idx(self) -> int:
        return int(self._state.anchor_idx)

    @property
    def fallback_active(self) -> bool:
        _w, fb = self.snapshot()
        return fb

    @property
    def n_updates(self) -> int:
        return int(self._state.n_updates)

    @property
    def n_poison_clipped(self) -> int:
        return int(self._state.n_poison_clipped)

    @property
    def n_fallback_trips(self) -> int:
        return int(self._state.n_fallback_trips)

    @property
    def quarantined(self) -> list[bool]:
        return [bool(self._state.quarantined[i]) for i in range(self._n_experts)]

    def __repr__(self) -> str:
        return (
            f"SafeFuser(n_experts={self._n_experts}, anchor_idx={self.anchor_idx}, "
            f"n_updates={self.n_updates}, fallback_active={self.fallback_active})"
        )


class EchoPool:
    """Provenance-deduplicated multi-agent evidence pool with Beta-Bernoulli peer reliability.

    Each unique ``(claim_id, evidence_id)`` pair is accumulated at most once
    (eliminating echo-chamber double-counting and overwrite churn), while
    :meth:`resolve` tracks empirical accuracy per source to down-weight noisy
    or Byzantine agents.
    """

    __slots__ = ("_lib", "_state", "_out", "_n_sources", "_n_claims")

    def __init__(
        self,
        n_sources: int,
        n_claims: int = 64,
        prior_prob: float = 0.5,
    ) -> None:
        self._lib = get_lib()
        self._state = eproc_echopool_t()
        self._out = eproc_echo_out_t()
        rc = self._lib.eproc_echopool_init(
            byref(self._state),
            c_uint32(int(n_sources)),
            c_uint32(int(n_claims)),
            c_double(float(prior_prob)),
        )
        check_rc(rc, "EchoPool.__init__")
        self._n_sources = int(self._state.n_sources)
        self._n_claims = int(self._state.n_claims)

    def submit(
        self,
        source_id: int,
        claim_id: int,
        evidence_id: int | str | bytes,
        llr: float,
    ) -> EchoSubmitResult:
        """Submit log-likelihood ratio evidence for ``claim_id`` from ``source_id``.

        Duplicate ``(claim_id, evidence_id)`` echoes are dropped idempotently
        (``accepted=False``) without altering the posterior. String or bytes
        ``evidence_id`` values are deterministically hashed via BLAKE2b-64.
        """
        src = int(source_id)
        clm = int(claim_id)
        if src < 0 or src >= self._n_sources:
            raise ValueError(f"source_id {src} out of range [0, {self._n_sources})")
        if clm < 0 or clm >= self._n_claims:
            raise ValueError(f"claim_id {clm} out of range [0, {self._n_claims})")
        ev_u64 = _normalize_evidence_id(evidence_id)
        rc = self._lib.eproc_echopool_submit(
            byref(self._state),
            c_uint32(src),
            c_uint32(clm),
            c_uint64(ev_u64),
            c_double(float(llr)),
            byref(self._out),
        )
        check_rc(rc, "EchoPool.submit")
        return EchoSubmitResult(
            accepted=bool(self._out.accepted),
            posterior_prob=float(self._out.posterior_prob),
            log_odds=float(self._out.log_odds),
            source_reliability=float(self._out.source_reliability),
            unique_count=int(self._out.unique_count),
        )

    def resolve(self, claim_id: int, outcome_positive: bool) -> None:
        """Resolve a claim's ground truth and update Beta-Bernoulli source reliabilities."""
        clm = int(claim_id)
        if clm < 0 or clm >= self._n_claims:
            raise ValueError(f"claim_id {clm} out of range [0, {self._n_claims})")
        rc = self._lib.eproc_echopool_resolve(
            byref(self._state), c_uint32(clm), c_int32(1 if outcome_positive else 0)
        )
        check_rc(rc, "EchoPool.resolve")

    def posterior(self, claim_id: int) -> float:
        """Return the current posterior probability for ``claim_id``."""
        clm = int(claim_id)
        if clm < 0 or clm >= self._n_claims:
            raise ValueError(f"claim_id {clm} out of range [0, {self._n_claims})")
        return float(self._lib.eproc_echopool_posterior(byref(self._state), c_uint32(clm)))

    def source_reliability(self, source_id: int) -> float:
        """Return the estimated reliability weight in ``[0, 1]`` for ``source_id``."""
        src = int(source_id)
        if src < 0 or src >= self._n_sources:
            raise ValueError(f"source_id {src} out of range [0, {self._n_sources})")
        return float(
            self._lib.eproc_echopool_source_reliability(byref(self._state), c_uint32(src))
        )

    @property
    def n_sources(self) -> int:
        return self._n_sources

    @property
    def n_claims(self) -> int:
        return self._n_claims

    @property
    def n_accepted(self) -> int:
        return int(self._state.n_accepted)

    @property
    def n_echo_dropped(self) -> int:
        return int(self._state.n_echo_dropped)

    @property
    def n_table_full(self) -> int:
        return int(self._state.n_table_full)

    def __repr__(self) -> str:
        return (
            f"EchoPool(n_sources={self._n_sources}, n_claims={self._n_claims}, "
            f"accepted={self.n_accepted}, echo_dropped={self.n_echo_dropped})"
        )


import os as _os  # noqa: E402

if not _os.environ.get("EPROC_DISABLE_NATIVE"):
    try:
        from . import _native as _nat  # noqa: E402
    except ImportError:
        _nat = None
    if _nat is not None:
        SafeFuser = _nat.SafeFuser  # type: ignore[misc,assignment]
        SafeFuserStep = _nat.SafeFuserStep  # type: ignore[misc,assignment]
        EchoPool = _nat.EchoPool  # type: ignore[misc,assignment]
        EchoSubmitResult = _nat.EchoSubmitResult  # type: ignore[misc,assignment]

