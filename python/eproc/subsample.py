"""Causal online subsampling (VarOpt, Online RLS) and hierarchical multi-resolution VOI selection."""

from __future__ import annotations

from ctypes import byref, c_double, c_uint32, c_uint64
from dataclasses import asdict, dataclass
from typing import Any

from ._ffi import (
    _check_len,
    check_rc,
    eproc_rls_out_t,
    eproc_rls_t,
    eproc_sift_action_t,
    eproc_sift_t,
    eproc_varopt_t,
    get_lib,
)

_SIFT_ACTION_NAMES: dict[int, str] = {
    0: "site",
    1: "channel",
    2: "ticks",
}


@dataclass(slots=True, frozen=True)
class VarOptItem:
    """Single retained item inside a :class:`VarOptSampler` reservoir."""

    item_id: int
    orig_weight: float
    adj_weight: float
    value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class RLSStepResult:
    """Per-item inclusion decision and Horvitz-Thompson weight from :class:`OnlineRLSSampler`."""

    keep: bool
    leverage: float
    prob: float
    ht_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class SiftAction:
    """Optimal next sensing action chosen by :class:`SiftVOISelector`."""

    action_type: int
    action_name: str
    target_idx: int
    evoi_per_cost: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VarOptSampler:
    """Variance-Optimal weighted reservoir sampler (Cohen et al. 2011).

    Maintains at most ``k`` items with exact Horvitz-Thompson adjusted weights,
    guaranteeing unbiased subset sum and weighted mean estimates.
    """

    __slots__ = ("_lib", "_state")

    def __init__(self, k: int, seed: int = 42) -> None:
        self._lib = get_lib()
        self._state = eproc_varopt_t()
        rc = self._lib.eproc_varopt_init(
            byref(self._state), c_uint32(int(k)), c_uint64(int(seed) & ((1 << 64) - 1))
        )
        check_rc(rc, "VarOptSampler.__init__")

    def push(self, item_id: int, weight: float, value: float) -> None:
        """Stream one weighted item ``(item_id, weight, value)`` into the reservoir."""
        if int(item_id) < 0:
            raise ValueError(f"item_id must be non-negative, got {item_id}")
        rc = self._lib.eproc_varopt_push(
            byref(self._state),
            c_uint64(int(item_id)),
            c_double(float(weight)),
            c_double(float(value)),
        )
        check_rc(rc, "VarOptSampler.push")

    def estimate_sum(self) -> float:
        """Return the unbiased Horvitz-Thompson estimate of ``sum_i w_i * v_i``."""
        return float(self._lib.eproc_varopt_estimate_sum(byref(self._state)))

    def estimate_mean(self) -> float:
        """Return the Horvitz-Thompson estimate of the weighted mean."""
        return float(self._lib.eproc_varopt_estimate_mean(byref(self._state)))

    def items(self) -> list[VarOptItem]:
        """Return all items currently held in the reservoir."""
        n = int(self._state.count)
        return [
            VarOptItem(
                item_id=int(self._state.items[i].item_id),
                orig_weight=float(self._state.items[i].orig_weight),
                adj_weight=float(self._state.items[i].adj_weight),
                value=float(self._state.items[i].value),
            )
            for i in range(n)
        ]

    @property
    def k(self) -> int:
        return int(self._state.k)

    @property
    def count(self) -> int:
        return int(self._state.count)

    @property
    def n_heavy(self) -> int:
        return int(self._state.n_heavy)

    @property
    def tau(self) -> float:
        return float(self._state.tau)

    @property
    def total_weight(self) -> float:
        return float(self._state.total_weight + self._state.total_weight_comp)

    @property
    def n_seen(self) -> int:
        return int(self._state.n_seen)

    def __repr__(self) -> str:
        return (
            f"VarOptSampler(k={self.k}, count={self.count}, n_seen={self.n_seen}, "
            f"total_weight={self.total_weight:.4g})"
        )


class OnlineRLSSampler:
    """Online Ridge Leverage Score (RLS) causal subsampler.

    Computes ``F_{t-1}``-predictable ridge leverage scores via Sherman-Morrison
    rank-1 inverse updates and returns Horvitz-Thompson weights ``1/p`` on kept
    feature vectors.
    """

    __slots__ = ("_lib", "_state", "_out", "_d")

    def __init__(
        self,
        d: int,
        lambda_reg: float = 1.0,
        oversample: float = 2.0,
        min_prob: float = 0.02,
        seed: int = 42,
    ) -> None:
        self._lib = get_lib()
        self._state = eproc_rls_t()
        self._out = eproc_rls_out_t()
        rc = self._lib.eproc_rls_init(
            byref(self._state),
            c_uint32(int(d)),
            c_double(float(lambda_reg)),
            c_double(float(oversample)),
            c_double(float(min_prob)),
            c_uint64(int(seed) & ((1 << 64) - 1)),
        )
        check_rc(rc, "OnlineRLSSampler.__init__")
        self._d = int(self._state.d)

    def step(self, x: Any) -> RLSStepResult:
        """Score one ``d``-dimensional feature vector and sample causally."""
        ptr, _n, _owner = _check_len(x, self._d, "OnlineRLSSampler.step")
        rc = self._lib.eproc_rls_step(byref(self._state), ptr, byref(self._out))
        check_rc(rc, "OnlineRLSSampler.step")
        return RLSStepResult(
            keep=bool(self._out.keep),
            leverage=float(self._out.leverage),
            prob=float(self._out.prob),
            ht_weight=float(self._out.ht_weight),
        )

    @property
    def d(self) -> int:
        return self._d

    @property
    def n_seen(self) -> int:
        return int(self._state.n_seen)

    @property
    def n_kept(self) -> int:
        return int(self._state.n_kept)

    @property
    def keep_rate(self) -> float:
        seen = self.n_seen
        return (self.n_kept / seen) if seen > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"OnlineRLSSampler(d={self._d}, n_seen={self.n_seen}, "
            f"n_kept={self.n_kept})"
        )


class SiftVOISelector:
    """Hierarchical multi-resolution Bayes Value-of-Information (VOI) selector.

    Selects whether to query a coarse site summary, a channel summary, or raw
    ticks across ``n_sites * per_site`` channels to maximize expected value of
    information per unit tool/attention cost.
    """

    __slots__ = ("_lib", "_state", "_action_out", "_n_sites", "_per_site")

    def __init__(
        self,
        n_sites: int,
        per_site: int = 8,
        signal_mean: float = 1.5,
        noise_sd: float = 1.0,
    ) -> None:
        self._lib = get_lib()
        self._state = eproc_sift_t()
        self._action_out = eproc_sift_action_t()
        rc = self._lib.eproc_sift_init(
            byref(self._state),
            c_uint32(int(n_sites)),
            c_uint32(int(per_site)),
            c_double(float(signal_mean)),
            c_double(float(noise_sd)),
        )
        check_rc(rc, "SiftVOISelector.__init__")
        self._n_sites = int(self._state.n_sites)
        self._per_site = int(self._state.per_site)

    def set_noise_scale(self, ch: int, scale: float) -> None:
        """Set a per-channel noise standard deviation multiplier."""
        idx = int(ch)
        if idx < 0 or idx >= self.n_channels:
            raise ValueError(f"channel index {idx} out of range [0, {self.n_channels})")
        rc = self._lib.eproc_sift_set_noise_scale(
            byref(self._state), c_uint32(idx), c_double(float(scale))
        )
        check_rc(rc, "SiftVOISelector.set_noise_scale")

    def observe_channel(self, ch: int, obs_mean: float, n_ticks: int = 1) -> None:
        """Update posterior belief for channel ``ch`` given an observed mean over ``n_ticks``."""
        idx = int(ch)
        if idx < 0 or idx >= self.n_channels:
            raise ValueError(f"channel index {idx} out of range [0, {self.n_channels})")
        rc = self._lib.eproc_sift_observe_channel(
            byref(self._state),
            c_uint32(idx),
            c_double(float(obs_mean)),
            c_uint32(int(n_ticks)),
        )
        check_rc(rc, "SiftVOISelector.observe_channel")

    def observe_site(self, site: int, site_mean: float) -> None:
        """Update posterior belief for site ``site`` given a coarse site mean."""
        idx = int(site)
        if idx < 0 or idx >= self._n_sites:
            raise ValueError(f"site index {idx} out of range [0, {self._n_sites})")
        rc = self._lib.eproc_sift_observe_site(
            byref(self._state), c_uint32(idx), c_double(float(site_mean))
        )
        check_rc(rc, "SiftVOISelector.observe_site")

    def select_next(
        self,
        cost_site: float = 1.0,
        cost_channel: float = 1.0,
        cost_ticks: float = 8.0,
        ticks_per_read: int = 8,
    ) -> SiftAction:
        """Choose the next sensing action maximizing EVOI per unit cost."""
        rc = self._lib.eproc_sift_select_next(
            byref(self._state),
            c_double(float(cost_site)),
            c_double(float(cost_channel)),
            c_double(float(cost_ticks)),
            c_uint32(int(ticks_per_read)),
            byref(self._action_out),
        )
        check_rc(rc, "SiftVOISelector.select_next")
        atype = int(self._action_out.action_type)
        return SiftAction(
            action_type=atype,
            action_name=_SIFT_ACTION_NAMES.get(atype, f"unknown({atype})"),
            target_idx=int(self._action_out.target_idx),
            evoi_per_cost=float(self._action_out.evoi_per_cost),
        )

    @property
    def n_sites(self) -> int:
        return self._n_sites

    @property
    def per_site(self) -> int:
        return self._per_site

    @property
    def n_channels(self) -> int:
        return self._n_sites * self._per_site

    def __repr__(self) -> str:
        return f"SiftVOISelector(n_sites={self._n_sites}, per_site={self._per_site})"


import os as _os  # noqa: E402

if not _os.environ.get("EPROC_DISABLE_NATIVE"):
    try:
        from . import _native as _nat  # noqa: E402
    except ImportError:
        _nat = None
    if _nat is not None:
        VarOptSampler = _nat.VarOptSampler  # type: ignore[misc,assignment]
        VarOptItem = _nat.VarOptItem  # type: ignore[misc,assignment]
        OnlineRLSSampler = _nat.OnlineRLSSampler  # type: ignore[misc,assignment]
        RLSStepResult = _nat.RLSStepResult  # type: ignore[misc,assignment]
        SiftVOISelector = _nat.SiftVOISelector  # type: ignore[misc,assignment]
        SiftAction = _nat.SiftAction  # type: ignore[misc,assignment]

