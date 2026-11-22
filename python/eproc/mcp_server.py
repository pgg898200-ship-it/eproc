"""Zero-dependency JSON-RPC 2.0 MCP stdio server for ``eproc``.

Supports both MCP ``2025-11-25`` and ``2026-07-28`` handshakes and discovery
(``initialize``, ``server/discover``, ``notifications/initialized``, ``ping``,
``tools/list`` with ``ttlMs`` / ``cacheScope``, and ``tools/call``), including
multi-turn stateful sessions via optional ``session_id``.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
import json
import math
import sys
from typing import Any, TextIO

from . import __version__

SUPPORTED_PROTOCOL_VERSIONS: tuple[str, ...] = (
    "2026-07-28",
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
DEFAULT_PROTOCOL_VERSION = "2026-07-28"
TOOL_LIST_TTL_MS = 300_000
TOOL_LIST_CACHE_SCOPE = "public"

# Standard JSON-RPC 2.0 error codes
ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603


def _sanitize_json(value: Any) -> Any:
    """Recursively replace non-finite floats (NaN/Inf) with ``None`` for strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(v) for v in value]
    return value


def _dumps(obj: Any) -> str:
    return json.dumps(_sanitize_json(obj), allow_nan=False, separators=(",", ":"))


def _require_number(val: Any, name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"Parameter {name!r} must be a number")
    return float(val)


def _require_finite_float(val: Any, name: str) -> float:
    out = _require_number(val, name)
    if not math.isfinite(out):
        raise ValueError(f"Parameter {name!r} must be finite, got {val!r}")
    return out


def _require_bool(args: dict[str, Any], name: str, default: bool = False) -> bool:
    if name not in args or args[name] is None:
        return default
    val = args[name]
    if not isinstance(val, bool):
        raise ValueError(f"Parameter {name!r} must be a boolean")
    return val


def _as_float_list(val: Any, name: str) -> list[float]:
    if not isinstance(val, list):
        raise ValueError(f"Parameter {name!r} must be a list of numbers")
    out: list[float] = []
    for idx, item in enumerate(val):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"Parameter {name!r}[{idx}] must be numeric")
        out.append(float(item))
    return out


def _optional_session_id(args: dict[str, Any]) -> str | None:
    sid = args.get("session_id")
    if sid is None:
        return None
    if not isinstance(sid, str) or not sid.strip():
        raise ValueError("Parameter 'session_id' must be a non-empty string")
    return sid


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "eproc_etest",
        "description": (
            "Anytime-valid ONS betting e-process hypothesis test for bounded streams "
            "X_t in [lower, upper]. Valid under continuous peeking at every observation "
            "(Ville inequality). Pass optional session_id to stream across turns."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Observations to ingest (may be empty to query an existing session).",
                },
                "mu0": {
                    "type": "number",
                    "description": "Null hypothesis mean threshold (required for new sessions / stateless calls).",
                },
                "alpha": {"type": "number", "default": 0.05, "description": "Significance level in (0, 1)."},
                "lower": {"type": "number", "default": 0.0, "description": "Lower bound of observations."},
                "upper": {"type": "number", "default": 1.0, "description": "Upper bound of observations."},
                "side": {
                    "type": "string",
                    "enum": ["upper", "lower", "two_sided"],
                    "default": "upper",
                    "description": "Alternative hypothesis direction.",
                },
                "session_id": {"type": "string", "description": "Optional handle for stateful multi-turn streaming."},
                "reset": {"type": "boolean", "default": False, "description": "Reset existing session before updating."},
            },
        },
        "annotations": {
            "readOnlyHint": False,
            "idempotentHint": False,
            "ttlMs": TOOL_LIST_TTL_MS,
            "cacheScope": TOOL_LIST_CACHE_SCOPE,
        },
    },
    {
        "name": "eproc_confidence_sequence",
        "description": (
            "Anytime-valid O(1) Empirical-Bernstein or Normal-Mixture confidence sequence "
            "with running-intersection tightening. Pass optional session_id for multi-turn updates."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Observations to ingest.",
                },
                "alpha": {"type": "number", "default": 0.05},
                "lower": {"type": "number", "default": 0.0},
                "upper": {"type": "number", "default": 1.0},
                "mode": {"type": "string", "enum": ["eb", "normal"], "default": "eb"},
                "sigma": {"type": "number", "default": 1.0},
                "v_opt": {"type": "number", "default": 100.0},
                "session_id": {"type": "string"},
                "reset": {"type": "boolean", "default": False},
            },
        },
        "annotations": {
            "readOnlyHint": False,
            "idempotentHint": False,
            "ttlMs": TOOL_LIST_TTL_MS,
            "cacheScope": TOOL_LIST_CACHE_SCOPE,
        },
    },
    {
        "name": "eproc_race",
        "description": (
            "Anytime-valid multi-arm agent/model racer with automatic early elimination "
            "and FWER control. Suggests the next arm to sample and certifies the winner."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "n_arms": {"type": "integer", "minimum": 2, "maximum": 64},
                "names": {"type": "array", "items": {"type": "string"}},
                "alpha": {"type": "number", "default": 0.05},
                "lower": {"type": "number", "default": 0.0},
                "upper": {"type": "number", "default": 1.0},
                "margin": {"type": "number", "default": 0.0},
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "arm": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
                            "reward": {"type": "number"},
                        },
                        "required": ["arm", "reward"],
                    },
                    "description": "List of {arm, reward} observations.",
                },
                "arms": {"type": "array", "items": {"oneOf": [{"type": "integer"}, {"type": "string"}]}},
                "rewards": {"type": "array", "items": {"type": "number"}},
                "session_id": {"type": "string"},
                "reset": {"type": "boolean", "default": False},
            },
        },
        "annotations": {
            "readOnlyHint": False,
            "idempotentHint": False,
            "ttlMs": TOOL_LIST_TTL_MS,
            "cacheScope": TOOL_LIST_CACHE_SCOPE,
        },
    },
    {
        "name": "eproc_conformal",
        "description": (
            "Multi-rate crossing-free AgACI + Conformal PID online interval calibrator "
            "for up to 5 simultaneous coverage levels."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "alphas": {
                    "type": "array",
                    "items": {"type": "number"},
                    "default": [0.50, 0.20, 0.10, 0.05, 0.01],
                },
                "window_size": {"type": "integer", "default": 128, "minimum": 1, "maximum": 256},
                "pid_ki": {"type": "number", "default": 0.05},
                "scores": {"type": "array", "items": {"type": "number"}},
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pred": {"type": "number"},
                            "y": {"type": "number"},
                            "scale": {"type": "number", "default": 1.0},
                        },
                        "required": ["pred", "y"],
                    },
                },
                "predict": {
                    "type": "object",
                    "properties": {
                        "pred": {"type": "number"},
                        "scale": {"type": "number", "default": 1.0},
                    },
                    "required": ["pred"],
                },
                "session_id": {"type": "string"},
                "reset": {"type": "boolean", "default": False},
            },
        },
        "annotations": {
            "readOnlyHint": False,
            "idempotentHint": False,
            "ttlMs": TOOL_LIST_TTL_MS,
            "cacheScope": TOOL_LIST_CACHE_SCOPE,
        },
    },
    {
        "name": "eproc_cmonoid_merge",
        "description": (
            "Bit-exact associative and commutative distributed 640-byte conformal monoid "
            "merge and leave-one-out subtraction across score shards or base64 wire payloads."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "shards": {
                    "type": "array",
                    "description": "List of shards, each either an array of score numbers or a base64 640-byte monoid.",
                    "items": {
                        "oneOf": [
                            {"type": "array", "items": {"type": "number"}},
                            {"type": "string"},
                        ]
                    },
                },
                "subtract_shard": {
                    "oneOf": [
                        {"type": "array", "items": {"type": "number"}},
                        {"type": "string"},
                    ],
                    "description": "Optional local shard to subtract for exact leave-one-out calibration.",
                },
                "quantiles": {
                    "type": "array",
                    "items": {"type": "number"},
                    "default": [0.50, 0.90, 0.95],
                },
                "scale_hint": {"type": "number", "default": 1.0},
                "include_wire_b64": {"type": "boolean", "default": True},
            },
            "required": ["shards"],
        },
        "annotations": {
            "readOnlyHint": True,
            "idempotentHint": True,
            "ttlMs": TOOL_LIST_TTL_MS,
            "cacheScope": TOOL_LIST_CACHE_SCOPE,
        },
    },
    {
        "name": "eproc_safe_fuse",
        "description": (
            "Byzantine-safe Bernstein Online Aggregation (BOA) with 3-gate Ville e-process "
            "non-inferiority fallback, poisoning isolation, and lock-free seqlocked weights."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "n_experts": {"type": "integer", "minimum": 1, "maximum": 64},
                "anchor_idx": {"type": "integer", "default": 0, "minimum": 0},
                "alpha_deg": {"type": "number", "default": 0.05},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "preds": {"type": "array", "items": {"type": "number"}},
                            "y": {"type": "number"},
                        },
                        "required": ["preds", "y"],
                    },
                },
                "predict": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional expert prediction vector to fuse after updates.",
                },
                "session_id": {"type": "string"},
                "reset": {"type": "boolean", "default": False},
            },
        },
        "annotations": {
            "readOnlyHint": False,
            "idempotentHint": False,
            "ttlMs": TOOL_LIST_TTL_MS,
            "cacheScope": TOOL_LIST_CACHE_SCOPE,
        },
    },
    {
        "name": "eproc_echopool",
        "description": (
            "Provenance-deduplicated multi-agent evidence pool with Beta-Bernoulli peer "
            "reliability tracking. Eliminates echo double-counting and overwrite churn."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "n_sources": {"type": "integer", "default": 16, "minimum": 1, "maximum": 256},
                "n_claims": {"type": "integer", "default": 64, "minimum": 1, "maximum": 256},
                "prior_prob": {"type": "number", "default": 0.5},
                "submissions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "integer"},
                            "claim_id": {"type": "integer"},
                            "evidence_id": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
                            "llr": {"type": "number"},
                        },
                        "required": ["source_id", "claim_id", "evidence_id", "llr"],
                    },
                },
                "resolutions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_id": {"type": "integer"},
                            "outcome_positive": {"type": "boolean"},
                        },
                        "required": ["claim_id", "outcome_positive"],
                    },
                },
                "query_claims": {"type": "array", "items": {"type": "integer"}},
                "query_sources": {"type": "array", "items": {"type": "integer"}},
                "session_id": {"type": "string"},
                "reset": {"type": "boolean", "default": False},
            },
        },
        "annotations": {
            "readOnlyHint": False,
            "idempotentHint": False,
            "ttlMs": TOOL_LIST_TTL_MS,
            "cacheScope": TOOL_LIST_CACHE_SCOPE,
        },
    },
]

_TOOL_NAMES = frozenset(tool["name"] for tool in TOOL_DEFINITIONS)


class McpServer:
    """Stateful JSON-RPC 2.0 MCP server exposing ``eproc`` statistical engines."""

    def __init__(self, max_sessions: int = 1024) -> None:
        self._max_sessions = max(1, int(max_sessions))
        self._sessions: OrderedDict[str, tuple[str, Any]] = OrderedDict()
        self._protocol_version: str = DEFAULT_PROTOCOL_VERSION

    # ------------------------------------------------------------------
    # Session Management (LRU bounded)
    # ------------------------------------------------------------------

    def _get_session(self, sid: str, kind: str) -> Any | None:
        entry = self._sessions.get(sid)
        if entry is None:
            return None
        stored_kind, obj = entry
        if stored_kind != kind:
            raise ValueError(
                f"Session {sid!r} belongs to tool {stored_kind!r}, not {kind!r} "
                f"(pass reset=true to replace it)"
            )
        self._sessions.move_to_end(sid)
        return obj

    def _put_session(self, sid: str, kind: str, obj: Any) -> None:
        self._sessions[sid] = (kind, obj)
        self._sessions.move_to_end(sid)
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)

    # ------------------------------------------------------------------
    # Tool Implementations
    # ------------------------------------------------------------------

    def _tool_etest(self, args: dict[str, Any]) -> dict[str, Any]:
        from .eprocess import EProcess

        sid = _optional_session_id(args)
        reset = _require_bool(args, "reset", False)
        ep: EProcess | None = None
        if sid is not None and not reset:
            ep = self._get_session(sid, "eproc_etest")

        if ep is None:
            if "mu0" not in args:
                raise ValueError("Missing required parameter 'mu0'")
            mu0 = _require_finite_float(args["mu0"], "mu0")
            alpha = _require_finite_float(args.get("alpha", 0.05), "alpha")
            lower = _require_finite_float(args.get("lower", 0.0), "lower")
            upper = _require_finite_float(args.get("upper", 1.0), "upper")
            side = args.get("side", "upper")
            ep = EProcess(mu0=mu0, alpha=alpha, lower=lower, upper=upper, side=side)
            if sid is not None:
                self._put_session(sid, "eproc_etest", ep)

        values = args.get("values")
        if values is not None:
            xs = _as_float_list(values, "values")
            res = ep.update_batch(xs)
        else:
            res = ep.result()

        verdict = "REJECT" if res.rejected else "CONTINUE"
        payload = res.to_dict()
        payload.update(
            {
                "verdict": verdict,
                "rejected_at": ep.rejected_at,
                "n_invalid": ep.n_invalid,
                "mu0": ep.mu0,
                "alpha": ep.alpha,
                "side": ep.side,
            }
        )
        if sid is not None:
            payload["session_id"] = sid
        return payload

    def _tool_confidence_sequence(self, args: dict[str, Any]) -> dict[str, Any]:
        from .eprocess import ConfidenceSequence

        sid = _optional_session_id(args)
        reset = _require_bool(args, "reset", False)
        cs: ConfidenceSequence | None = None
        if sid is not None and not reset:
            cs = self._get_session(sid, "eproc_confidence_sequence")

        if cs is None:
            alpha = _require_finite_float(args.get("alpha", 0.05), "alpha")
            lower = _require_finite_float(args.get("lower", 0.0), "lower")
            upper = _require_finite_float(args.get("upper", 1.0), "upper")
            mode = str(args.get("mode", "eb"))
            sigma = _require_finite_float(args.get("sigma", 1.0), "sigma")
            v_opt = _require_finite_float(args.get("v_opt", 100.0), "v_opt")
            cs = ConfidenceSequence(
                alpha=alpha,
                lower=lower,
                upper=upper,
                mode=mode,
                sigma=sigma,
                v_opt=v_opt,
            )
            if sid is not None:
                self._put_session(sid, "eproc_confidence_sequence", cs)

        values = args.get("values")
        if values is not None:
            xs = _as_float_list(values, "values")
            res = cs.update_batch(xs)
        else:
            res = cs.query()

        payload = res.to_dict()
        payload.update({"alpha": cs.alpha, "mode": cs.mode})
        if sid is not None:
            payload["session_id"] = sid
        return payload

    def _tool_race(self, args: dict[str, Any]) -> dict[str, Any]:
        from .eprocess import AgentRace

        sid = _optional_session_id(args)
        reset = _require_bool(args, "reset", False)
        race: AgentRace | None = None
        if sid is not None and not reset:
            race = self._get_session(sid, "eproc_race")

        if race is None:
            names = args.get("names")
            if names is not None:
                if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
                    raise ValueError("Parameter 'names' must be a list of strings")
                raw_n = args.get("n_arms", len(names))
                if isinstance(raw_n, bool) or not isinstance(raw_n, int):
                    raise ValueError("Parameter 'n_arms' must be an integer")
                n_arms = int(raw_n)
            elif "n_arms" in args:
                if isinstance(args["n_arms"], bool) or not isinstance(args["n_arms"], int):
                    raise ValueError("Parameter 'n_arms' must be an integer")
                n_arms = int(args["n_arms"])
            else:
                raise ValueError("Missing required parameter 'n_arms' (or 'names')")

            alpha = _require_finite_float(args.get("alpha", 0.05), "alpha")
            lower = _require_finite_float(args.get("lower", 0.0), "lower")
            upper = _require_finite_float(args.get("upper", 1.0), "upper")
            margin = _require_finite_float(args.get("margin", 0.0), "margin")
            race = AgentRace(
                n_arms=n_arms,
                alpha=alpha,
                lower=lower,
                upper=upper,
                margin=margin,
                names=names,
            )
            if sid is not None:
                self._put_session(sid, "eproc_race", race)

        observations = args.get("observations")
        if observations is not None:
            if not isinstance(observations, list):
                raise ValueError("Parameter 'observations' must be a list")
            arm_list: list[int | str] = []
            rew_list: list[float] = []
            for idx, item in enumerate(observations):
                if not isinstance(item, dict) or "arm" not in item or "reward" not in item:
                    raise ValueError(f"observations[{idx}] must have 'arm' and 'reward'")
                arm_val = item["arm"]
                if isinstance(arm_val, bool) or not isinstance(arm_val, (int, str)):
                    raise ValueError(f"observations[{idx}].arm must be int or str")
                rew_val = _require_number(item["reward"], f"observations[{idx}].reward")
                arm_list.append(arm_val)
                rew_list.append(rew_val)
            if arm_list:
                race.update_batch(arm_list, rew_list)

        if "arms" in args or "rewards" in args:
            arms_seq = args.get("arms")
            rewards_seq = args.get("rewards")
            if not isinstance(arms_seq, list) or not isinstance(rewards_seq, list):
                raise ValueError("Parameters 'arms' and 'rewards' must both be lists")
            for idx, arm_val in enumerate(arms_seq):
                if isinstance(arm_val, bool) or not isinstance(arm_val, (int, str)):
                    raise ValueError(f"arms[{idx}] must be int or str")
            rew_floats = _as_float_list(rewards_seq, "rewards")
            race.update_batch(arms_seq, rew_floats)

        status = race.status()
        payload: dict[str, Any] = {
            "status": status.to_dict(),
            "suggested_arm": race.suggest_arm(),
            "active_arms": race.active_arms,
            "arms": [race.arm_state(i).to_dict() for i in range(race.n_arms)],
        }
        if sid is not None:
            payload["session_id"] = sid
        return payload

    def _tool_conformal(self, args: dict[str, Any]) -> dict[str, Any]:
        from .conformal import MultiRateConformal

        sid = _optional_session_id(args)
        reset = _require_bool(args, "reset", False)
        cp: MultiRateConformal | None = None
        if sid is not None and not reset:
            cp = self._get_session(sid, "eproc_conformal")

        if cp is None:
            alphas = args.get("alphas", [0.50, 0.20, 0.10, 0.05, 0.01])
            alpha_vals = _as_float_list(alphas, "alphas")
            win = args.get("window_size", 128)
            if isinstance(win, bool) or not isinstance(win, int):
                raise ValueError("Parameter 'window_size' must be an integer")
            pid_ki = _require_finite_float(args.get("pid_ki", 0.05), "pid_ki")
            cp = MultiRateConformal(alphas=alpha_vals, window_size=win, pid_ki=pid_ki)
            if sid is not None:
                self._put_session(sid, "eproc_conformal", cp)

        if "scores" in args and args["scores"] is not None:
            scores = _as_float_list(args["scores"], "scores")
            cp.update_batch(scores)

        if "observations" in args and args["observations"] is not None:
            obs = args["observations"]
            if not isinstance(obs, list):
                raise ValueError("Parameter 'observations' must be a list")
            derived_scores: list[float] = []
            for idx, item in enumerate(obs):
                if not isinstance(item, dict) or "pred" not in item or "y" not in item:
                    raise ValueError(f"observations[{idx}] must have 'pred' and 'y'")
                pred = _require_number(item["pred"], f"observations[{idx}].pred")
                y = _require_number(item["y"], f"observations[{idx}].y")
                scale = _require_finite_float(item.get("scale", 1.0), f"observations[{idx}].scale")
                if scale <= 0.0:
                    raise ValueError(f"observations[{idx}].scale must be finite and > 0")
                derived_scores.append(abs(y - pred) / scale)
            if derived_scores:
                cp.update_batch(derived_scores)

        # Query current metadata via predict(0.0) or requested prediction
        pred_req = args.get("predict")
        if pred_req is not None:
            if not isinstance(pred_req, dict) or "pred" not in pred_req:
                raise ValueError("Parameter 'predict' must be an object with 'pred'")
            p_val = _require_finite_float(pred_req["pred"], "predict.pred")
            s_val = _require_finite_float(pred_req.get("scale", 1.0), "predict.scale")
            pred_res = cp.predict(p_val, scale=s_val)
            prediction_dict: dict[str, Any] | None = pred_res.to_dict()
        else:
            pred_res = cp.predict(0.0, scale=1.0)
            prediction_dict = None

        payload: dict[str, Any] = {
            "alphas": list(cp.alphas),
            "n_updates": cp.n_updates,
            "quantiles": list(pred_res.quantiles),
            "eff_alphas": list(pred_res.eff_alphas),
            "emp_coverage": list(pred_res.emp_coverage),
        }
        if prediction_dict is not None:
            payload["prediction"] = prediction_dict
        if sid is not None:
            payload["session_id"] = sid
        return payload

    def _tool_cmonoid_merge(self, args: dict[str, Any]) -> dict[str, Any]:
        from .conformal import ConformalMonoid

        if "shards" not in args or not isinstance(args["shards"], list):
            raise ValueError("Missing required parameter 'shards' (list of score lists or base64 monoids)")
        shards = args["shards"]
        if len(shards) == 0:
            raise ValueError("Parameter 'shards' must not be empty")

        scale_hint = _require_finite_float(args.get("scale_hint", 1.0), "scale_hint")
        quantiles_req = _as_float_list(args.get("quantiles", [0.50, 0.90, 0.95]), "quantiles")
        include_wire_b64 = _require_bool(args, "include_wire_b64", True)

        def _build_monoid(spec: Any, label: str) -> ConformalMonoid:
            if isinstance(spec, str):
                try:
                    raw = base64.b64decode(spec, validate=True)
                except Exception as exc:
                    raise ValueError(f"Invalid base64 wire payload in {label}: {exc}") from exc
                return ConformalMonoid.from_bytes(raw)
            if isinstance(spec, list):
                scores = _as_float_list(spec, label)
                m = ConformalMonoid(scale_hint=scale_hint)
                m.add_batch(scores)
                return m
            raise ValueError(f"{label} must be a list of numbers or a base64 wire string")

        acc = _build_monoid(shards[0], "shards[0]")
        for i in range(1, len(shards)):
            nxt = _build_monoid(shards[i], f"shards[{i}]")
            acc = acc.merge(nxt)

        if "subtract_shard" in args and args["subtract_shard"] is not None:
            local = _build_monoid(args["subtract_shard"], "subtract_shard")
            acc = acc.subtract(local)

        q_out = {str(q): acc.quantile(q) for q in quantiles_req}
        payload: dict[str, Any] = {
            "count": acc.count,
            "wire_size": acc.wire_size,
            "mean": acc.mean,
            "min_val": acc.min_val,
            "max_val": acc.max_val,
            "quantiles": q_out,
        }
        if include_wire_b64:
            payload["wire_b64"] = base64.b64encode(acc.to_bytes()).decode("ascii")
        return payload

    def _tool_safe_fuse(self, args: dict[str, Any]) -> dict[str, Any]:
        from .fusion import SafeFuser

        sid = _optional_session_id(args)
        reset = _require_bool(args, "reset", False)
        fuser: SafeFuser | None = None
        if sid is not None and not reset:
            fuser = self._get_session(sid, "eproc_safe_fuse")

        steps = args.get("steps")
        predict_vec = args.get("predict")

        if fuser is None:
            n_experts = args.get("n_experts")
            if n_experts is None:
                if isinstance(steps, list) and steps and isinstance(steps[0], dict) and isinstance(steps[0].get("preds"), list):
                    n_experts = len(steps[0]["preds"])
                elif isinstance(predict_vec, list) and predict_vec:
                    n_experts = len(predict_vec)
                else:
                    raise ValueError("Missing required parameter 'n_experts'")
            if isinstance(n_experts, bool) or not isinstance(n_experts, int):
                raise ValueError("Parameter 'n_experts' must be an integer")
            anchor_idx = args.get("anchor_idx", 0)
            if isinstance(anchor_idx, bool) or not isinstance(anchor_idx, int):
                raise ValueError("Parameter 'anchor_idx' must be an integer")
            alpha_deg = _require_finite_float(args.get("alpha_deg", 0.05), "alpha_deg")
            fuser = SafeFuser(n_experts=n_experts, anchor_idx=anchor_idx, alpha_deg=alpha_deg)
            if sid is not None:
                self._put_session(sid, "eproc_safe_fuse", fuser)

        last_step: dict[str, Any] | None = None
        if steps is not None:
            if not isinstance(steps, list):
                raise ValueError("Parameter 'steps' must be a list")
            for idx, item in enumerate(steps):
                if not isinstance(item, dict) or "preds" not in item or "y" not in item:
                    raise ValueError(f"steps[{idx}] must have 'preds' and 'y'")
                preds = _as_float_list(item["preds"], f"steps[{idx}].preds")
                y_val = _require_number(item["y"], f"steps[{idx}].y")
                step_res = fuser.update(preds, y_val)
                last_step = step_res.to_dict()

        weights, fb = fuser.snapshot()
        payload: dict[str, Any] = {
            "n_experts": fuser.n_experts,
            "anchor_idx": fuser.anchor_idx,
            "weights": weights,
            "fallback_active": fb,
            "quarantined": fuser.quarantined,
            "n_updates": fuser.n_updates,
            "n_poison_clipped": fuser.n_poison_clipped,
            "n_fallback_trips": fuser.n_fallback_trips,
        }
        if last_step is not None:
            payload["last_step"] = last_step
        if predict_vec is not None:
            p_list = _as_float_list(predict_vec, "predict")
            payload["prediction"] = fuser.predict(p_list)
        if sid is not None:
            payload["session_id"] = sid
        return payload

    def _tool_echopool(self, args: dict[str, Any]) -> dict[str, Any]:
        from .fusion import EchoPool

        sid = _optional_session_id(args)
        reset = _require_bool(args, "reset", False)
        pool: EchoPool | None = None
        if sid is not None and not reset:
            pool = self._get_session(sid, "eproc_echopool")

        if pool is None:
            n_sources = args.get("n_sources", 16)
            n_claims = args.get("n_claims", 64)
            if isinstance(n_sources, bool) or not isinstance(n_sources, int):
                raise ValueError("Parameter 'n_sources' must be an integer")
            if isinstance(n_claims, bool) or not isinstance(n_claims, int):
                raise ValueError("Parameter 'n_claims' must be an integer")
            prior_prob = _require_finite_float(args.get("prior_prob", 0.5), "prior_prob")
            pool = EchoPool(n_sources=n_sources, n_claims=n_claims, prior_prob=prior_prob)
            if sid is not None:
                self._put_session(sid, "eproc_echopool", pool)

        sub_results: list[dict[str, Any]] = []
        touched_claims: set[int] = set()
        touched_sources: set[int] = set()

        submissions = args.get("submissions")
        if submissions is not None:
            if not isinstance(submissions, list):
                raise ValueError("Parameter 'submissions' must be a list")
            for idx, item in enumerate(submissions):
                if not isinstance(item, dict):
                    raise ValueError(f"submissions[{idx}] must be an object")
                for key in ("source_id", "claim_id", "evidence_id", "llr"):
                    if key not in item:
                        raise ValueError(f"submissions[{idx}] missing required key {key!r}")
                src = item["source_id"]
                clm = item["claim_id"]
                if isinstance(src, bool) or not isinstance(src, int):
                    raise ValueError(f"submissions[{idx}].source_id must be an integer")
                if isinstance(clm, bool) or not isinstance(clm, int):
                    raise ValueError(f"submissions[{idx}].claim_id must be an integer")
                ev = item["evidence_id"]
                if isinstance(ev, bool) or not isinstance(ev, (int, str)):
                    raise ValueError(f"submissions[{idx}].evidence_id must be int or str")
                llr = _require_number(item["llr"], f"submissions[{idx}].llr")
                res = pool.submit(src, clm, ev, llr)
                sub_results.append(res.to_dict())
                touched_claims.add(clm)
                touched_sources.add(src)

        resolutions = args.get("resolutions")
        if resolutions is not None:
            if not isinstance(resolutions, list):
                raise ValueError("Parameter 'resolutions' must be a list")
            for idx, item in enumerate(resolutions):
                if not isinstance(item, dict) or "claim_id" not in item or "outcome_positive" not in item:
                    raise ValueError(f"resolutions[{idx}] must have 'claim_id' and 'outcome_positive'")
                clm = item["claim_id"]
                if isinstance(clm, bool) or not isinstance(clm, int):
                    raise ValueError(f"resolutions[{idx}].claim_id must be an integer")
                outcome_pos = item["outcome_positive"]
                if not isinstance(outcome_pos, bool):
                    raise ValueError(f"resolutions[{idx}].outcome_positive must be a boolean")
                pool.resolve(clm, outcome_pos)
                touched_claims.add(clm)

        q_claims = args.get("query_claims")
        if q_claims is not None:
            if not isinstance(q_claims, list) or not all(
                isinstance(c, int) and not isinstance(c, bool) for c in q_claims
            ):
                raise ValueError("Parameter 'query_claims' must be a list of integers")
            claim_ids = q_claims
        else:
            claim_ids = sorted(touched_claims)

        q_sources = args.get("query_sources")
        if q_sources is not None:
            if not isinstance(q_sources, list) or not all(
                isinstance(s, int) and not isinstance(s, bool) for s in q_sources
            ):
                raise ValueError("Parameter 'query_sources' must be a list of integers")
            source_ids = q_sources
        else:
            source_ids = sorted(touched_sources)

        payload: dict[str, Any] = {
            "n_accepted": pool.n_accepted,
            "n_echo_dropped": pool.n_echo_dropped,
            "n_table_full": pool.n_table_full,
            "submissions": sub_results,
            "posteriors": {str(c): pool.posterior(c) for c in claim_ids},
            "source_reliabilities": {str(s): pool.source_reliability(s) for s in source_ids},
        }
        if sid is not None:
            payload["session_id"] = sid
        return payload

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "eproc_etest": self._tool_etest,
            "eproc_confidence_sequence": self._tool_confidence_sequence,
            "eproc_race": self._tool_race,
            "eproc_conformal": self._tool_conformal,
            "eproc_cmonoid_merge": self._tool_cmonoid_merge,
            "eproc_safe_fuse": self._tool_safe_fuse,
            "eproc_echopool": self._tool_echopool,
        }
        return handlers[name](args)

    # ------------------------------------------------------------------
    # JSON-RPC 2.0 Request Dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _ok(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": req_id, "error": err}

    def handle_request(self, req: Any) -> dict[str, Any] | None:
        """Process one decoded JSON-RPC 2.0 request object and return a response dict or ``None``."""
        if not isinstance(req, dict):
            return self._error(None, ERR_INVALID_REQUEST, "Invalid Request: expected JSON object")

        has_id = "id" in req
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params")

        if req.get("jsonrpc") != "2.0" or not isinstance(method, str) or not method:
            return self._error(
                req_id if has_id else None,
                ERR_INVALID_REQUEST,
                "Invalid Request: missing jsonrpc='2.0' or method",
            )

        if params is None:
            params = {}
        elif not isinstance(params, dict):
            return self._error(
                req_id if has_id else None,
                ERR_INVALID_PARAMS,
                "Invalid params: expected object",
            )

        # Valid JSON-RPC 2.0 notifications (no id) must never receive a response
        if not has_id:
            return None

        if method == "initialize":
            client_ver = params.get("protocolVersion")
            if isinstance(client_ver, str) and client_ver in SUPPORTED_PROTOCOL_VERSIONS:
                negotiated = client_ver
            else:
                negotiated = DEFAULT_PROTOCOL_VERSION
            self._protocol_version = negotiated
            return self._ok(
                req_id,
                {
                    "protocolVersion": negotiated,
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "eproc",
                        "version": __version__,
                    },
                    "instructions": (
                        "Sub-microsecond anytime-valid statistical inference engine (C11). "
                        "Use optional session_id on streaming tools to accumulate observations across turns."
                    ),
                },
            )

        if method == "server/discover":
            return self._ok(
                req_id,
                {
                    "protocolVersions": list(SUPPORTED_PROTOCOL_VERSIONS[:2]),
                    "serverInfo": {
                        "name": "eproc",
                        "version": __version__,
                    },
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "tools": TOOL_DEFINITIONS,
                    "ttlMs": TOOL_LIST_TTL_MS,
                    "cacheScope": TOOL_LIST_CACHE_SCOPE,
                },
            )

        if method == "ping":
            return self._ok(req_id, {})

        if method == "tools/list":
            return self._ok(
                req_id,
                {
                    "tools": TOOL_DEFINITIONS,
                    "ttlMs": TOOL_LIST_TTL_MS,
                    "cacheScope": TOOL_LIST_CACHE_SCOPE,
                    "_meta": {
                        "ttlMs": TOOL_LIST_TTL_MS,
                        "cacheScope": TOOL_LIST_CACHE_SCOPE,
                    },
                },
            )

        if method == "tools/call":
            tool_name = params.get("name")
            if not isinstance(tool_name, str) or tool_name not in _TOOL_NAMES:
                return self._error(
                    req_id,
                    ERR_INVALID_PARAMS,
                    f"Unknown or missing tool name: {tool_name!r}",
                )
            raw_args = params.get("arguments", {})
            if raw_args is None:
                raw_args = {}
            if not isinstance(raw_args, dict):
                return self._error(req_id, ERR_INVALID_PARAMS, "Tool 'arguments' must be a JSON object")

            try:
                payload = _sanitize_json(self._dispatch_tool(tool_name, raw_args))
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                err_payload = {"error": str(exc), "tool": tool_name}
                return self._ok(
                    req_id,
                    {
                        "content": [{"type": "text", "text": _dumps(err_payload)}],
                        "structuredContent": err_payload,
                        "isError": True,
                    },
                )
            except Exception as exc:
                return self._error(req_id, ERR_INTERNAL, f"Internal tool error: {exc}")

            return self._ok(
                req_id,
                {
                    "content": [{"type": "text", "text": _dumps(payload)}],
                    "structuredContent": payload,
                    "isError": False,
                },
            )

        return self._error(req_id, ERR_METHOD_NOT_FOUND, f"Method not found: {method}")

    def handle_line(self, line: str) -> str | None:
        """Parse one NDJSON line, handle it, and serialize any response line."""
        stripped = line.strip()
        if not stripped:
            return None
        try:
            req = json.loads(stripped)
        except json.JSONDecodeError as exc:
            return _dumps(self._error(None, ERR_PARSE, f"Parse error: {exc.msg}"))
        resp = self.handle_request(req)
        if resp is None:
            return None
        return _dumps(resp)

    def serve_stdio(
        self,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> int:
        """Run the newline-delimited JSON-RPC 2.0 loop over stdin/stdout."""
        in_stream = sys.stdin if stdin is None else stdin
        out_stream = sys.stdout if stdout is None else stdout
        for line in in_stream:
            reply = self.handle_line(line)
            if reply is not None:
                out_stream.write(reply + "\n")
                out_stream.flush()
        return 0


def serve_stdio(stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    """Launch a fresh :class:`McpServer` over stdio."""
    return McpServer().serve_stdio(stdin=stdin, stdout=stdout)


def main(argv: list[str] | None = None) -> int:
    _ = argv
    return serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
