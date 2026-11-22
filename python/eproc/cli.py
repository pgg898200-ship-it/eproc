"""Fast zero-dependency CLI for ``eproc`` streaming inference, racing, conformal, fusion, and MCP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, TextIO

from . import __version__


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(v) for v in value]
    return value


def _emit_json(obj: Any, out: TextIO) -> None:
    out.write(json.dumps(_sanitize_json(obj), allow_nan=False) + "\n")


def _open_input(path: str | None, stdin: TextIO) -> tuple[TextIO, bool]:
    if path is None or path == "-":
        return stdin, False
    return open(path, "r", encoding="utf-8"), True


def _extract_numeric_from_obj(obj: Any, field: str | None) -> float:
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj)
    if isinstance(obj, dict):
        if field is not None:
            if field not in obj:
                raise ValueError(f"Field {field!r} not found in JSON object")
            return float(obj[field])
        for candidate in ("value", "score", "reward", "metric", "x", "y", "pass"):
            if candidate in obj and isinstance(obj[candidate], (int, float)) and not isinstance(obj[candidate], bool):
                return float(obj[candidate])
        for val in obj.values():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
        raise ValueError("No numeric field found in JSON object; specify --field")
    raise ValueError(f"Cannot extract numeric observation from {type(obj).__name__}")


def _iter_scalar_stream(
    stream: TextIO,
    fmt: str = "auto",
    field: str | None = None,
) -> Iterator[float]:
    csv_header: list[str] | None = None
    col_idx: int | None = None
    if field is not None and field.isdigit():
        col_idx = int(field)

    for raw_line in stream:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        use_json = fmt == "jsonl" or (fmt == "auto" and line[0] in ("{", "["))
        if use_json:
            obj = json.loads(line)
            if isinstance(obj, list):
                for item in obj:
                    yield _extract_numeric_from_obj(item, field)
            else:
                yield _extract_numeric_from_obj(obj, field)
            continue

        use_csv = fmt == "csv" or (fmt == "auto" and "," in line)
        if use_csv:
            row = next(csv.reader([line]))
            if csv_header is None:
                try:
                    float(row[0].strip())
                except ValueError:
                    csv_header = [c.strip() for c in row]
                    if field is not None and col_idx is None:
                        if field not in csv_header:
                            raise ValueError(f"CSV column {field!r} not found in header {csv_header}")
                        col_idx = csv_header.index(field)
                    continue
            idx = 0 if col_idx is None else col_idx
            if idx >= len(row):
                raise ValueError(f"CSV row has only {len(row)} columns, requested index {idx}")
            yield float(row[idx].strip())
            continue

        for tok in line.split():
            yield float(tok)


# ---------------------------------------------------------------------------
# Subcommand Handlers
# ---------------------------------------------------------------------------


def _cmd_gate(args: argparse.Namespace, stdin: TextIO, stdout: TextIO, stderr: TextIO) -> int:
    from .eprocess import EProcess

    ep = EProcess(
        mu0=args.mu0,
        alpha=args.alpha,
        lower=args.lower,
        upper=args.upper,
        side=args.side,
    )
    stream, should_close = _open_input(args.input, stdin)
    try:
        batch: list[float] = []
        for val in _iter_scalar_stream(stream, fmt=args.format, field=args.field):
            if args.stop_early:
                res = ep.update(val)
                if res.rejected:
                    break
            else:
                batch.append(val)
                if len(batch) >= 4096:
                    ep.update_batch(batch)
                    batch.clear()
        if batch:
            ep.update_batch(batch)
    finally:
        if should_close:
            stream.close()

    res = ep.result()
    if not res.rejected:
        verdict = "CONTINUE"
    elif ep.side == "upper":
        verdict = "PASS"
    elif ep.side == "lower":
        verdict = "FAIL"
    else:
        verdict = "PASS" if res.mean >= ep.mu0 else "FAIL"

    payload = res.to_dict()
    payload.update(
        {
            "verdict": verdict,
            "mu0": ep.mu0,
            "alpha": ep.alpha,
            "side": ep.side,
            "rejected_at": ep.rejected_at,
            "n_invalid": ep.n_invalid,
        }
    )
    if args.json:
        _emit_json(payload, stdout)
    else:
        stdout.write(
            f"VERDICT: {verdict} | n={res.n} mean={res.mean:.6g} "
            f"e_value={res.e_value:.6g} p_value={res.p_value:.6g} "
            f"rejected={res.rejected} (mu0={ep.mu0}, alpha={ep.alpha}, side={ep.side})\n"
        )
    return 0


def _cmd_race(args: argparse.Namespace, stdin: TextIO, stdout: TextIO, stderr: TextIO) -> int:
    from .eprocess import AgentRace

    names = [s.strip() for s in args.names.split(",") if s.strip()] if args.names else None
    n_arms = args.arms if args.arms is not None else (len(names) if names else None)
    if n_arms is None:
        raise ValueError("Specify --arms N or --names a,b,c")

    race = AgentRace(
        n_arms=n_arms,
        alpha=args.alpha,
        lower=args.lower,
        upper=args.upper,
        margin=args.margin,
        names=names,
    )
    stream, should_close = _open_input(args.input, stdin)
    csv_header: list[str] | None = None
    try:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line[0] == "{":
                obj = json.loads(line)
                arm = obj[args.arm_field]
                reward = float(obj[args.reward_field])
            else:
                row = next(csv.reader([line]))
                if csv_header is None and len(row) >= 2:
                    try:
                        float(row[1].strip())
                    except ValueError:
                        csv_header = [c.strip() for c in row]
                        continue
                raw_arm = row[0].strip()
                arm = int(raw_arm) if raw_arm.isdigit() and names is None else raw_arm
                reward = float(row[1].strip())
            st = race.update(arm, reward)
            if args.stop_early and st.winner is not None:
                break
    finally:
        if should_close:
            stream.close()

    status = race.status()
    arms_detail = [race.arm_state(i).to_dict() for i in range(race.n_arms)]
    payload = {
        "status": status.to_dict(),
        "suggested_arm": race.suggest_arm(),
        "active_arms": race.active_arms,
        "arms": arms_detail,
    }
    if args.json:
        _emit_json(payload, stdout)
    else:
        win_lbl = (
            f"{status.winner} ({status.winner_name})"
            if status.winner is not None and status.winner_name
            else str(status.winner)
        )
        stdout.write(
            f"RACE: total_obs={status.total_obs} active={status.n_active}/{status.n_arms} "
            f"winner={win_lbl} leader={status.leader} next_arm={race.suggest_arm()}\n"
        )
        for a in arms_detail:
            tag = "ACTIVE" if a["active"] else f"ELIM@{a['eliminated_at']}"
            label = f"{a['arm']}:{a['name']}" if a["name"] else str(a["arm"])
            stdout.write(
                f"  arm {label:<12} [{tag:<10}] n={a['n']:<5} "
                f"mean={a['mean']:.4f} cs=[{a['low']:.4f}, {a['high']:.4f}]\n"
            )
    return 0


def _cmd_conformal(args: argparse.Namespace, stdin: TextIO, stdout: TextIO, stderr: TextIO) -> int:
    from .conformal import MultiRateConformal

    if not math.isfinite(args.scale) or args.scale <= 0.0:
        raise ValueError(f"--scale must be finite and > 0, got {args.scale!r}")

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    cp = MultiRateConformal(alphas=alphas, window_size=args.window, pid_ki=args.pid_ki)

    stream, should_close = _open_input(args.input, stdin)
    try:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            pred_val: float | None = None
            scale_val = 1.0
            if line[0] == "{":
                obj = json.loads(line)
                raw_scale = obj.get(args.scale_field, 1.0)
                if isinstance(raw_scale, bool) or not isinstance(raw_scale, (int, float)):
                    raise ValueError(f"Field {args.scale_field!r} must be a positive finite number")
                scale_val = float(raw_scale)
                if not math.isfinite(scale_val) or scale_val <= 0.0:
                    raise ValueError(
                        f"Field {args.scale_field!r} must be finite and > 0, got {scale_val!r}"
                    )
                if args.score_field in obj:
                    score = float(obj[args.score_field])
                    if args.pred_field in obj:
                        pred_val = float(obj[args.pred_field])
                elif args.pred_field in obj and args.y_field in obj:
                    pred_val = float(obj[args.pred_field])
                    y_val = float(obj[args.y_field])
                    score = abs(y_val - pred_val) / scale_val
                else:
                    raise ValueError(
                        f"JSON record must include {args.score_field!r} or ({args.pred_field!r}, {args.y_field!r})"
                    )
            else:
                score = float(line.split(",")[0].strip())

            if args.stream:
                p_out = cp.predict(0.0 if pred_val is None else pred_val, scale=scale_val)
                _emit_json({"step": cp.n_updates + 1, "score": score, **p_out.to_dict()}, stdout)
            cp.update(score)
    finally:
        if should_close:
            stream.close()

    if not args.stream or args.predict is not None:
        pred_pt = 0.0 if args.predict is None else float(args.predict)
        pred = cp.predict(pred_pt, scale=args.scale)
        payload = {
            "n_updates": cp.n_updates,
            "alphas": list(cp.alphas),
            "quantiles": list(pred.quantiles),
            "eff_alphas": list(pred.eff_alphas),
            "emp_coverage": list(pred.emp_coverage),
        }
        if args.predict is not None:
            payload["prediction"] = pred.to_dict()

        if args.json:
            _emit_json(payload, stdout)
        else:
            stdout.write(f"CONFORMAL: n_updates={cp.n_updates}\n")
            for a, q, lo, hi, cov in zip(
                pred.alphas, pred.quantiles, pred.lows, pred.highs, pred.emp_coverage
            ):
                extra = f" interval=[{lo:.4f}, {hi:.4f}]" if args.predict is not None else ""
                stdout.write(
                    f"  cov={1.0 - a:>5.1%} (alpha={a:.3f}) q={q:.4f} emp_cov={cov:.3f}{extra}\n"
                )
    return 0


def _cmd_fuse(args: argparse.Namespace, stdin: TextIO, stdout: TextIO, stderr: TextIO) -> int:
    from .fusion import SafeFuser

    fuser: SafeFuser | None = None
    if args.experts is not None:
        fuser = SafeFuser(args.experts, anchor_idx=args.anchor, alpha_deg=args.alpha_deg)

    last_step: dict[str, Any] | None = None
    stream, should_close = _open_input(args.input, stdin)
    try:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line[0] == "{":
                obj = json.loads(line)
                preds = [float(v) for v in obj[args.preds_field]]
                y_true = float(obj[args.y_field])
            else:
                parts = [float(tok.strip()) for tok in line.split(",")]
                if len(parts) < 2:
                    raise ValueError("CSV row must have at least one expert prediction and y_true")
                preds = parts[:-1]
                y_true = parts[-1]

            if fuser is None:
                fuser = SafeFuser(len(preds), anchor_idx=args.anchor, alpha_deg=args.alpha_deg)

            step = fuser.update(preds, y_true)
            last_step = step.to_dict()
            if args.stream:
                _emit_json(last_step, stdout)
    finally:
        if should_close:
            stream.close()

    if fuser is None:
        raise ValueError("No input rows received and --experts not specified")

    weights, fb = fuser.snapshot()
    payload: dict[str, Any] = {
        "n_experts": fuser.n_experts,
        "anchor_idx": fuser.anchor_idx,
        "n_updates": fuser.n_updates,
        "weights": weights,
        "fallback_active": fb,
        "quarantined": fuser.quarantined,
        "n_poison_clipped": fuser.n_poison_clipped,
        "n_fallback_trips": fuser.n_fallback_trips,
        "last_step": last_step,
    }
    if not args.stream:
        if args.json:
            _emit_json(payload, stdout)
        else:
            w_str = ", ".join(f"{w:.4f}" for w in weights)
            stdout.write(
                f"FUSE: n_updates={fuser.n_updates} fallback={fb} "
                f"poison_clipped={fuser.n_poison_clipped} weights=[{w_str}]\n"
            )
    return 0


def _cmd_subsample(args: argparse.Namespace, stdin: TextIO, stdout: TextIO, stderr: TextIO) -> int:
    stream, should_close = _open_input(args.input, stdin)
    try:
        if args.method == "varopt":
            from .subsample import VarOptSampler

            vo = VarOptSampler(k=args.k, seed=args.seed)
            auto_id = 0
            for raw_line in stream:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line[0] == "{":
                    obj = json.loads(line)
                    item_id = int(obj.get("id", obj.get("item_id", auto_id)))
                    weight = float(obj.get("weight", 1.0))
                    value = float(obj["value"])
                else:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        item_id = int(parts[0])
                        weight = float(parts[1])
                        value = float(parts[2])
                    elif len(parts) == 2:
                        item_id = auto_id
                        weight = float(parts[0])
                        value = float(parts[1])
                    else:
                        item_id = auto_id
                        weight = 1.0
                        value = float(parts[0])
                vo.push(item_id, weight, value)
                auto_id += 1

            payload: dict[str, Any] = {
                "method": "varopt",
                "k": vo.k,
                "count": vo.count,
                "n_seen": vo.n_seen,
                "total_weight": vo.total_weight,
                "estimate_sum": vo.estimate_sum(),
                "estimate_mean": vo.estimate_mean(),
            }
            if args.json:
                _emit_json(payload, stdout)
            else:
                stdout.write(
                    f"VAROPT: count={vo.count}/{vo.k} n_seen={vo.n_seen} "
                    f"est_sum={payload['estimate_sum']:.6g} est_mean={payload['estimate_mean']:.6g}\n"
                )
            return 0

        # Online RLS mode
        from .subsample import OnlineRLSSampler

        rls: OnlineRLSSampler | None = None
        if args.dim is not None:
            rls = OnlineRLSSampler(
                d=args.dim,
                lambda_reg=args.lambda_reg,
                oversample=args.oversample,
                min_prob=args.min_prob,
                seed=args.seed,
            )

        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line[0] == "{":
                obj = json.loads(line)
                vec = [float(v) for v in obj["x"]]
            elif line[0] == "[":
                vec = [float(v) for v in json.loads(line)]
            else:
                vec = [float(tok.strip()) for tok in line.split(",")]
            if rls is None:
                rls = OnlineRLSSampler(
                    d=len(vec),
                    lambda_reg=args.lambda_reg,
                    oversample=args.oversample,
                    min_prob=args.min_prob,
                    seed=args.seed,
                )
            step = rls.step(vec)
            if args.stream:
                _emit_json(step.to_dict(), stdout)

        if rls is None:
            raise ValueError("No feature vectors received and --dim not specified")

        payload = {
            "method": "rls",
            "d": rls.d,
            "n_seen": rls.n_seen,
            "n_kept": rls.n_kept,
            "keep_rate": rls.keep_rate,
        }
        if not args.stream:
            if args.json:
                _emit_json(payload, stdout)
            else:
                stdout.write(
                    f"RLS: d={rls.d} n_seen={rls.n_seen} n_kept={rls.n_kept} "
                    f"keep_rate={rls.keep_rate:.4f}\n"
                )
        return 0
    finally:
        if should_close:
            stream.close()


def _cmd_bench(args: argparse.Namespace, stdin: TextIO, stdout: TextIO, stderr: TextIO) -> int:
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir.parents[1] / "benchmarks" / "bench_suite.py" if len(pkg_dir.parents) >= 2 else None,
        Path.cwd() / "benchmarks" / "bench_suite.py",
    ]
    bench_script = next((p for p in candidates if p is not None and p.is_file()), None)
    if bench_script is None:
        stderr.write("error: benchmarks/bench_suite.py not found\n")
        return 1

    env = os.environ.copy()
    py_root = str(pkg_dir.parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{py_root}:{existing}" if existing else py_root
    proc = subprocess.run([sys.executable, str(bench_script), *args.bench_args], env=env)
    return int(proc.returncode)


def _cmd_mcp(args: argparse.Namespace, stdin: TextIO, stdout: TextIO, stderr: TextIO) -> int:
    from .mcp_server import serve_stdio

    return serve_stdio(stdin=stdin, stdout=stdout)


def _cmd_version(args: argparse.Namespace, stdin: TextIO, stdout: TextIO, stderr: TextIO) -> int:
    from ._ffi import abi_version, get_lib_path, simd_flag_names, simd_flags

    abi = abi_version()
    mask = simd_flags()
    simd_names = simd_flag_names(mask)
    lib_path = get_lib_path()
    payload = {
        "version": __version__,
        "abi_version": abi,
        "simd_flags": mask,
        "simd": simd_names if simd_names else ["scalar"],
        "library": lib_path,
        "python": sys.version.split()[0],
    }
    if args.json:
        _emit_json(payload, stdout)
    else:
        simd_str = ",".join(payload["simd"])
        stdout.write(
            f"eproc {__version__} (ABI v{abi}, SIMD: {simd_str}, lib: {lib_path})\n"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eproc",
        description="Sub-microsecond anytime-valid online statistical inference engine (C11).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # gate
    p_gate = sub.add_parser("gate", help="Anytime-valid e-process gating over a stream.")
    p_gate.add_argument("--mu0", type=float, required=True, help="Null mean threshold.")
    p_gate.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05).")
    p_gate.add_argument("--lower", type=float, default=0.0, help="Observation lower bound.")
    p_gate.add_argument("--upper", type=float, default=1.0, help="Observation upper bound.")
    p_gate.add_argument("--side", choices=("upper", "lower", "two_sided"), default="upper")
    p_gate.add_argument("--field", type=str, default=None, help="JSON key or CSV column name/index.")
    p_gate.add_argument("--format", choices=("auto", "jsonl", "csv", "numbers"), default="auto")
    p_gate.add_argument("--input", "-i", type=str, default=None, help="Input file path (default: stdin).")
    p_gate.add_argument("--stop-early", action="store_true", help="Stop reading immediately on rejection.")
    p_gate.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    p_gate.set_defaults(func=_cmd_gate)

    # race
    p_race = sub.add_parser("race", help="Anytime-valid multi-arm elimination race.")
    p_race.add_argument("--arms", type=int, default=None, help="Number of arms.")
    p_race.add_argument("--names", type=str, default=None, help="Comma-separated arm names.")
    p_race.add_argument("--alpha", type=float, default=0.05)
    p_race.add_argument("--lower", type=float, default=0.0)
    p_race.add_argument("--upper", type=float, default=1.0)
    p_race.add_argument("--margin", type=float, default=0.0)
    p_race.add_argument("--arm-field", type=str, default="arm")
    p_race.add_argument("--reward-field", type=str, default="reward")
    p_race.add_argument("--input", "-i", type=str, default=None)
    p_race.add_argument("--stop-early", action="store_true")
    p_race.add_argument("--json", action="store_true")
    p_race.set_defaults(func=_cmd_race)

    # conformal
    p_conf = sub.add_parser("conformal", help="Multi-rate crossing-free online conformal calibration.")
    p_conf.add_argument("--alphas", type=str, default="0.50,0.20,0.10,0.05,0.01")
    p_conf.add_argument("--window", type=int, default=128)
    p_conf.add_argument("--pid-ki", type=float, default=0.05)
    p_conf.add_argument("--score-field", type=str, default="score")
    p_conf.add_argument("--pred-field", type=str, default="pred")
    p_conf.add_argument("--y-field", type=str, default="y")
    p_conf.add_argument("--scale-field", type=str, default="scale")
    p_conf.add_argument("--predict", type=float, default=None, help="Emit intervals around this forecast.")
    p_conf.add_argument("--scale", type=float, default=1.0)
    p_conf.add_argument("--input", "-i", type=str, default=None)
    p_conf.add_argument("--stream", action="store_true", help="Emit JSONL causal intervals per tick.")
    p_conf.add_argument("--json", action="store_true")
    p_conf.set_defaults(func=_cmd_conformal)

    # fuse
    p_fuse = sub.add_parser("fuse", help="Byzantine-safe BOA multi-agent prediction fusion.")
    p_fuse.add_argument("--experts", type=int, default=None)
    p_fuse.add_argument("--anchor", type=int, default=0)
    p_fuse.add_argument("--alpha-deg", type=float, default=0.05)
    p_fuse.add_argument("--preds-field", type=str, default="preds")
    p_fuse.add_argument("--y-field", type=str, default="y")
    p_fuse.add_argument("--input", "-i", type=str, default=None)
    p_fuse.add_argument("--stream", action="store_true")
    p_fuse.add_argument("--json", action="store_true")
    p_fuse.set_defaults(func=_cmd_fuse)

    # subsample
    p_sub = sub.add_parser("subsample", help="Causal VarOpt or Online RLS subsampling.")
    p_sub.add_argument("--method", choices=("varopt", "rls"), default="varopt")
    p_sub.add_argument("--k", type=int, default=64, help="VarOpt reservoir capacity.")
    p_sub.add_argument("--dim", type=int, default=None, help="RLS feature dimension.")
    p_sub.add_argument("--lambda-reg", type=float, default=1.0)
    p_sub.add_argument("--oversample", type=float, default=2.0)
    p_sub.add_argument("--min-prob", type=float, default=0.02)
    p_sub.add_argument("--seed", type=int, default=42)
    p_sub.add_argument("--input", "-i", type=str, default=None)
    p_sub.add_argument("--stream", action="store_true")
    p_sub.add_argument("--json", action="store_true")
    p_sub.set_defaults(func=_cmd_subsample)

    # bench
    p_bench = sub.add_parser("bench", help="Run the eproc benchmark suite.")
    p_bench.add_argument("bench_args", nargs=argparse.REMAINDER)
    p_bench.set_defaults(func=_cmd_bench)

    # mcp
    p_mcp = sub.add_parser("mcp", help="Launch the JSON-RPC 2.0 MCP 2026-07-28 stdio server.")
    p_mcp.set_defaults(func=_cmd_mcp)

    # version
    p_ver = sub.add_parser("version", help="Show version, C ABI version, and SIMD flags.")
    p_ver.add_argument("--json", action="store_true")
    p_ver.set_defaults(func=_cmd_version)

    return parser


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    in_s = sys.stdin if stdin is None else stdin
    out_s = sys.stdout if stdout is None else stdout
    err_s = sys.stderr if stderr is None else stderr

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args, in_s, out_s, err_s))
    except (ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
        err_s.write(f"eproc error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
