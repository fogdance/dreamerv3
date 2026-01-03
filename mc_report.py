# mc_report.py
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_all_jsonl(run_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(run_dir / "episodes_w*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"No episodes_w*.jsonl found under: {run_dir}")

    dfs = []
    for p in paths:
        df = pd.read_json(p, lines=True)
        df["__src_file"] = Path(p).name
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)

    # Normalize numeric columns (coerce errors -> NaN)
    num_cols = [
        "equity", "return_pct", "max_drawdown_pct", "profit_factor",
        "trades_opened", "fee_total", "stop_loss_fired", "take_profit_fired",
        "length", "score", "steps",
    ]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Normalize bool columns
    for c in ["terminated", "truncated"]:
        if c in out.columns:
            out[c] = out[c].astype(bool)

    return out


def _summarize_series(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    q = np.quantile(x, [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) >= 2 else 0.0,
        "min": float(np.min(x)),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p10": float(q[2]),
        "p50": float(q[3]),
        "p90": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "max": float(np.max(x)),
    }


def _var_cvar(x: pd.Series, alpha: float = 0.05) -> dict:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"var": None, "cvar": None}
    var = float(np.quantile(x, alpha))
    tail = x[x <= var]
    cvar = float(np.mean(tail)) if len(tail) else float("nan")
    return {"var": var, "cvar": cvar}


def _fmt(d: dict, pct: bool = False) -> str:
    if not d or d.get("n", 0) == 0:
        return "n=0"
    mul = 100.0 if pct else 1.0
    unit = "%" if pct else ""
    return (
        f"n={d['n']} mean={d['mean']*mul:.4f}{unit} std={d['std']*mul:.4f}{unit} "
        f"p05={d['p05']*mul:.4f}{unit} p50={d['p50']*mul:.4f}{unit} p95={d['p95']*mul:.4f}{unit} "
        f"min={d['min']*mul:.4f}{unit} max={d['max']*mul:.4f}{unit}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str, help="MC run dir containing episodes_w*.jsonl and summary.json")
    ap.add_argument("--save", action="store_true", help="Save report.csv and report.json into run_dir")
    ap.add_argument("--alpha", type=float, default=0.05, help="VaR/CVaR alpha, default 0.05")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    df = _read_all_jsonl(run_dir)

    # Basic integrity
    n = len(df)
    n_term = int(df["terminated"].sum()) if "terminated" in df.columns else 0
    n_trunc = int(df["truncated"].sum()) if "truncated" in df.columns else 0

    # Core metrics summaries
    s_ret = _summarize_series(df.get("return_pct", pd.Series(dtype=float)))
    s_eq = _summarize_series(df.get("equity", pd.Series(dtype=float)))
    s_dd = _summarize_series(df.get("max_drawdown_pct", pd.Series(dtype=float)))
    s_pf = _summarize_series(df.get("profit_factor", pd.Series(dtype=float)))
    s_tr = _summarize_series(df.get("trades_opened", pd.Series(dtype=float)))
    s_fee = _summarize_series(df.get("fee_total", pd.Series(dtype=float)))

    # Risk stats on daily returns
    rets = pd.to_numeric(df.get("return_pct", pd.Series(dtype=float)), errors="coerce").astype(float)
    rets = rets[np.isfinite(rets)]
    p_loss = float((rets < 0).mean()) if len(rets) else float("nan")
    varcvar = _var_cvar(df.get("return_pct", pd.Series(dtype=float)), alpha=args.alpha)

    # Stops / take profits
    sl_total = int(pd.to_numeric(df.get("stop_loss_fired", 0), errors="coerce").fillna(0).sum())
    tp_total = int(pd.to_numeric(df.get("take_profit_fired", 0), errors="coerce").fillna(0).sum())
    sl_days = int((pd.to_numeric(df.get("stop_loss_fired", 0), errors="coerce").fillna(0) > 0).sum())
    tp_days = int((pd.to_numeric(df.get("take_profit_fired", 0), errors="coerce").fillna(0) > 0).sum())

    # Best / worst days
    worst = df.loc[df["return_pct"].astype(float).idxmin()] if "return_pct" in df.columns and df["return_pct"].notna().any() else None
    best = df.loc[df["return_pct"].astype(float).idxmax()] if "return_pct" in df.columns and df["return_pct"].notna().any() else None

    # Print report
    print("=" * 80)
    print(f"MC Report: {run_dir}")
    print(f"Episodes: {n} | terminated={n_term} truncated={n_trunc}")
    print("-" * 80)
    print(f"Return_pct:      {_fmt(s_ret, pct=True)}")
    print(f"Equity:          {_fmt(s_eq, pct=False)}")
    print(f"MaxDrawdown_pct: {_fmt(s_dd, pct=False)}  (already in % units from env)")
    print(f"ProfitFactor:    {_fmt(s_pf, pct=False)}")
    print(f"Trades_opened:   {_fmt(s_tr, pct=False)}")
    print(f"Fee_total:       {_fmt(s_fee, pct=False)}")
    print("-" * 80)
    if np.isfinite(p_loss):
        print(f"P(return<0):     {p_loss*100:.2f}%")
    print(f"VaR@{args.alpha:.2f}:        {varcvar['var']*100:.4f}% (on return_pct)")
    print(f"CVaR@{args.alpha:.2f}:       {varcvar['cvar']*100:.4f}% (avg of worst tail)")
    print("-" * 80)
    print(f"StopLoss fired:  total={sl_total}  days_with_SL>0={sl_days}/{n}")
    print(f"TakeProfit fired: total={tp_total}  days_with_TP>0={tp_days}/{n}")
    print("-" * 80)
    if worst is not None:
        print("Worst day:")
        print(
            f"  return={float(worst['return_pct'])*100:.4f}% "
            f"dd={float(worst.get('max_drawdown_pct', np.nan)):.4f} "
            f"pf={float(worst.get('profit_factor', np.nan)):.4f} "
            f"trades={float(worst.get('trades_opened', np.nan)):.0f} "
            f"file={worst.get('__src_file','')}, worker={worst.get('worker','')}, ep={worst.get('episode_index','')}"
        )
    if best is not None:
        print("Best day:")
        print(
            f"  return={float(best['return_pct'])*100:.4f}% "
            f"dd={float(best.get('max_drawdown_pct', np.nan)):.4f} "
            f"pf={float(best.get('profit_factor', np.nan)):.4f} "
            f"trades={float(best.get('trades_opened', np.nan)):.0f} "
            f"file={best.get('__src_file','')}, worker={best.get('worker','')}, ep={best.get('episode_index','')}"
        )
    print("=" * 80)

    report = {
        "run_dir": str(run_dir),
        "episodes": int(n),
        "terminated": int(n_term),
        "truncated": int(n_trunc),
        "summary": {
            "return_pct": s_ret,
            "equity": s_eq,
            "max_drawdown_pct": s_dd,
            "profit_factor": s_pf,
            "trades_opened": s_tr,
            "fee_total": s_fee,
        },
        "risk": {
            "p_return_negative": p_loss,
            f"var_{args.alpha}": varcvar["var"],
            f"cvar_{args.alpha}": varcvar["cvar"],
        },
        "stops": {
            "stop_loss_fired_total": sl_total,
            "take_profit_fired_total": tp_total,
            "days_with_stop_loss": sl_days,
            "days_with_take_profit": tp_days,
        },
    }

    if args.save:
        # save merged episodes (csv) + report json
        out_csv = run_dir / "episodes_merged.csv"
        out_json = run_dir / "report.json"
        df.to_csv(out_csv, index=False)
        with open(out_json, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[OK] wrote {out_csv}")
        print(f"[OK] wrote {out_json}")

# python mc_report.py /home/v/logdir/future_monte_carlo/mc/20251231_205606 --save

if __name__ == "__main__":
    main()
