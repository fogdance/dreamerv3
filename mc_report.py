from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


# --- NEW: canonical keys we care about (episode-level) ---
EXTRA_EP_KEYS = [
    "total_trades",
    "winning_trades",
    "win_rate",
    "trades_closed",
    "opens_per_1000_steps",
    "fee_drag_ratio",
    "expectancy",
    "invalid_action",
    "invalid_action_total",
    "invalid_action_ratio",
]

CORE_KEYS = [
    "equity",
    "return_pct",
    "max_drawdown_pct",
    "profit_factor",
    "trades_opened",
    "fee_total",
    "stop_loss_fired",
    "take_profit_fired",
    "length",
    "score",
    "steps",
]

ALL_KEYS = CORE_KEYS + EXTRA_EP_KEYS


def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ensure_canonical_cols(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """
    Ensure df has canonical columns for keys.
    Accepts aliases like:
      - key
      - log/env/key
      - env/key
    If only alias exists, copy to canonical key.
    """
    alias_prefixes = ["log/env/", "env/", "log_env/", "log-env/"]

    for k in keys:
        if k in df.columns:
            continue

        found = None
        # exact alias matches
        for pref in alias_prefixes:
            ak = f"{pref}{k}"
            if ak in df.columns:
                found = ak
                break

        # sometimes flattened with dot notation
        if found is None:
            for pref in ["log.env.", "env."]:
                ak = f"{pref}{k}"
                if ak in df.columns:
                    found = ak
                    break

        if found is not None:
            df[k] = df[found]

    return df


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

    # NEW: normalize canonical columns (handles log/env/* aliases)
    out = _ensure_canonical_cols(out, ALL_KEYS)

    # Normalize numeric columns (coerce errors -> NaN)
    for c in ALL_KEYS:
        if c in out.columns:
            out[c] = _coerce_numeric(out[c])

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


def _safe_idx(df: pd.DataFrame, col: str, fn: str):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return None
    if fn == "min":
        return int(pd.to_numeric(df[col], errors="coerce").astype(float).idxmin())
    if fn == "max":
        return int(pd.to_numeric(df[col], errors="coerce").astype(float).idxmax())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str, help="MC run dir containing episodes_w*.jsonl and summary.json")
    ap.add_argument("--save", action="store_true", help="Save episodes_merged.csv and report.json into run_dir")
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

    # NEW: extra summaries
    s_total_trades = _summarize_series(df.get("total_trades", pd.Series(dtype=float)))
    s_win_trades = _summarize_series(df.get("winning_trades", pd.Series(dtype=float)))
    s_win_rate = _summarize_series(df.get("win_rate", pd.Series(dtype=float)))
    s_tr_closed = _summarize_series(df.get("trades_closed", pd.Series(dtype=float)))
    s_opens_1k = _summarize_series(df.get("opens_per_1000_steps", pd.Series(dtype=float)))
    s_fee_drag = _summarize_series(df.get("fee_drag_ratio", pd.Series(dtype=float)))
    s_exp = _summarize_series(df.get("expectancy", pd.Series(dtype=float)))
    s_invalid_total = _summarize_series(df.get("invalid_action_total", pd.Series(dtype=float)))
    s_invalid_ratio = _summarize_series(df.get("invalid_action_ratio", pd.Series(dtype=float)))

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
    worst_i = _safe_idx(df, "return_pct", "min")
    best_i = _safe_idx(df, "return_pct", "max")
    worst = df.loc[worst_i] if worst_i is not None else None
    best = df.loc[best_i] if best_i is not None else None

    # NEW: diagnostics around PF / fees
    pf = pd.to_numeric(df.get("profit_factor", pd.Series(dtype=float)), errors="coerce").astype(float)
    pf_valid = pf[np.isfinite(pf)]
    pf_valid_n = int(len(pf_valid))
    pf_none_n = int(n - pf_valid_n)

    total_trades = pd.to_numeric(df.get("total_trades", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(float)
    winning_trades = pd.to_numeric(df.get("winning_trades", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(float)

    # episodes where all close trades are wins (but PF may be None)
    all_wins = (total_trades > 0) & (winning_trades >= total_trades)
    all_wins_n = int(all_wins.sum())

    # fee-drag suspicion: all wins but return < 0 (net negative)
    ret = pd.to_numeric(df.get("return_pct", pd.Series(dtype=float)), errors="coerce").astype(float)
    fee_drag_loss = all_wins & (ret < 0)
    fee_drag_loss_n = int(fee_drag_loss.sum())

    # open without close anomaly
    tr_open = pd.to_numeric(df.get("trades_opened", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(float)
    tr_close = pd.to_numeric(df.get("trades_closed", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(float)
    open_no_close = (tr_open > 0) & (tr_close == 0)
    open_no_close_n = int(open_no_close.sum())

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
    print(f"Trades_closed:   {_fmt(s_tr_closed, pct=False)}")
    print(f"Total_trades:    {_fmt(s_total_trades, pct=False)}")
    print(f"Winning_trades:  {_fmt(s_win_trades, pct=False)}")
    print(f"Win_rate:        {_fmt(s_win_rate, pct=False)}")
    print(f"Expectancy:      {_fmt(s_exp, pct=False)}")
    print(f"Fee_total:       {_fmt(s_fee, pct=False)}")
    print(f"Fee_drag_ratio:  {_fmt(s_fee_drag, pct=False)}")
    print(f"Opens/1000steps:  {_fmt(s_opens_1k, pct=False)}")
    print(f"Invalid_total:   {_fmt(s_invalid_total, pct=False)}")
    print(f"Invalid_ratio:   {_fmt(s_invalid_ratio, pct=False)}")
    print("-" * 80)
    if np.isfinite(p_loss):
        print(f"P(return<0):     {p_loss*100:.2f}%")
    print(f"VaR@{args.alpha:.2f}:        {varcvar['var']*100:.4f}% (on return_pct)")
    print(f"CVaR@{args.alpha:.2f}:       {varcvar['cvar']*100:.4f}% (avg of worst tail)")
    print("-" * 80)
    print(f"StopLoss fired:  total={sl_total}  days_with_SL>0={sl_days}/{n}")
    print(f"TakeProfit fired: total={tp_total}  days_with_TP>0={tp_days}/{n}")
    print("-" * 80)
    print("Diagnostics:")
    print(f"  ProfitFactor valid: {pf_valid_n}/{n}  (None/NaN: {pf_none_n})")
    print(f"  All-wins episodes (winning_trades==total_trades>0): {all_wins_n}/{n}")
    print(f"  All-wins but return<0 (fee drag suspicion): {fee_drag_loss_n}/{n}")
    print(f"  trades_opened>0 but trades_closed==0 (anomaly): {open_no_close_n}/{n}")
    print("-" * 80)

    def _print_day(tag: str, row: pd.Series):
        r = float(row.get("return_pct", np.nan)) * 100.0
        dd = float(row.get("max_drawdown_pct", np.nan))
        pfv = float(row.get("profit_factor", np.nan))
        tr_o = float(row.get("trades_opened", np.nan))
        tr_c = float(row.get("trades_closed", np.nan))
        tt = float(row.get("total_trades", np.nan))
        wr = float(row.get("win_rate", np.nan))
        fee = float(row.get("fee_total", np.nan))
        fdr = float(row.get("fee_drag_ratio", np.nan))
        inv = float(row.get("invalid_action_total", np.nan))
        print(f"{tag} day:")
        print(
            f"  return={r:.4f}% dd={dd:.4f} pf={pfv:.4f} "
            f"open={tr_o:.0f} close={tr_c:.0f} total_trades={tt:.0f} win_rate={wr:.4f} "
            f"fee={fee:.4f} fee_drag={fdr:.4f} invalid_total={inv:.0f} "
            f"file={row.get('__src_file','')}, worker={row.get('worker','')}, ep={row.get('episode_index','')}"
        )

    if worst is not None:
        _print_day("Worst", worst)
    if best is not None:
        _print_day("Best", best)
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
            "trades_closed": s_tr_closed,
            "total_trades": s_total_trades,
            "winning_trades": s_win_trades,
            "win_rate": s_win_rate,
            "expectancy": s_exp,
            "fee_total": s_fee,
            "fee_drag_ratio": s_fee_drag,
            "opens_per_1000_steps": s_opens_1k,
            "invalid_action_total": s_invalid_total,
            "invalid_action_ratio": s_invalid_ratio,
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
        "diagnostics": {
            "profit_factor_valid_n": pf_valid_n,
            "profit_factor_none_n": pf_none_n,
            "all_wins_n": all_wins_n,
            "all_wins_but_return_negative_n": fee_drag_loss_n,
            "open_no_close_n": open_no_close_n,
        },
    }

    if args.save:
        out_csv = run_dir / "episodes_merged.csv"
        out_json = run_dir / "report.json"
        df.to_csv(out_csv, index=False)
        with open(out_json, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[OK] wrote {out_csv}")
        print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
