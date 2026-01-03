# dreamerv3/scripts/monte_carlo.py
import collections
import json
import time
from functools import partial as bind

import elements
import embodied
import numpy as np


def _scalar(x, default=np.nan):
  if x is None:
    return default
  try:
    if isinstance(x, np.ndarray):
      if x.shape == ():
        return x.item()
      # safety: if accidentally non-scalar, take last element
      return float(np.asarray(x).reshape(-1)[-1])
    return x
  except Exception:
    return default


def _get(tran, key, default=np.nan):
  return _scalar(tran.get(key, None), default=default)


def monte_carlo(make_agent, make_env, make_logger, args):
  """
  Monte Carlo evaluation runner (no training, no replay, no DB).

  Outputs:
    {logdir}/mc/{run_id}/episodes_w{worker}.jsonl
    {logdir}/mc/{run_id}/summary.json
  """

  agent = make_agent()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step

  # ---- load checkpoint ----
  if args.from_checkpoint:
    regex = args.get('from_checkpoint_regex', None)
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=regex)))
  else:
    raise ValueError("MC requires --run.from_checkpoint (policy must be loaded)")

  # ---- MC config ----
  target_episodes = int(args.get("mc_episodes", 200))
  chunk_steps = int(args.get("mc_chunk_steps", 10))  # keep small to reduce overshoot
  policy_mode = args.get("mc_policy_mode", "eval")   # try eval; if your agent uses 'report', change it
  run_id = args.get("mc_run_id", time.strftime("%Y%m%d_%H%M%S"))

  outdir = logdir / "mc" / run_id
  outdir.mkdir(parents=True, exist_ok=True)

  # ---- per-worker episode state ----
  ep_len = collections.defaultdict(int)
  ep_score = collections.defaultdict(float)
  ep_index = collections.defaultdict(int)

  # ---- file handles (one file per worker => no lock) ----
  files = {}

  def _file(worker):
    if worker not in files:
      path = outdir / f"episodes_w{worker}.jsonl"
      files[worker] = open(str(path), "a", buffering=1)
    return files[worker]

  total_done = {"n": 0}

  # ---- driver ----
  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)

  driver.on_step(lambda tran, _: step.increment())

  def on_step(tran, worker):
    # episode bookkeeping
    if tran.get("is_first", False):
      ep_len[worker] = 0
      ep_score[worker] = 0.0

    ep_len[worker] += 1
    ep_score[worker] += float(_get(tran, "reward", 0.0))

    if tran.get("is_last", False):
      # terminated vs truncated: dreamerv3 convention usually uses is_terminal
      terminated = bool(_get(tran, "is_terminal", False))
      trunc_raw = tran.get("log/env/is_truncated", None)
      if trunc_raw is None:
        truncated = bool(tran.get("is_last", False) and not terminated)
      else:
        truncated = bool(np.asarray(trunc_raw).item())

      row = {
        "run_id": run_id,
        "worker": int(worker),
        "episode_index": int(ep_index[worker]),
        "checkpoint": str(args.from_checkpoint),
        "policy_mode": str(policy_mode),
        "steps": int(step),

        # episode stats
        "length": int(ep_len[worker]),
        "score": float(ep_score[worker]),
        "terminated": terminated,
        "truncated": truncated,
        "log_is_truncated": None if trunc_raw is None else bool(np.asarray(trunc_raw).item()),

        # ---- metrics from env info (your env already emits log/env/*) ----
        "equity": float(_get(tran, "log/env/equity", np.nan)),
        "return_pct": float(_get(tran, "log/env/return_pct", np.nan)),
        "max_drawdown_pct": float(_get(tran, "log/env/max_drawdown_pct", np.nan)),
        "profit_factor": float(_get(tran, "log/env/profit_factor", np.nan)),
        "trades_opened": float(_get(tran, "log/env/trades_opened", np.nan)),
        "fee_total": float(_get(tran, "log/env/fee_total", np.nan)),

        # counters (these are top-level in your env info)
        "stop_loss_fired": int(_get(tran, "stop_loss_fired", 0)),
        "take_profit_fired": int(_get(tran, "take_profit_fired", 0)),
      }

      _file(worker).write(json.dumps(row, ensure_ascii=False) + "\n")
      ep_index[worker] += 1
      total_done["n"] += 1

  driver.on_step(on_step)

  # ---- run loop ----
  print(f"[MC] start run_id={run_id} target_episodes={target_episodes} envs={args.envs} mode={policy_mode}")
  policy = lambda *x: agent.policy(*x, mode=policy_mode)

  driver.reset(agent.init_policy)
  while total_done["n"] < target_episodes:
    driver(policy, steps=chunk_steps)

  # ---- close files ----
  for f in files.values():
    try:
      f.close()
    except Exception:
      pass

  # ---- build summary (read jsonl back; 2000 rows is tiny) ----
  import glob
  rows = []
  for p in glob.glob(str(outdir / "episodes_w*.jsonl")):
    with open(p, "r") as f:
      for line in f:
        line = line.strip()
        if line:
          rows.append(json.loads(line))

  def summarize(key):
    xs = np.array([r.get(key, np.nan) for r in rows], dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
      return {"n": 0}
    q05, q50, q95 = np.quantile(xs, [0.05, 0.5, 0.95])
    out = {
      "n": int(xs.size),
      "mean": float(xs.mean()),
      "std": float(xs.std(ddof=1)) if xs.size >= 2 else 0.0,
      "p05": float(q05),
      "p50": float(q50),
      "p95": float(q95),
      "min": float(xs.min()),
      "max": float(xs.max()),
    }
    return out

  # VaR/CVaR for return_pct (5%)
  rets = np.array([r.get("return_pct", np.nan) for r in rows], dtype=np.float64)
  rets = rets[np.isfinite(rets)]
  if rets.size:
    var5 = float(np.quantile(rets, 0.05))
    cvar5 = float(rets[rets <= var5].mean()) if np.any(rets <= var5) else float("nan")
    p_loss = float((rets < 0).mean())
  else:
    var5, cvar5, p_loss = float("nan"), float("nan"), float("nan")

  summary = {
    "run_id": run_id,
    "checkpoint": str(args.from_checkpoint),
    "target_episodes": int(target_episodes),
    "collected_episodes": int(len(rows)),
    "policy_mode": str(policy_mode),
    "metrics": {
      "return_pct": summarize("return_pct"),
      "equity": summarize("equity"),
      "max_drawdown_pct": summarize("max_drawdown_pct"),
      "profit_factor": summarize("profit_factor"),
      "trades_opened": summarize("trades_opened"),
      "fee_total": summarize("fee_total"),
    },
    "risk": {
      "VaR_5_return_pct": var5,
      "CVaR_5_return_pct": cvar5,
      "P_return_negative": p_loss,
    },
  }

  with open(str(outdir / "summary.json"), "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

  print(f"[MC] done. wrote: {outdir}/episodes_w*.jsonl and summary.json")
  logger.close()
