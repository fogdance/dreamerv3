#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
根据 episode 总 reward 过滤 replay：

支持两种模式：
1) 固定阈值：--reward_threshold X
   - 保留 episode_total_reward >= X 的 episode

2) Top X%：--top_percent P
   - 先遍历一遍所有 episode，统计 total_reward 分布
   - 计算出第 (100 - P) 分位数作为阈值
   - 第二遍按该阈值重建新的 replay 目录

注意：
- 只切在单个 chunk 内完整结束的 episode（跨 chunk 的尾巴丢弃）
"""

import importlib
import os
import pathlib
import sys
from functools import partial as bind

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import argparse
from pathlib import Path

import elements
import embodied
import numpy as np

from embodied.core import chunk as chunklib
from embodied.core import replay as replaylib


def iter_chunks(src_replay_dir: Path):
  """按文件名排序依次加载所有 chunk，yield Chunk 对象。"""
  files = sorted(src_replay_dir.glob("*.npz"))
  print(f"[filter] Found {len(files)} chunk files in {src_replay_dir}")

  for i, fn in enumerate(files):
    ch = chunklib.Chunk.load(fn, error="none")
    if ch is None:
      print(f"[filter] Skip corrupted chunk: {fn.name}")
      continue
    if not ch.data:
      print(f"[filter] Empty chunk (no data): {fn.name}")
      continue
    yield ch


def collect_episode_returns(src_replay_dir: str):
  """
  第一遍扫描：只统计每个完整 episode 的 total reward，返回一个 np.ndarray。
  规则和 build_good_replay 完全一致：只保留“完全落在单个 chunk 内”的 episode。
  """
  src_replay_dir = elements.Path(src_replay_dir)

  all_returns = []
  total_eps = 0

  for chunk_idx, ch in enumerate(iter_chunks(src_replay_dir)):
    data = ch.data
    T = ch.length

    reward = np.asarray(data["reward"], dtype=np.float32)
    is_first = np.asarray(data["is_first"], dtype=bool)
    is_last = np.asarray(data["is_last"], dtype=bool)

    assert reward.shape[0] == T
    assert is_first.shape[0] == T
    assert is_last.shape[0] == T

    ep_start = None

    for t in range(T):
      if is_first[t]:
        ep_start = t

      if ep_start is not None and is_last[t]:
        ep_end = t
        total_eps += 1

        ep_return = float(reward[ep_start:ep_end + 1].sum())
        all_returns.append(ep_return)

        ep_start = None

  all_returns = np.array(all_returns, dtype=np.float32)
  print("==========================================================")
  print(f"[filter] collect_episode_returns: total complete episodes = {total_eps}")
  if total_eps > 0:
    print(
      f"[filter] reward stats: "
      f"min={all_returns.min():.4f}, "
      f"max={all_returns.max():.4f}, "
      f"mean={all_returns.mean():.4f}"
    )
  print("==========================================================")
  return all_returns


def build_good_replay(
    src_replay_dir: str,
    dst_replay_dir: str | None,
    reward_threshold: float,
    replay_length: int = 65,
    chunksize: int = 1024,
    seed: int = 0,
):
  """
  从 src_replay_dir 读取所有 chunk，筛掉 episode 总 reward < 阈值 的 episode，
  用 Dreamer 的 Replay 写一套新的 replay 目录到 dst_replay_dir。
  """

  src_replay_dir = elements.Path(src_replay_dir)

  if dst_replay_dir is None:
    # 默认同级目录 good_replay
    parent = src_replay_dir.parent
    dst_replay_dir = parent / "good_replay"
  else:
    dst_replay_dir = elements.Path(dst_replay_dir)

  print("==========================================================")
  print(f"[filter] Source replay dir: {src_replay_dir}")
  print(f"[filter] Target replay dir: {dst_replay_dir}")
  print(f"[filter] Reward threshold: {reward_threshold}")
  print(f"[filter] replay.length (seq length): {replay_length}")
  print(f"[filter] chunksize: {chunksize}")
  print("==========================================================")

  # 新建一个“纯离线”的 Replay，只负责接收 add() 并 save()，online=False
  good_replay = replaylib.Replay(
      length=replay_length,
      capacity=None,
      directory=dst_replay_dir,
      chunksize=chunksize,
      online=False,
      selector=None,
      save_wait=True,
      name="good_replay",
      seed=seed,
  )

  total_eps = 0
  kept_eps = 0
  total_steps = 0
  kept_steps = 0

  # 这里只用一个 worker=0，把所有好 episode 全灌到同一个“流”里
  worker_id = 0

  for chunk_idx, ch in enumerate(iter_chunks(src_replay_dir)):
    data = ch.data
    T = ch.length

    # 三个关键字段（你示例里已经确认了）
    reward = np.asarray(data["reward"], dtype=np.float32)
    is_first = np.asarray(data["is_first"], dtype=bool)
    is_last = np.asarray(data["is_last"], dtype=bool)

    assert reward.shape[0] == T
    assert is_first.shape[0] == T
    assert is_last.shape[0] == T

    # 在当前 chunk 内按 is_first / is_last 切 episode
    ep_start = None

    for t in range(T):
      if is_first[t]:
        # 开启一个新的 episode
        ep_start = t

      if ep_start is not None and is_last[t]:
        # 找到一个完整落在 [ep_start, t] 的 episode
        ep_end = t
        total_eps += 1

        ep_return = float(reward[ep_start:ep_end + 1].sum())

        # 更新统计
        total_steps += (ep_end - ep_start + 1)

        if ep_return >= reward_threshold:
          kept_eps += 1
          kept_steps += (ep_end - ep_start + 1)

          # 把这个 episode 的 step 逐步灌进新的 Replay
          for tau in range(ep_start, ep_end + 1):
            # 每个 step 是一个 dict(key -> 单步 value)
            step = {k: v[tau] for k, v in data.items()}
            good_replay.add(step, worker=worker_id)

        # 关闭这个 episode（无论是否被保留）
        ep_start = None

    # 如果 ep_start 不为 None，说明这个 episode 跨 chunk 边界，我们在这个版本里直接丢弃尾巴部分
    # （只严格保留“完整地落在单个 chunk 内”的 episode）
    if ep_start is not None:
      # episode 跨 chunk 边界，本版本直接丢弃不处理
      pass

    if (chunk_idx + 1) % 100 == 0:
      print(
        f"[filter] Processed {chunk_idx + 1} chunks | "
        f"eps: {total_eps} (kept {kept_eps}) | "
        f"steps: {total_steps} (kept {kept_steps})"
      )

  print("==========================================================")
  print("[filter] Done scanning source replay.")
  print(
    f"[filter] Total episodes: {total_eps}, kept: {kept_eps} "
    f"({kept_eps / max(1, total_eps) * 100:.2f}%)"
  )
  print(
    f"[filter] Total steps: {total_steps}, kept: {kept_steps} "
    f"({kept_steps / max(1, total_steps) * 100:.2f}%)"
  )
  print("==========================================================")

  # 保存新的 replay
  print("[filter] Saving new replay chunks...")
  good_replay.save()
  stats = good_replay.stats()
  print(
    f"[filter] New replay stats: "
    f"items={stats['items']} chunks={stats['chunks']} "
    f"ram_gb={stats['ram_gb']:.2f}"
  )
  print("[filter] All done.")


def main():
  parser = argparse.ArgumentParser(
      description=(
        "Filter DreamerV3 replay by episode total reward "
        "and build a new replay directory.\n\n"
        "支持两种模式：\n"
        "  1) --reward_threshold X   固定阈值\n"
        "  2) --top_percent P        只保留 reward 排名前 P% 的 episode"
      )
  )
  parser.add_argument(
      "--src_replay",
      type=str,
      required=True,
      help="原始 replay 目录，例如 /data/logdir/xxx/replay",
  )
  parser.add_argument(
      "--dst_replay",
      type=str,
      default=None,
      help="新的 replay 目录，默认是 src_replay 同级目录下的 good_replay",
  )
  parser.add_argument(
      "--reward_threshold",
      type=float,
      default=None,
      help="episode 总 reward 阈值，>= 这个值的 episode 才会被保留；"
           "如果同时指定了 --top_percent，则优先使用 top_percent 模式计算阈值。",
  )
  parser.add_argument(
      "--top_percent",
      type=float,
      default=None,
      help="保留 reward 排名前 top_percent%% 的 episode，例如 10 表示 Top 10%%；"
           "内部会先遍历一次 replay 统计 reward 分布，再计算对应分位数作为阈值。",
  )
  parser.add_argument(
      "--replay_length",
      type=int,
      default=65,
      help="Replay.length（序列长度），要和训练/离线预训练 config 保持一致，"
           "通常是 batch_length + replay_context，比如 64 + 1 = 65",
  )
  parser.add_argument(
      "--chunksize",
      type=int,
      default=1024,
      help="新的 replay chunk 大小，建议沿用原来的 1024",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=0,
      help="Replay 内部采样器的随机种子（虽然这里只用来写 chunk，不采样）",
  )

  args = parser.parse_args()

  # 决定使用哪种模式：top_percent 优先
  if args.top_percent is not None:
    if not (0 < args.top_percent < 100):
      raise ValueError("--top_percent 必须在 (0, 100) 之间，例如 10 表示 Top 10%")

    returns = collect_episode_returns(args.src_replay)
    if returns.size == 0:
      raise RuntimeError(
        "[filter] 没有找到任何完整 episode（可能都是跨 chunk），无法根据 top_percent 计算阈值。"
      )

    # Top P% => 取第 (100 - P) 分位数作为阈值
    q = 100.0 - float(args.top_percent)
    threshold = float(np.percentile(returns, q))

    print("==========================================================")
    print(
      f"[filter] top_percent={args.top_percent}% => "
      f"use reward_threshold = percentile(q={q:.2f}) = {threshold:.6f}"
    )
    print("==========================================================")

    reward_threshold = threshold

  else:
    if args.reward_threshold is None:
      raise ValueError(
        "必须指定 --reward_threshold 或 --top_percent 其中一个；"
        "如果你想用 Top X% 模式，请用 --top_percent，例如 --top_percent 10"
      )
    reward_threshold = float(args.reward_threshold)

  build_good_replay(
      src_replay_dir=args.src_replay,
      dst_replay_dir=args.dst_replay,
      reward_threshold=reward_threshold,
      replay_length=args.replay_length,
      chunksize=args.chunksize,
      seed=args.seed,
  )

# python scripts/filter_replay_by_reward.py   --src_replay /data/logdir/50m_future_jm_0121_1/replay   --top_percent 10   --replay_length 65   --chunksize 1024

if __name__ == "__main__":
  main()
