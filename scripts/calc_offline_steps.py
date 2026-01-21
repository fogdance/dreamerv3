#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
根据 offline 数据量估算 DreamerV3 offline_pretrain 需要的 run.steps。

公式：
    run.steps ≈ UTD * kept_steps / (batch_size * batch_length)

参数：
  - kept_steps: 过滤后 replay 里保留的总步数（脚本 filter_replay_by_reward 最后一行打印的 kept_steps）
  - batch_size, batch_length: 和 offline_pretrain 配置里的保持一致
  - utd: 想要的 update-to-data ratio，可以给多个值，方便对比

UTD ≈ 5–10：比较保守，不太容易严重过拟合
UTD ≈ 20–50：更激进一点，但对 50M 大模型也还算合理
超过 100：就要非常小心了，除非你有很强的正则 / 数据增强 / 早停  
"""

import argparse


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--kept_steps", type=float, required=True,
                      help="过滤后保留下来的总环境步数 (Total kept steps)")
  parser.add_argument("--batch_size", type=int, required=True)
  parser.add_argument("--batch_length", type=int, required=True)
  parser.add_argument("--utd", type=float, nargs="+", default=[5, 10, 20, 30, 50],
                      help="希望尝试的 UTD 列表，比如: --utd 10 20 30")

  args = parser.parse_args()

  batch_steps = args.batch_size * args.batch_length
  print("====================================================")
  print(f"kept_steps     = {args.kept_steps}")
  print(f"batch_size     = {args.batch_size}")
  print(f"batch_length   = {args.batch_length}")
  print(f"batch_steps    = batch_size * batch_length = {batch_steps}")
  print("====================================================")
  for u in args.utd:
    run_steps = u * args.kept_steps / batch_steps
    print(f"UTD = {u:6.2f}  =>  run.steps ≈ {run_steps:10.1f}  (建议取整: {int(run_steps)})")

# python scripts/calc_offline_steps.py \
#   --kept_steps 892825 \
#   --batch_size 32 \
#   --batch_length 64 \
#   --utd 10 20 30


if __name__ == "__main__":
  main()
