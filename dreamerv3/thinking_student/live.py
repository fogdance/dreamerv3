"""Live paper trading (user 2026-10-30): the frozen policy trades a live env
(markets/live/*_env.yaml: live_mode, gym envs/entry_live.py) through
run.collect, exactly as the student of a PAPER exam. Nothing is learned.
The account starts empty at the latest DB bar; a stopped run resumes from
collect's state.pkl (decisions it catches up on send no orders).

  python -m dreamerv3.thinking_student.live --policy DIR --out DIR \\
      --env configs/thinking_student/markets/live/CZCE_FG701_env.yaml
"""
import argparse
from pathlib import Path


def main(argv=None):
  p = argparse.ArgumentParser()
  p.add_argument('--policy', required=True)
  p.add_argument('--env', required=True)
  p.add_argument('--out', required=True)
  p.add_argument('--chunk', type=int, default=30, help='decisions per saved chunk')
  p.add_argument('--device')
  args = p.parse_args(argv)
  from .run import collect, load_policy
  from gym_trading_env.research import ts_v1_corpus as C
  from gym_trading_env.research.ts_v1_account import StudentEnv
  agent, meta = load_policy(args.policy, args.device)
  collect(agent, StudentEnv(C.native_env(Path(args.env).resolve())), Path(args.out) / 'collect',
          10 ** 9, 'policy', args.chunk, meta['policy'])


if __name__ == '__main__':
  main()
