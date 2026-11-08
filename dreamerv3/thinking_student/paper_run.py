"""One paper run of frozen policies on several products (user 2026-11-08: "后续品种可能增加 ... 不然到时候
不好维护"): the PAPER exams of every policy on every contract of each product's paper pool (its product
profile, configs products/<code>.yaml), one paper report per product and, with --review, the review
results of the policies' formal run under the next free result ids. In place of copying run.sh.

  cd gym-trading-env && PYTHONPATH=.:src:../dreamerv3 python -m dreamerv3.thinking_student.paper_run \\
      --out DIR --strategy STRATEGY --products fg,sa --run FORMAL_RUN --policy pi_0005 --policy pi_0009 [--review]

Writes DIR/<product>/<policy>/<contract>/ (exam), DIR/<product>/paper_report.{json,md}, DIR/logs/ and
DIR/paper_run.json (commits, strategy, pools, review ids). A finished exam (exam.json) is kept, so a
rerun continues. Run it from the gym repo: the exams run there as in the earlier run.sh scripts.
"""
import argparse
import concurrent.futures as cf
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[2] / 'configs' / 'thinking_student'


def log(out, text):
  line = f'{time.strftime("%F %T")} {text}'
  print(line, flush=True)
  with (out / 'logs' / 'run.log').open('a') as f:
    f.write(line + '\n')


def commits():
  def head(path):
    return subprocess.run(['git', '-C', str(path), 'rev-parse', '--short', 'HEAD'], capture_output=True,
                          text=True).stdout.strip()
  def dirty(path):
    return bool(subprocess.run(['git', '-C', str(path), 'status', '--porcelain'], capture_output=True,
                               text=True).stdout.strip())
  dreamer = Path(__file__).resolve().parents[2]
  return dict(gym=head('.'), gym_dirty=dirty('.'), dreamerv3=head(dreamer), dreamerv3_dirty=dirty(dreamer))


def env():
  return dict(os.environ, PYTHONPATH='.:src:../dreamerv3', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
              XLA_PYTHON_CLIENT_PREALLOCATE='false')


def exam(job):
  out, product, tag, policy, pool, contract, strategy = job
  folder = out / product / tag / contract
  if (folder / 'exam.json').exists():
    return job, 0
  if folder.exists():
    subprocess.run(['rm', '-rf', str(folder)], check=True)
  with (out / 'logs' / f'{product}_{tag}_{contract}.log').open('w') as f:
    code = subprocess.call([sys.executable, '-u', '-m', 'dreamerv3.thinking_student.exam', '--policy', str(policy),
                            '--out', str(folder), '--paper-pool', str(pool), '--contract', contract,
                            '--strategy', str(strategy)], stdout=f, stderr=subprocess.STDOUT, env=env())
  return job, code


def next_result(run):
  """The next free review result id of the formal run (rNNNN after the highest of any policy)."""
  ids = [int(p.name[1:]) for p in (run / 'review' / 'student').glob('*/PAPER/r[0-9]*') if re.fullmatch(r'r\d+', p.name)]
  return f'r{max(ids, default=0) + 1:04d}'


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--out', required=True)
  p.add_argument('--strategy', required=True, help='paper strategy (one for every product; costs and ticks from each product profile)')
  p.add_argument('--products', required=True, help='product codes, e.g. fg,sa (configs products/<code>.yaml)')
  p.add_argument('--run', required=True, help='formal run of the policies (its policies/ and review/)')
  p.add_argument('--policy', action='append', required=True, help='a policy of RUN/policies, e.g. pi_0005')
  p.add_argument('--workers', type=int, default=10)
  p.add_argument('--review', action='store_true', help='register the review results (next free ids)')
  p.add_argument('--configs', default=str(CONFIGS))
  a = p.parse_args(argv)
  from gym_trading_env.envs.entry_product import load_profile
  out, run, strategy = Path(a.out).resolve(), Path(a.run).resolve(), Path(a.strategy).resolve()
  (out / 'logs').mkdir(parents=True, exist_ok=True)
  products = [x.strip().lower() for x in a.products.split(',') if x.strip()]
  profiles = {x: load_profile(Path(a.configs) / 'products' / f'{x}.yaml') for x in products}
  if any('paper_pool' not in v for v in profiles.values()):
    p.error('every product profile needs a paper_pool')
  policies = {name: run / 'policies' / name for name in a.policy}
  missing = [str(v) for v in policies.values() if not v.is_dir()]
  if missing:
    p.error(f'no such policies: {missing}')
  state = dict(command=sys.argv, commits=commits(), strategy=str(strategy), run=str(run),
               products={x: dict(profile=v['path'], pool=v['paper_pool']) for x, v in profiles.items()},
               policies={k: str(v) for k, v in policies.items()})
  log(out, f'start {json.dumps(state["commits"])}')
  jobs = [(out, x, tag, path, profiles[x]['paper_pool'], r['contract'], strategy)
          for x in products for tag, path in policies.items()
          for r in json.loads(Path(profiles[x]['paper_pool']).read_text())['files']]
  failed = []
  with cf.ThreadPoolExecutor(max(1, a.workers)) as pool:
    for (_, x, tag, _, _, contract, _), code in pool.map(exam, jobs):
      log(out, f'exam {x} {tag} {contract}: exit {code}')
      if code:
        failed.append(f'{x} {tag} {contract}')
  if failed:
    log(out, f'failed exams {failed}; no reports')
    return 1
  for x in products:
    folder = out / x
    with (out / 'logs' / f'{x}_report.log').open('w') as f:
      code = subprocess.call([sys.executable, '-m', 'dreamerv3.thinking_student.paper', '--run', str(folder),
                              '--pool', profiles[x]['paper_pool'], *policies], stdout=f, stderr=subprocess.STDOUT,
                             env=env())
    log(out, f'report {x}: exit {code}')
    if code:
      return 1
  if a.review:
    state['review'] = {}
    (run / 'review').mkdir(exist_ok=True)
    with (run / 'review' / '.paper_run.lock').open('w') as lock:
      fcntl.flock(lock, fcntl.LOCK_EX)  # one id per product, even with paper runs side by side
      for x in products:
        result = next_result(run)
        for tag in policies:
          with (out / 'logs' / 'review.log').open('a') as f:
            code = subprocess.call([sys.executable, '-m', 'gym_trading_env.research.review_export', 'paper',
                                    '--input', str(run), '--results', str(out / x / tag), '--policy', tag,
                                    '--result', result], stdout=f, stderr=subprocess.STDOUT, env=env())
          log(out, f'review {x} {tag} {result}: exit {code}')
          if code:
            return 1
        state['review'][x] = result
    with (out / 'logs' / 'review_validate.log').open('w') as f:
      code = subprocess.call([sys.executable, '-m', 'gym_trading_env.research.review_export', 'validate',
                              '--input', str(run)], stdout=f, stderr=subprocess.STDOUT, env=env())
    log(out, f'review validate: exit {code}')
    if code:
      return 1
  (out / 'paper_run.json').write_text(json.dumps(state, indent=1))
  log(out, 'done')
  return 0


if __name__ == '__main__':
  sys.exit(main())
