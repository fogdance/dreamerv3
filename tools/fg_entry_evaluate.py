"""Evaluate a formal Dreamer checkpoint and three baselines on identical FG data."""
import argparse
from pathlib import Path
import numpy as np
import elements
from ruamel import yaml
from dreamerv3.main import make_agent
from embodied.run.action_mask_guard import require_formal_actor
from gym_trading_env.research.fg_entry_v1 import run_policy, write_json


def main():
  parser=argparse.ArgumentParser(__doc__)
  parser.add_argument('--model-config',required=True)
  parser.add_argument('--checkpoint',required=True)
  parser.add_argument('--env-config',required=True)
  parser.add_argument('--output',required=True)
  parser.add_argument('--seed',type=int,default=0)
  args=parser.parse_args()
  config=elements.Config(yaml.YAML(typ='safe').load(Path(args.model_config).read_text()))
  if config.task not in ('gymnasium_FixedDirectionEntryEnv-v0','gymnasium_FGEntryDatasetEnv-v0'):
    raise ValueError('FG evaluation requires the FG two-action profile')
  config=config.update({'task':'gymnasium_FixedDirectionEntryEnv-v0',
      'env.gymnasium.config_path':str(Path(args.env_config).resolve())})
  agent=make_agent(config)
  elements.checkpoint.load(args.checkpoint,dict(agent=agent.load))
  require_formal_actor(agent,'FG entry comparison')
  carry=[agent.init_policy(1)]
  first=[True]
  def policy(obs):
    batch={k:np.asarray(v)[None] for k,v in obs.items()}
    batch.update(reward=np.zeros(1,np.float32),is_first=np.array(first),
                 is_last=np.zeros(1,bool),is_terminal=np.zeros(1,bool))
    carry[0],actions,_=agent.policy(carry[0],batch,mode='eval')
    first[0]=False
    return int(actions['action'][0])
  root=Path(args.output)
  results={}
  for strategy in ('always_add','random_add','rule_add'):
    results[strategy]=run_policy(args.env_config,root/strategy,strategy=strategy,seed=args.seed)
  results['dreamerv3']=run_policy(args.env_config,root/'dreamerv3',policy=policy,seed=args.seed)
  report=results['dreamerv3']
  decisions=report['legal_add_decisions']
  frequency=1-report['voluntary_waits']/decisions if decisions else 0
  results['matched_random_add']=run_policy(args.env_config,root/'matched_random_add',
      strategy='random_add',seed=args.seed,random_probability=frequency)
  write_json(root/'comparison.json',dict(results=results,matched_add_probability=frequency,
      checkpoint=str(Path(args.checkpoint).resolve()),
      timing_ability_verdict='REQUIRES_OUT_OF_SAMPLE_MULTI_SEED_COMPARISON'))
  print(root/'comparison.json')


if __name__=='__main__':
  main()
