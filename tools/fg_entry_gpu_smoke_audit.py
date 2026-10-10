"""Audit checkpoints created by our bounded GPU smoke, without further training."""
import json
import math
import pickle
from pathlib import Path
import sys
import numpy as np


def audit(root):
  root=Path(root)
  checkpoint=root/'ckpt'/(root/'ckpt/latest').read_text().strip()
  load=lambda name: pickle.loads((checkpoint/f'{name}.pkl').read_bytes())
  step,driver,model=load('step'),load('driver'),load('agent')
  account=driver['envs'][0]['env']['account']
  before=json.loads((root/'smoke_before_resume.json').read_text())
  assert step==30 and before['current_step']==19 and account['current_step']==29
  assert account['account_sequence']==before['account_sequence']==1
  for key in ('fills','ledger_entries','reward_events'):
    assert account[key][:len(before[key])]==before[key],key
  assert model['counters']['updates']>0
  assert all(np.isfinite(value).all() for value in model['params'].values())
  losses={key:value for line in (root/'metrics.jsonl').read_text().splitlines()
      for key,value in json.loads(line).items() if key.startswith('train/loss/')}
  assert losses and all(math.isfinite(value) for value in losses.values())
  result=dict(formal_training=False,environment_steps=step,
      updates=model['counters']['updates'],finite_loss_metrics=losses,
      financial_history_preserved=True,cursor_before=before['current_step'],
      cursor_after=account['current_step'],checkpoint=str(checkpoint))
  (root/'smoke_result.json').write_text(json.dumps(result,indent=2))
  print(json.dumps(result,indent=2))


if __name__=='__main__':
  audit(sys.argv[1])
