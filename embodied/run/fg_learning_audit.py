"""Observe real Actor/Critic changes without changing training objectives."""
import json
from pathlib import Path
import numpy as np


def selected(agent):
  data=agent.save()
  params={k:np.asarray(v) for k,v in data['params'].items()
          if '/pol/' in k or '/val/' in k or k.startswith(('pol/','val/'))}
  if not params:
    raise ValueError('No real Actor/Critic parameter keys found')
  return data['counters'],params


def snapshot_initial(agent, logdir):
  root=Path(logdir)
  if (root/'learning_initial.npz').exists():
    return
  counters,params=selected(agent)
  np.savez(root/'learning_initial.npz',**params)
  (root/'learning_initial_counters.json').write_text(json.dumps(counters))


def summarize_learning(agent, logdir, step):
  root=Path(logdir)
  counters,params=selected(agent)
  initial=np.load(root/'learning_initial.npz')
  baseline=json.loads((root/'learning_initial_counters.json').read_text())
  result=dict(env_interactions=step,updates=counters['updates']-baseline['updates'],parameters={})
  for group,name in [('pol','actor'),('val','critic')]:
    keys=[k for k in params if f'/{group}/' in k or k.startswith(group+'/')]
    finite=bool(keys) and all(np.isfinite(params[k]).all() for k in keys)
    delta=sum(float(np.square(params[k].astype(np.float64)-initial[k].astype(np.float64)).sum()) for k in keys)
    result['parameters'][name]=dict(finite=finite,l2_delta=float(np.sqrt(delta)),keys=len(keys))
  result['actor_gate_enabled']=bool(agent.get_avail_actor_enabled())
  result['parameter_changes_observed']=result['updates']>0 and all(v['finite'] and v['l2_delta']>0 for v in result['parameters'].values())
  result['actual_learning']=result['parameter_changes_observed'] and result['actor_gate_enabled']
  result['limitation']='Parameter movement alone does not prove profitable timing; disabled gate forbids a policy-learning claim.'
  (root/'learning_audit.json').write_text(json.dumps(result,indent=2))
