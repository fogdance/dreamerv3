"""One isolated original Quality Actor/critic update; never installs candidate state."""
from __future__ import annotations

import copy
from collections import deque
import datetime
import decimal
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys

import numpy as np

PROTOCOL = 'FG_S3_E_QUALITY_ISOLATED_RL_PROBE_V1'
MAX_CANDIDATE_UPDATES = 2
_EXECUTED = {}


def state_digest(value):
    """Typed byte digest for the existing full native state, including RNG/slots."""
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,np.ndarray):
            h.update(b'array'+str(x.dtype).encode()+repr(x.shape).encode());h.update(x.tobytes())
        elif isinstance(x,np.generic):visit(np.asarray(x))
        elif isinstance(x,dict):
            h.update((type(x).__module__+'.'+type(x).__qualname__).encode()+b':dict')
            if hasattr(x,'default_factory'):
                factory=x.default_factory
                visit(None if factory is None else factory.__module__+'.'+factory.__qualname__)
            for key in sorted(x,key=lambda k:(type(k).__name__,str(k))):visit(key);visit(x[key])
        elif isinstance(x,(list,tuple,deque)):
            h.update(type(x).__name__.encode()+str(len(x)).encode())
            for item in x:visit(item)
        elif isinstance(x,(set,frozenset)):
            visit(sorted(x,key=lambda k:(type(k).__name__,str(k))))
        elif isinstance(x,bytes):h.update(b'bytes'+str(len(x)).encode()+x)
        elif x is None:h.update(b'none')
        elif type(x) is bool:h.update(b'bool'+bytes([x]))
        elif type(x) is int:h.update(b'int'+str(x).encode()+b';')
        elif type(x) is float:h.update(b'float'+struct.pack('!d',x))
        elif isinstance(x,str):h.update(b'str'+str(len(x.encode())).encode()+x.encode())
        elif isinstance(x,(datetime.datetime,datetime.date,datetime.time)):
            visit(type(x).__name__);visit(x.isoformat());visit(getattr(x,'fold',0))
        elif isinstance(x,decimal.Decimal):visit('Decimal');visit(x.as_tuple())
        elif type(x).__name__=='UUID' and hasattr(x,'value'):
            visit('native_uuid');visit(x.value)
        elif hasattr(x,'__dict__'):
            visit(type(x).__module__+'.'+type(x).__qualname__);visit(vars(x))
        else:raise TypeError('Unsupported native state digest: '+str(type(x)))
    visit(value)
    return h.hexdigest()


@dataclass(frozen=True)
class NumericProbe:
    report: dict


def require_numeric_probe(probe):
    registered=_EXECUTED.get(id(probe))
    if (registered is None or registered[0] is not probe
            or registered[1]!=state_digest(probe.report)):
        raise ValueError('ONLINE_ACTUAL_ISOLATED_RL_PROBE_REQUIRED')
    return copy.deepcopy(probe.report)


def isolated_rl_probe(snapshot,settings,batch,*,directory,seed,controlled_small=False,context=None):
    """Only this actual child execution registers a numeric result in the parent."""
    from gym_trading_env.research.fg_s3_e_quality_course_preflight import _local_path
    from gym_trading_env.research.fg_replay import digest
    from gym_trading_env.research.fg_s3_e_quality_foundation import execution_identity
    root=_local_path(directory);root.mkdir(parents=True,exist_ok=False)
    payload=dict(protocol=PROTOCOL,agent=snapshot,settings=settings,batch=batch,seed=seed,
        controlled_small=controlled_small,context=context,code=execution_identity())
    input_path=root/'candidate_input.pkl'
    with input_path.open('xb') as stream:pickle.dump(payload,stream,protocol=5)
    source_root=Path(__file__).resolve().parents[2]
    gym_root=source_root.parent/'gym-trading-env'
    command=[sys.executable,'-B','-m','embodied.run.fg_s3_e_quality_online_probe',
        '--input',str(input_path),'--output',str(root/'candidate_report.json')]
    env={**os.environ,'PYTHONPATH':os.pathsep.join(map(str,(gym_root,gym_root/'src',source_root))),
        'JAX_PLATFORMS':'cpu','CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1',
        'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(root/'.matplotlib')}
    env.pop('LD_LIBRARY_PATH',None)
    with (root/'candidate.log').open('x') as stream:
        process=subprocess.run(command,cwd=source_root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
    if process.returncode:
        raise ValueError('ONLINE_ISOLATED_RL_PROBE_FAILED: '+str(root/'candidate.log'))
    report=json.loads((root/'candidate_report.json').read_text())
    if (report.get('protocol')!=PROTOCOL or report.get('input_sha256')!=digest(input_path)
            or report.get('code')!=payload['code'] or report.get('code')!=execution_identity()
            or report.get('context')!=context or report.get('source_agent_sha256')!=state_digest(snapshot)
            or report.get('batch_sha256')!=state_digest(batch)
            or type(report.get('actual_candidate_updates')) is not int
            or not 1<=report['actual_candidate_updates']<=MAX_CANDIDATE_UPDATES
            or report.get('ready') is not True):
        raise ValueError('ONLINE_ISOLATED_PROBE_EXECUTION_BINDING_MISMATCH')
    report.update(command=command,exit_code=process.returncode,
        candidate_report_sha256=digest(root/'candidate_report.json'),log_sha256=digest(root/'candidate.log'))
    result=NumericProbe(report);_EXECUTED[id(result)]=(result,state_digest(report))
    return result


def _candidate(input_path,output_path):
    from gym_trading_env.research.fg_s3_e_quality_actor import create_agent
    from gym_trading_env.research.fg_s3_v2_gpu_check import gpu_batch
    from gym_trading_env.research.fg_s3_e_quality_foundation import execution_identity
    from gym_trading_env.research.fg_replay import digest
    from dreamerv3.fg_s3a_actor import parameter_digests
    from .fg_checkpoint import host
    payload=pickle.loads(input_path.read_bytes())
    if payload['protocol']!=PROTOCOL or payload['code']!=execution_identity():
        raise ValueError('ONLINE_PROBE_CODE_OR_PROTOCOL_CHANGED')
    agent=create_agent(payload['settings'],output_path.parent/'candidate_model',
        controlled_small=payload['controlled_small'],execution_mode='ONLINE_RL_CONTROLLED')
    agent.load_exact(payload['agent'])
    if state_digest(agent.save_exact())!=state_digest(payload['agent']):
        raise ValueError('ONLINE_PROBE_AGENT_CLONE_NOT_EXACT')
    agent.set_avail_actor_enabled(True)  # Isolated candidate only; no external authority.
    before=parameter_digests(agent.params)
    allowed=('quality_actor/','val/','slowval/','slowval_count/','retnorm/','valnorm/','advnorm/')
    def permitted(key):
        return (key.startswith(allowed) or agent.model.f3_state_owner[key] in ('CLOCK','quality_actor/')
            or key.startswith('opt/') and 'val/' in key)
    update_records=[];ready=False
    for number in range(MAX_CANDIDATE_UPDATES):
        _,_,metrics=agent.train(agent.init_train(agent.config.batch_size),
            gpu_batch(agent,payload['batch'],payload['seed']+number))
        metrics={key:np.asarray(value).tolist() for key,value in host(metrics).items()}
        update_records.append(metrics)
        after=parameter_digests(agent.params)
        frozen=set(before)==set(after) and all(before[k]==after[k] for k in before if not permitted(k))
        actor=any(before[k]!=after[k] for k in before if k.startswith('quality_actor/'))
        critic=any(before[k]!=after[k] for k in before if k.startswith('val/'))
        valid=bool(metrics.get('f4/rl_active') and metrics.get('f3/finite') and metrics.get('f3/updated')
            and np.isfinite(metrics['opt/grad_norm']) and np.isfinite(metrics['f4/value_loss'])
            and metrics['f4/imag_invalid']==0 and frozen
            and int(agent.n_updates)==payload['agent']['counters']['updates']+number+1)
        ready=valid and actor and critic
        if ready or not valid:break
    report=dict(protocol=PROTOCOL,input_sha256=digest(input_path),code=execution_identity(),context=payload['context'],
        source_agent_sha256=state_digest(payload['agent']),batch_sha256=state_digest(payload['batch']),
        ready=ready,actual_candidate_updates=len(update_records),candidate_update_budget=MAX_CANDIDATE_UPDATES,
        actor_changed=actor,value_changed=critic,frozen_equal=frozen,update_records=update_records,
        before=before,after=after,metrics=metrics,candidate_discarded=True,live_rl_updates=0)
    with output_path.open('x') as stream:json.dump(report,stream,sort_keys=True,indent=2);stream.write('\n')
    if not ready:
        print(json.dumps(dict(ready=False,actor_changed=actor,value_changed=critic,frozen_equal=frozen,
            source_updates=payload['agent']['counters']['updates'],actual_updates=int(agent.n_updates),
            metrics={k:metrics[k] for k in ('f4/rl_active','f3/finite','f3/updated','opt/grad_norm','f4/value_loss','f4/imag_invalid')},
            unexpected_changed=[k for k in before if not permitted(k) and before[k]!=after[k]])))
        raise ValueError('ONLINE_CANDIDATE_ACTOR_CRITIC_OR_MASK_FAILED')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    _candidate(args.input,args.output)
