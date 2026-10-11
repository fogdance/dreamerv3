"""Later D5 only: independent causal DEV using original execution components."""
from pathlib import Path
import pickle

from embodied.run.fg_s3a_checkpoint import load_source,audit_bc
from gym_trading_env.research.fg_replay import (check_files,collect_trace,digest,reconcile_trace,signed,write_json,SINGLE3_SCHEMA)
from gym_trading_env.research.fg_single3 import load_env
from gym_trading_env.research.fg_single3_evaluation import (ActorPolicy,analyze_trace,embodied_runtime,require_dev)
from gym_trading_env.research.fg_single3_replay import Single3BehaviorPolicy


def evaluate(source,labels,prepared,output,strategy,*,checkpoint=None,device='cpu'):
    if source['diagnostic']:raise ValueError('Formal DEV cannot use diagnostic random models')
    prepared=Path(prepared);dev=require_dev(prepared/'dev_manifest.json')
    row=dev['rows'][0];root=Path(__file__).resolve().parents[2]
    output=Path(output)
    if output.exists():raise FileExistsError('DEV output must be a new independent directory')
    agent=None;model_hash=None
    if strategy in ['original','bc']:
        agent=load_source(source,output,device=device)
        model_hash=source['manifest_sha256']
        if strategy=='bc':
            if not checkpoint:raise ValueError('BC DEV requires --checkpoint')
            m=audit_bc(checkpoint)
            if (m['identity']['source_manifest_sha256']!=model_hash or m['identity']['diagnostic']
                    or m['identity']['label_manifest_sha256']!=digest(Path(labels)/'manifest.json')):
                raise ValueError('BC evaluation source/label identity mismatch')
            state=pickle.loads((Path(checkpoint)/'state.pkl').read_bytes())
            agent.load_exact(state['agent']);agent.sync_policy_from_train(reset_counter=True)
            model_hash=digest(Path(checkpoint)/'manifest.json')
        policy_factory=lambda _:ActorPolicy(agent)
        spec=dict(id=strategy,kind='neural_actor',seed_components=[0])
    else:
        kinds=dict(always_wait='always_wait',always_add='always_add',random_add='probabilistic',delay='single_delay')
        if strategy not in kinds:raise ValueError('Unknown causal baseline')
        policy_factory=Single3BehaviorPolicy
        spec=dict(id=strategy,kind=kinds[strategy],seed_components=[0],probability=.5,wait_minutes=15)
    factory=lambda:load_env(row,dev['frozen_inputs'])
    identity=dict(protocol='FG_S3A_DEV_V1',role='DEV',dev_hash=dev['content_hash'],
                  model_checkpoint_sha256=model_hash,conditional_direction_only=True)
    collect_trace(row['env_config'],spec,output,identity,str(root),env_factory=factory,
        protocol='fg_single3_v1',policy_factory=policy_factory,runtime_factory=embodied_runtime)
    reconciliation=reconcile_trace(output,str(root),env_factory=factory,schema=SINGLE3_SCHEMA,runtime_factory=embodied_runtime)
    if reconciliation['status']!='PASS':raise RuntimeError('DEV original-Env reconciliation failed')
    write_json(output/'reconciliation.json',signed(reconciliation))
    summary=analyze_trace(output,dev,strategy)
    check_files(dev['frozen_inputs'])
    result=signed(dict(status='PASS',identity=identity,summary=summary,
        oracle_labels_used=False,test_prices_accessed=False,
        files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}))
    write_json(output/'evaluation_receipt.json',result)
    return result
