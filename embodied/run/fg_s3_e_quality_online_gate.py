"""Independent evidence gates; numeric DEV permission is never formal admission."""
from __future__ import annotations

import copy
import math

from gym_trading_env.envs.entry_contract import fingerprint

NAMES=('COLLECTION_VALID','SUPERVISED_STABLE','WM_VALID_FOR_RL','RL_NUMERIC_READY')
_REVIEWS={}


class OnlineGates:
    protocol='FG_S3_E_QUALITY_ONLINE_GATES_V3'

    def __init__(self,identity,mask_policy,*,wm_policy=None,collection_root=None):
        import elements
        from .action_mask_warmup import ActionMaskWarmup
        from .fg_s3_gate import gate_policy
        if mask_policy['protocol']!='FG_SINGLE3_MASK_GATE_V1':raise ValueError('ONLINE_ORIGINAL_FORMAL_MASK_POLICY_REQUIRED')
        checked=gate_policy(dict(agent=dict(avail_threshold=mask_policy['classification_threshold']),
            action_mask_warmup=mask_policy['config']),diagnostic=False,
            source_opening=mask_policy['source_opening'])
        if checked!=mask_policy:raise ValueError('ONLINE_MASK_POLICY_NOT_CANONICAL')
        self.policy=copy.deepcopy(mask_policy);self.wm_policy=copy.deepcopy(wm_policy)
        if wm_policy is not None and (set(wm_policy)!={'max_reward_mae','max_continue_mae'} or
            any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in wm_policy.values())):
            raise ValueError('ONLINE_WM_CALIBRATION_POLICY_INVALID')
        self.mask=ActionMaskWarmup.for_single3_1m(elements.Config(mask_policy['config']),False)
        if self.mask.opening_thresholds!=mask_policy['opening']['thresholds']:
            raise ValueError('ONLINE_MASK_POLICY_CHANGED')
        self.identity=fingerprint(dict(protocol=self.protocol,execution=identity,mask=mask_policy,wm=wm_policy))
        self.ready=dict.fromkeys(NAMES,False);self.evidence={};self.samples=set()
        self.exposure=dict(updates=0,add=0,wait=0,rejections=0)
        self.wm_identity=None;self.formal_admitted=False
        self.collection_root=collection_root;self.run_identity=identity

    def collection(self,collector,evidence):
        from gym_trading_env.research.fg_s3_e_quality_online_evidence import (
            require_account_execution,portable_account_report,prefix_identity)
        report=require_account_execution(evidence);receipt=report['receipt']
        legal=all(not r['action'] or r['legal_add'] for r in collector.rows)
        teacher_free=not collector.teacher_enabled and collector.policy.beta==0 and not collector.query_seconds
        good=bool(collector.counts['env_steps'] and legal and teacher_free
            and receipt['run_identity']==self.run_identity and receipt['prefix']==prefix_identity(collector)
            and receipt['reconciliation']['status']=='PASS')
        self.ready['COLLECTION_VALID']=good
        if good:
            self.evidence['collection']=portable_account_report(evidence,self.collection_root)
        return good

    def collection_advanced(self,market_step):
        current=self.evidence.get('collection',{}).get('receipt',{}).get('verified_through_market_step',-1)
        self.ready['COLLECTION_VALID']=bool(current==market_step)

    def collection_through(self):
        return self.evidence.get('collection',{}).get('receipt',{}).get('verified_through_market_step',0)

    def supervised(self,metrics,*,frozen_equal):
        import numpy as np
        good=bool(metrics['f3/finite']) and frozen_equal and np.isfinite(metrics['opt/loss']) and np.isfinite(metrics['opt/grad_norm'])
        if not good:self.exposure['rejections']+=1
        if good and metrics['f3/updated']:
            self.exposure['updates']+=1
            self.exposure['add']+=int(metrics['f3/natural_add'])
            self.exposure['wait']+=int(metrics['f3/natural_wait'])
        self.ready['SUPERVISED_STABLE']=bool(all(self.exposure[k]>0 for k in ('updates','add','wait')) and not self.exposure['rejections'])
        self.evidence['supervised']=copy.deepcopy(self.exposure)

    def calibration(self,record):
        from gym_trading_env.research.fg_s3_e_quality_online_checks import require_calibration
        r=require_calibration(record)
        if self.wm_identity is not None and r['wm_identity']!=self.wm_identity:
            self.ready['WM_VALID_FOR_RL']=False;self.formal_admitted=False
            raise ValueError('ONLINE_CALIBRATION_MODEL_IDENTITY_CHANGED')
        self.wm_identity=r['wm_identity']
        self.samples.update(map(tuple,r['native_owner_steps']) if 'native_owner_steps' in r
            else ((r['source_identity'],x) for x in r['native_stepids']))
        self.mask.update(r['metrics'],len(self.samples),allow_activation=False)
        good=bool(self.wm_policy is not None and r['finite'] and
            r['reward_mae']<=self.wm_policy['max_reward_mae'] and r['continue_mae']<=self.wm_policy['max_continue_mae']
            and self.mask.ready(len(self.samples)) and self.mask.consecutive_passes>=self.mask.required_passes)
        self.ready['WM_VALID_FOR_RL']=good
        self.evidence['calibration']=r
        return good

    def numeric(self,probe):
        from .fg_s3_e_quality_online_probe import require_numeric_probe
        record=require_numeric_probe(probe)
        context=record.get('context') or {}
        if self.ready['WM_VALID_FOR_RL'] and (context.get('run_identity')!=self.run_identity
                or context.get('wm_identity')!=self.wm_identity
                or context.get('mask_identity')!=fingerprint(self.policy)):
            raise ValueError('ONLINE_NUMERIC_PROBE_WM_OR_RUN_MISMATCH')
        metrics=record['metrics']
        good=self._numeric_valid(metrics,actor_changed=record['actor_changed'],
            value_changed=record['value_changed'],frozen_equal=record['frozen_equal'])
        self.ready['RL_NUMERIC_READY']=good
        self.evidence['numeric']={key:copy.deepcopy(record[key]) for key in (
            'protocol','source_agent_sha256','batch_sha256','context','code','input_sha256',
            'candidate_report_sha256','actual_candidate_updates','actor_changed','value_changed',
            'frozen_equal','ready','metrics','before','after','candidate_discarded','live_rl_updates')}
        return good

    @staticmethod
    def _numeric_valid(metrics,*,actor_changed,value_changed,frozen_equal):
        import numpy as np
        return bool(metrics.get('f4/rl_active',False) and metrics['f3/finite'] and metrics['f3/updated']
            and np.isfinite(metrics['opt/grad_norm']) and np.isfinite(metrics['f4/value_loss'])
            and metrics['f4/imag_invalid']==0 and actor_changed and value_changed and frozen_equal)

    def observe_rl(self,metrics,*,frozen_equal):
        # Monitoring a live update cannot bootstrap technical readiness.
        if not self._numeric_valid(metrics,actor_changed=True,value_changed=True,frozen_equal=frozen_equal):
            self.ready['RL_NUMERIC_READY']=False;self.formal_admitted=False
            raise ValueError('ONLINE_LIVE_RL_NUMERIC_OR_OWNER_VIOLATION')

    def reset_technical(self):
        import elements
        from .action_mask_warmup import ActionMaskWarmup
        self.mask=ActionMaskWarmup.for_single3_1m(elements.Config(self.policy['config']),False)
        self.samples.clear();self.wm_identity=None;self.formal_admitted=False
        for key,name in (('WM_VALID_FOR_RL','calibration'),('RL_NUMERIC_READY','numeric')):
            self.ready[key]=False;self.evidence.pop(name,None)

    def review_rl(self,preparation,path,approved_sha256,*,authorized=False,expected_contract=None):
        """Explicit operator-reviewed digest, independent of technical Gates.

        A file saying PASS is not authority. The caller must supply its reviewed
        digest and separate RL authorization; the native F3 proof remains required.
        This method does not produce approvals or alter the technical thresholds.
        """
        import json
        import re
        from gym_trading_env.research.fg_replay import digest
        from gym_trading_env.research.fg_s3_e_quality_course_preflight import _local_path
        from gym_trading_env.research.fg_s3_e_quality_actor import require_training_authority
        if authorized is not True:raise ValueError('ONLINE_SEPARATE_RL_AUTHORIZATION_REQUIRED')
        require_training_authority(preparation,authorized=True)
        if (not all(self.ready[k] for k in ('SUPERVISED_STABLE','WM_VALID_FOR_RL','RL_NUMERIC_READY'))
                or self.collection_through()<=0):raise ValueError('ONLINE_RL_GATE_CLOSED')
        if expected_contract is None:raise ValueError('ONLINE_SEALED_REVIEW_CONTRACT_REQUIRED')
        path=_local_path(path)
        if (not isinstance(approved_sha256,str) or re.fullmatch('[0-9a-f]{64}',approved_sha256) is None
                or digest(path)!=approved_sha256):raise ValueError('ONLINE_REVIEWED_RL_DIGEST_REQUIRED')
        review=json.loads(path.read_text())
        fields={'protocol','verdict','reviewer_role','review_id','packet_path','packet_sha256','review_contract'}
        if (type(review) is not dict or set(review)!=fields
                or review['protocol']!='FG_S3_E_QUALITY_FORMAL_RL_REVIEW_V2'
                or review['verdict']!='APPROVED_FOR_CONTROLLED_RL'
                or review['reviewer_role']!='RESEARCH_OPERATOR'
                or not isinstance(review['review_id'],str) or not review['review_id'].strip()
                or review['review_contract']!=expected_contract):
            raise ValueError('ONLINE_RL_REVIEW_BINDING_MISMATCH')
        packet_path=_local_path(review['packet_path'])
        if digest(packet_path)!=review['packet_sha256']:raise ValueError('ONLINE_RL_REVIEW_PACKET_CHANGED')
        packet=json.loads(packet_path.read_text())
        if (packet.get('protocol')!='FG_S3_E_QUALITY_RL_REVIEW_PACKET_V1'
                or packet.get('status')!='AWAIT_OPERATOR_REVIEW'
                or packet.get('review_contract')!=expected_contract):
            raise ValueError('ONLINE_RL_REVIEW_PACKET_BINDING_MISMATCH')
        token=object();_REVIEWS[token]=(self,copy.deepcopy(expected_contract),approved_sha256,
            dict(path=str(path),packet_path=str(packet_path),packet_sha256=review['packet_sha256']))
        return token

    def require_rl(self,preparation,*,authorized=False,review=None):
        from gym_trading_env.research.fg_s3_e_quality_actor import require_training_authority
        blockers=[key for key in ('SUPERVISED_STABLE','WM_VALID_FOR_RL','RL_NUMERIC_READY') if not self.ready[key]]
        if self.collection_through()<=0:blockers.append('COLLECTION_PREFIX_VERIFIED')
        if blockers:raise ValueError('ONLINE_RL_GATE_CLOSED: '+','.join(blockers))
        # Reuse the in-process native proof/authorization mechanism, not a
        # JSON "PASS" or the old bundled ONLINE_ADAPT flags.
        require_training_authority(preparation,authorized=authorized)
        reviewed=_REVIEWS.get(review)
        # The approved prefix remains a prefix after native collection grows;
        # current COLLECTION_VALID stays false until another actual reconcile.
        # The explicit review permits guarded native continuation and requires
        # final full-account verification. It never certifies an unverified tail.
        if (not reviewed or reviewed[0] is not self or reviewed[1]['wm_identity']!=self.wm_identity
                or reviewed[1]['gate_identity']!=self.identity
                or reviewed[1]['mask_identity']!=fingerprint(self.policy)):
            self.formal_admitted=False
            raise ValueError('ONLINE_INDEPENDENT_RL_RESEARCH_REVIEW_REQUIRED')
        from gym_trading_env.research.fg_replay import digest
        from gym_trading_env.research.fg_s3_e_quality_course_preflight import _local_path
        if (digest(_local_path(reviewed[3]['path']))!=reviewed[2]
                or digest(_local_path(reviewed[3]['packet_path']))!=reviewed[3]['packet_sha256']):
            self.formal_admitted=False
            raise ValueError('ONLINE_EXTERNAL_RL_REVIEW_CHANGED_DURING_RUN')
        self.formal_admitted=True

    def save(self):
        return copy.deepcopy(dict(protocol=self.protocol,identity=self.identity,ready=self.ready,
            evidence=self.evidence,mask=self.mask.save(),samples=sorted(self.samples),exposure=self.exposure,
            wm_identity=self.wm_identity,formal_admitted=False))
    def validate_snapshot(self,state=None):
        if state is None:state=self.save()
        if state.get('protocol')!=self.protocol or state.get('identity')!=self.identity or set(state.get('ready',{}))!=set(NAMES):
            raise ValueError('ONLINE_GATE_RESUME_IDENTITY_MISMATCH')
        if any(type(v) is not bool for v in state['ready'].values()):raise ValueError('ONLINE_GATE_INVALID_STATE')
        if any(state['ready'][key] and name not in state['evidence'] for key,name in zip(NAMES,('collection','supervised','calibration','numeric'))):
            raise ValueError('ONLINE_GATE_MISSING_EVIDENCE')
        if self.wm_identity is not None and state.get('wm_identity')!=self.wm_identity:
            raise ValueError('ONLINE_GATE_WM_IDENTITY_CHANGED')
        if 'collection' in state['evidence']:
            from gym_trading_env.research.fg_s3_e_quality_online_evidence import validate_account_report
            validate_account_report(state['evidence']['collection'],self.collection_root,self.run_identity)
    def load(self,state):
        self.validate_snapshot(state);self.ready=copy.deepcopy(state['ready']);self.evidence=copy.deepcopy(state['evidence'])
        self.mask.load(state['mask']);self.samples=set(map(tuple,state['samples']));self.exposure=copy.deepcopy(state['exposure'])
        self.wm_identity=state['wm_identity'];self.formal_admitted=False

    def report(self):
        return dict(protocol=self.protocol,ready=copy.deepcopy(self.ready),blockers=[k for k,v in self.ready.items() if not v],
            mask_policy=copy.deepcopy(self.policy),wm_policy=copy.deepcopy(self.wm_policy),
            formal_rl_admitted=self.formal_admitted and all(self.ready[k] for k in (
                'SUPERVISED_STABLE','WM_VALID_FOR_RL','RL_NUMERIC_READY')),unique_calibration_owner_steps=len(self.samples),
            collection_verified_through_market_step=self.collection_through(),
            collection_status=self.evidence.get('collection',{}).get('receipt',{}).get('collection_status','NOT_VERIFIED'),
            prediction_q_decision_fusion=False)
