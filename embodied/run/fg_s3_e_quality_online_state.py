"""Small saveables used by the ORIGINAL FullCheckpoint and Replay."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


class ChunkIdentity:
    """Recoverable native chunk identity; never uses a global RNG override."""
    protocol='FG_S3_E_QUALITY_NATIVE_CHUNK_SEQUENCE_V1'

    def __init__(self,execution_identity):
        self.identity=execution_identity;self.sequence=0

    def allocate(self):
        import elements
        value=hashlib.sha256(f'{self.protocol}:{self.identity}:{self.sequence}'.encode()).digest()[:16]
        time=f'seq{self.sequence:020d}'  # Logical archive ordering, not market/wall time.
        self.sequence+=1
        return elements.UUID(value),time

    def save(self):return dict(protocol=self.protocol,identity=self.identity,sequence=self.sequence)
    def validate_snapshot(self,state=None):
        if state is None:state=self.save()
        if (state.get('protocol')!=self.protocol or state.get('identity')!=self.identity
                or type(state.get('sequence')) is not int or state['sequence']<0):
            raise ValueError('ONLINE_REPLAY_CHUNK_IDENTITY_MISMATCH')
    def load(self,state):self.validate_snapshot(state);self.sequence=state['sequence']


class JournalState:
    """Committed log prefixes; preserve crash tails instead of discarding them."""
    def __init__(self,root,names):self.root=Path(root);self.names=tuple(names)
    def save(self):
        from gym_trading_env.research.fg_s3_e_quality_online_data import journal_boundary
        return {name:journal_boundary(self.root/name) for name in self.names}
    def validate_snapshot(self,state=None):
        from gym_trading_env.research.fg_s3_e_quality_online_data import validate_journal_boundary
        if state is None:state=self.save()
        if set(state)!=set(self.names):raise ValueError('ONLINE_LOG_SET_MISMATCH')
        for name,value in state.items():validate_journal_boundary(self.root/name,value)
    def load(self,state):
        from gym_trading_env.research.fg_s3_e_quality_online_data import restore_journal_boundary
        self.validate_snapshot(state)
        for name,value in state.items():restore_journal_boundary(self.root/name,value)


class OnlinePhase:
    """A collected cycle can resume midway through label publication/updates."""
    def __init__(self):self.state=dict(stage='IDLE',end=0,updates_done=0)
    def save(self):return dict(self.state)
    def validate_snapshot(self,state=None):
        state=self.state if state is None else state
        if (set(state)!={'stage','end','updates_done'} or state['stage'] not in ('IDLE','COLLECT','PUBLISH','UPDATE','FINALIZE')
                or any(type(state[k]) is not int or state[k]<0 for k in ('end','updates_done'))):
            raise ValueError('ONLINE_PHASE_STATE_INVALID')
    def load(self,state):self.validate_snapshot(state);self.state=dict(state)


class SourceExposures:
    """Actual consumed loss-eligible exposures, separate from Reader prefetch."""
    def __init__(self):self.state={}
    def add(self,audit,data,start):
        import numpy as np
        from dreamerv3.fg_s3_e_quality_actor import valid_l2_targets
        labels={k:data[k][:,start:] for k in ('teacher_action','teacher_valid')}
        valid=valid_l2_targets(labels,*[data[k][:,start:] for k in ('action_mask','is_first','is_last','is_terminal')])
        for i,row in enumerate(audit):
            key=json.dumps([row.get('role','NATURAL'),row['corpus_id'],row['source_identity']],separators=(',',':'))
            counts=self.state.setdefault(key,dict(windows=0,target_positions=0,valid_add=0,valid_wait=0))
            counts['windows']+=1;counts['target_positions']+=int(valid.shape[1])
            counts['valid_add']+=int((valid[i] & (labels['teacher_action'][i]==1)).sum())
            counts['valid_wait']+=int((valid[i] & (labels['teacher_action'][i]==0)).sum())
    def save(self):return copy.deepcopy(self.state)
    def validate_snapshot(self,state=None):
        state=self.state if state is None else state
        for key,value in state.items():
            if (len(json.loads(key))!=3 or set(value)!={'windows','target_positions','valid_add','valid_wait'}
                    or any(type(v) is not int or v<0 for v in value.values())
                    or value['valid_add']+value['valid_wait']>value['target_positions']):
                raise ValueError('ONLINE_SOURCE_EXPOSURE_STATE_INVALID')
    def load(self,state):self.validate_snapshot(state);self.state=copy.deepcopy(state)
