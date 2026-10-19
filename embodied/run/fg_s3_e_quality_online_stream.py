"""Composition over native historical and live lifecycle streams."""
import numpy as np


class OnlineHistoryStream:
    """Source-level mixture of existing lifecycle streams, no Replay/trainer."""
    protocol='FG_S3_E_QUALITY_ONLINE_HISTORY_STREAM_V1'

    def __init__(self,history,online,*,history_ratio,seed):
        import copy
        from gym_trading_env.envs.entry_contract import fingerprint
        if not 0<history_ratio<1:raise ValueError('ONLINE_HISTORY_RATIO_REQUIRED')
        self.history=history;self.online=online;self.ratio=history_ratio;self.rng=np.random.default_rng(seed)
        self.identity=fingerprint(dict(protocol=self.protocol,history=history.identity_hash,
            online=online.identity,history_ratio=history_ratio,seed=seed))
        self.last_batch_audit=[];self.counts=dict(history_batches=0,online_batches=0)

    @property
    def readers(self):return [*self.history.readers,self.online]
    def __iter__(self):return self
    def __next__(self):
        import copy
        before=self.save()
        try:
            use_history=self.rng.random()<self.ratio
            if use_history:
                data,audit=self.history.sample();self.counts['history_batches']+=1
            else:
                data,audit=self.online.sample();self.counts['online_batches']+=1
                if 'AUXILIARY' in self.history.role_streams:
                    historical,ha=self.history.sample()
                    idx=[i for i,r in enumerate(ha) if r['role']=='AUXILIARY']
                    data={k:np.concatenate([v,historical[k][idx]],axis=0) for k,v in data.items()}
                    audit.extend(ha[i] for i in idx)
            self.last_batch_audit=copy.deepcopy(audit)
            return data
        except BaseException:
            self.load(before)
            raise
    def save(self):
        import copy
        return dict(protocol=self.protocol,identity=self.identity,rng=copy.deepcopy(self.rng.bit_generator.state),
            history=self.history.save(),online=self.online.save(),counts=copy.deepcopy(self.counts),
            last_batch_audit=copy.deepcopy(self.last_batch_audit))
    save_full=save
    def validate_snapshot(self,state=None):
        if state is not None and (state.get('protocol')!=self.protocol or state.get('identity')!=self.identity):
            raise ValueError('ONLINE_MIX_RESTORE_IDENTITY_MISMATCH')
        self.history.validate_snapshot(None if state is None else state['history'])
        self.online.validate_snapshot(None if state is None else state['online'])
    def load(self,state):
        import copy
        self.validate_snapshot(state)
        self.history.load(state['history']);self.online.load(state['online'])
        self.rng.bit_generator.state=copy.deepcopy(state['rng']);self.counts=copy.deepcopy(state['counts'])
        self.last_batch_audit=copy.deepcopy(state['last_batch_audit'])
    load_full=load
    def coverage(self):
        return dict(history=self.history.coverage(),online=self.online.coverage(),
            selected_batches=dict(self.counts),denominators='Delivered source windows, not independent market N')
    def close(self):self.history.close();self.online.close()
