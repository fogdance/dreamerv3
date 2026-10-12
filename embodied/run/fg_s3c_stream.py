"""Transparent lifecycle wrapper inside Agent's checkpointed Prefetch."""
from embodied.run.fg_s3_teacher_adapter import attach_teacher


class TeacherJoinStream:
    def __init__(self,source,catalog,*,mode):
        self.source,self.catalog,self.mode=source,catalog,mode
        self.iterator=None

    def __iter__(self):
        self.iterator=iter(self.source)
        return self

    def __next__(self):
        if self.iterator is None: raise RuntimeError('Start teacher stream before consumption')
        return attach_teacher(next(self.iterator),self.catalog,mode=self.mode,sampling_flags=True)

    def _state(self,full):
        child=self.source.save_full() if full and hasattr(self.source,'save_full') else self.source.save()
        return dict(catalog_sha256=self.catalog.manifest_sha256,mode=self.mode,source=child)

    def save(self):return self._state(False)

    def save_full(self):return self._state(True)

    def load(self,state):
        if state['catalog_sha256']!=self.catalog.manifest_sha256 or state['mode']!=self.mode:
            raise ValueError('Teacher stream identity differs')
        self.source.load(state['source'])

    load_full=load
