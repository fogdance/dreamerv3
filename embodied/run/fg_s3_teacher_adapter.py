"""Pure host-side teacher join. No JAX, RNG, cache mutation or private labels."""
import numpy as np

from gym_trading_env.research.fg_s3_teacher_catalog import DatasetIdentityError


def attach_teacher(batch, catalog, *, mode, sampling_flags=False):
    ids = np.asarray(batch['stepid'])
    if ids.dtype != np.uint8 or ids.ndim != 3 or ids.shape[-1] != 20:
        raise DatasetIdentityError('BATCH_STEPID_SCHEMA')
    if mode not in ('train', 'report'):
        raise ValueError('Unknown teacher mode')
    if 'teacher_action' in batch or 'teacher_valid' in batch:
        raise DatasetIdentityError('ALREADY_JOINED_BATCH')
    shape = ids.shape[:2]
    action, valid = np.zeros(shape, np.int32), np.zeros(shape, bool)
    if mode == 'train':
        if sampling_flags:
            consec=np.asarray(batch.get('consec'))
            if consec.shape!=shape or consec.dtype!=np.int32:
                raise DatasetIdentityError('BATCH_CONSEC_SCHEMA')
        fields = {'action': ((), np.int32), 'action_mask': ((2,), np.float32),
                  'position_state': ((9,), np.float32),
                  **{k: ((), np.bool_) for k in ('is_first', 'is_last', 'is_terminal')}}
        for name, (tail, dtype) in fields.items():
            x = np.asarray(batch[name])
            if x.shape != (*shape, *tail) or x.dtype != dtype:
                raise DatasetIdentityError('BATCH_FIELD_SCHEMA: '+name)
        for pos in np.ndindex(shape):
            match = catalog.lookup(ids[pos])
            row = match.record
            if row is None:
                continue
            # Check every P4 row, even excluded rows, before applying sentinel.
            pairs = [('action', 'replay_action'), ('action_mask', 'mask'),
                     ('position_state', 'position_state'), ('is_first', 'is_first'),
                     ('is_last', 'is_last'), ('is_terminal', 'is_terminal')]
            for name, field in pairs:
                expected=row[field]
                # Replay.sample() explicitly creates a reset at sequence start.
                # Authenticate that documented transform; never loosen other
                # flags or rewrite the native observation/teacher definition.
                if sampling_flags and name=='is_first' and pos[1]==0 and consec[pos]==0:
                    expected=True
                if not np.array_equal(batch[name][pos], expected):
                    raise DatasetIdentityError('TEACHER_BATCH_MISMATCH: '+name)
            if row['label_valid']:
                action[pos] = row['label']
                valid[pos] = True
    return {**batch, 'teacher_action': action, 'teacher_valid': valid}
