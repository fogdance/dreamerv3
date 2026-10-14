"""Real Ninjax/JAX CPU encoder, not a data aggregation acceptance test."""
import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from dreamerv3.fg_s3_e_encoder import S3EMultiscaleEncoder


def setup(single=False):
    space = {k: elements.Space(np.float32, shape) for k, shape in dict(
        market_seq_e_shared=(60,18), market_context_15m=(24,7),
        position_state=(9,), action_mask=(2,)).items()}
    enc = S3EMultiscaleEncoder(space, units=16, layers=3, name='enc')
    leading = (2,) if single else (2,3)
    rng = np.random.default_rng(70)
    obs = {k:jnp.array(rng.normal(size=(*leading,*s.shape)),jnp.float32) for k,s in space.items()}
    obs['market_context_15m'] = obs['market_context_15m'].at[...,6].set(1.)
    reset = jnp.zeros(leading,bool)
    fn = nj.pure(lambda o:enc({},o,reset,False,single))
    params, (_, entries, tokens) = fn({},obs,seed=11,create=True,modify=True)
    return enc,fn,params,obs,tokens


@pytest.mark.parametrize('single',[False,True])
def test_EM03_EM04_shapes_registered_branches_and_context_effect(single):
    enc,fn,p,o,tokens=setup(single)
    assert tokens.shape == (*o['action_mask'].shape[:-1],16)
    assert enc.initial(2) == enc.entry_space == enc.truncate({}) == {}
    assert all(any(k.startswith(prefix) for k in p) for prefix in
               ('enc/minute/','enc/context15/','enc/fusion/'))
    changed={**o,'market_context_15m':o['market_context_15m'].at[...,0].add(1.)}
    out=fn(p,changed,seed=11)[1][-1]
    assert np.isfinite(out).all() and not np.array_equal(out,tokens)


def test_EM05_invalid_rows_and_all_invalid_ignore_values_and_bias():
    _,fn,p,o,_=setup()
    p={k:jnp.ones_like(v)*3 if k.startswith('enc/context15/') and k.endswith('/bias') else v for k,v in p.items()}
    zero={**o,'market_context_15m':jnp.zeros_like(o['market_context_15m'])}
    polluted={**zero,'market_context_15m':zero['market_context_15m'].at[...,:6].set(999.)}
    np.testing.assert_array_equal(fn(p,zero,seed=11)[1][-1],fn(p,polluted,seed=11)[1][-1])
    partial=o['market_context_15m'].at[...,:12,:].set(0.)
    polluted=partial.at[...,:12,:6].set(999.)
    np.testing.assert_array_equal(fn(p,{**o,'market_context_15m':partial},seed=11)[1][-1],
                                 fn(p,{**o,'market_context_15m':polluted},seed=11)[1][-1])


def test_EM05_shape_dtype_future_keys_and_jit_nonfinite_rejected():
    _,fn,p,o,_=setup()
    for bad in ({**o,'market_context_15m':o['market_context_15m'][...,:6]},
                {**o,'market_seq_e_shared':o['market_seq_e_shared'].astype(jnp.int32)},
                {**o,'outcome_event':jnp.ones((2,3))}):
        with pytest.raises(ValueError):fn(p,bad,seed=11)
    call=jax.jit(lambda state,obs:fn(state,obs,seed=11)[1][-1])
    for context in (o['market_context_15m'].at[...,0,0].set(jnp.nan),
                    o['market_context_15m'].at[...,6].set(.5)):
        with pytest.raises(Exception,match='nonfinite|nonbinary'):
            call(p,{**o,'market_context_15m':context}).block_until_ready()


def test_EM07_wm_style_gradient_reaches_all_three_branches():
    _,fn,p,o,_=setup()
    loss=lambda state:jnp.square(fn(state,o,seed=11)[1][-1]).mean()
    grads=jax.grad(loss)(p)
    for prefix in ('enc/minute/','enc/context15/','enc/fusion/'):
        values=[np.asarray(v) for k,v in grads.items() if k.startswith(prefix)]
        assert all(np.isfinite(v).all() for v in values)
        assert sum(np.linalg.norm(v) for v in values)>0
