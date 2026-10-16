"""Reviewed V1 quality networks, detached prediction fusion and joint losses.

This module contains no future-data reader. Labels enter only the loss; the
exact same prediction/fusion function serves training and causal policy.
"""
from __future__ import annotations

import math

import elements
import embodied
import jax
import jax.numpy as jnp
import numpy as np
import optax

sg = jax.lax.stop_gradient


def scaled_features(features,scale):
    detached=sg(features)
    return detached+scale*(features-detached)

DEFAULTS = dict(quality_alpha=.8, quality_kappa=1., quality_wmax=2.,
    quality_strict_pair_lambda=.1, quality_pair_lambda=.05, quality_soft_lambda=.10,
    quality_outcome_lambda=.20, quality_score_lambda=.20, quality_q_c1_lambda=.10,
    quality_q_v2_lambda=.10, quality_mono_lambda=.02, quality_repr_grad_scale=.1,
    quality_pair_tau=1., quality_q_scale=1.)


def quality_extensions(settings):
    supervision=settings.get('quality_supervision_protocol')
    if supervision is not None:
        from gym_trading_env.research.fg_s3_e_quality_supervision import PROTOCOL
        if supervision != PROTOCOL:
            raise ValueError('Unknown Quality supervision protocol')
        forbidden=('quality_q_c1_lambda','quality_q_v2_lambda','quality_soft_lambda')
        if any(settings.get(k,0.) != 0. for k in forbidden):
            raise ValueError('E_QUALITY_SUPERVISION_Q_SOFT_FORBIDDEN')
    stage=settings.get('quality_stage','HEADS_ONLY')
    if stage not in ('HEADS_ONLY','CONTROLLED_FULL'):
        raise ValueError('Unknown Quality-first research stage')
    admitted=settings.get('quality_prediction_admitted',False)
    if type(admitted) is not bool or stage=='CONTROLLED_FULL' and not admitted:
        raise ValueError('CONTROLLED_FULL requires verified prediction admission')
    if stage=='HEADS_ONLY' and admitted:
        raise ValueError('HEADS_ONLY does not certify predictive research acceptance')
    variant = settings.get('quality_variant','FULL')
    if variant not in ('B1','B2','B3','FULL'): raise ValueError('Unknown quality ablation')
    if stage=='HEADS_ONLY' and variant!='FULL':
        raise ValueError('B1/B2/B3 are later controlled teaching ablations')
    result = dict(s3_v2_mode='quality_v2',quality_variant=variant,
                  quality_stage=stage,quality_prediction_admitted=admitted)
    if supervision is not None:
        result['quality_supervision_protocol']=supervision
    decision=settings.get('quality_decision_admitted',False)
    if type(decision) is not bool or stage=='HEADS_ONLY' and decision:
        raise ValueError('HEADS_ONLY does not admit quality decision interventions')
    result['quality_decision_admitted']=decision
    interval=settings.get('quality_gradient_every_updates',0 if stage=='HEADS_ONLY' else 100)
    if type(interval) is not int or interval<0: raise ValueError('Invalid quality gradient sampling interval')
    result['quality_gradient_every_updates']=interval
    for k,v in DEFAULTS.items():
        default=0. if supervision and k in forbidden else v
        x = settings.get(k,default)
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x < 0:
            raise ValueError('Invalid quality setting: '+k)
        result[k] = float(x)
    if (not 0 <= result['quality_alpha'] <= 1 or not 0 <= result['quality_repr_grad_scale'] <= 1
            or result['quality_wmax'] < 1 or result['quality_pair_tau'] <= 0 or result['quality_q_scale'] <= 0):
        raise ValueError('Quality alpha/gradient/temperature/weight range')
    for key,default in (('quality_fuse_outcome',stage=='CONTROLLED_FULL' and variant in ('B3','FULL')),
                        ('quality_fuse_score',stage=='CONTROLLED_FULL' and variant in ('B3','FULL')),
                        ('quality_q_admitted',False),('quality_weighting',variant != 'B1')):
        value = settings.get(key,default)
        if type(value) is not bool: raise ValueError('Quality switch requires explicit boolean: '+key)
        result[key] = value
    if stage=='HEADS_ONLY':
        if result['quality_fuse_outcome'] or result['quality_fuse_score'] or result['quality_q_admitted']:
            raise ValueError('HEADS_ONLY forbids prediction influence on Actor')
        result['quality_repr_grad_scale']=0.
        for k in ('quality_strict_pair_lambda','quality_pair_lambda','quality_soft_lambda',
                  'quality_q_c1_lambda','quality_q_v2_lambda'):
            result[k]=0.
    if result['quality_q_admitted'] and variant != 'FULL':
        raise ValueError('Only admitted FULL may fuse numeric Q')
    if any(result[k] for k in ('quality_fuse_outcome','quality_fuse_score','quality_q_admitted')) and not decision:
        raise ValueError('DECISION_GATE_PENDING: forecast accuracy does not admit Actor intervention')
    if variant in ('B1','B2') and (result['quality_fuse_outcome'] or result['quality_fuse_score']):
        raise ValueError('B1/B2 require prediction-to-decision fusion OFF')
    if variant == 'B1':
        for k in ('quality_pair_lambda','quality_soft_lambda','quality_outcome_lambda',
                  'quality_score_lambda','quality_q_c1_lambda','quality_q_v2_lambda','quality_mono_lambda'):
            result[k] = 0.
    return result


def head_modules(agent, config):
    """Reuse Dreamer policy presets; no parallel network-default parser."""
    outs = {k:config.policy_dist_disc if s.discrete else config.policy_dist_cont for k,s in agent.act_space.items()}
    agent.quality_actor = embodied.jax.MLPHead(agent.act_space,outs,**config.policy,name='quality_actor')
    common = dict(layers=1,units=config.policy.units,act=config.policy.act,
                  norm=config.policy.norm,winit=config.policy.winit,outscale=1.)
    for name,shape in (('event',(7,4)),('touch',(4,2)),('adverse',(4,)),
                        ('score',()),('q_c1',(2,)),('q_v2',(2,))):
        setattr(agent,'quality_'+name,embodied.jax.MLPHead(
            elements.Space(np.float32,shape),'mse',**common,name='quality_'+name))
    agent.modules.extend([agent.quality_actor,*[getattr(agent,'quality_'+n)
        for n in ('event','touch','adverse','score','q_c1','q_v2')]])


def predict(agent, features, bdims):
    return {name:getattr(agent,'quality_'+name)(features,bdims).pred().astype(jnp.float32)
            for name in ('event','touch','adverse','score','q_c1','q_v2')}


def fuse_logits(raw, predictions, mask, config):
    """All corrections are predicted, stopped, bounded and run-configured."""
    p = jax.tree.map(sg,predictions)
    event = jax.nn.softmax(p['event'],-1)
    score = 100 * jax.nn.sigmoid(p['score'])
    delta_o = .5*jnp.clip(event[...,5,0]-6*event[...,5,1]-.5*jax.nn.sigmoid(p['adverse'][...,0]),-1.,1.)
    delta_s = .5*jnp.clip((score-60)/40,-1.,1.)
    delta_q = .5*jnp.tanh((p['q_v2'][...,1]-p['q_v2'][...,0])/config['quality_q_scale'])
    legal = jnp.all(mask>.5,-1)
    potential={'quality/diagnostic_delta_O':jnp.where(legal,delta_o,0.),
               'quality/diagnostic_delta_S':jnp.where(legal,delta_s,0.),
               'quality/diagnostic_delta_Q':jnp.where(legal,delta_q,0.)}
    gate=config['quality_decision_admitted']
    delta_o = jnp.where(legal & gate & config['quality_fuse_outcome'],delta_o,0.)
    delta_s = jnp.where(legal & gate & config['quality_fuse_score'],delta_s,0.)
    delta_q = jnp.where(legal & gate & config['quality_q_admitted'],delta_q,0.)
    final = raw.at[...,1].add(delta_o+delta_s+delta_q)
    z = raw[...,1]-raw[...,0]
    return final,{**potential,'quality/base_policy_logit':z,'quality/delta_O':delta_o,
        'quality/delta_S':delta_s,'quality/delta_Q':delta_q,
        'quality/z_final':final[...,1]-final[...,0], 'quality/score_prediction':score,
        'quality/outcome_probs':event,'quality/q_c1_prediction':p['q_c1'],
        'quality/touch_probability':jax.nn.sigmoid(p['touch']),
        'quality/adverse_probability':jax.nn.sigmoid(p['adverse']),
        'quality/q_v2_prediction':p['q_v2'],
        'quality/pADD_raw':jax.nn.softmax(final,-1)[...,1]}


def policy_forward(agent, features, mask, bdims):
    predictions = predict(agent,features,bdims)
    head_only=agent.quality['quality_stage']=='HEADS_ONLY'
    actor=agent.pol if head_only else agent.quality_actor
    raw = actor(sg(features),bdims)[agent.action_key].logits
    _ = agent.quality_actor.values  # Allocate/access the frozen Full module too.
    final,diagnostics = fuse_logits(raw,predictions,mask,agent.quality)
    dist = agent._masked_policy({agent.action_key:embodied.jax.outs.Categorical(final)},mask,False)
    action = dist[agent.action_key]
    diagnostics.update({'quality/pADD_after_mask_unimix':action.prob(jnp.ones(raw.shape[:-1],jnp.int32)),
                        'quality/argmax_action':action.pred()})
    return dist,diagnostics,predictions


def mean(value, valid, weight=None):
    valid = jnp.asarray(valid,bool)
    w = valid.astype(jnp.float32) if weight is None else jnp.where(valid,weight,0.)
    denominator=jnp.where(w.sum()>0,w.sum(),1.)
    return (jnp.where(valid,value,0.)*w).sum()/denominator


def pair_identity(labels, first):
    owner = labels['quality_owner']
    segment = jnp.cumsum(first.astype(jnp.int32),-1)
    return ((owner[:,:,None,:] == owner[:,None,:,:]).all(-1)
            & (segment[:,:,None] == segment[:,None,:])
            & (owner != 0).any(-1)[:,:,None] & (owner != 0).any(-1)[:,None,:])


def teaching_objective(dist, labels, y, valid, first, config, bc_lambda=1.):
    """Natural anchor is never weighted. All pairs stay within real lineage."""
    nll = -dist.logp(jnp.where(valid,y,0))
    safe = jnp.where(valid[...,None],dist.logits,0.)
    # Ranking can also use certified outcome-only candidates without strict BC.
    z = dist.logits[...,1]-dist.logits[...,0]
    z = jnp.where(labels['quality_outcome_valid'] | valid,z,0.)
    padd = jax.nn.softmax(safe,-1)[...,1]
    pos,neg = valid & (y==1),valid & (y==0)
    natural = mean(nll,valid)
    score = labels['quality_score']
    known = labels['quality_score_valid']
    risk = labels['quality_sl'] | labels['quality_early5']
    high = labels['quality_tp'] & known & (score >= 80)
    wq = jnp.where(pos & known,1+.5*score/100,
         jnp.where(neg & known & risk,1.5,jnp.where(neg & high,.35,1.)))
    if not config['quality_weighting']: wq=jnp.ones_like(wq)
    difficulty = sg(jnp.where(y==1,1-padd,padd))
    we2 = jnp.clip(1+config['quality_kappa']*difficulty,1,config['quality_wmax'])
    weights = sg(jnp.clip(wq*we2,.25,2.5))
    both = pos.any() & neg.any()
    balanced = jnp.where(both,.5*(mean(nll,pos,weights)+mean(nll,neg,weights)),natural)
    identity = pair_identity(labels,first)
    strict_pair = identity & pos[:,:,None] & neg[:,None,:]
    gap = z[:,:,None]-z[:,None,:]
    pair_weights = sg(jnp.sqrt(weights[:,:,None]*weights[:,None,:]))
    strict = mean(jax.nn.softplus(-gap/config['quality_pair_tau']),strict_pair,pair_weights)
    qs = labels['quality_score_valid'] & labels['quality_outcome_valid']
    s_gap = score[:,:,None]-score[:,None,:]
    # Count each directed high->low pair once; |gap| threshold is 15 points.
    quality_pair = identity & qs[:,:,None] & qs[:,None,:] & (s_gap >= 15)
    quality = mean(jax.nn.softplus(-gap),quality_pair)
    qvalid = labels['quality_soft_valid'] & (labels['quality_soft_source']==2)
    target=sg(labels['quality_soft_target'])
    soft_valid=labels['quality_soft_valid']
    # These log probabilities already include exactly one mask/unimix.
    logprobs = jax.nn.log_softmax(jnp.where(soft_valid[...,None],dist.logits,0.),-1)
    soft_nll = -target*logprobs[...,1]-(1-target)*logprobs[...,0]
    soft = mean(jnp.where(soft_valid,soft_nll,0.),soft_valid)
    mix = config['quality_alpha']*natural+(1-config['quality_alpha'])*balanced
    loss = bc_lambda*mix + config['quality_strict_pair_lambda']*strict + config['quality_pair_lambda']*quality + config['quality_soft_lambda']*soft
    if config['quality_stage']=='HEADS_ONLY': loss=jnp.float32(0)
    predicted = padd > .5
    metrics = {'quality/natural_loss':natural,'quality/balanced_loss':balanced,
        'quality/strict_pair_loss':strict,'quality/quality_pair_loss':quality,'quality/soft_loss':soft,
        'quality/strict_pair_count':strict_pair.sum(),'quality/quality_pair_count':quality_pair.sum(),
        'quality/soft_valid_count':soft_valid.sum(),'quality/soft_q_source_count':qvalid.sum(),
        'quality/soft_heuristic_source_count':(soft_valid & ~qvalid).sum(),
        'quality/weight_mean':mean(weights,valid),
        'quality/weight_variance':mean((weights-mean(weights,valid))**2,valid),
        'quality/quality_weight_variance':mean((wq-mean(wq,valid))**2,valid),
        'quality/weight_pos_variance':mean((weights-mean(weights,pos))**2,pos),
        'quality/weight_neg_variance':mean((weights-mean(weights,neg))**2,neg),
        'quality/quality_weight_pos_variance':mean((wq-mean(wq,pos))**2,pos),
        'quality/quality_weight_neg_variance':mean((wq-mean(wq,neg))**2,neg),
        'bc/nll':natural,'bc/valid_count':valid.sum(),'bc/add_count':pos.sum(),'bc/wait_count':neg.sum(),
        'bc/tp':(pos & predicted).sum(),'bc/fn':(pos & ~predicted).sum(),
        'bc/fp':(neg & predicted).sum(),'bc/tn':(neg & ~predicted).sum(),
        'bc/accuracy':mean((predicted==(y==1)).astype(jnp.float32),valid),'bc/p_add_mean':mean(padd,valid)}
    return loss,metrics


def active_branches(labels, y, valid, first, config, bc_lambda):
    identity = pair_identity(labels,first)
    strict_pairs = identity & (valid & (y==1))[:,:,None] & (valid & (y==0))[:,None,:]
    score = labels['quality_score']
    qs = labels['quality_score_valid'] & labels['quality_outcome_valid']
    qp = identity & qs[:,:,None] & qs[:,None,:] & (score[:,:,None]-score[:,None,:] >= 15)
    soft = labels['quality_soft_valid']
    actor = ((bc_lambda > 0) & valid.any()) | ((config['quality_strict_pair_lambda']>0) & strict_pairs.any())
    actor |= ((config['quality_pair_lambda']>0) & qp.any()) | ((config['quality_soft_lambda']>0) & soft.any())
    actor &= config['quality_stage']=='CONTROLLED_FULL'
    return {'quality_actor/':actor,
        'quality_event/':(config['quality_outcome_lambda']>0) & labels['quality_event_valid'].any()
                          | ((config['quality_mono_lambda']>0) & (labels['quality_event_valid'][...,:5].sum(-1)>1).any()),
        'quality_touch/':(config['quality_outcome_lambda']>0) & labels['quality_touch_valid'].any(),
        'quality_adverse/':(config['quality_outcome_lambda']>0) & labels['quality_adverse_valid'].any(),
        'quality_score/':(config['quality_score_lambda']>0) & labels['quality_score_valid'].any(),
        'quality_q_c1/':(config['quality_q_c1_lambda']>0) & (valid.any() | labels['quality_q_c1_valid'].any()),
        'quality_q_v2/':(config['quality_q_v2_lambda']>0) & labels['quality_q_v2_valid'].any()}


def prediction_objective(pred, labels, y, strict_valid, config):
    event_valid = labels['quality_event_valid']
    event_y = labels['quality_event']
    ce = optax.softmax_cross_entropy_with_integer_labels(pred['event'],event_y)
    event = mean(ce,event_valid)
    touch = mean(optax.sigmoid_binary_cross_entropy(pred['touch'],labels['quality_touch']),labels['quality_touch_valid'])
    adverse = mean(optax.sigmoid_binary_cross_entropy(pred['adverse'],labels['quality_adverse']),labels['quality_adverse_valid'])
    outcome = event+.5*touch+.5*adverse
    score_pred = jax.nn.sigmoid(pred['score'])
    score_target = labels['quality_score']/100
    score_per = optax.huber_loss(score_pred,score_target,delta=.1)+.1*optax.sigmoid_binary_cross_entropy(
        (score_pred-.8)/.1,(labels['quality_score']>=80).astype(jnp.float32))
    score = mean(score_per,labels['quality_score_valid'])
    dc1 = pred['q_c1'][...,1]-pred['q_c1'][...,0]
    rank = mean(jnp.where(y==1,jax.nn.softplus(-dc1),jax.nn.relu(dc1)**2),strict_valid)
    c1_value = mean(optax.huber_loss(pred['q_c1'],labels['quality_q_c1'],delta=1).mean(-1),labels['quality_q_c1_valid'])
    qv2 = mean(optax.huber_loss(pred['q_v2'],labels['quality_q_v2'],delta=1).mean(-1),labels['quality_q_v2_valid'])
    probs = jax.nn.softmax(pred['event'],-1)
    idx = jnp.arange(5)
    valid_pairs = (idx[:,None] < idx[None,:]) & event_valid[...,:5,None] & event_valid[...,None,:5]
    violation = jax.nn.relu(probs[...,:5,None,:3]-probs[...,None,:5,:3])**2
    mono = mean(violation,jnp.broadcast_to(valid_pairs[...,None],violation.shape))
    open_violation = mean(jax.nn.relu(probs[...,None,:5,3]-probs[...,:5,None,3]),valid_pairs)
    loss = (config['quality_outcome_lambda']*outcome+config['quality_score_lambda']*score
        +config['quality_q_c1_lambda']*(rank+c1_value)+config['quality_q_v2_lambda']*qv2
        +config['quality_mono_lambda']*mono)
    metrics = {'quality/outcome_loss':outcome,'quality/event_loss':event,'quality/touch_loss':touch,
        'quality/adverse_loss':adverse,'quality/score_loss':score,'quality/q_c1_rank_loss':rank,
        'quality/q_c1_value_loss':c1_value,'quality/q_v2_loss':qv2,'quality/mono_loss':mono,
        'quality/open_monotonic_violation':open_violation,
        'quality/event_valid_count':event_valid.sum(),'quality/touch_valid_count':labels['quality_touch_valid'].sum(),
        'quality/adverse_valid_count':labels['quality_adverse_valid'].sum(),
        'quality/score_valid_count':labels['quality_score_valid'].sum(),
        'quality/q_c1_value_valid_count':labels['quality_q_c1_valid'].sum(),
        'quality/q_v2_value_valid_count':labels['quality_q_v2_valid'].sum(),
        'quality/score_mae':100*mean(jnp.abs(score_pred-score_target),labels['quality_score_valid']),
        'quality/q_v2_delta_mse':mean(((pred['q_v2'][...,1]-pred['q_v2'][...,0])-
            (labels['quality_q_v2'][...,1]-labels['quality_q_v2'][...,0]))**2,labels['quality_q_v2_valid'])}
    for h in range(7):
        for c,name in enumerate(('tp','sl','other','open')):
            metrics[f'quality/h{h}_{name}_count']=(event_valid[...,h] & (event_y[...,h]==c)).sum()
    return loss,metrics
