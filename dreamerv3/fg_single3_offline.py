"""Strict admission for Single3 pilots and the frozen C1_M1_1M experiment."""
import json
import importlib.metadata
from pathlib import Path
from decimal import Decimal, InvalidOperation


def update_budget(value):
  """Accept numeric YAML values without truncating fractional/invalid budgets."""
  if isinstance(value, bool):
    raise ValueError("Single3 update budget must be a positive integer")
  try:
    number = Decimal(str(value))
  except InvalidOperation as exc:
    raise ValueError("Single3 update budget must be a positive integer") from exc
  if not number.is_finite() or number <= 0 or number != number.to_integral_value():
    raise ValueError("Single3 update budget must be a positive integer")
  return int(number)


def validate(config, *, allow_unapproved=False, persist=True):
  options=config.fg_single3_offline
  if not options.enabled:
    return None
  if not options.authorized and not allow_unapproved:
    raise ValueError('Single3 GPU pilot is NOT_RUN: user Reward/Replay/Oracle acceptance is required')
  if (not config.run.fg_single3_offline or config.script!='offline_pretrain' or config.task!='gymnasium_FixedDirectionEntryEnv-v0'
      or config.run.from_checkpoint or config.run.fg_stage_transfer or config.run.fg_full_checkpoint):
    raise ValueError('Single3 pilot requires joint offline training and a fresh/same-identity checkpoint')
  budget = update_budget(config.run.steps)
  profile = options.get('profile', 'pilot_2k')
  budgets = {'pilot_2k': 2000, 'C1_M1_1M': 62500, 'C1_M1_OFFLINE_10K': 10000}
  if profile not in budgets or budget != budgets[profile]:
    raise ValueError('Single3 baseline profile/update budget changed')
  full_profile = profile != 'pilot_2k'
  key_steps = [0,2000,10000] if profile == 'C1_M1_OFFLINE_10K' else [0,2000,10000,25000,50000,62500]
  if full_profile and (
      options.candidate != 'C1' or options.pool != 'C1_natural_oracle'
      or config.agent.replay_context_mode != 'recompute_prefix'
      or not config.agent.update_diagnostics or not config.jax.sync_train_metrics
      or not config.run.fg_single3_audit or not config.run.fg_keep_all_checkpoints
      or config.run.fg_report_every_steps != 100 or config.run.fg_save_every_steps != 2000
      or list(config.run.fg_key_steps) != key_steps
      or Path(config.logdir).name != profile):
    raise ValueError('Single3 frozen experiment, causal input or audit configuration changed')
  if (config.batch_size!=16 or config.batch_length!=64 or config.replay_context!=1
      or config.consec_train!=1 or config.consec_report!=1 or config.report_length!=64
      or config.agent.horizon!=333 or config.agent.imag_length!=15 or not config.agent.contdisc
      or config.seed!=0 or config.seed_protocol.replay_seed!=202
      or config.replay.online or config.replay.fracs.uniform!=1
      or not config.action_mask_warmup.enabled or not config.action_mask_warmup.add_recall_enabled
      or config.agent.avail_actor_enabled or config.agent.avail_always_available_index!=0):
    raise ValueError('Single3 joint pilot baseline or ADD-recall Mask Gate changed')
  from gym_trading_env.research.fg_single3_dataset import validate_dataset
  from gym_trading_env.research.fg_single3_coverage import require_coverage
  from gym_trading_env.research.fg_replay import read_manifest,check_files,digest
  from gym_trading_env.envs.entry_contract import fingerprint,read_yaml
  root=Path(options.dataset).resolve()
  dataset=validate_dataset(root)
  coverage=require_coverage(root,options.coverage)
  horizon_path=Path(options.horizon).resolve()
  horizon=read_manifest(horizon_path/'manifest.json')
  if (horizon.get('stage')!='H' or horizon.get('status')!='PASS' or horizon.get('pairs')!=576
      or horizon['dataset_manifest_sha256']!=digest(root/'manifest.json')
      or horizon['coverage_manifest_sha256']!=digest(Path(options.coverage)/'manifest.json')):
    raise ValueError('Single3 H Gate identity mismatch')
  check_files(horizon['frozen_inputs'])
  check_files({str(horizon_path/p):h for p,h in horizon['files'].items()})
  if not dataset['full_training_fold'] or options.candidate not in ('C0','C1','C2','C3'):
    raise ValueError('Single3 requires full TRAIN and exactly one frozen Reward')
  pool=options.pool
  if pool not in (options.candidate+'_natural',options.candidate+'_natural_oracle'):
    raise ValueError('Single3 M0/M1 pool does not match the one selected Reward')
  members=[dataset['traces'][i] for i in dataset['pool_members'][pool]]
  capacity=sum(r['sequence_starts'] for r in members)
  if float(config.replay.size)!=capacity or Path(config.run.replay_dir).resolve()!=root/dataset['views'][pool]:
    raise ValueError('Single3 needs the complete audited Uniform pool capacity/directory')
  inputs=read_manifest(root/'inputs.json')
  engine={p:h for p,h in inputs['provenance']['code_files'].items()
      if '/gym_trading_env/envs/' in p or '/gym_trading_env/utils/' in p
      or p.endswith('/single3_reward.py') or p.endswith('/fg_reward_candidates.py')}
  check_files(engine)
  view=inputs['training_views'][options.candidate]
  if Path(config.fg_entry.manifest).resolve()!=Path(view['path']).resolve():
    raise ValueError('Single3 observation TRAIN manifest and Reward differ')
  selected=Path(config.env.gymnasium.config_path).resolve()
  if selected not in {Path(r['env_config']).resolve() for r in view['rows']}:
    raise ValueError('Agent observation spaces would use a foreign Env')
  env=read_yaml(selected)
  if env['entry'].get('protocol')!='fg_single3_v1':raise ValueError('Legacy six-lot Env cannot feed Single3')
  # Agent architecture remains the existing baseline. Record its resolved
  # configuration together with data/pool/Reward/Obs, seed and budget.
  from embodied.run.fg_checkpoint import learning_spec,plain
  versions={name:importlib.metadata.version(name) for name in
      ('numpy','jax','jaxlib','optax','elements','portal','cloudpickle','gymnasium','pandas')}
  identity=dict(packages=versions, learning=learning_spec(config), code_sha256={
      str(p.relative_to(Path(__file__).parents[1])):digest(p) for p in (
          Path(__file__), Path(__file__).with_name('agent.py'),
          Path(__file__).with_name('rssm.py'), Path(__file__).with_name('main.py'),
          Path(__file__).with_name('configuration.py'),
          Path(__file__).with_name('fg_entry_guard.py'),
          Path(__file__).with_name('configs.yaml'),
          Path(__file__).parents[1]/'embodied/jax/agent.py',
          Path(__file__).parents[1]/'embodied/jax/opt.py',
          Path(__file__).parents[1]/'embodied/core/replay.py',
          Path(__file__).parents[1]/'embodied/core/streams.py',
          Path(__file__).parents[1]/'embodied/run/fg_checkpoint.py',
          Path(__file__).parents[1]/'embodied/run/fg_single3_audit.py',
          Path(__file__).parents[1]/'embodied/run/action_mask_warmup.py',
          Path(__file__).parents[1]/'embodied/run/offline_pretrain.py')}, protocol='FG_SINGLE3_OFFLINE_V1',dataset_sha256=digest(root/'manifest.json'),
      coverage_sha256=digest(Path(options.coverage)/'manifest.json'),horizon_sha256=digest(horizon_path/'manifest.json'),
      pool=pool,candidate=options.candidate,reward_params_hash=members[0]['reward_params_hash'],
      observation_identity=members[0]['observation_identity'],capacity=capacity,
      model_seed=0,replay_seed=202,gradient_updates=budget,profile=profile,
      training_positions=budget*16*64,
      replay_context_mode=config.agent.get("replay_context_mode", "stored"),
      context_initialization="initial_WAIT_then_one_causal_prefix_stop_gradient" if full_profile else "stored_latent",
      checkpoint_schedule=dict(report=config.run.fg_report_every_steps,save=config.run.fg_save_every_steps,
          keys=list(config.run.fg_key_steps),keep_all=config.run.get("fg_keep_all_checkpoints",False)),agent_config=dict(config.agent),
      batch_size=16,batch_length=64,replay_context=1,mode='world_model_only_actor_critic_frozen')
  if full_profile:
    from embodied.run.action_mask_warmup import ActionMaskWarmup
    gate = ActionMaskWarmup.for_single3_1m(config.action_mask_warmup)
    identity['mask_gate_opening'] = dict(thresholds=gate.opening_thresholds,
        consecutive_reports=gate.required_passes,
        report_every_updates=config.run.fg_report_every_steps,
        automatic_activation=False)
  # Compare the same JSON representation in launcher, CLI and checkpoint tags.
  identity=plain(identity)
  identity_hash=fingerprint(identity)
  if options.identity_hash and options.identity_hash!=identity_hash:
    raise ValueError('Frozen Single3 checkpoint/pilot identity differs')
  logdir=Path(config.logdir).expanduser()
  tag=logdir/'single3_identity.json'
  if tag.exists():
    existing=json.loads(tag.read_text())
    if existing!=dict(identity=identity,identity_hash=identity_hash):
      raise ValueError('Cannot resume a legacy/other-Reward/other-pool checkpoint')
  elif logdir.exists() and any(logdir.iterdir()):
    raise ValueError('Existing result directory has no matching Single3 checkpoint identity')
  elif persist:
    logdir.mkdir(parents=True,exist_ok=True)
    temporary=logdir/'.single3_identity.pending'
    temporary.write_text(json.dumps(dict(identity=identity,identity_hash=identity_hash),sort_keys=True,indent=2))
    temporary.replace(tag)
  checkpoint_root=logdir/'ckpt'
  expected=dict(protocol='FG_SINGLE3_OFFLINE_V1',identity_hash=identity_hash)
  if checkpoint_root.exists() and any(checkpoint_root.iterdir()):
    # Inspect identities before reading any unrelated old checkpoint payload.
    for meta_path in checkpoint_root.glob('step_*/manifest.json'):
      if json.loads(meta_path.read_text()).get('identity')!=expected:
        raise ValueError('Checkpoint belongs to a legacy/other Single3 experiment')
    from embodied.run.fg_checkpoint import FullCheckpoint
    FullCheckpoint(checkpoint_root,None,expected,()).latest()
  return dict(identity=identity,identity_hash=identity_hash)
