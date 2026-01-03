# Switch to a text TTY (Ctrl+Alt+F3) or SSH in, then:
避免GUI抢占导致崩溃
tmux new -s d3

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
# JAX/XLA memory settings (see §3)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

python -u dreamerv3/main.py \
  --logdir ~/logdir/forex \
  --configs forex \
  --run.from_checkpoint ~/logdir/forex/ckpt/$(cat ~/logdir/forex/ckpt/latest) \
  2>&1 | stdbuf -oL -eL tee -a ~/logdir/forex/train.log

训练模式,csv
python dreamerv3/main.py --logdir ~/logdir/future --configs future

训练模式，live
python dreamerv3/main.py --logdir ~/logdir/future --configs future_live 

live
python dreamerv3/main.py --logdir ~/logdir/future_live --configs future_live --run.from_checkpoint ~/logdir/future/ckpt/$(cat ~/logdir/future/ckpt/latest) --script live_trading

tmux ls

分离 
ctrl-b d

tmux attach -t d3
