# Aftertrace

Aftertrace is a controlled episodic-memory benchmark with a symbolic world
generator, compact decoder readers, and preregistered calibration/evaluation
scripts. Install the project environment with `python -m pip install -e .`.

## Run calibration on a remote GPU

On the GPU host, clone or sync this repository, install its environment, and
confirm that PyTorch sees the GPU with `python -c 'import torch; print(torch.cuda.is_available())'`.
Start a persistent terminal session so calibration can continue after SSH
disconnects:

```sh
tmux new -s aftertrace-calibration
cd /path/to/aftertrace
source .venv/bin/activate
mkdir -p runs
python scripts/calibrate.py \
  --sizes r2m r5m r14m r31m \
  --seeds 11 29 \
  --device cuda \
  --steps 15000 \
  --output-dir runs/calibration 2>&1 | tee runs/calibration.log
```

Detach with `Ctrl-b`, then `d`; the job continues in tmux. Reconnect with
`tmux attach -t aftertrace-calibration` and follow progress with
`tail -f runs/calibration.log`. Calibration checkpoints under the output
directory, so rerunning the same command resumes completed training steps.

From the local machine, copy the generated results back with:

```sh
rsync -avP -- rsync-user@gpu-host:/path/to/aftertrace/runs/ ./runs/
```

Replace `rsync-user`, `gpu-host`, and the remote checkout path with the values
for your machine. `-a` preserves the directory tree and timestamps, `-v` shows
transfers, and `-P` keeps partial files and reports progress. The calibration
table, plot, checkpoints, and logs are written below `runs/calibration/`.
