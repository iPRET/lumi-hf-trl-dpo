#!/bin/bash
#
# Runs inside the Singularity container on each node.
# Sets up library paths and launches the training script via accelerate.

export CPATH=$HOME/local/include:$CPATH
export LIBRARY_PATH=$HOME/local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
export CFLAGS="-I$HOME/local/include $CFLAGS"
export LDFLAGS="-L$HOME/local/lib $LDFLAGS"

accelerate launch \
  --num_machines 8 \
  --num_processes 64 \
  --machine_rank "$SLURM_NODEID" \
  --main_process_ip "$HEAD_NODE" \
  --main_process_port 29500 \
  train_dpo.py
#^  --num_processes means "node count" * "processes per node".
