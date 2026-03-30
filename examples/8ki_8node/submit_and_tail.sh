#!/usr/bin/env bash
#
# Convenience wrapper: submits a SLURM job and tails its output once running.
#
# Usage: ./submit_and_tail.sh launch.sh

if [ -z "$1" ]; then
  echo "Usage: $0 <sbatch_script>"
  exit 1
fi

echo "Submitting $1..."
JOB_SUBMISSION_OUTPUT=$(sbatch "$1")
echo "$JOB_SUBMISSION_OUTPUT"

JOB_ID=$(echo "$JOB_SUBMISSION_OUTPUT" | awk '{print $4}')
if [ -z "$JOB_ID" ]; then
  echo "Could not parse Job ID. Exiting."
  exit 1
fi

echo "Job ID: $JOB_ID"

while true; do
  sleep 2
  JOB_STATE=$(squeue -j "$JOB_ID" -h -o "%T")

  if [ -z "$JOB_STATE" ]; then
    echo "Job $JOB_ID is no longer in the queue."
    break
  fi

  if [ "$JOB_STATE" == "RUNNING" ]; then
    echo "Job $JOB_ID running. Tailing output..."
    break
  fi

  echo "State: $JOB_STATE (checking in 2s...)"
done

sleep 2
tail -f "slurm-${JOB_ID}.out"
