#!/usr/bin/env bash
set -euo pipefail

# Directory of this script
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python script to run
MAIN_PY="$ROOT_DIR/main.py"

# Log directory
LOG_DIR="$ROOT_DIR/history/experiment_logs"
mkdir -p "$LOG_DIR"

echo "Running experiments from $MAIN_PY"

# Run adv-node = 0 once with noise 0 (timestamped logfile)
# echo "Running --adv-node=0 --adv-noise=0"
# ts=$(date +"%Y%m%d_%H%M%S")
# LOGFILE="$LOG_DIR/run_advnode0_noise0_${ts}.log"
# echo "Logging to $LOGFILE"
# python3 "$MAIN_PY" --parallel --adv-node 0 --adv-noise 0 > "$LOGFILE" 2>&1 || {
#   echo "Run adv-node=0 failed, see $LOGFILE"
# }

# adv-node values and noise values for them
ADV_NODES=(5 10)
NOISES=(30)

for node in "${ADV_NODES[@]}"; do
  for noise in "${NOISES[@]}"; do
    echo "Running --adv-node=$node --adv-noise=$noise"
    ts=$(date +"%Y%m%d_%H%M%S")
    LOGFILE="$LOG_DIR/run_advnode${node}_noise${noise}_${ts}.log"
    echo "Logging to $LOGFILE"
    python3 "$MAIN_PY" --parallel --adv-node "$node" --adv-noise "$noise" > "$LOGFILE" 2>&1 || {
      echo "Run adv-node=$node noise=$noise failed, see $LOGFILE"
    }
  done
done

echo "All requested runs submitted. Logs are in $LOG_DIR"
