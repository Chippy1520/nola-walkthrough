#!/usr/bin/env bash
set -euo pipefail
D=$(cd -- "$(dirname -- "$0")" && pwd)
PY=/home/chip/Desktop/Robotics/research-lab/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
# Refuse to train if the pilot is incomplete or fails artifact validation.
python3 "$D/verify_pair.py" "$D/eurosat"
# Preserve any previous output; do not launch a duplicate workload.
test ! -e "$D/eurosat-full" && test ! -e "$D/eurosat-full.log"
set -o noclobber
{
  flock -n /home/chip/nola-walkthrough/.offline-nola-gpu.lock "$PY" -u "$D/run_local.py" --dataset eurosat --out "$D/eurosat-full" --train 0 --val 0 --test 0 --epochs 2 --microbatch 8 --effective-batch 512 --feature-batch 16
  python3 "$D/verify_pair.py" "$D/eurosat-full" > "$D/eurosat-full/verification.txt"
} > "$D/eurosat-full.log" 2>&1
