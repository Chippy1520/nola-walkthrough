#!/usr/bin/env bash
set -euo pipefail
D=$(cd -- "$(dirname -- "$0")" && pwd)
ROOT=$HOME/nola-walkthrough/data/cifar10
PY=$HOME/Desktop/Robotics/research-lab/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
exec > >(tee -a "$D/cifar10.log") 2>&1
printf '%s Waiting for verified CIFAR-10 archive and extraction.\n' "$(date -Is)"
for ((i=0;i<120;i++)); do
  [[ -f "$ROOT/cifar-10-python.tar.gz" && -f "$ROOT/cifar-10-batches-py/test_batch" ]] && break
  sleep 5
done
printf '%s  %s\n' c58f30108f718f92721af3b95e74349a "$ROOT/cifar-10-python.tar.gz" | md5sum -c -
printf '%s Archive verified; waiting for the desktop GPU lock.\n' "$(date -Is)"
(
  flock -w 14400 9
  test ! -e "$D/cifar10"
  printf '%s Starting matched CIFAR-10 teaching pilot.\n' "$(date -Is)"
  "$PY" -u "$D/run_local.py" --dataset cifar10 --out "$D/cifar10" --cifar-root "$ROOT" --train 540 --val 200 --test 300 --epochs 2 --microbatch 8 --effective-batch 32 --feature-batch 16
  python3 "$D/verify_pair.py" "$D/cifar10" | tee "$D/cifar10/verification.txt"
  printf '%s CIFAR-10 pilot verified.\n' "$(date -Is)"
) 9>"$HOME/nola-walkthrough/.offline-nola-gpu.lock"
