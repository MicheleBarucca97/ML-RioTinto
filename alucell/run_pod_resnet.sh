#!/bin/bash
# ---------------------------------------------------------------------------
# POD-ResNet surrogate for the three Alucell fields.
#
#   ./run_pod_resnet.sh prep     build the three POD datasets
#   ./run_pod_resnet.sh train    train the three networks
#   ./run_pod_resnet.sh eval     score them on the held-out test split
#   ./run_pod_resnet.sh all      all three, in order
#
# Each stage logs to logs/<field>_<stage>.log and stops on the first failure.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

MASTER=../report_ML/master_ml.h5
MANIFEST=../report_ML/manifest.csv

# Dead-anode runs (one anode at 0 A) are a real operating regime, so they stay
# in the dataset.  Drop this flag to reproduce the coverage report's bulk-only
# population instead.
DEAD_FLAG=--include-dead

#            name        mapping           n_modes   (n99 from coverage report)
FIELDS=(
  "midacd     velocity_midacd   64"   # n99 = 35
  "full3d     velocity_full     64"   # n99 = 37
  "interface  interface         48"   # n99 = 24
)

mkdir -p data models logs plots

prep() {
  for spec in "${FIELDS[@]}"; do
    read -r name mapping kmodes <<< "$spec"
    echo "=== prepare: $name ($mapping, k=$kmodes) ==="
    python prepare_alucell.py \
      --master   "$MASTER" \
      --manifest "$MANIFEST" \
      --mapping  "$mapping" \
      --delta --pod --n-modes "$kmodes" \
      $DEAD_FLAG \
      --output   "data/${name}_pod_delta.h5" 2>&1 | tee "logs/${name}_prep.log"
  done
}

train() {
  for spec in "${FIELDS[@]}"; do
    read -r name mapping kmodes <<< "$spec"
    echo "=== train: $name ==="
    python train.py --config "config_alucell_${name}.yaml" \
      2>&1 | tee "logs/${name}_train.log"
  done
}

evaluate() {
  for spec in "${FIELDS[@]}"; do
    read -r name mapping kmodes <<< "$spec"
    echo "=== eval: $name ==="
    python evaluate.py --config "config_alucell_${name}.yaml" \
      2>&1 | tee "logs/${name}_eval.log"
  done
}

case "${1:-all}" in
  prep)  prep ;;
  train) train ;;
  eval)  evaluate ;;
  all)   prep; train; evaluate ;;
  *)     echo "usage: $0 {prep|train|eval|all}" >&2; exit 2 ;;
esac

echo
echo "=== artefacts ==="
ls -la models/best_model_ResFFNN__*_pod_delta.pth 2>/dev/null || true
