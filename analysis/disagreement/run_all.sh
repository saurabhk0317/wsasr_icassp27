#!/usr/bin/env bash
# ============================================================================
# Full star-vs-validator-disagreement study, steps 1-5, all RESPIN languages.
#
#   1. labels         label_disagreement.py   (built on demand by step 2)
#   2. posteriors     run_star_posteriors.sh  (GPU; ~1 min per model per split)
#   3-5. analysis     analyze.py --split dev / test (per language + pooled)
#
# READ-ONLY on WSASR/exp_final. Everything is written under this directory.
# Resumable: finished posteriors are skipped on a re-run.
#
# Usage (from a screen session):
#   ./run_all.sh                       # all 9 langs, dev+test, GPU 0
#   ./run_all.sh --gpus "0 1"          # one job per GPU
#   ./run_all.sh --langs "kn te mr"    # a subset
# Expected runtime: ~70 min on one GPU for everything (~35 min on two),
# plus ~1-2 min of analysis per language per split.
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
langs="bh bn ch hi kn mg mr mt te"
splits="dev test"
gpus="0"
n_boot=300
while [ $# -gt 0 ]; do
  case "$1" in
    --langs) langs="$2"; shift 2 ;;
    --splits) splits="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --n_boot) n_boot="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done

"${HERE}/run_star_posteriors.sh" --langs "${langs}" --splits "${splits}" --gpus "${gpus}"

set +u
# shellcheck disable=SC1091
source /home3/Saurabh/tools/espnet_may22_2026/tools/venv/bin/activate
set -u
for split in ${splits}; do
  echo "=== analysis: ${split} ==="
  python "${HERE}/analyze.py" --split "${split}" --langs "${langs}" --n_boot "${n_boot}" \
      2>&1 | grep -v "Warning" | tee "${HERE}/logs/analyze_${split}.log"
done
echo "reports: ${HERE}/results/{dev,test}_report.md"
