#!/usr/bin/env bash
# ============================================================================
# Validator-disagreement analysis behind the paper's tab:disagree.
#
#   disputed = at least one validator differs from the reference AND the
#              validators do not all supply the same realisation; unanimous
#              corrections are dropped from both pools (--drop_consensus)
#   probe    = w=0 for characters and words
#
# Needs posteriors/ from run_star_posteriors.sh.  Writes
#   labels_mutual/                 labels
#   results_mutual_drop/           per-language means, by-dissenter split,
#                                  pooled means with bootstrap CIs
#   results_mutual_drop/bootstrap_pvalues.txt   10000-resample p-values
#
# Usage: ./run_mutual.sh [--n_boot 300]
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=/home3/Saurabh/tools/espnet_may22_2026/tools/venv/bin/activate
n_boot=300
[ "${1:-}" = "--n_boot" ] && n_boot="$2"
set +u; source "${VENV}"; set -u
cd "${HERE}"
mkdir -p logs

for lang in bh bn ch hi kn mg mr mt te; do
  for split in dev test; do
    [ -f "labels_mutual/${lang}_${split}.pkl.gz" ] || \
      python label_disagreement_mutual.py --lang "${lang}" --split "${split}"
  done
done

for split in test dev; do
  python analyze.py --split "${split}" --labels_dir labels_mutual \
    --results_dir results_mutual_drop --word_measure p_word_w0 --drop_consensus \
    --n_boot "${n_boot}" > "logs/analyze_mutual_drop_${split}.log" 2>&1 &
done
wait

python mean_diff_bootstrap.py --labels_dir labels_mutual --word_probe word_bp_w0 \
  --drop_consensus --out results_mutual_drop/mean_diff_bootstrap.txt \
  > logs/boot_mutual_drop.log 2>&1
# Disputed units only, so the result is the same with or without
# --drop_consensus; the shipped file was produced without it.
python bootstrap_pvalues.py --labels_dir labels_mutual --word_probe word_bp_w0 \
  --out results_mutual_drop/bootstrap_pvalues.txt > logs/pvalues.log 2>&1
echo "all done"
