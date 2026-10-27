#!/usr/bin/env bash
# ============================================================================
# Star-disagreement study under the MUTUAL-DISSENT definition of "disputed",
# with a uniform w=0 probe at both granularities.
#
#   disputed = at least one validator differs from the reference AND the
#              validators do not all supply the same realisation
#   probe    = w=0 for characters AND words (the original run used w=-2 for
#              words)
#
# Reuses posteriors/ unchanged (tok_bp_w0 and word_bp_w0 were already
# computed), so no GPU and no model access is needed -- only relabelling and
# re-analysis.  Nothing under labels/ or results/ is touched.
#
#   labels_mutual/        new labels
#   results_mutual/       consensus units counted as AGREED (primary)
#   results_mutual_drop/  consensus units dropped entirely (sensitivity)
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
    --results_dir results_mutual --word_measure p_word_w0 --n_boot "${n_boot}" \
    > "logs/analyze_mutual_${split}.log" 2>&1 &
  python analyze.py --split "${split}" --labels_dir labels_mutual \
    --results_dir results_mutual_drop --word_measure p_word_w0 --drop_consensus \
    --n_boot "${n_boot}" > "logs/analyze_mutual_drop_${split}.log" 2>&1 &
done
wait

python mean_diff_bootstrap.py --labels_dir labels_mutual --word_probe word_bp_w0 \
  --out results_mutual/mean_diff_bootstrap.txt > logs/boot_mutual.log 2>&1
python mean_diff_bootstrap.py --labels_dir labels_mutual --word_probe word_bp_w0 \
  --drop_consensus --out results_mutual_drop/mean_diff_bootstrap.txt \
  > logs/boot_mutual_drop.log 2>&1
echo "all done"
