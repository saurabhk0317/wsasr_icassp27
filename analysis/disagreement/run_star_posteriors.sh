#!/usr/bin/env bash
# ============================================================================
# Step 2 driver: star posteriors for every (language, split, model).
#
# Reads models from WSASR/exp_final/respin/<lang>_s5/asr_otcrecip_respin_<lang>_<method>
# (READ-ONLY) and writes posteriors/<lang>_<split>_<method>.pkl.gz here.
# A cell whose output already exists is skipped, so the script can be
# re-run after an interruption.  Labels (step 1) are built first if missing.
#
# Runtime: ~2.5 min per model per split on one GPU (kn test, 2161 utts);
# all 9 langs x 4 models x {dev,test} ~ 3 h on one GPU.  --gpus "0 1" runs
# one job per GPU in parallel.
#
# Usage:
#   ./run_star_posteriors.sh                                  # everything, GPU 0
#   ./run_star_posteriors.sh --langs "kn te" --splits test --gpus "0 1"
#   ./run_star_posteriors.sh --kn_word bpn35                  # which kn otc_word run
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
EXP=/home3/Saurabh/exp/espnet/RESPIN/WSASR/exp_final/respin
VENV=/home3/Saurabh/tools/espnet_may22_2026/tools/venv/bin/activate

langs="bh bn ch hi kn mg mr mt te"
splits="dev test"
methods="ctc otc_word otc_token otc_recip otc_word_tuned otc_token_word otc_token_word_recip"
gpus="0"
kn_word="bpn35"        # kn has an otc_word sweep; bpn19/25/30 never converged

while [ $# -gt 0 ]; do
  case "$1" in
    --langs) langs="$2"; shift 2 ;;
    --splits) splits="$2"; shift 2 ;;
    --methods) methods="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --kn_word) kn_word="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done

# the venv's activate script reads unset variables (LD_LIBRARY_PATH), so it
# cannot be sourced under `set -u`
set +u
# shellcheck disable=SC1090
source "${VENV}"
set -u
mkdir -p "${HERE}/posteriors" "${HERE}/logs" "${HERE}/labels"

jobs_file=$(mktemp)
for lang in ${langs}; do
  for split in ${splits}; do
    [ -f "${HERE}/labels/${lang}_${split}.pkl.gz" ] || \
      python "${HERE}/label_disagreement.py" --lang "${lang}" --split "${split}"
    for m in ${methods}; do
      dir_m="${m}"
      [ "${lang}" = kn ] && [ "${m}" = otc_word ] && dir_m="otc_word_${kn_word}"
      mdir="${EXP}/${lang}_s5/asr_otcrecip_respin_${lang}_${dir_m}"
      out="${HERE}/posteriors/${lang}_${split}_${m}.pkl.gz"
      if [ ! -f "${mdir}/valid.loss.ave.pth" ]; then echo "MISSING model ${mdir}" >&2; continue; fi
      if [ -f "${out}" ]; then echo "skip (exists) ${out}"; continue; fi
      echo "${lang} ${split} ${m} ${mdir} ${out}" >> "${jobs_file}"
    done
  done
done
echo "$(wc -l < "${jobs_file}") job(s) to run on GPU(s): ${gpus}"

# one worker per GPU, each taking every n-th job
read -r -a G <<< "${gpus}"
n=${#G[@]}
pids=()
for k in "${!G[@]}"; do
  (
    i=0
    while read -r lang split m mdir out; do
      if [ $((i % n)) -eq "${k}" ]; then
        log="${HERE}/logs/${lang}_${split}_${m}.log"
        echo "[GPU ${G[$k]}] ${lang} ${split} ${m}"
        CUDA_VISIBLE_DEVICES="${G[$k]}" python "${HERE}/star_posteriors.py" \
          --model_dir "${mdir}" --lang "${lang}" --split "${split}" --method "${m}" \
          --device cuda --out "${out}" > "${log}" 2>&1 \
          || echo "  FAILED ${lang} ${split} ${m} -- see ${log}" >&2
      fi
      i=$((i + 1))
    done < "${jobs_file}"
  ) &
  pids+=($!)
done
wait "${pids[@]}"
rm -f "${jobs_file}"
echo "all done"
