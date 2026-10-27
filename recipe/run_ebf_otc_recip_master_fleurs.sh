#!/usr/bin/env bash
# ============================================================================
# asr_otc_recip | FLEURS | main WER table : 14 languages x 7 methods = 98 cells
#
# Features : frozen SSL -- XLS-R 300M (xlsr2_300m.pt) via the s3prl upstream.
#            The checkpoint downloads to ./hub on first use.
# Tokens   : char.  Word boundaries come from the <space> id.
# Epochs   : 30.   Epoch-schedule decay: 0.775.
#
# METHODS
#   ctc otc_word otc_token otc_recip   the original four (conf/fleurs_final)
#   otc_word_tuned  OTC-WORD, tuned weights: bypass -19 / 0.925, self-loop
#                   -1 / 0.975, star arcs from epoch 3 (star_start_epoch: 3)
#   otc_token_word  otc_granularity=dual: token bypass -25 / 0.775, word bypass
#                   -19 / 0.925, self-loop -5 / 0.775, star arcs from epoch 3
#   otc_token_word_recip  otc_recip (entropy-reciprocal, k 1.3) on the dual
#                   graph: token bypass -25 -> -0.01, self-loop -5 -> -0.001,
#                   word bypass -19 -> -1, star arcs from epoch 3
#   All three take every setting outside the loss block from the corpus's
#   other configs.  Epochs 1-2 train on the plain CTC graph; from epoch 3 the
#   epoch-decay clock (the first two) or the entropy schedule (the third)
#   drives the arc weights (see the configs' headers and ctc_otc_recip.py).
#
# NOTE ON DATA: the FLEURS dump is not present on every machine.  The data
# layout is controlled entirely by the flags below --
#
#   --data_root   corpus root                     (default: data_fleurs)
#   --set_tmpl    train/dev/test set name pattern (default: fleurs_%s_%s)
#
# and --langs takes whatever language token that layout uses.  The default
# language list is the 14 English names from the study; if the dump on the
# target machine uses locale codes (en_us, ru_ru, ...) pass them with --langs
# and they are used verbatim.  Nothing here hard-codes a path.
#
# RESUME: a cell whose score_wer/result.txt already exists is SKIPPED by
# default, with the reason printed.  --force reruns it.
#
# ENCODER (--encoder ebf | conformer), default ebf:
#   ebf        e_branchformer, conf/fleurs_final/<method>.yaml
#   conformer  conformer,      conf/fleurs_final/<method>_conformer.yaml
#              and "_conformer" in the asr_tag (same exp_<lang>_<suffix> dir)
# The conformer configs differ from their ebf counterparts only in the
# encoder block and in patience 5 (keep_nbest_models is 5 in both).  ebf runs
# carry no encoder tag, so existing experiment names are unchanged.
#
# Usage
#   ./run_ebf_otc_recip_master_fleurs.sh --dry-run
#   ./run_ebf_otc_recip_master_fleurs.sh --langs "Hindi Greek" --methods otc_recip
#   ./run_ebf_otc_recip_master_fleurs.sh --methods "otc_word_tuned otc_token_word otc_token_word_recip"
#   ./run_ebf_otc_recip_master_fleurs.sh --data_root /path/to/data_fleurs --gpu 2
#   ./run_ebf_otc_recip_master_fleurs.sh --encoder conformer --dry-run
# ============================================================================

. ./otc_recip/run_lib.sh

# ---- options ---------------------------------------------------------------
gpu=0
encoder=ebf                # ebf | conformer (see the header)
train_stage=10            # 10 = redo collect-stats for the SSL frontend
suffix=ssl_xlsr300m
seeds=20261
langs="English Russian Arabic Hindi Bengali Gujarati Punjabi Kannada \
       Malayalam Tamil Telugu Greek Armenian Georgian"   # verbatim
nbpe=10000                 # unused for token_type=char, but asr_respin.sh wants it
asr_script=asr_respin.sh
train_nj=12
inference_nj=8
ngpu=1
token_type=char
use_lm=false
feats_type=raw
audio_format=wav
speed_perturb_factors="0.9 1.0 1.1"
feats_normalize=utt_mvn
bpe_nlsyms=""
lm_config=conf/train_lm.yaml
data_root=data_fleurs
train_tmpl="fleurs_%s_train"
dev_tmpl="fleurs_%s_dev"
test_tmpl="fleurs_test_%s"
infer_model=valid.loss.ave.pth

otcr_expand_bool_flags "$@"
set -- "${OTCR_ARGV[@]}"
. utils/parse_options.sh

# Resolve the encoder.  ebf keeps the original config names and asr_tags.
case "${encoder}" in
  ebf)       cfg_enc="";           enc_tag="" ;;
  conformer) cfg_enc="_conformer"; enc_tag="conformer" ;;
  *)         otcr_die "--encoder must be 'ebf' or 'conformer', got '${encoder}'" ;;
esac

ALL_METHODS=(ctc otc_word otc_token otc_recip otc_word_tuned otc_token_word otc_token_word_recip)
otcr_filter "${methods}" "${ALL_METHODS[@]}"
RUN_METHODS=("${OTCR_FILTERED[@]}")

# Language tokens are taken VERBATIM -- they must match the directory names in
# the dump on this machine (data_fleurs/<lang>, fleurs_<lang>_train, ...).
# Override with --langs; nothing is translated or checked against a baked-in
# list, so locale codes (en_us, ru_ru, ...) work exactly as well as the names.
read -r -a RUN_LANGS <<< "${langs}"
[ ${#RUN_LANGS[@]} -gt 0 ] || otcr_die "--langs is empty"

OTCR_INFER_CONFIG=conf/fleurs/decode_ctc.yaml
otcr_require_config "${OTCR_INFER_CONFIG}"

echo "======================================================================"
echo "asr_otc_recip | FLEURS | main WER table"
echo "  encoder : ${encoder}"
echo "  methods : ${RUN_METHODS[*]}"
echo "  langs   : ${RUN_LANGS[*]}"
echo "  seeds   : ${seeds}"
echo "  cells   : $(( ${#RUN_METHODS[@]} * ${#RUN_LANGS[@]} * $(echo ${seeds} | wc -w) ))"
echo "  data    : ${data_root}"
echo "  gpu     : ${gpu}   dry-run: ${dry_run}   force: ${force}"
echo "======================================================================"

OTCR_CORPUS=fleurs
OTCR_EXPROOT=exp_fleurs
OTCR_SUFFIX="${suffix}"
OTCR_SCRIPT="${asr_script}"
OTCR_GPU="${gpu}"
OTCR_TRAIN_STAGE="${train_stage}"
OTCR_INFER_MODEL="${infer_model}"
OTCR_INFER_TAG="decode_ctc_asr_model_${infer_model%.pth}"
OTCR_VARIANT="${enc_tag}"

for lang in "${RUN_LANGS[@]}"; do
    # shellcheck disable=SC2059
    train_set=$(printf "${train_tmpl}" "${lang}")
    # shellcheck disable=SC2059
    valid_set=$(printf "${dev_tmpl}" "${lang}")
    # shellcheck disable=SC2059
    test_set=$(printf "${test_tmpl}" "${lang}")
    data_folder="${data_root}/${lang}"

    OTCR_LANG="${lang}"
    OTCR_EXPTAG="${lang}"
    OTCR_TESTSETS="${test_set}"

    OTCR_COMMON=(
        --tag "${lang}"
        --data_folder "${data_folder}"
        --dumpdir "dumps/dump_${lang}"
        --lang "${lang}"
        --auxiliary_data_tags ""
        --audio_format "${audio_format}"
        --use_lm "${use_lm}"
        --feats_normalize "${feats_normalize}"
        --lm_config "${lm_config}"
        --token_type "${token_type}"
        --nbpe "${nbpe}"
        --bpe_nlsyms "${bpe_nlsyms}"
        --feats_type "${feats_type}"
        --speed_perturb_factors "${speed_perturb_factors}"
        --train_set "${train_set}"
        --valid_set "${valid_set}"
        --test_sets "${valid_set}"
        --bpe_train_text "${data_folder}/${train_set}/text"
        --lm_train_text "${data_folder}/${train_set}/text"
        --local_score_opts "--score_lang_id false"
        --nj "${train_nj}"
        --ngpu "${ngpu}"
        --inference_nj "${inference_nj}"
    )

    for method in "${RUN_METHODS[@]}"; do
        for seed in ${seeds}; do
            OTCR_METHOD="${method}"
            OTCR_SEED="${seed}"
            OTCR_CONFIG="conf/fleurs_final/${method}${cfg_enc}.yaml"
            otcr_run_cell
        done
    done
done

otcr_summary
