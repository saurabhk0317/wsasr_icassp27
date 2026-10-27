#!/usr/bin/env bash
# ============================================================================
# asr_otc_recip | RESPIN | Table 1 : 9 languages x 7 methods = 63 cells
#
# Features : STFT filterbank, NO SSL.  A frozen SSL frontend gives no benefit
#            on RESPIN in our prior work, so the non-SSL setup is the one
#            reported.
# Tokens   : char (the <space> id is what word-granularity derives its word
#            boundaries from -- present in all 9 RESPIN char token lists).
# Epochs   : 70.   Epoch-schedule decay: 0.775.
#
# METHODS
#   ctc         CTC baseline -- the same k2 graph with the star arcs OFF
#   otc_word    OTC, one <star> arc per WORD    (icefall / Gao-2023 parity)
#   otc_token   OTC, one <star> arc per TOKEN   (-25/-5, epoch decay 0.775)
#   otc_recip   OTC, token graph, ENTROPY-RECIPROCAL schedule (k=1.3)
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
# RESUME: a cell whose score_wer/result.txt already exists is SKIPPED by
# default, with the reason printed.  --force reruns it.
#
# RESPIN is run with filterbank features only (no SSL variant):
#   conf/respin_final/<method>[_conformer].yaml, exp suffix stft_char.
#
# ENCODER (--encoder ebf | conformer), default ebf:
#
#   ebf        e_branchformer, conf/respin_final/<method>.yaml
#   conformer  conformer,      conf/respin_final/<method>_conformer.yaml
#              and "_conformer" in the asr_tag (same exp_<tag>_<suffix> dir)
#
# The conformer configs differ from their ebf counterparts only in the
# encoder block and in patience 5 / keep_nbest_models 5; batching, epochs,
# warmup, specaug and the loss are the same.  ebf runs carry no encoder tag,
# so existing experiment names are unchanged.
#
# Usage
#   ./run_ebf_otc_recip_master_respin.sh --dry-run
#   ./run_ebf_otc_recip_master_respin.sh --methods otc_recip
#   ./run_ebf_otc_recip_master_respin.sh --methods "otc_word_tuned otc_token_word otc_token_word_recip"
#   ./run_ebf_otc_recip_master_respin.sh --langs "hi kn" --gpu 1
#   ./run_ebf_otc_recip_master_respin.sh --methods "ctc otc_token" --force
#   ./run_ebf_otc_recip_master_respin.sh --encoder conformer --dry-run
#
# --langs is taken VERBATIM: pass whatever tokens name the language
# directories in your dump. Nothing is translated or validated against a
# baked-in list.
# ============================================================================

. ./otc_recip/run_lib.sh

# ---- options ---------------------------------------------------------------
gpu=1
encoder=ebf                # ebf | conformer (see the header)
train_stage=11             # features already dumped
suffix=stft_char
seeds=20261                # one seed here; the sweep lives in *_seeds.sh
langs="bn bh ch hi kn mg mt mr te"   # verbatim; must match the dump's dir names
nbpe=500                   # unused for token_type=char, but asr_respin.sh wants it
asr_script=asr_respin.sh
train_nj=12
inference_nj=16
ngpu=1
token_type=char
use_lm=false
feats_type=raw
audio_format=wav
speed_perturb_factors="0.9 1.0 1.1"
feats_normalize=utt_mvn
bpe_nlsyms=""
lm_config=conf/train_lm.yaml
data_root=data_respin       # corpus root; override for another machine
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
# the dump on this machine (data_respin/<lang>, IISc_RESPIN_train_<lang>_s5).
# The default is the 9 RESPIN languages as this dump names them:
#   bn bengali | bh bhojpuri | ch chhattisgarhi | hi hindi | kn kannada
#   mg magahi  | mt maithili | mr marathi       | te telugu
read -r -a RUN_LANGS <<< "${langs}"
[ ${#RUN_LANGS[@]} -gt 0 ] || otcr_die "--langs is empty"

OTCR_INFER_CONFIG=conf/respin/decode_ctc.yaml
otcr_require_config "${OTCR_INFER_CONFIG}"

echo "======================================================================"
echo "asr_otc_recip | RESPIN | Table 1"
echo "  features: filterbank (suffix ${suffix}, start stage ${train_stage})"
echo "  encoder : ${encoder}"
echo "  methods : ${RUN_METHODS[*]}"
echo "  langs   : ${RUN_LANGS[*]}"
echo "  seeds   : ${seeds}"
echo "  cells   : $(( ${#RUN_METHODS[@]} * ${#RUN_LANGS[@]} * $(echo ${seeds} | wc -w) ))"
echo "  gpu     : ${gpu}   dry-run: ${dry_run}   force: ${force}"
echo "======================================================================"

OTCR_CORPUS=respin
OTCR_EXPROOT=exp_respin
OTCR_SUFFIX="${suffix}"
OTCR_SCRIPT="${asr_script}"
OTCR_GPU="${gpu}"
OTCR_TRAIN_STAGE="${train_stage}"
OTCR_INFER_MODEL="${infer_model}"
OTCR_VARIANT="${enc_tag}"
OTCR_INFER_TAG="decode_ctc_asr_model_${infer_model%.pth}"

for code in "${RUN_LANGS[@]}"; do
    tag="${code}_s5"
    data_folder="${data_root}/${code}"
    train_set="IISc_RESPIN_train_${tag}"
    train_dev="IISc_RESPIN_dev_${code}"

    OTCR_LANG="${code}"
    OTCR_EXPTAG="${tag}"
    OTCR_TESTSETS="IISc_RESPIN_test_${code}"

    OTCR_COMMON=(
        --tag "${tag}"
        --data_folder "${data_folder}"
        --dumpdir "dumps/dump_${tag}"
        --lang "${code}"
        --auxiliary_data_tags ""
        --local_data_opts "--stage 0 --lang ${code}"
        --post_process_local_data_opts "--stage 2 --lang ${code}"
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
        --valid_set "${train_dev}"
        --test_sets "${train_dev}"
        --bpe_train_text "${data_folder}/${train_set}/text"
        --lm_train_text "${data_folder}/${train_set}/text"
        --local_score_opts "--score_lang_id true"
        --nj "${train_nj}"
        --ngpu "${ngpu}"
        --inference_nj "${inference_nj}"
    )

    for method in "${RUN_METHODS[@]}"; do
        for seed in ${seeds}; do
            OTCR_METHOD="${method}"
            OTCR_SEED="${seed}"
            OTCR_CONFIG="conf/respin_final/${method}${cfg_enc}.yaml"
            otcr_run_cell
        done
    done
done

otcr_summary
