#!/usr/bin/env bash
# ============================================================================
# asr_otc_recip | LibriSpeech train_clean_100 | main WER table : 1 set x 7 methods
#
# Features : frozen SSL -- wav2vec2_base_960 via the s3prl upstream.  The
#            checkpoint downloads to ./hub on first use.
# Tokens   : BPE-200 (sentencepiece).  Word boundaries come from the "▁"
#            piece prefix -- 118 of the 200 pieces carry it.
# Epochs   : 30.   Epoch-schedule decay: 0.775.
# Batching : batch_bins 16000000, accum_grad 4 -- IDENTICAL for all seven
#            methods.  (The earlier asr_otc configs ran CTC and OTC-WORD at
#            accum_grad 1 and the other two at 4, a 4x effective-batch
#            difference inside one column of the table.  Fixed here and
#            guarded by otc_recip/tests/test_config_within_corpus.py.)
#
# METHODS
#   ctc otc_word otc_token otc_recip   the original four (conf/libri_final)
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
# NOTE ON DATA: data_libri/ is not present on every machine.  Point --data_root
# at the LibriSpeech dump wherever it lives; nothing here hard-codes a path.
#
# RESUME: a cell whose score_wer/result.txt already exists is SKIPPED by
# default, with the reason printed.  --force reruns it.
#
# ENCODER (--encoder ebf | conformer), default ebf:
#   ebf        e_branchformer, conf/libri_final/<method>.yaml
#   conformer  conformer,      conf/libri_final/<method>_conformer.yaml
#              and "_conformer" in the asr_tag (same exp_<tag>_<suffix> dir)
# The conformer configs differ from their ebf counterparts only in the
# encoder block and in patience 5 (keep_nbest_models is 5 in both; batching
# stays 16000000 / 4).  ebf runs carry no encoder tag, so existing experiment
# names are unchanged.
#
# Usage
#   ./run_ebf_otc_recip_master_libri.sh --dry-run
#   ./run_ebf_otc_recip_master_libri.sh --methods "otc_token otc_recip"
#   ./run_ebf_otc_recip_master_libri.sh --methods "otc_word_tuned otc_token_word otc_token_word_recip"
#   ./run_ebf_otc_recip_master_libri.sh --data_root /path/to/data_libri --gpu 0
#   ./run_ebf_otc_recip_master_libri.sh --encoder conformer --dry-run
# ============================================================================

. ./otc_recip/run_lib.sh

# ---- options ---------------------------------------------------------------
gpu=0
encoder=ebf                # ebf | conformer (see the header)
train_stage=11            # 10 = redo collect-stats for the SSL frontend
suffix=""                  # "" -> ssl_w2v2base960h_bpe<nbpe>
seeds=20261
nbpe=200
asr_script=asr_respin.sh
train_nj=12
inference_nj=8
ngpu=1
token_type=bpe
use_lm=false
feats_type=raw
audio_format=wav
speed_perturb_factors="0.9 1.0 1.1"
feats_normalize=utt_mvn
bpe_nlsyms=""
lm_config=conf/train_lm.yaml
data_root=data_libri
train_set=train_clean_100
valid_set=dev
test_sets="test_clean test_other"
infer_model=valid.loss.ave.pth

otcr_expand_bool_flags "$@"
set -- "${OTCR_ARGV[@]}"
. utils/parse_options.sh

[ -n "${suffix}" ] || suffix="ssl_w2v2base960h_bpe${nbpe}"

# Resolve the encoder.  ebf keeps the original config names and asr_tags.
case "${encoder}" in
  ebf)       cfg_enc="";           enc_tag="" ;;
  conformer) cfg_enc="_conformer"; enc_tag="conformer" ;;
  *)         otcr_die "--encoder must be 'ebf' or 'conformer', got '${encoder}'" ;;
esac

ALL_METHODS=(ctc otc_word otc_token otc_recip otc_word_tuned otc_token_word otc_token_word_recip)
otcr_filter "${methods}" "${ALL_METHODS[@]}"
RUN_METHODS=("${OTCR_FILTERED[@]}")

OTCR_INFER_CONFIG=conf/libri/decode_ctc.yaml
otcr_require_config "${OTCR_INFER_CONFIG}"

echo "======================================================================"
echo "asr_otc_recip | LibriSpeech ${train_set} | main WER table"
echo "  encoder : ${encoder}"
echo "  methods : ${RUN_METHODS[*]}"
echo "  seeds   : ${seeds}"
echo "  cells   : $(( ${#RUN_METHODS[@]} * $(echo ${seeds} | wc -w) ))"
echo "  data    : ${data_root}"
echo "  gpu     : ${gpu}   dry-run: ${dry_run}   force: ${force}"
echo "======================================================================"

tag="libri100_en"

OTCR_CORPUS=libri
OTCR_LANG=en
OTCR_EXPROOT=exp_libri
OTCR_EXPTAG="${tag}"
OTCR_SUFFIX="${suffix}"
OTCR_SCRIPT="${asr_script}"
OTCR_GPU="${gpu}"
OTCR_TRAIN_STAGE="${train_stage}"
OTCR_INFER_MODEL="${infer_model}"
OTCR_INFER_TAG="decode_ctc_asr_model_${infer_model%.pth}"
OTCR_TESTSETS="${test_sets}"
OTCR_VARIANT="${enc_tag}"

OTCR_COMMON=(
    --tag "${tag}"
    --data_folder "${data_root}"
    --dumpdir "dumps/dump_${tag}"
    --lang en
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
    --bpe_train_text "${data_root}/${train_set}/text"
    --lm_train_text "${data_root}/${train_set}/text"
    --local_score_opts "--score_lang_id false"
    --nj "${train_nj}"
    --ngpu "${ngpu}"
    --inference_nj "${inference_nj}"
)

for method in "${RUN_METHODS[@]}"; do
    for seed in ${seeds}; do
        OTCR_METHOD="${method}"
        OTCR_SEED="${seed}"
        OTCR_CONFIG="conf/libri_final/${method}${cfg_enc}.yaml"
        otcr_run_cell
    done
done

otcr_summary
