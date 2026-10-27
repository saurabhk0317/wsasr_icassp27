#!/usr/bin/env bash
# ============================================================================
# asr_otc_recip | LABEL SMOOTHING (Table 3) : LibriSpeech only, 4 runs
#
# THE POINT.  "Isn't OTC just label smoothing?"  Both relax the training
# objective away from the exact reference, so the question is fair and a
# reviewer will ask it.  These four runs answer it with the star arcs OFF and
# a uniform-distribution regulariser ON:
#
#     L         = (1 - eps) * L_CTC + eps * L_uniform
#     L_uniform = - mean over VALID frames of [ (1/V) * sum_v log p_t(v) ]
#
# V = real_odim: blank INCLUDED, the <star> column EXCLUDED.
#
# NOTE: espnet's model_conf.lsm_weight smooths the ATTENTION DECODER's
# cross-entropy and does not touch CTC.  There is no CTC label smoothing in
# espnet, so it is implemented in espnet2/asr/ctc_otc_recip.py under
# ctc_conf.arc_schedule="label_smoothing" + ctc_conf.ctc_label_smoothing, and
# pinned down by otc_recip/tests/test_label_smoothing.py.
#
# SWEEP: eps in {0.005, 0.01, 0.02, 0.05} on train_clean_100.
# Each config differs from conf/libri_final/ctc.yaml ONLY by the added term.
#
# RESUME: a cell whose score_wer/result.txt already exists is SKIPPED by
# default.  --force reruns it.
#
# Usage
#   ./run_ebf_otc_recip_labelsmooth.sh --dry-run
#   ./run_ebf_otc_recip_labelsmooth.sh --eps "0.01 0.02"
#   ./run_ebf_otc_recip_labelsmooth.sh --data_root /path/to/data_libri --gpu 0
# ============================================================================

. ./otc_recip/run_lib.sh

gpu=0
train_stage=10             # 10 = redo collect-stats for the SSL frontend
suffix=""                  # "" -> ssl_w2v2base960h_bpe<nbpe>
seeds=20261
eps="0.005 0.01 0.02 0.05" # the Table-3 sweep; verbatim
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

# eps values are taken VERBATIM; each selects
# conf/label_smoothing/libri_ls<eps, "." -> "p">.yaml and the run aborts if
# that config does not exist, so a typo fails loudly instead of being skipped.
read -r -a RUN_EPS <<< "${eps}"
[ ${#RUN_EPS[@]} -gt 0 ] || otcr_die "--eps is empty"

OTCR_INFER_CONFIG=conf/libri/decode_ctc.yaml
otcr_require_config "${OTCR_INFER_CONFIG}"

echo "======================================================================"
echo "asr_otc_recip | LABEL SMOOTHING (Table 3) | LibriSpeech ${train_set}"
echo "  eps     : ${RUN_EPS[*]}"
echo "  seeds   : ${seeds}"
echo "  cells   : $(( ${#RUN_EPS[@]} * $(echo ${seeds} | wc -w) ))"
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
OTCR_METHOD=labelsmooth

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

for e in "${RUN_EPS[@]}"; do
    slug="${e//./p}"
    OTCR_VARIANT="ls${slug}"
    OTCR_CONFIG="conf/label_smoothing/libri_ls${slug}.yaml"
    for seed in ${seeds}; do
        OTCR_SEED="${seed}"
        otcr_run_cell
    done
done

otcr_summary
