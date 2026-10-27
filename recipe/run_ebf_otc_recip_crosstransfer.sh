#!/usr/bin/env bash
# ============================================================================
# asr_otc_recip | CROSS-TRANSFER SWEEPS on 10h subsets
#
# Three modes, run over the same controlled setup so they are comparable:
#
#   ctc         the reference line.  Star arcs off.  Nothing swept: exactly one
#               cell per corpus / language / variant / seed.
#   otc_token   sweeps the epoch-indexed geometric DECAY.  The decay is what
#               ties OTC-TOKEN to its epoch budget, so the sweep shows where the
#               optimum sits and how far it moves between feature types.
#   otc_recip   sweeps the entropy shaping exponent K (entropy_shape_k).  The
#               entropy schedule has no decay, so K is the closest thing it has
#               to a tunable; the sweep is evidence for how flat that axis is.
#
# THE POINT of running all three on one subset: if the OTC-TOKEN optimum shifts
# between non-SSL and SSL while the OTC-RECIP K optimum does not, that is the
# paper's claim measured directly rather than inferred from two points.
#
# ---------------------------------------------------------------------------
# OTC-WORD ARC ABLATION + OTC-TOKEN-WORD.  Five more modes, OPT-IN (never in
# the default --modes) and SSL ONLY (nonssl cells are skipped with a note).
# They form a pipeline; later stages use the best values of earlier ones:
#
#   word_sl         otc_word, SELF-LOOP ONLY (bypass arc off).
#                   self-loop weight in --word_sl_weights (3 1 0 -1 -3 -5),
#                   all at decay --word_sl_weight_decay (0.975).
#   word_sl_decay   otc_word, self-loop only, at the BEST word_sl weight;
#                   decay over --word_decays (1.0 .. 0.6, step 0.05).
#   word_byp        otc_word, BYPASS ONLY (self-loop arc off).
#                   weight --word_byp_weight (-19); decay over --word_decays.
#   word_both       otc_word, both arcs: the best self-loop (weight, decay)
#                   over ALL word_sl + word_sl_decay cells, and weight -19 at
#                   the best word_byp decay.
#   otc_token_word  otc_granularity=dual: token bypass --tw_token_weight /
#                   --tw_token_decay (-25 / 0.775), word bypass -19 at the best
#                   word_byp decay, self-loop --tw_self_loop_weight /
#                   --tw_self_loop_decay (-5 / 0.775).
#
# "BEST" = lowest DEV-SET WER (score_wer/result.txt of the dev decode), per
# corpus/language, averaged over --seeds.  A stage is BLOCKED -- reported and
# skipped, never guessed -- until every cell it depends on has a fully scored
# dev decode (result_lid.txt present).  Ties go to the first value in sweep
# order.  Under --action train (not --dry-run) the background decode queue is
# drained before a pick, so e.g. --modes "word_sl word_sl_decay" runs end to
# end in one invocation.  Overrides skip the lookup; they apply to EVERY
# selected corpus/language, so select one with --corpora/--<corpus>_langs:
#   --word_sl_best_weight W                  used by word_sl_decay
#   --word_both_sl_weight W --word_both_sl_decay D   (together) used by word_both
#   --word_byp_best_decay D                  used by word_both and otc_token_word
# DELAYED STAR ARCS.  word_byp, word_both and otc_token_word train with
# --ctc_conf star_start_epoch=<--word_star_start_epoch> (default 3): epochs
# before it use the plain CTC graph and the star arcs exist from then on, with the
# epoch-decay clock starting there (see ctc_otc_recip.py).  An untrained model
# sends most lattice paths through the star arcs, which kept some runs from
# converging.  word_sl and word_sl_decay are NOT delayed: those sweeps were
# already run with arcs from epoch 1, and their remaining cells must match.
# Run TAGS DO NOT encode the delay (so finished cells keep being found); the
# delay is recorded in each run's config.yaml.  --word_star_start_epoch 1
# restores arcs from the start for these three modes.
#
# If the best self-loop weight is 0, word_sl_decay runs nothing (0 * decay^e
# is 0 for every decay, so the cells would be identical) and word_both uses
# that weight at --word_sl_weight_decay.
#
# OTC-TOKEN-WORD ON THE ENTROPY SCHEDULE.  One more opt-in, ssl-only mode with
# no picks of its own:
#
#   otc_token_word_recip  otc_granularity=dual under arc_schedule=entropy_reciprocal:
#                   ONE entropy-driven progress for all three arcs, each gliding
#                   log-geometrically from its start to its end weight:
#                     token bypass  --twr_token_start/_end   (-25 -> -0.01)
#                     self-loop     --twr_sl_start/_end      (-5  -> -0.001)
#                     word bypass   --twr_word_start/_end    (-19 -> -1)
#                   sweeps entropy_shape_k over --twr_ks (0.8 0.9 1 1.2 1.3 1.4);
#                   star arcs from --word_star_start_epoch.
#
#   Base config conf/cross_transfer/<corpus>_otc_recip_ssl.yaml (the otc_recip
#   entropy keys: soft, dev, alpha 0.3, low 0.01, eps 1e-4; its token and
#   self-loop start/end are the same -25/-0.01 and -5/-0.001) with
#     --ctc_conf otc_granularity=dual, the six start/end weights,
#     entropy_shape_k=<K> and star_start_epoch.
#   A start and its end must share a sign and be non-zero: the schedule
#   interpolates log|w|, which cannot cross or reach 0.
#   Tag: otc_token_word_recip_ssl_tok<S>to<E>_word<S>to<E>_sl<S>to<E>_k<K>
#
# All five word-pipeline modes reuse conf/cross_transfer/<corpus>_otc_token_ssl.yaml and set the
# loss through --ctc_conf, as the sweeps above do:
#   word_sl/_decay  otc_granularity=word allow_bypass_arc=false allow_self_loop_arc=true
#                   initial_self_loop_weight=W self_loop_weight_decay=D
#   word_byp        otc_granularity=word allow_bypass_arc=true allow_self_loop_arc=false
#                   initial_bypass_weight=-19 bypass_weight_decay=D
#   word_both       otc_granularity=word, both arcs, all four keys
#   otc_token_word  otc_granularity=dual, both arcs, the four token/self-loop
#                   keys plus initial_word_bypass_weight / word_bypass_weight_decay
#   (word_byp, word_both and otc_token_word also get star_start_epoch, above)
# Run tags (otcrecip_<corpus>_<lang>_<method>_<variant>_seed<N>; "p" = ".",
# "m" = "-"):
#   otc_word_ssl_slonly_sl<W>_sld<D>        otc_word_ssl_byponly_byp<W>_bypd<D>
#   otc_word_ssl_both_sl<W>_sld<D>_byp<W>_bypd<D>
#   otc_token_word_ssl_tok<W>_tokd<D>_word<W>_wordd<D>_sl<W>_sld<D>
#
# NO PER-VALUE CONFIG FILES.  The swept parameter is injected at run time:
#     otc_token   --ctc_conf bypass_weight_decay=<D> --ctc_conf self_loop_weight_decay=<D>
#     otc_recip   --ctc_conf entropy_shape_k=<K>
# espnet's NestedDictAction MERGES a "key=value" form into the block loaded
# from --config rather than replacing it, so only the named key moves and
# everything else comes from the shipped config.
#
# ---------------------------------------------------------------------------
# WHAT RUNS.  corpora x languages x variants x modes x sweep values x seeds.
#
#   --action             train | decode   (default: train)
#   --corpora            space-separated (default: respin libri fleurs)
#   --variants           subset of "nonssl ssl" (default: both)
#   --modes              subset of "ctc otc_token otc_recip" (default: all), plus
#                        the opt-in "word_sl word_sl_decay word_byp word_both
#                        otc_token_word otc_token_word_recip" (see OTC-WORD ARC
#                        ABLATION and OTC-TOKEN-WORD ON THE ENTROPY SCHEDULE)
#   --decays             explicit decay list; overrides the range below
#   --decay_from/_to/_step   default 1.0 / 0.6 / 0.025
#   --ks                 explicit K list; overrides the range below
#   --k_from/_to/_step   default 0.8 / 1.4 / 0.1  -> 0.8 0.9 1 1.1 1.2 1.3 1.4
#   --<corpus>_langs     language tokens, VERBATIM (must match the dump's dirs)
#   --<corpus>_* (below) per-corpus dataset names, so a subset can be selected
#
# A range whose step does not divide it exactly is a WARNING, not a silent
# truncation: the sweep stops short of the far endpoint and says so.
#
# ---------------------------------------------------------------------------
# TWO ACTIONS (--action train | decode), and one run does exactly one of them.
# They are separated because they want different machines: training holds the
# GPU and must be serialised, whereas decoding is CPU work that can be left to
# grind through a backlog.
#
#   --action train    Trains every selected cell that has no model yet, one at
#                     a time.  Cells that already have a model are left alone.
#                     When a training finishes, that run's DEV and TEST decodes
#                     are handed to the background queue and the next training
#                     starts immediately -- it does not wait for the scoring.
#
#   --action decode   Trains nothing.  Walks every selected cell that already
#                     HAS a model and finishes whichever of its dev/test decodes
#                     is incomplete, one at a time in the foreground.  Cells
#                     still waiting to be trained are counted, not decoded.
#
# So the intended loop is: leave "--action train" running on the GPU box, and
# run "--action decode" whenever the score table needs filling in -- including
# for runs trained earlier, or on another machine.
#
# DECODE SETS are the corpus's dev set followed by its test set(s):
#   respin  IISc_RESPIN_dev_<lang>   IISc_RESPIN_test_<lang>
#   libri   dev                      <libri_testsets>  (default test_clean;
#                                     pass "test_clean test_other" for both)
#   fleurs  fleurs_<lang>_dev        fleurs_test_<lang>
# Each set is a separate job with its own state, so a cell whose dev decode
# landed but whose test decode did not is finished off rather than redone.
# --<corpus>_testsets "" decodes the dev set only.
#
# WHAT "INCOMPLETE" MEANS.  Two files in the set's output directory say how far
# that set got, and --action decode resumes from there rather than redoing work:
#
#   text + result_lid.txt   inference AND scoring are done      -> skip
#   text only               inference done, scoring is not      -> stage 13 only
#   neither                 nothing usable                      -> stages 12-13
#
# "text" is stage 12's hypothesis file.  "result_lid.txt" is the marker for
# scoring because asr_respin.sh writes it LAST in stage 13, after every
# score_<type>/result.txt -- so score_wer/result.txt alone can mean a scoring
# pass that died partway through, whereas result_lid.txt cannot.  It is written
# for every corpus here, not only the ones with --score_lang_id true.
#
# The output directory is <expdir>/<inference_tag>/<set>/ for a TEST set but
# <expdir>/<inference_tag>/org/<valid_set>/ for the DEV set: asr_respin.sh
# takes the valid set out of --test_sets, flips eval_valid_set, and decodes it
# under "org/".  xt_decode_subdir() is the single place that knows this.
#
# Rescoring is cheap and inference is not, so the "text only" case is worth the
# extra state: it turns a killed or half-finished scoring pass into seconds of
# work instead of a full re-decode.  --force ignores all of this and reruns
# stages 12-13.
#
# Decoding always runs on CPU (--gpu_inference false) so it never contends with
# training for the GPU.
#
# WHERE THE BACKGROUND QUEUE APPLIES: --action train ONLY.  It exists for one
# reason -- so the next training starts the moment the previous one finishes
# instead of waiting on its scoring.  Under --action decode there is nothing to
# overlap with (the decodes ARE the work, and they run one at a time), so they
# run in the FOREGROUND, where the output is live and Ctrl-C stops the decode
# directly.  --max_decode_jobs (default 1) therefore only means anything under
# --action train, where 0 makes even those decodes inline.
#
# WHERE TO LOOK WHILE A DECODE RUNS.  Two paths are printed as each job starts:
#   wrapper log   logs_otc_recip/<corpus>/<tag>.decode_<set>.log
#                 asr_respin.sh's own stdout (".score_<set>.log" for a rescore)
#   espnet  log   <expdir>/<inference_tag>/<subdir>/logdir/asr_inference.*.log
#                 espnet's per-job inference logs -- the real decoding progress,
#                 and the traceback when a job dies.  Stage 13 writes no such
#                 directory, so a rescore prints only the wrapper log.
#
# Each BACKGROUND decode writes its exit status to <log>.status; that file, not
# the shell's job table, is the completion signal.  `wait` is ambiguous here:
# `wait -n` reaps a job, and a later `wait <pid>` on the same PID reports "not
# a child" rather than the real status, which would show a successful decode as
# a failure.
#
# RESUME: under --action train a cell is skipped when its model exists; under
# --action decode a set is skipped when it has both marker files, and resumed
# at stage 13 when it has only "text".  --force overrides both.  A selection
# matching NO cell is an ERROR, not a silent no-op.  Interrupting terminates the running decode.
#
# MULTIPLE GPUS (--action train).  --gpu takes a space-separated list, e.g.
# --gpu "0 1".  Each GPU runs ONE training at a time: a cell starts on the
# first free GPU (CUDA_VISIBLE_DEVICES=<that id>), and when a training
# finishes its dev/test decodes are queued exactly as above and the GPU takes
# the next cell.  Cells still start in enumeration order.  With a single id
# (the default "0") the order of work is the same as before.  Trainings run in
# the background with their exit status in
# logs_otc_recip/<corpus>/<tag>.train.status (the completion signal, as for
# decodes); the word-pipeline picks and the end of the run wait for every
# running training first, and an interrupt terminates them (whole process
# tree) along with the decodes.  More GPUs finish trainings faster than one
# decode slot may score them; raise --max_decode_jobs if the queue backs up.
#
# This asynchronous behaviour is local to THIS script: it overrides
# otcr_run_cell after sourcing otc_recip/run_lib.sh, so the other run scripts
# keep the original train-then-decode-in-sequence behaviour.
#
# ---------------------------------------------------------------------------
# CONFIG RESOLUTION.  One file per (corpus, mode, variant):
#     conf/cross_transfer/<corpus>_<mode>_<variant>.yaml
# These are sweep-specific and deliberately NOT the main-table configs under
# conf/<corpus>_final/, which this script never reads.  Held fixed across all
# 18, so that a shift in an optimum is attributable to the front-end and the
# epoch budget alone:
#     initial_bypass_weight    -25.0  (both variants; the SSL main-table configs
#                                      use -15, which would confound the sweep)
#     initial_self_loop_weight  -5.0
#     batch_bins / accum_grad   6000000 / 1  (these are 10h subsets)
#     warmup_steps              5000 non-SSL, 3000 SSL
#
# DATASETS.  Defaults point at the 10h subsets:
#   respin  IISc_RESPIN_train_<lang>_<respin_tag>   (respin_tag=s5_ran_10h)
#   libri   <libri_trainset>                        (train_clean_100_ran_10h)
#   fleurs  printf <fleurs_train_tmpl> <lang>       (fleurs_<lang>_train)
#
# Usage
#   ./run_ebf_otc_recip_crosstransfer.sh --dry-run
#   ./run_ebf_otc_recip_crosstransfer.sh --corpora libri --variants ssl
#   ./run_ebf_otc_recip_crosstransfer.sh --action decode --corpora libri
#   ./run_ebf_otc_recip_crosstransfer.sh --action decode --libri_testsets "test_clean test_other"
#   ./run_ebf_otc_recip_crosstransfer.sh --modes "ctc otc_recip"
#   ./run_ebf_otc_recip_crosstransfer.sh --modes otc_recip --k_step 0.2
#   ./run_ebf_otc_recip_crosstransfer.sh --modes otc_token --decays "0.7 0.9"
#   ./run_ebf_otc_recip_crosstransfer.sh --variants ssl --modes "word_sl word_byp"
#   ./run_ebf_otc_recip_crosstransfer.sh --variants ssl --corpora libri --modes otc_token_word_recip
#   ./run_ebf_otc_recip_crosstransfer.sh --gpu "0 1" --max_decode_jobs 2 --variants ssl \
#       --modes "word_sl word_byp word_sl_decay word_both otc_token_word"
#   ./run_ebf_otc_recip_crosstransfer.sh --variants ssl --corpora libri \
#       --modes "word_sl_decay word_both otc_token_word"
#   ./run_ebf_otc_recip_crosstransfer.sh --variants ssl --corpora libri \
#       --modes word_both --word_byp_best_decay 0.9 --word_both_sl_weight -1 --word_both_sl_decay 0.85
# ============================================================================

. ./otc_recip/run_lib.sh

gpu=0                      # one id, or several: --gpu "0 1" (one training per GPU)
action=train               # train | decode   (see the header)
seeds=20261
corpora="respin libri fleurs"
variants="nonssl ssl"
modes="ctc otc_token otc_recip"

# ---- otc_token: decay range -----------------------------------------------
decays=""                  # explicit list; when set it overrides the range
decay_from=1.0
decay_to=0.6
decay_step=0.025

# ---- otc_recip: entropy_shape_k range --------------------------------------
ks=""                      # explicit list; when set it overrides the range
k_from=0.8
k_to=1.4
k_step=0.1

# ---- OTC-word arc ablation + otc_token_word (opt-in, ssl only; see header) --
word_sl_weights="3 1 0 -1 -3 -5"   # word_sl: self-loop weights
word_sl_weight_decay=0.975          # word_sl: their decay
word_byp_weight=-19                 # word_byp / word_both / otc_token_word
word_decays=""                      # explicit list; overrides the range below
word_decay_from=1.0                 # word_sl_decay and word_byp decays
word_decay_to=0.90
word_decay_step=0.025
tw_token_weight=-25                 # otc_token_word: token bypass
tw_token_decay=0.775
tw_self_loop_weight=-5              # otc_token_word: self-loop
tw_self_loop_decay=0.775
# otc_token_word_recip (dual graph, entropy_reciprocal; see the header)
twr_ks="0.8 0.9 1.0 1.1 1.2 1.3 1.4"   # entropy_shape_k sweep
twr_token_start=-25                 # token bypass: start -> end
twr_token_end=-0.01
twr_sl_start=-5                     # self-loop: start -> end
twr_sl_end=-0.001
twr_word_start=-19                  # word bypass: start -> end
twr_word_end=-1
# Manual picks; "" = choose the lowest dev WER from the finished cells.
word_sl_best_weight=""              # -> word_sl_decay
word_both_sl_weight=""              # -> word_both (with word_both_sl_decay)
word_both_sl_decay=""
word_byp_best_decay=""              # -> word_both and otc_token_word
# First epoch with star arcs for word_byp / word_both / otc_token_word.
word_star_start_epoch=3

# ---- per corpus: languages and dataset names.  Verbatim; 10h subsets. ------
respin_langs="mt"
libri_langs="en"
fleurs_langs="Hindi"

respin_tag=s5_ran_10h                    # -> IISc_RESPIN_train_<lang>_<tag>
libri_trainset=train_clean_100_ran_10h
# The exp tag must encode the SUBSET, not just the corpus.  The full-data runs
# already occupy exp_libri/exp_libri100_en_*, and reusing that tag for the 10h
# sweep would make the resume logic treat 100h results as 10h ones.
libri_exptag=libri100_ran_10h
fleurs_train_tmpl="fleurs_%s_train"
fleurs_dev_tmpl="fleurs_%s_dev"
fleurs_test_tmpl="fleurs_test_%s"

# ---- test sets: decoded after the dev set.  Empty -> dev set only. ---------
# LibriSpeech ships two; only test_clean is decoded by default because the
# sweep is 25 cells per corpus/variant and each extra set is another 25 jobs.
respin_testsets_tmpl="IISc_RESPIN_test_%s"
libri_testsets="test_clean"
fleurs_testsets_tmpl="${fleurs_test_tmpl}"

asr_script=asr_respin.sh
max_decode_jobs=1          # --action train only: concurrent background
                           # decodes, 0 = inline.  --action decode always runs
                           # its decodes inline, one at a time.
train_stage=11             # both variants; lower to 10 to redo collect-stats
train_nj=12
inference_nj=32
ngpu=1
use_lm=false
feats_type=raw
audio_format=wav
speed_perturb_factors="0.9 1.0 1.1"
feats_normalize=utt_mvn
bpe_nlsyms=""
lm_config=conf/train_lm.yaml
infer_model=valid.loss.ave.pth

respin_data_root=data_respin
libri_data_root=data_libri
fleurs_data_root=data_fleurs

libri_nbpe_ssl=200
libri_nbpe_nonssl=200

otcr_expand_bool_flags "$@"
set -- "${OTCR_ARGV[@]}"
. utils/parse_options.sh

[ -n "${corpora}" ]  || otcr_die "--corpora is empty"
[ -n "${variants}" ] || otcr_die "--variants is empty"
[ -n "${modes}" ]    || otcr_die "--modes is empty"

for v in ${variants}; do
    case "${v}" in
        nonssl|ssl) ;;
        *) otcr_die "--variants may only contain 'nonssl' and 'ssl', got '${v}'" ;;
    esac
done
XT_WORD_MODES="word_sl word_sl_decay word_byp word_both otc_token_word otc_token_word_recip"
for m in ${modes}; do
    case "${m}" in
        ctc|otc_token|otc_recip) ;;
        word_sl|word_sl_decay|word_byp|word_both|otc_token_word|otc_token_word_recip) ;;
        *) otcr_die "--modes may only contain 'ctc', 'otc_token', 'otc_recip' or one of '${XT_WORD_MODES}', got '${m}'" ;;
    esac
done
# Run order: the original modes as given, then the word pipeline in dependency
# order, so a stage always comes after the stages it picks its values from.
RUN_MODES=""
for m in ${modes}; do
    case " ${XT_WORD_MODES} " in *" ${m} "*) ;; *) RUN_MODES+="${m} " ;; esac
done
for m in ${XT_WORD_MODES}; do
    case " ${modes} " in *" ${m} "*) RUN_MODES+="${m} " ;; esac
done
case "${action}" in
    train|decode) ;;
    *) otcr_die "--action must be 'train' or 'decode', got '${action}'" ;;
esac
case "${max_decode_jobs}" in
    ''|*[!0-9]*) otcr_die "--max_decode_jobs must be a non-negative integer, got '${max_decode_jobs}'" ;;
esac
read -r -a XT_GPUS <<< "${gpu}"
[ ${#XT_GPUS[@]} -gt 0 ] || otcr_die "--gpu is empty"
for g in "${XT_GPUS[@]}"; do
    case "${g}" in ''|*[!0-9]*) otcr_die "--gpu must be GPU ids separated by spaces, got '${gpu}'" ;; esac
done
_seen_gpus=" "
for g in "${XT_GPUS[@]}"; do
    case "${_seen_gpus}" in *" ${g} "*) otcr_die "--gpu lists GPU ${g} twice" ;; esac
    _seen_gpus+="${g} "
done

# Backgrounding a decode is only worth its machinery when there is something to
# overlap it with, and that is exactly --action train: the next training must
# not wait on the previous one's scoring.  Under --action decode the decodes are
# the only work and run one at a time, so backgrounding them would buy nothing
# and cost the live output and a direct Ctrl-C.  Zero here means "inline".
if [ "${action}" = "decode" ]; then
    max_decode_jobs=0
fi

# --------------------------------------------------------------------------
# xt_numlist <name> <from> <to> <step>
#   Echoes an inclusive numeric range, trimmed of trailing zeros so 1.00 -> 1
#   and 0.90 -> 0.9.  That keeps run tags ("decay0p7") identical to the ones
#   earlier runs already use, so completed cells are still found and skipped.
#   Warns on stderr if the step does not land exactly on the far endpoint.
# --------------------------------------------------------------------------
xt_numlist() {
    local name="$1" a="$2" b="$3" s="$4" out miss
    local v
    for v in "${a}" "${b}" "${s}"; do
        case "${v}" in
            ''|*[!0-9.]*|*.*.*) otcr_die "--${name}_* must be numbers, got '${v}'" ;;
        esac
    done
    out="$(awk -v a="${a}" -v b="${b}" -v s="${s}" 'BEGIN{
        if (s <= 0) { print "STEP_NOT_POSITIVE"; exit }
        dir = (b >= a) ? 1 : -1
        n = int(((dir>0 ? b-a : a-b) / s) + 1e-9)
        for (i = 0; i <= n; i++) {
            v = a + dir*i*s
            t = sprintf("%.4f", v); sub(/0+$/, "", t); sub(/\.$/, "", t)
            printf "%s ", t
        }
        last = a + dir*n*s
        if ((last - b)^2 > 1e-12) printf "\nENDPOINT_MISSED:%s", sprintf("%.4f", last)
    }')"
    case "${out}" in
        *STEP_NOT_POSITIVE*) otcr_die "--${name}_step must be > 0, got '${s}'" ;;
    esac
    miss="$(printf '%s' "${out}" | sed -n '2p')"
    if [ -n "${miss}" ]; then
        echo "WARNING: --${name}_step ${s} does not divide ${a}..${b} exactly;" >&2
        echo "         the sweep stops at ${miss#ENDPOINT_MISSED:} and never reaches ${b}." >&2
    fi
    printf '%s' "${out}" | head -1
}

if [ -z "${decays}" ]; then
    decays="$(xt_numlist decay "${decay_from}" "${decay_to}" "${decay_step}")"
fi
if [ -z "${ks}" ]; then
    ks="$(xt_numlist k "${k_from}" "${k_to}" "${k_step}")"
fi
read -r -a RUN_DECAYS <<< "${decays}"
read -r -a RUN_KS     <<< "${ks}"
[ ${#RUN_DECAYS[@]} -gt 0 ] || otcr_die "--decays is empty"
[ ${#RUN_KS[@]}     -gt 0 ] || otcr_die "--ks is empty"

# --------------------------------------------------------------------------
# xt_num <flag> <value>
#   Canonical signed decimal (1.0 -> 1, 0.950 -> 0.95, -19.0 -> -19), so a
#   value always produces the same run tag however it was typed.
# xt_slug <value>
#   Tag-safe form: "." -> "p", "-" -> "m"  (-0.5 -> m0p5).
# --------------------------------------------------------------------------
xt_num() {
    case "$2" in
        ''|-|.|-.|*[!0-9.-]*|*.*.*|?*-*) otcr_die "--$1 must be a number, got '$2'" ;;
    esac
    awk -v v="$2" 'BEGIN{ t = sprintf("%.4f", v); sub(/0+$/, "", t); sub(/\.$/, "", t)
                          if (t == "-0") t = "0"; print t }'
}
xt_slug() { local s="${1//./p}"; printf '%s' "${s//-/m}"; }

if [ -z "${word_decays}" ]; then
    word_decays="$(xt_numlist word_decay "${word_decay_from}" "${word_decay_to}" "${word_decay_step}")"
fi
RUN_WORD_DECAYS=(); for v in ${word_decays};     do RUN_WORD_DECAYS+=("$(xt_num word_decays "${v}")"); done
RUN_SL_WEIGHTS=();  for v in ${word_sl_weights}; do RUN_SL_WEIGHTS+=("$(xt_num word_sl_weights "${v}")"); done
[ ${#RUN_WORD_DECAYS[@]} -gt 0 ] || otcr_die "--word_decays is empty"
[ ${#RUN_SL_WEIGHTS[@]}  -gt 0 ] || otcr_die "--word_sl_weights is empty"
word_sl_weight_decay="$(xt_num word_sl_weight_decay "${word_sl_weight_decay}")"
word_byp_weight="$(xt_num word_byp_weight "${word_byp_weight}")"
tw_token_weight="$(xt_num tw_token_weight "${tw_token_weight}")"
tw_token_decay="$(xt_num tw_token_decay "${tw_token_decay}")"
tw_self_loop_weight="$(xt_num tw_self_loop_weight "${tw_self_loop_weight}")"
tw_self_loop_decay="$(xt_num tw_self_loop_decay "${tw_self_loop_decay}")"
RUN_TWR_KS=(); for v in ${twr_ks}; do RUN_TWR_KS+=("$(xt_num twr_ks "${v}")"); done
[ ${#RUN_TWR_KS[@]} -gt 0 ] || otcr_die "--twr_ks is empty"
for _n in twr_token_start twr_token_end twr_sl_start twr_sl_end twr_word_start twr_word_end; do
    eval "${_n}=\$(xt_num ${_n} \"\${${_n}}\")"
done
# entropy_reciprocal interpolates log|w| from start to end: both must be
# non-zero and share a sign, or the weight would silently flip or be undefined.
for _arc in token sl word; do
    eval "_a=\${twr_${_arc}_start}; _b=\${twr_${_arc}_end}"
    awk -v a="${_a}" -v b="${_b}" 'BEGIN { exit !(a != 0 && b != 0 && a * b > 0) }' || otcr_die \
        "--twr_${_arc}_start (${_a}) and --twr_${_arc}_end (${_b}) must be non-zero and of the same sign"
done
[ -z "${word_sl_best_weight}" ] || word_sl_best_weight="$(xt_num word_sl_best_weight "${word_sl_best_weight}")"
[ -z "${word_both_sl_weight}" ] || word_both_sl_weight="$(xt_num word_both_sl_weight "${word_both_sl_weight}")"
[ -z "${word_both_sl_decay}" ]  || word_both_sl_decay="$(xt_num word_both_sl_decay "${word_both_sl_decay}")"
[ -z "${word_byp_best_decay}" ] || word_byp_best_decay="$(xt_num word_byp_best_decay "${word_byp_best_decay}")"
if { [ -n "${word_both_sl_weight}" ] && [ -z "${word_both_sl_decay}" ]; } || \
   { [ -z "${word_both_sl_weight}" ] && [ -n "${word_both_sl_decay}" ]; }; then
    otcr_die "--word_both_sl_weight and --word_both_sl_decay must be given together"
fi
case "${word_star_start_epoch}" in
    ''|*[!0-9]*|0) otcr_die "--word_star_start_epoch must be an integer >= 1, got '${word_star_start_epoch}'" ;;
esac
word_star_start_epoch=$((10#${word_star_start_epoch}))

# ===========================================================================
# Decode queue: FIFO, drained by at most --max_decode_jobs background workers.
#
# Entries hold the shell-quoted command (run through eval) plus the metadata
# needed to report the outcome.  Quoting with %q and evaluating is what lets a
# full argv live in a single array slot; bash has no array of arrays.
# ===========================================================================
XT_Q_CMD=(); XT_Q_TAG=(); XT_Q_LOG=(); XT_Q_ST=(); XT_Q_DIR=(); XT_Q_ELOG=()
XT_R_PID=(); XT_R_TAG=(); XT_R_LOG=(); XT_R_ST=(); XT_R_DIR=(); XT_R_ELOG=()
XT_DEC_OK=0; XT_DEC_FAIL=0; XT_DEC_QUEUED=0

xt_arr() { local n="$1[@]"; printf '%s\n' "${!n:-}"; }

xt_enqueue_decode() {   # <tag> <log> <status> <expdir> <espnet_log> <cmd...>
    local tag="$1" log="$2" st="$3" dir="$4" elog="$5"; shift 5
    XT_Q_CMD+=("$(printf '%q ' "$@")")
    XT_Q_TAG+=("${tag}"); XT_Q_LOG+=("${log}")
    XT_Q_ST+=("${st}");   XT_Q_DIR+=("${dir}")
    XT_Q_ELOG+=("${elog}")
    XT_DEC_QUEUED=$((XT_DEC_QUEUED + 1))
}

xt_report_decode() {   # <tag> <log> <expdir> <rc>
    if [ "$4" -eq 0 ]; then
        echo "  [decode ok]     $1"
        XT_DEC_OK=$((XT_DEC_OK + 1))
    else
        echo "  [decode FAILED] $1 (exit $4) -- see $2" >&2
        tail -n 15 "$2" >&2 2>/dev/null || true
        XT_DEC_FAIL=$((XT_DEC_FAIL + 1))
    fi
    return 0
}

# Harvest every running decode whose status file has appeared. Non-blocking.
xt_reap_decodes() {
    local i rc
    local -a kp=() kt=() kl=() ks_=() kd=() ke=()
    for i in "${!XT_R_PID[@]}"; do
        if [ -f "${XT_R_ST[$i]}" ]; then
            rc="$(cat "${XT_R_ST[$i]}" 2>/dev/null || echo 1)"
            case "${rc}" in ''|*[!0-9]*) rc=1 ;; esac
            wait "${XT_R_PID[$i]}" 2>/dev/null || true
            rm -f "${XT_R_ST[$i]}"
            xt_report_decode "${XT_R_TAG[$i]}" "${XT_R_LOG[$i]}" "${XT_R_DIR[$i]}" "${rc}"
        else
            kp+=("${XT_R_PID[$i]}"); kt+=("${XT_R_TAG[$i]}"); kl+=("${XT_R_LOG[$i]}")
            ks_+=("${XT_R_ST[$i]}"); kd+=("${XT_R_DIR[$i]}"); ke+=("${XT_R_ELOG[$i]}")
        fi
    done
    XT_R_PID=(${kp[@]+"${kp[@]}"});  XT_R_TAG=(${kt[@]+"${kt[@]}"})
    XT_R_LOG=(${kl[@]+"${kl[@]}"});  XT_R_ST=(${ks_[@]+"${ks_[@]}"})
    XT_R_DIR=(${kd[@]+"${kd[@]}"});  XT_R_ELOG=(${ke[@]+"${ke[@]}"})
}

# Launch queued decodes until the slot limit is reached.
xt_launch_decodes() {
    local cmd tag log st dir elog pid
    while [ "${#XT_Q_CMD[@]}" -gt 0 ] && [ "${#XT_R_PID[@]}" -lt "${max_decode_jobs}" ]; do
        cmd="${XT_Q_CMD[0]}"; tag="${XT_Q_TAG[0]}"; log="${XT_Q_LOG[0]}"
        st="${XT_Q_ST[0]}";   dir="${XT_Q_DIR[0]}"; elog="${XT_Q_ELOG[0]}"
        XT_Q_CMD=("${XT_Q_CMD[@]:1}"); XT_Q_TAG=("${XT_Q_TAG[@]:1}")
        XT_Q_LOG=("${XT_Q_LOG[@]:1}"); XT_Q_ST=("${XT_Q_ST[@]:1}")
        XT_Q_DIR=("${XT_Q_DIR[@]:1}"); XT_Q_ELOG=("${XT_Q_ELOG[@]:1}")
        mkdir -p "$(dirname "${log}")"
        rm -f "${st}"
        # `set +e` is REQUIRED: the subshell inherits errexit from the top of
        # this script, so a decode exiting non-zero would kill the subshell
        # before it wrote its status file, and the waiter -- which treats that
        # file as the completion signal -- would block forever on one failure.
        ( set +e
          eval "${cmd}" > "${log}" 2>&1
          rc=$?
          printf '%s' "${rc}" > "${st}.part"
          mv "${st}.part" "${st}" ) &
        pid=$!
        XT_R_PID+=("${pid}"); XT_R_TAG+=("${tag}"); XT_R_LOG+=("${log}")
        XT_R_ST+=("${st}");   XT_R_DIR+=("${dir}"); XT_R_ELOG+=("${elog}")
        echo "  [decode start]  ${tag}  (pid ${pid}, $((${#XT_Q_CMD[@]})) still queued)"
        echo "      wrapper log : ${log}"
        [ -n "${elog}" ] && echo "      espnet  log : ${elog}"
    done
}

# Keep the slot busy without blocking the caller.
xt_pump_decodes() { xt_reap_decodes; xt_launch_decodes; }

# Run the queue to completion.
xt_drain_decodes() {
    local total=$(( ${#XT_Q_CMD[@]} + ${#XT_R_PID[@]} ))
    if [ "${total}" -gt 0 ]; then
        otcr_hr
        echo "Draining the decode queue: ${#XT_R_PID[@]} running, ${#XT_Q_CMD[@]} queued."
    fi
    while :; do
        xt_pump_decodes
        [ "${#XT_Q_CMD[@]}" -eq 0 ] && [ "${#XT_R_PID[@]}" -eq 0 ] && break
        # `wait -n` returns as soon as any child exits, so there is no polling
        # latency.  Its status is ignored: it may refer to a job already
        # accounted for, and the status file is authoritative.  `|| true` is
        # required because set -e would otherwise abort on a failed decode.
        wait -n 2>/dev/null || true
    done
}

# Terminate a background job and everything it started.  Killing only the
# wrapper subshell would leave asr_respin.sh and the python trainer holding the
# GPU.  Children first, so none of them is re-parented before it is found.
xt_kill_tree() {   # <pid>
    local child
    for child in $(pgrep -P "$1" 2>/dev/null || true); do
        xt_kill_tree "${child}"
    done
    kill "$1" 2>/dev/null || true
}

xt_kill_decodes() {
    if [ "${#XT_T_PID[@]}" -gt 0 ]; then
        echo >&2
        echo "Interrupted: terminating ${#XT_T_PID[@]} running training(s)" \
             "on GPU(s) ${XT_T_GPU[*]}." >&2
        local tpid
        for tpid in "${XT_T_PID[@]}"; do
            [ -n "${tpid}" ] && xt_kill_tree "${tpid}"
        done
        echo "Those cells have no model; rerun to train them (resume picks up" \
             "from their last checkpoint)." >&2
    fi
    if [ "${#XT_R_PID[@]}" -gt 0 ]; then
        echo >&2
        echo "Interrupted: terminating ${#XT_R_PID[@]} running decode(s)," \
             "${#XT_Q_CMD[@]} still queued." >&2
        local pid
        for pid in $(xt_arr XT_R_PID); do
            [ -n "${pid}" ] && kill "${pid}" 2>/dev/null || true
        done
        echo "Those cells are left unscored; rerun to redo them (no --force needed)." >&2
    fi
    exit 130
}
trap xt_kill_decodes INT TERM

# ===========================================================================
# Training pool: one background training per GPU in XT_GPUS.
#
# A training's decodes are queued when it FINISHES, but by then the enumeration
# has moved on and the per-cell globals (OTCR_COMMON, XT_DECODE_SETS, ...)
# describe a different cell.  So the cell's context is captured with
# `declare -p` at launch and swapped back in, around xt_submit_decode, when the
# training is reaped -- the decodes are then built by exactly the same code as
# before.
# ===========================================================================
XT_T_PID=(); XT_T_GPU=(); XT_T_TAG=(); XT_T_ST=(); XT_T_LOG=(); XT_T_ROOT=()
XT_T_DIR=(); XT_T_LOGDIR=(); XT_T_CTX=()
XT_CTX_VARS="OTCR_COMMON XT_DECODE_SETS XT_VALID_SET OTCR_CONFIG OTCR_INFER_CONFIG"

xt_capture_ctx() {   # prints eval-able GLOBAL assignments of XT_CTX_VARS
    # shellcheck disable=SC2086
    { declare -p ${XT_CTX_VARS} 2>/dev/null || true; } \
        | sed -e 's/^declare -a /declare -ga /' -e 's/^declare -- /declare -g /'
}

# Harvest every training whose status file has appeared.  Non-blocking.
xt_reap_trainings() {
    local i rc saved set_name
    local -a kp=() kg=() kt=() ks_=() kl=() kr=() kd=() klg=() kc=()
    for i in "${!XT_T_PID[@]}"; do
        if [ ! -f "${XT_T_ST[$i]}" ]; then
            kp+=("${XT_T_PID[$i]}"); kg+=("${XT_T_GPU[$i]}"); kt+=("${XT_T_TAG[$i]}")
            ks_+=("${XT_T_ST[$i]}"); kl+=("${XT_T_LOG[$i]}"); kr+=("${XT_T_ROOT[$i]}")
            kd+=("${XT_T_DIR[$i]}"); klg+=("${XT_T_LOGDIR[$i]}"); kc+=("${XT_T_CTX[$i]}")
            continue
        fi
        rc="$(cat "${XT_T_ST[$i]}" 2>/dev/null || echo 1)"
        case "${rc}" in ''|*[!0-9]*) rc=1 ;; esac
        wait "${XT_T_PID[$i]}" 2>/dev/null || true
        rm -f "${XT_T_ST[$i]}"
        otcr_hr
        if [ "${rc}" -ne 0 ]; then
            echo "  FAILED training ${XT_T_TAG[$i]} (GPU ${XT_T_GPU[$i]}, exit ${rc}) -- see ${XT_T_LOG[$i]}" >&2
            tail -n 25 "${XT_T_LOG[$i]}" >&2 || true
            XT_N_TRAIN_FAIL=$((XT_N_TRAIN_FAIL + 1))
            continue
        fi
        if [ ! -f "${XT_T_DIR[$i]}/${OTCR_INFER_MODEL}" ]; then
            echo "  FAILED ${XT_T_DIR[$i]}/${OTCR_INFER_MODEL} was not produced" >&2
            XT_N_TRAIN_FAIL=$((XT_N_TRAIN_FAIL + 1))
            continue
        fi
        XT_N_TRAINED=$((XT_N_TRAINED + 1))
        # Swap in the finished cell's context, queue its decodes, swap back.
        saved="$(xt_capture_ctx)"
        eval "${XT_T_CTX[$i]}"
        echo "  trained ${XT_T_TAG[$i]} (GPU ${XT_T_GPU[$i]}); queueing decodes for ${XT_DECODE_SETS[*]}"
        # The model is new, so every set runs stages 12-13 in full regardless of
        # any hypothesis or score left over from an earlier training of this
        # cell -- the resume markers describe the OLD model and must not be
        # trusted.
        for set_name in "${XT_DECODE_SETS[@]}"; do
            xt_submit_decode "${XT_T_TAG[$i]}" "${XT_T_ROOT[$i]}" "${XT_T_DIR[$i]}" \
                             "${XT_T_LOGDIR[$i]}" "${set_name}" 12
        done
        eval "${saved}"
    done
    XT_T_PID=(${kp[@]+"${kp[@]}"});   XT_T_GPU=(${kg[@]+"${kg[@]}"})
    XT_T_TAG=(${kt[@]+"${kt[@]}"});   XT_T_ST=(${ks_[@]+"${ks_[@]}"})
    XT_T_LOG=(${kl[@]+"${kl[@]}"});   XT_T_ROOT=(${kr[@]+"${kr[@]}"})
    XT_T_DIR=(${kd[@]+"${kd[@]}"});   XT_T_LOGDIR=(${klg[@]+"${klg[@]}"})
    XT_T_CTX=(${kc[@]+"${kc[@]}"})
}

# First GPU with no running training -> XT_FREE_GPU, or return 1.
xt_find_free_gpu() {
    local g b busy
    for g in "${XT_GPUS[@]}"; do
        busy=false
        for b in ${XT_T_GPU[@]+"${XT_T_GPU[@]}"}; do [ "${b}" = "${g}" ] && busy=true; done
        if ! ${busy}; then XT_FREE_GPU="${g}"; return 0; fi
    done
    return 1
}

# Block until a GPU is free, reaping finished trainings (which queues their
# decodes) and keeping the decode slots busy meanwhile.
xt_wait_for_gpu() {
    local said=false
    while :; do
        xt_reap_trainings
        xt_pump_decodes
        xt_find_free_gpu && return 0
        if ! ${said}; then
            echo "  waiting for a free GPU (busy: ${XT_T_GPU[*]})"
            said=true
        fi
        # Poll, not `wait -n`: a training that exits just before `wait -n` is
        # called can leave it blocked until some OTHER child exits, which with
        # one GPU never happens.  The status files are authoritative.
        sleep 5
    done
}

# Wait for every running training (and queue their decodes).
xt_drain_trainings() {
    [ "${#XT_T_PID[@]}" -gt 0 ] || return 0
    otcr_hr
    echo "Waiting for ${#XT_T_PID[@]} running training(s) on GPU(s) ${XT_T_GPU[*]}."
    while [ "${#XT_T_PID[@]}" -gt 0 ]; do
        xt_reap_trainings
        xt_pump_decodes
        [ "${#XT_T_PID[@]}" -eq 0 ] && break
        sleep 5   # poll; see xt_wait_for_gpu
    done
}

xt_launch_training() {   # <gpu> <asr_tag> <expdir_root> <expdir> <logdir> <train_log> <cmd...>
    local g="$1" tag="$2" root="$3" dir="$4" logdir="$5" log="$6"; shift 6
    local st="${logdir}/${tag}.train.status" pid
    rm -f "${st}"
    # `set +e` for the same reason as the decode workers: the status file must
    # be written even when the training fails.
    ( set +e
      CUDA_VISIBLE_DEVICES="${g}" "$@" > "${log}" 2>&1
      rc=$?
      printf '%s' "${rc}" > "${st}.part"
      mv "${st}.part" "${st}" ) &
    pid=$!
    XT_T_PID+=("${pid}"); XT_T_GPU+=("${g}"); XT_T_TAG+=("${tag}"); XT_T_ST+=("${st}")
    XT_T_LOG+=("${log}"); XT_T_ROOT+=("${root}"); XT_T_DIR+=("${dir}")
    XT_T_LOGDIR+=("${logdir}"); XT_T_CTX+=("$(xt_capture_ctx)")
    echo "  ...training on GPU ${g} (pid ${pid}; tail -f ${log})"
}

# ===========================================================================
# Per-cell driver.  Replaces the run_lib.sh version in two ways: training and
# decoding are selected by --action rather than always run back to back, and a
# decode is queued to the background pool instead of blocking the caller.
#
# XT_DECODE_SETS holds this cell's dev set followed by its test set(s); each is
# a separate job with its own scored state.
# ===========================================================================
XT_N_TRAINED=0; XT_N_SKIP=0; XT_N_TRAIN_FAIL=0; XT_N_PENDING=0; XT_N_SCORED=0
XT_N_RESCORE=0

# A set's OUTPUT DIRECTORY is not always named after the set.  asr_respin.sh
# (line ~324) notices that --test_sets contains the valid set, removes it from
# test_sets, sets eval_valid_set=true, and then decodes it as "org/<valid_set>"
# -- so the dev results land in <inference_tag>/org/dev, not <inference_tag>/dev.
# Test sets keep their own name.  Everything that inspects or reports a decode
# directory must go through here, or it looks in a directory that will never
# exist and re-decodes work that is already done.
xt_decode_subdir() {   # <set>
    if [ "$1" = "${XT_VALID_SET}" ]; then echo "org/$1"; else echo "$1"; fi
}

# How far one set got, from the two files asr_respin.sh leaves behind:
#
#   text            written by stage 12 (inference)
#   result_lid.txt  written by stage 13, AFTER every score_<type>/result.txt,
#                   so it is the only file that means "scoring finished" --
#                   score_wer/result.txt is written early enough that it also
#                   appears when a scoring pass died partway through.
#
# Echoes the stage this set should start from, or "done".
xt_decode_state() {   # <expdir> <set>
    local d="$1/${OTCR_INFER_TAG}/$(xt_decode_subdir "$2")"
    if [ -f "${d}/text" ]; then
        [ -f "${d}/result_lid.txt" ] && { echo done; return 0; }
        echo 13; return 0      # inference is there; only rescore
    fi
    echo 12                    # nothing usable; decode and score
}

# Build the decode argv for one set into XT_DECODE_CMD.  The trailing
# --test_sets wins over the one already in OTCR_COMMON: parse_options.sh
# assigns as it walks the argv, so the last occurrence of a flag is the one
# that survives.
xt_decode_cmd() {   # <asr_tag> <expdir_root> <set> [from_stage]
    XT_DECODE_CMD=("./${OTCR_SCRIPT}"
        --stage "${4:-12}" --stop_stage 13
        "${OTCR_COMMON[@]}"
        --expdir "$2"
        --asr_config "${OTCR_CONFIG}"
        --inference_config "${OTCR_INFER_CONFIG}"
        --asr_tag "$1"
        --inference_tag "${OTCR_INFER_TAG}"
        --inference_asr_model "${OTCR_INFER_MODEL}"
        --test_sets "$3"
        --skip_train true
        --asr_task asr_otc_recip
        --gpu_inference false)
}

xt_submit_decode() {   # <asr_tag> <expdir_root> <expdir> <logdir> <set> [from]
    local tag="$1" root="$2" dir="$3" logdir="$4" set_name="$5" from="${6:-12}"
    local label="${tag} [${set_name}]" kind=decode what=decode

    # espnet's own per-job inference logs, written by stage 12 via run.pl:
    #   <expdir>/<inference_tag>/<set>/logdir/asr_inference.<JOB>.log
    # These carry the real decoding progress and the traceback when a job
    # dies; the wrapper log only holds asr_respin.sh's own stdout, which on a
    # failure says little more than "check the logs".  Stage 13 has no such
    # directory -- sclite's output lands in the wrapper log -- so a rescore
    # leaves this empty rather than printing a path that will not exist.
    local elog=""
    [ "${from}" = "12" ] && \
        elog="${dir}/${OTCR_INFER_TAG}/$(xt_decode_subdir "${set_name}")/logdir/asr_inference.*.log"

    if [ "${from}" = "13" ]; then
        # Separate log: a rescore must not clobber the record of the inference
        # pass whose output it is scoring.
        kind=score; what=rescore
        label="${label} rescore-only (text present, result_lid.txt missing)"
        XT_N_RESCORE=$((XT_N_RESCORE + 1))
    fi
    local log="${logdir}/${tag}.${kind}_${set_name}.log"
    local st="${logdir}/${tag}.${kind}_${set_name}.status"

    xt_decode_cmd "${tag}" "${root}" "${set_name}" "${from}"

    if ${dry_run}; then
        echo "  [would ${what}] ${label}"
        echo "      wrapper log : ${log}"
        [ -n "${elog}" ] && echo "      espnet  log : ${elog}"
        XT_DEC_QUEUED=$((XT_DEC_QUEUED + 1))
        return 0
    fi

    mkdir -p "${logdir}"
    if [ "${max_decode_jobs}" -eq 0 ]; then
        echo "  [${what} start]  ${label}"
        echo "      wrapper log : ${log}"
        [ -n "${elog}" ] && echo "      espnet  log : ${elog}"
        if ! "${XT_DECODE_CMD[@]}" > "${log}" 2>&1; then
            echo "  [decode FAILED] ${label} -- see ${log}" >&2
            [ -n "${elog}" ] && echo "                  and ${elog}" >&2
            tail -n 15 "${log}" >&2 2>/dev/null || true
            XT_DEC_FAIL=$((XT_DEC_FAIL + 1))
        else
            echo "  [decode ok]     ${label}"
            XT_DEC_OK=$((XT_DEC_OK + 1))
        fi
        return 0
    fi

    xt_enqueue_decode "${label}" "${log}" "${st}" "${dir}" "${elog}" \
                      "${XT_DECODE_CMD[@]}"
    xt_pump_decodes
}

otcr_run_cell() {
    local variant_tag=""
    [ -n "${OTCR_VARIANT}" ] && variant_tag="_${OTCR_VARIANT}"
    local asr_tag="otcrecip_${OTCR_CORPUS}_${OTCR_LANG}_${OTCR_METHOD}${variant_tag}_seed${OTCR_SEED}"
    local expdir_root="${OTCR_EXPROOT}/exp_${OTCR_EXPTAG}_${OTCR_SUFFIX}"
    local expdir="${expdir_root}/asr_${asr_tag}"

    otcr_require_config "${OTCR_CONFIG}"

    local logdir="${OTCR_LOGROOT}/${OTCR_CORPUS}"
    local train_log="${logdir}/${asr_tag}.train.log"

    local has_model=false
    [ -f "${expdir}/${OTCR_INFER_MODEL}" ] && has_model=true

    local set_name

    # ------------------------------ decode ---------------------------------
    # Score what is already trained.  A cell with no model is somebody else's
    # job (--action train); it is counted so the summary says how much of the
    # sweep is still missing.
    if [ "${action}" = "decode" ]; then
        ${has_model} || { XT_N_PENDING=$((XT_N_PENDING + 1)); return 0; }
        local from
        for set_name in "${XT_DECODE_SETS[@]}"; do
            if ${force}; then
                from=12          # --force redoes inference as well as scoring
            else
                from="$(xt_decode_state "${expdir}" "${set_name}")"
                if [ "${from}" = "done" ]; then
                    XT_N_SCORED=$((XT_N_SCORED + 1))
                    continue
                fi
            fi
            xt_submit_decode "${asr_tag}" "${expdir_root}" "${expdir}" \
                             "${logdir}" "${set_name}" "${from}"
        done
        return 0
    fi

    # ------------------------------ train ----------------------------------
    if ${has_model} && ! ${force}; then
        XT_N_SKIP=$((XT_N_SKIP + 1))
        return 0
    fi

    local -a train_cmd
    train_cmd=("./${OTCR_SCRIPT}"
        --stage "${OTCR_TRAIN_STAGE}" --stop_stage 11
        "${OTCR_COMMON[@]}"
        --expdir "${expdir_root}"
        --asr_config "${OTCR_CONFIG}"
        --inference_config "${OTCR_INFER_CONFIG}"
        --asr_tag "${asr_tag}"
        --inference_tag "${OTCR_INFER_TAG}"
        --inference_asr_model "${OTCR_INFER_MODEL}"
        --asr_args "--seed ${OTCR_SEED}${OTCR_ASR_ARGS:+ ${OTCR_ASR_ARGS}}"
        --skip_train false
        --asr_task asr_otc_recip
        --gpu_inference false)

    otcr_hr
    echo "TRAIN ${asr_tag}"
    echo "  config : ${OTCR_CONFIG}"
    echo "  expdir : ${expdir}"
    echo "  logs   : ${train_log}"
    echo "  decode : ${XT_DECODE_SETS[*]}"

    if ${dry_run}; then
        echo "  DRY-RUN commands (not executed):"
        if [ "${#XT_GPUS[@]}" -eq 1 ]; then
            echo "    CUDA_VISIBLE_DEVICES=${OTCR_GPU}"
        else
            echo "    CUDA_VISIBLE_DEVICES=<first free of: ${OTCR_GPU}>"
        fi
        otcr_show "${train_cmd[@]}"
        echo "    # the decodes below are QUEUED and run in the background"
        for set_name in "${XT_DECODE_SETS[@]}"; do
            xt_decode_cmd "${asr_tag}" "${expdir_root}" "${set_name}"
            otcr_show "${XT_DECODE_CMD[@]}"
            XT_DEC_QUEUED=$((XT_DEC_QUEUED + 1))
        done
        XT_N_TRAINED=$((XT_N_TRAINED + 1))
        return 0
    fi

    mkdir -p "${logdir}"

    # One training per GPU: wait for a free one (reaping finished trainings and
    # queueing their decodes meanwhile), then start this cell in the background.
    # Its decodes are queued by xt_reap_trainings once it finishes.
    xt_wait_for_gpu
    xt_launch_training "${XT_FREE_GPU}" "${asr_tag}" "${expdir_root}" "${expdir}" \
                       "${logdir}" "${train_log}" "${train_cmd[@]}"
}

# ===========================================================================
# OTC-word arc ablation helpers (see OTC-WORD ARC ABLATION in the header).
# They read the per-cell globals xt_visit_cells has set for the current
# corpus/language (OTCR_EXPROOT, OTCR_EXPTAG, OTCR_SUFFIX, OTCR_CORPUS,
# OTCR_LANG, XT_VALID_SET) and its local ${variant}.
# ===========================================================================
XT_N_BLOCKED=0; XT_N_NONSSL=0

# Same asr_tag / expdir rule as otcr_run_cell.
xt_expdir() {   # <method> <variant_tag> <seed>
    echo "${OTCR_EXPROOT}/exp_${OTCR_EXPTAG}_${OTCR_SUFFIX}/asr_otcrecip_${OTCR_CORPUS}_${OTCR_LANG}_$1_$2_seed$3"
}

# Dev-set WER of a cell, or nothing unless its dev scoring FINISHED
# (result_lid.txt is written last by stage 13; see xt_decode_state).
xt_dev_wer() {   # <expdir>
    local d="$1/${OTCR_INFER_TAG}/$(xt_decode_subdir "${XT_VALID_SET}")"
    [ -f "${d}/result_lid.txt" ] && [ -f "${d}/score_wer/result.txt" ] || return 0
    { grep -m1 "Sum/Avg" "${d}/score_wer/result.txt" || true; } \
        | tr -d '|' | awk 'NF >= 8 { print $8 }'
}

# Under --action train the trainings of earlier stages may still be running and
# their dev decodes may still be queued; let both land before a pick reads
# their scores.
xt_drain_before_pick() {
    if [ "${action}" = "train" ] && ! ${dry_run}; then
        xt_drain_trainings
        if [ "${max_decode_jobs}" -gt 0 ]; then
            xt_drain_decodes
        fi
    fi
}

# xt_pick_best <what> <candidate>...     candidate = "<value>|<method>|<variant_tag>"
#   Lowest mean dev WER over ${seeds}; ties keep the earlier candidate.  Sets
#   XT_BEST and returns 0, or prints what is missing and returns 1.
xt_pick_best() {
    local what="$1"; shift
    local cand val method vtag seed dir wer sum n best_val="" best_wer="" mean
    local -a missing=() table=()
    for cand in "$@"; do
        IFS='|' read -r val method vtag <<< "${cand}"
        sum=0; n=0
        for seed in ${seeds}; do
            dir="$(xt_expdir "${method}" "${vtag}" "${seed}")"
            wer="$(xt_dev_wer "${dir}")"
            if [ -z "${wer}" ]; then missing+=("${dir}"); continue; fi
            sum="$(awk -v a="${sum}" -v b="${wer}" 'BEGIN { print a + b }')"
            n=$((n + 1))
        done
        [ "${n}" -gt 0 ] || { table+=("$(printf '%-14s %s' "${val}" "-")"); continue; }
        mean="$(awk -v a="${sum}" -v n="${n}" 'BEGIN { printf "%.4f", a / n }')"
        table+=("$(printf '%-14s %s' "${val}" "${mean}")")
        if [ -z "${best_wer}" ] || awk -v a="${mean}" -v b="${best_wer}" 'BEGIN { exit !(a < b) }'; then
            best_wer="${mean}"; best_val="${val}"
        fi
    done
    otcr_hr
    echo "PICK  ${what}   (${OTCR_CORPUS}/${OTCR_LANG}, dev WER, mean over seeds: ${seeds})"
    local row; for row in "${table[@]}"; do echo "        ${row}"; done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "  BLOCKED: ${#missing[@]} cell(s) have no finished dev decode yet, e.g."
        local i; for i in "${!missing[@]}"; do [ "${i}" -lt 5 ] && echo "      ${missing[$i]}"; done
        echo "  (train/decode those first, or pass the value by hand; see the header)"
        return 1
    fi
    XT_BEST="${best_val}"
    echo "  -> best ${what}: ${XT_BEST}  (dev WER ${best_wer})"
    return 0
}

xt_tag_sl()   { echo "${variant}_slonly_sl$(xt_slug "$1")_sld$(xt_slug "$2")"; }
xt_tag_byp()  { echo "${variant}_byponly_byp$(xt_slug "$1")_bypd$(xt_slug "$2")"; }
xt_tag_both() { echo "${variant}_both_sl$(xt_slug "$1")_sld$(xt_slug "$2")_byp$(xt_slug "$3")_bypd$(xt_slug "$4")"; }
xt_tag_tw()   { echo "${variant}_tok$(xt_slug "$1")_tokd$(xt_slug "$2")_word$(xt_slug "$3")_wordd$(xt_slug "$4")_sl$(xt_slug "$5")_sld$(xt_slug "$6")"; }

xt_is_zero() { awk -v v="$1" 'BEGIN { exit !(v == 0) }'; }

# Best word_sl weight -> XT_BEST_SL_W.
xt_best_sl_weight() {
    if [ -n "${word_sl_best_weight}" ]; then
        XT_BEST_SL_W="${word_sl_best_weight}"
        echo "  word_sl weight fixed by --word_sl_best_weight: ${XT_BEST_SL_W}"
        return 0
    fi
    local w; local -a c=()
    for w in "${RUN_SL_WEIGHTS[@]}"; do
        c+=("${w}|otc_word|$(xt_tag_sl "${w}" "${word_sl_weight_decay}")")
    done
    xt_pick_best "self-loop-only weight (decay ${word_sl_weight_decay})" "${c[@]}" || return 1
    XT_BEST_SL_W="${XT_BEST}"
}

# Best self-loop-only (weight, decay) over word_sl + word_sl_decay -> XT_BEST_SL_W/_D.
xt_best_sl_pair() {
    if [ -n "${word_both_sl_weight}" ]; then
        XT_BEST_SL_W="${word_both_sl_weight}"; XT_BEST_SL_D="${word_both_sl_decay}"
        echo "  self-loop fixed by --word_both_sl_weight/_decay: ${XT_BEST_SL_W} / ${XT_BEST_SL_D}"
        return 0
    fi
    xt_best_sl_weight || return 1
    local w d; local -a c=()
    for w in "${RUN_SL_WEIGHTS[@]}"; do
        c+=("${w}:${word_sl_weight_decay}|otc_word|$(xt_tag_sl "${w}" "${word_sl_weight_decay}")")
    done
    if ! xt_is_zero "${XT_BEST_SL_W}"; then
        for d in "${RUN_WORD_DECAYS[@]}"; do
            [ "${d}" = "${word_sl_weight_decay}" ] && continue   # already a word_sl cell
            c+=("${XT_BEST_SL_W}:${d}|otc_word|$(xt_tag_sl "${XT_BEST_SL_W}" "${d}")")
        done
    fi
    xt_pick_best "self-loop-only weight:decay" "${c[@]}" || return 1
    XT_BEST_SL_W="${XT_BEST%%:*}"; XT_BEST_SL_D="${XT_BEST#*:}"
}

# Best word_byp decay -> XT_BEST_BYP_D.
xt_best_byp_decay() {
    if [ -n "${word_byp_best_decay}" ]; then
        XT_BEST_BYP_D="${word_byp_best_decay}"
        echo "  bypass decay fixed by --word_byp_best_decay: ${XT_BEST_BYP_D}"
        return 0
    fi
    local d; local -a c=()
    for d in "${RUN_WORD_DECAYS[@]}"; do
        c+=("${d}|otc_word|$(xt_tag_byp "${word_byp_weight}" "${d}")")
    done
    xt_pick_best "bypass-only decay (weight ${word_byp_weight})" "${c[@]}" || return 1
    XT_BEST_BYP_D="${XT_BEST}"
}

xt_tag_twr() {   # <k>
    echo "${variant}_tok$(xt_slug "${twr_token_start}")to$(xt_slug "${twr_token_end}")_word$(xt_slug "${twr_word_start}")to$(xt_slug "${twr_word_end}")_sl$(xt_slug "${twr_sl_start}")to$(xt_slug "${twr_sl_end}")_k$(xt_slug "$1")"
}

xt_run_seeds() {   # <method> <variant_tag> <asr_args>
    OTCR_METHOD="$1"; OTCR_VARIANT="$2"; OTCR_ASR_ARGS="$3"
    for seed in ${seeds}; do
        OTCR_SEED="${seed}"; otcr_run_cell
    done
}

xt_args_sl_only() {   # <weight> <decay>
    echo "--ctc_conf otc_granularity=word --ctc_conf allow_bypass_arc=false --ctc_conf allow_self_loop_arc=true --ctc_conf initial_self_loop_weight=$1 --ctc_conf self_loop_weight_decay=$2"
}
# " --ctc_conf star_start_epoch=N", or nothing when N is 1 (the model default),
# so an undelayed run's arguments are exactly what they were before.
xt_args_star_start() {
    [ "${word_star_start_epoch}" -eq 1 ] || printf ' --ctc_conf star_start_epoch=%s' "${word_star_start_epoch}"
}
xt_args_byp_only() {   # <weight> <decay>
    echo "--ctc_conf otc_granularity=word --ctc_conf allow_bypass_arc=true --ctc_conf allow_self_loop_arc=false --ctc_conf initial_bypass_weight=$1 --ctc_conf bypass_weight_decay=$2$(xt_args_star_start)"
}
xt_args_both() {   # <sl_w> <sl_d> <byp_w> <byp_d>
    echo "--ctc_conf otc_granularity=word --ctc_conf allow_bypass_arc=true --ctc_conf allow_self_loop_arc=true --ctc_conf initial_self_loop_weight=$1 --ctc_conf self_loop_weight_decay=$2 --ctc_conf initial_bypass_weight=$3 --ctc_conf bypass_weight_decay=$4$(xt_args_star_start)"
}
xt_args_twr() {   # <k>   (on the otc_recip base config)
    echo "--ctc_conf otc_granularity=dual --ctc_conf allow_bypass_arc=true --ctc_conf allow_self_loop_arc=true --ctc_conf initial_bypass_weight=${twr_token_start} --ctc_conf bypass_end=${twr_token_end} --ctc_conf initial_word_bypass_weight=${twr_word_start} --ctc_conf word_bypass_end=${twr_word_end} --ctc_conf initial_self_loop_weight=${twr_sl_start} --ctc_conf self_loop_end=${twr_sl_end} --ctc_conf entropy_shape_k=$1$(xt_args_star_start)"
}
xt_args_tw() {   # <tok_w> <tok_d> <word_w> <word_d> <sl_w> <sl_d>
    echo "--ctc_conf otc_granularity=dual --ctc_conf allow_bypass_arc=true --ctc_conf allow_self_loop_arc=true --ctc_conf initial_bypass_weight=$1 --ctc_conf bypass_weight_decay=$2 --ctc_conf initial_word_bypass_weight=$3 --ctc_conf word_bypass_weight_decay=$4 --ctc_conf initial_self_loop_weight=$5 --ctc_conf self_loop_weight_decay=$6$(xt_args_star_start)"
}

# ===========================================================================
# Cell enumeration.  Called twice, once per phase, so the per-corpus data
# layout is written exactly once.
# ===========================================================================
xt_visit_cells() {
    local corpus variant lang mode val slug tag data_folder train_set valid_set nbpe
    local test_sets_cell t

    for corpus in ${corpora}; do
        eval "local corpus_langs=\${${corpus}_langs:-}"
        [ -n "${corpus_langs}" ] || otcr_die \
            "unknown corpus '${corpus}' (no --${corpus}_langs is defined). This
  script knows respin, libri and fleurs; check the spelling of --corpora, or
  add a case branch plus a <corpus>_langs default to support another."
        OTCR_CORPUS="${corpus}"

        for variant in ${variants}; do
            for lang in ${corpus_langs}; do
                OTCR_LANG="${lang}"
                case "${corpus}" in
                  respin)
                    OTCR_EXPROOT=exp_respin
                    OTCR_INFER_CONFIG=conf/respin/decode_ctc.yaml
                    [ "${variant}" = "ssl" ] && OTCR_SUFFIX=ssl_xlsr300m || OTCR_SUFFIX=stft_char
                    tag="${lang}_${respin_tag}"
                    OTCR_EXPTAG="${tag}"
                    data_folder="${respin_data_root}/${lang}"
                    train_set="IISc_RESPIN_train_${tag}"
                    valid_set="IISc_RESPIN_dev_${lang}"
                    # shellcheck disable=SC2059
                    test_sets_cell=$([ -n "${respin_testsets_tmpl}" ] \
                        && printf "${respin_testsets_tmpl}" "${lang}" || echo "")
                    OTCR_COMMON=(
                        --tag "${tag}" --data_folder "${data_folder}"
                        --dumpdir "dumps/dump_${tag}" --lang "${lang}"
                        --auxiliary_data_tags ""
                        --local_data_opts "--stage 0 --lang ${lang}"
                        --post_process_local_data_opts "--stage 2 --lang ${lang}"
                        --token_type char --nbpe 500
                        --local_score_opts "--score_lang_id true"
                    ) ;;
                  libri)
                    OTCR_EXPROOT=exp_libri
                    OTCR_INFER_CONFIG=conf/libri/decode_ctc.yaml
                    if [ "${variant}" = "ssl" ]; then
                        nbpe="${libri_nbpe_ssl}";    OTCR_SUFFIX="ssl_w2v2base960h_bpe${nbpe}"
                    else
                        nbpe="${libri_nbpe_nonssl}"; OTCR_SUFFIX="stft_bpe${nbpe}"
                    fi
                    tag="${libri_exptag}_${lang}"
                    OTCR_EXPTAG="${tag}"
                    data_folder="${libri_data_root}"
                    train_set="${libri_trainset}"
                    valid_set="dev"
                    test_sets_cell="${libri_testsets}"
                    OTCR_COMMON=(
                        --tag "${tag}" --data_folder "${data_folder}"
                        --dumpdir "dumps/dump_${tag}" --lang "${lang}"
                        --auxiliary_data_tags ""
                        --token_type bpe --nbpe "${nbpe}"
                        --local_score_opts "--score_lang_id false"
                    ) ;;
                  fleurs)
                    OTCR_EXPROOT=exp_fleurs
                    OTCR_INFER_CONFIG=conf/fleurs/decode_ctc.yaml
                    [ "${variant}" = "ssl" ] && OTCR_SUFFIX=ssl_xlsr300m || OTCR_SUFFIX=stft_char
                    OTCR_EXPTAG="${lang}"
                    data_folder="${fleurs_data_root}/${lang}"
                    # shellcheck disable=SC2059
                    train_set=$(printf "${fleurs_train_tmpl}" "${lang}")
                    # shellcheck disable=SC2059
                    valid_set=$(printf "${fleurs_dev_tmpl}" "${lang}")
                    # shellcheck disable=SC2059
                    test_sets_cell=$([ -n "${fleurs_testsets_tmpl}" ] \
                        && printf "${fleurs_testsets_tmpl}" "${lang}" || echo "")
                    OTCR_COMMON=(
                        --tag "${lang}" --data_folder "${data_folder}"
                        --dumpdir "dumps/dump_${lang}" --lang "${lang}"
                        --auxiliary_data_tags ""
                        --token_type char --nbpe 10000
                        --local_score_opts "--score_lang_id false"
                    ) ;;
                  *)
                    otcr_die "unknown corpus '${corpus}'. This script knows the data layout of
  respin, libri and fleurs; add a case branch to support another one." ;;
                esac

                OTCR_COMMON+=(
                    --audio_format "${audio_format}"
                    --use_lm "${use_lm}"
                    --feats_normalize "${feats_normalize}"
                    --lm_config "${lm_config}"
                    --bpe_nlsyms "${bpe_nlsyms}"
                    --feats_type "${feats_type}"
                    --speed_perturb_factors "${speed_perturb_factors}"
                    --train_set "${train_set}"
                    --valid_set "${valid_set}"
                    --test_sets "${valid_set}"
                    --bpe_train_text "${data_folder}/${train_set}/text"
                    --lm_train_text "${data_folder}/${train_set}/text"
                    --nj "${train_nj}"
                    --ngpu "${ngpu}"
                    --inference_nj "${inference_nj}"
                )
                # Decode targets: the dev set first, then the test set(s).
                # Each is scored separately, so a half-finished cell is
                # completed rather than redone.
                XT_DECODE_SETS=("${valid_set}")
                for t in ${test_sets_cell}; do XT_DECODE_SETS+=("${t}"); done
                # xt_decode_subdir needs this to spot the "org/" case.
                XT_VALID_SET="${valid_set}"
                otcr_require_config "${OTCR_INFER_CONFIG}"

                for mode in ${RUN_MODES}; do
                    OTCR_METHOD="${mode}"
                    case " ${XT_WORD_MODES} " in
                      *" ${mode} "*)
                        # Opt-in word pipeline: ssl only, on the otc_token base.
                        if [ "${variant}" != "ssl" ]; then
                            XT_N_NONSSL=$((XT_N_NONSSL + 1))
                            continue
                        fi
                        if [ "${mode}" = "otc_token_word_recip" ]; then
                            OTCR_CONFIG="conf/cross_transfer/${corpus}_otc_recip_${variant}.yaml"
                        else
                            OTCR_CONFIG="conf/cross_transfer/${corpus}_otc_token_${variant}.yaml"
                        fi ;;
                      *)
                        OTCR_CONFIG="conf/cross_transfer/${corpus}_${mode}_${variant}.yaml" ;;
                    esac
                    otcr_require_config "${OTCR_CONFIG}"

                    case "${mode}" in
                      ctc)
                        # Nothing to sweep: one cell per corpus/lang/variant.
                        OTCR_VARIANT="${variant}"
                        OTCR_ASR_ARGS=""
                        for seed in ${seeds}; do
                            OTCR_SEED="${seed}"; otcr_run_cell
                        done ;;
                      otc_token)
                        for val in "${RUN_DECAYS[@]}"; do
                            slug="${val//./p}"
                            OTCR_VARIANT="${variant}_decay${slug}"
                            OTCR_ASR_ARGS="--ctc_conf bypass_weight_decay=${val} --ctc_conf self_loop_weight_decay=${val}"
                            for seed in ${seeds}; do
                                OTCR_SEED="${seed}"; otcr_run_cell
                            done
                        done ;;
                      otc_recip)
                        for val in "${RUN_KS[@]}"; do
                            slug="${val//./p}"
                            OTCR_VARIANT="${variant}_k${slug}"
                            OTCR_ASR_ARGS="--ctc_conf entropy_shape_k=${val}"
                            for seed in ${seeds}; do
                                OTCR_SEED="${seed}"; otcr_run_cell
                            done
                        done ;;
                      word_sl)
                        for val in "${RUN_SL_WEIGHTS[@]}"; do
                            xt_run_seeds otc_word "$(xt_tag_sl "${val}" "${word_sl_weight_decay}")" \
                                "$(xt_args_sl_only "${val}" "${word_sl_weight_decay}")"
                        done ;;
                      word_sl_decay)
                        xt_drain_before_pick
                        if ! xt_best_sl_weight; then
                            XT_N_BLOCKED=$((XT_N_BLOCKED + 1)); continue
                        fi
                        if xt_is_zero "${XT_BEST_SL_W}"; then
                            otcr_hr
                            echo "SKIP  word_sl_decay: best self-loop weight is 0, so every decay"
                            echo "      gives the same weight (0) and the cells would be identical."
                            continue
                        fi
                        for val in "${RUN_WORD_DECAYS[@]}"; do
                            [ "${val}" = "${word_sl_weight_decay}" ] && continue  # a word_sl cell
                            xt_run_seeds otc_word "$(xt_tag_sl "${XT_BEST_SL_W}" "${val}")" \
                                "$(xt_args_sl_only "${XT_BEST_SL_W}" "${val}")"
                        done ;;
                      word_byp)
                        for val in "${RUN_WORD_DECAYS[@]}"; do
                            xt_run_seeds otc_word "$(xt_tag_byp "${word_byp_weight}" "${val}")" \
                                "$(xt_args_byp_only "${word_byp_weight}" "${val}")"
                        done ;;
                      word_both)
                        xt_drain_before_pick
                        if ! xt_best_sl_pair || ! xt_best_byp_decay; then
                            XT_N_BLOCKED=$((XT_N_BLOCKED + 1)); continue
                        fi
                        xt_run_seeds otc_word \
                            "$(xt_tag_both "${XT_BEST_SL_W}" "${XT_BEST_SL_D}" "${word_byp_weight}" "${XT_BEST_BYP_D}")" \
                            "$(xt_args_both "${XT_BEST_SL_W}" "${XT_BEST_SL_D}" "${word_byp_weight}" "${XT_BEST_BYP_D}")" ;;
                      otc_token_word)
                        xt_drain_before_pick
                        if ! xt_best_byp_decay; then
                            XT_N_BLOCKED=$((XT_N_BLOCKED + 1)); continue
                        fi
                        xt_run_seeds otc_token_word \
                            "$(xt_tag_tw "${tw_token_weight}" "${tw_token_decay}" "${word_byp_weight}" "${XT_BEST_BYP_D}" "${tw_self_loop_weight}" "${tw_self_loop_decay}")" \
                            "$(xt_args_tw "${tw_token_weight}" "${tw_token_decay}" "${word_byp_weight}" "${XT_BEST_BYP_D}" "${tw_self_loop_weight}" "${tw_self_loop_decay}")" ;;
                      otc_token_word_recip)
                        for val in "${RUN_TWR_KS[@]}"; do
                            xt_run_seeds otc_token_word_recip "$(xt_tag_twr "${val}")" \
                                "$(xt_args_twr "${val}")"
                        done ;;
                    esac
                done
            done
        done
    done
}

# --------------------------------------------------------------------------
n_lang=0
for c in ${corpora}; do
    eval "_l=\${${c}_langs:-}"
    [ -n "${_l}" ] && n_lang=$((n_lang + $(echo ${_l} | wc -w)))
done
n_cell_per_lang=0
n_word_cell_per_lang=0      # ssl only
n_decay_not_sl=0
for d in "${RUN_WORD_DECAYS[@]}"; do [ "${d}" = "${word_sl_weight_decay}" ] || n_decay_not_sl=$((n_decay_not_sl + 1)); done
for m in ${modes}; do
    case "${m}" in
        ctc)       n_cell_per_lang=$((n_cell_per_lang + 1)) ;;
        otc_token) n_cell_per_lang=$((n_cell_per_lang + ${#RUN_DECAYS[@]})) ;;
        otc_recip) n_cell_per_lang=$((n_cell_per_lang + ${#RUN_KS[@]})) ;;
        word_sl)        n_word_cell_per_lang=$((n_word_cell_per_lang + ${#RUN_SL_WEIGHTS[@]})) ;;
        word_sl_decay)  n_word_cell_per_lang=$((n_word_cell_per_lang + n_decay_not_sl)) ;;
        word_byp)       n_word_cell_per_lang=$((n_word_cell_per_lang + ${#RUN_WORD_DECAYS[@]})) ;;
        word_both|otc_token_word) n_word_cell_per_lang=$((n_word_cell_per_lang + 1)) ;;
        otc_token_word_recip) n_word_cell_per_lang=$((n_word_cell_per_lang + ${#RUN_TWR_KS[@]})) ;;
    esac
done
n_ssl=0; case " ${variants} " in *" ssl "*) n_ssl=1 ;; esac

echo "======================================================================"
echo "asr_otc_recip | CROSS-TRANSFER SWEEPS (10h subsets)"
echo "  corpora  : ${corpora}"
for _c in ${corpora}; do
    eval "_l=\${${_c}_langs:-<unknown corpus>}"
    printf '    %-8s langs="%s"\n' "${_c}" "${_l}"
done
echo "  variants : ${variants}"
echo "  modes    : ${modes}"
for _m in ${modes}; do
    case "${_m}" in
        ctc)       printf '    %-10s no sweep (1 cell)\n' "ctc" ;;
        otc_token) printf '    %-10s decay %s   (%d values)\n' "otc_token" "${decays}" "${#RUN_DECAYS[@]}" ;;
        otc_recip) printf '    %-10s k     %s   (%d values)\n' "otc_recip" "${ks}" "${#RUN_KS[@]}" ;;
        word_sl)        printf '    %-14s self-loop only, weight %s at decay %s   (ssl only)\n' "word_sl" "${RUN_SL_WEIGHTS[*]}" "${word_sl_weight_decay}" ;;
        word_sl_decay)  printf '    %-14s self-loop only, best word_sl weight, decay %s   (ssl only)\n' "word_sl_decay" "${RUN_WORD_DECAYS[*]}" ;;
        word_byp)       printf '    %-14s bypass only, weight %s, decay %s, arcs from epoch %s   (ssl only)\n' "word_byp" "${word_byp_weight}" "${RUN_WORD_DECAYS[*]}" "${word_star_start_epoch}" ;;
        word_both)      printf '    %-14s both arcs: best self-loop + bypass %s at best decay, arcs from epoch %s   (ssl only)\n' "word_both" "${word_byp_weight}" "${word_star_start_epoch}" ;;
        otc_token_word) printf '    %-14s dual: token %s/%s, word %s/best, self-loop %s/%s, arcs from epoch %s   (ssl only)\n' "otc_token_word" "${tw_token_weight}" "${tw_token_decay}" "${word_byp_weight}" "${tw_self_loop_weight}" "${tw_self_loop_decay}" "${word_star_start_epoch}" ;;
        otc_token_word_recip) printf '    %-14s dual, entropy_reciprocal: token %s->%s, word %s->%s, self-loop %s->%s, k %s, arcs from epoch %s   (ssl only)\n' "otc_token_word_recip" "${twr_token_start}" "${twr_token_end}" "${twr_word_start}" "${twr_word_end}" "${twr_sl_start}" "${twr_sl_end}" "${RUN_TWR_KS[*]}" "${word_star_start_epoch}" ;;
    esac
done
echo "  seeds    : ${seeds}"
echo "  cells    : $(( (n_lang * $(echo ${variants} | wc -w) * n_cell_per_lang + n_lang * n_ssl * n_word_cell_per_lang) * $(echo ${seeds} | wc -w) ))"
echo "  action   : ${action}"
if [ "${action}" = "decode" ]; then
    echo "  decoding : dev + test sets, sequential, in the foreground"
elif [ "${max_decode_jobs}" -eq 0 ]; then
    echo "  decoding : dev + test sets, inline after each training"
else
    echo "  decoding : dev + test sets, backgrounded after each training," \
         "${max_decode_jobs} at a time"
fi
echo "  gpu      : ${gpu}   dry-run: ${dry_run}   force: ${force}"
echo "======================================================================"

OTCR_SCRIPT="${asr_script}"
OTCR_GPU="${gpu}"
OTCR_TRAIN_STAGE="${train_stage}"
OTCR_INFER_MODEL="${infer_model}"
OTCR_INFER_TAG="decode_ctc_asr_model_${infer_model%.pth}"

otcr_hr
if [ "${action}" = "decode" ]; then
    echo "DECODE: scoring every trained cell whose dev or test decode is missing"
else
    echo "TRAIN: training every cell with no model; each finished run's dev and"
    echo "       test decodes are queued and the next training starts at once"
fi
xt_visit_cells

if ! ${dry_run} && [ "${action}" = "train" ]; then
    xt_drain_trainings
fi
if ! ${dry_run} && [ "${max_decode_jobs}" -gt 0 ]; then
    xt_drain_decodes
fi

otcr_hr
if [ "${action}" = "decode" ]; then
    if ${dry_run}; then
        echo "DRY RUN: ${XT_DEC_QUEUED} job(s) would run" \
             "(${XT_N_RESCORE} of them stage 13 only)," \
             "${XT_N_SCORED} already complete," \
             "${XT_N_PENDING} cell(s) not trained yet."
    else
        echo "Decodes  : ${XT_DEC_OK} ok, ${XT_DEC_FAIL} failed," \
             "${XT_N_SCORED} skipped as already complete;" \
             "${XT_N_RESCORE} needed stage 13 only"
        echo "Untrained: ${XT_N_PENDING} cell(s) have no model;" \
             "run again with --action train"
    fi
else
    if ${dry_run}; then
        echo "DRY RUN: ${XT_N_TRAINED} training(s) would run," \
             "${XT_DEC_QUEUED} decode(s) would follow them," \
             "${XT_N_SKIP} cell(s) already trained."
    else
        echo "Trained : ${XT_N_TRAINED} (${XT_N_TRAIN_FAIL} failed)," \
             "${XT_N_SKIP} skipped as already trained"
        echo "Decodes : ${XT_DEC_OK} ok, ${XT_DEC_FAIL} failed"
    fi
fi
if [ "${XT_N_BLOCKED}" -gt 0 ]; then
    echo "Blocked : ${XT_N_BLOCKED} word-pipeline stage(s) waiting on dev WERs of" \
         "earlier stages (see the PICK lines above)"
fi
if [ "${XT_N_NONSSL}" -gt 0 ]; then
    echo "Note    : word_sl/word_sl_decay/word_byp/word_both/otc_token_word/otc_token_word_recip are ssl-only;" \
         "${XT_N_NONSSL} nonssl (mode, language) pair(s) skipped"
fi
echo "Collect WERs with:  python ../analysis/tables/extract_wer.py  (set ROOT first)"
