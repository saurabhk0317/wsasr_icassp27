#!/usr/bin/env bash
# ============================================================================
# asr_otc_recip | SCHEDULE-SENSITIVITY SWEEPS (paper Fig. 2, fig:sweep)
#
# Random 10 h subset of LibriSpeech train-clean-100, frozen SSL front-end
# (wav2vec 2.0 Base, BPE 200), WER on the combined dev-clean + dev-other set.
# Three modes, run over the same controlled setup so they are comparable:
#
#   ctc         the dashed reference line.  Star arcs off.  Nothing swept:
#               exactly one cell per seed.
#   otc_token   Fig. 2(a): sweeps the epoch-indexed geometric decay tau of
#               OTC-T_e (bypass and self-loop decay set together).
#   otc_recip   Fig. 2(b): sweeps the entropy shaping exponent kappa
#               (entropy_shape_k) of OTC-T_H.
#
# The selected values (tau = 0.775, kappa = 1.3) are the ones used by every
# config under conf/<corpus>_final/.  analysis/tables/plot_hparam_sweep.py
# draws the figure from the dev WERs.
#
# NO PER-VALUE CONFIG FILES.  The swept parameter is injected at run time:
#     otc_token   --ctc_conf bypass_weight_decay=<D> --ctc_conf self_loop_weight_decay=<D>
#     otc_recip   --ctc_conf entropy_shape_k=<K>
# espnet's NestedDictAction MERGES a "key=value" form into the block loaded
# from --config rather than replacing it, so only the named key moves and
# everything else comes from the shipped config.
#
# ---------------------------------------------------------------------------
# WHAT RUNS.  modes x sweep values x seeds.
#
#   --action             train | decode   (default: train)
#   --modes              subset of "ctc otc_token otc_recip" (default: all)
#   --decays             explicit decay list; overrides the range below
#   --decay_from/_to/_step   default 1.0 / 0.6 / 0.025  (17 values)
#   --ks                 explicit K list; overrides the range below
#   --k_from/_to/_step   default 0.8 / 1.4 / 0.1  -> 0.8 0.9 1 1.1 1.2 1.3 1.4
#   --libri_trainset     the 10 h training subset (default train_clean_100_ran_10h)
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
# DECODE SETS are "dev" (dev-clean + dev-other, the set Fig. 2 reports)
# followed by any --libri_testsets (default none).  Each set is a separate job
# with its own state, so a cell whose dev decode landed but whose test decode
# did not is finished off rather than redone.
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
# here even though --score_lang_id is false.
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
# decodes); the end of the run waits for every
# running training first, and an interrupt terminates them (whole process
# tree) along with the decodes.  More GPUs finish trainings faster than one
# decode slot may score them; raise --max_decode_jobs if the queue backs up.
#
# This asynchronous behaviour is local to THIS script: it overrides
# otcr_run_cell after sourcing otc_recip/run_lib.sh, so the other run scripts
# keep the original train-then-decode-in-sequence behaviour.
#
# ---------------------------------------------------------------------------
# CONFIG RESOLUTION.  One file per mode:
#     conf/sweep/libri_<mode>_ssl.yaml
# These are sweep-specific and deliberately NOT the main-table configs under
# conf/<corpus>_final/, which this script never reads.  Held fixed across the
# three:
#     initial_bypass_weight    -25.0
#     initial_self_loop_weight  -5.0
#     batch_bins / accum_grad   6000000 / 1  (10 h subset)
#     warmup_steps              3000
#
# Usage
#   ./run_ebf_otc_recip_sweep.sh --dry-run
#   ./run_ebf_otc_recip_sweep.sh
#   ./run_ebf_otc_recip_sweep.sh --action decode
#   ./run_ebf_otc_recip_sweep.sh --modes "ctc otc_recip"
#   ./run_ebf_otc_recip_sweep.sh --modes otc_token --decays "0.775 0.8"
#   ./run_ebf_otc_recip_sweep.sh --gpu "0 1" --max_decode_jobs 2
# ============================================================================

. ./otc_recip/run_lib.sh

gpu=0                      # one id, or several: --gpu "0 1" (one training per GPU)
action=train               # train | decode   (see the header)
seeds=20261
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

# ---- dataset: 10 h subset of train-clean-100 --------------------------------
libri_lang=en
libri_trainset=train_clean_100_ran_10h
# The exp tag must encode the SUBSET, not just the corpus.  The full-data runs
# already occupy exp_libri/exp_libri100_en_*, and reusing that tag for the 10h
# sweep would make the resume logic treat 100h results as 10h ones.
libri_exptag=libri100_ran_10h
# Fig. 2 reports the dev set only; list test sets here to decode them as well.
libri_testsets=""

asr_script=asr_respin.sh
max_decode_jobs=1          # --action train only: concurrent background
                           # decodes, 0 = inline.  --action decode always runs
                           # its decodes inline, one at a time.
train_stage=11             # lower to 10 to redo collect-stats
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
libri_data_root=data_libri
libri_nbpe=200

otcr_expand_bool_flags "$@"
set -- "${OTCR_ARGV[@]}"
. utils/parse_options.sh

[ -n "${modes}" ] || otcr_die "--modes is empty"
for m in ${modes}; do
    case "${m}" in
        ctc|otc_token|otc_recip) ;;
        *) otcr_die "--modes may only contain 'ctc', 'otc_token' and 'otc_recip', got '${m}'" ;;
    esac
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
# Cell enumeration: LibriSpeech, SSL front-end, every mode x value x seed.
# ===========================================================================
xt_visit_cells() {
    local mode val slug tag data_folder train_set valid_set t

    OTCR_CORPUS=libri
    OTCR_LANG="${libri_lang}"
    OTCR_EXPROOT=exp_libri
    OTCR_INFER_CONFIG=conf/libri/decode_ctc.yaml
    OTCR_SUFFIX="ssl_w2v2base960h_bpe${libri_nbpe}"
    tag="${libri_exptag}_${libri_lang}"
    OTCR_EXPTAG="${tag}"
    data_folder="${libri_data_root}"
    train_set="${libri_trainset}"
    valid_set="dev"
    OTCR_COMMON=(
        --tag "${tag}" --data_folder "${data_folder}"
        --dumpdir "dumps/dump_${tag}" --lang "${libri_lang}"
        --auxiliary_data_tags ""
        --token_type bpe --nbpe "${libri_nbpe}"
        --local_score_opts "--score_lang_id false"
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
    # Decode targets: the dev set first, then any test set(s).  Each is scored
    # separately, so a half-finished cell is completed rather than redone.
    XT_DECODE_SETS=("${valid_set}")
    for t in ${libri_testsets}; do XT_DECODE_SETS+=("${t}"); done
    # xt_decode_subdir needs this to spot the "org/" case.
    XT_VALID_SET="${valid_set}"
    otcr_require_config "${OTCR_INFER_CONFIG}"

    for mode in ${modes}; do
        OTCR_METHOD="${mode}"
        OTCR_CONFIG="conf/sweep/libri_${mode}_ssl.yaml"
        otcr_require_config "${OTCR_CONFIG}"

        case "${mode}" in
          ctc)
            OTCR_VARIANT=ssl
            OTCR_ASR_ARGS=""
            for seed in ${seeds}; do
                OTCR_SEED="${seed}"; otcr_run_cell
            done ;;
          otc_token)
            for val in "${RUN_DECAYS[@]}"; do
                slug="${val//./p}"
                OTCR_VARIANT="ssl_decay${slug}"
                OTCR_ASR_ARGS="--ctc_conf bypass_weight_decay=${val} --ctc_conf self_loop_weight_decay=${val}"
                for seed in ${seeds}; do
                    OTCR_SEED="${seed}"; otcr_run_cell
                done
            done ;;
          otc_recip)
            for val in "${RUN_KS[@]}"; do
                slug="${val//./p}"
                OTCR_VARIANT="ssl_k${slug}"
                OTCR_ASR_ARGS="--ctc_conf entropy_shape_k=${val}"
                for seed in ${seeds}; do
                    OTCR_SEED="${seed}"; otcr_run_cell
                done
            done ;;
        esac
    done
}

# --------------------------------------------------------------------------
n_cell=0
for m in ${modes}; do
    case "${m}" in
        ctc)       n_cell=$((n_cell + 1)) ;;
        otc_token) n_cell=$((n_cell + ${#RUN_DECAYS[@]})) ;;
        otc_recip) n_cell=$((n_cell + ${#RUN_KS[@]})) ;;
    esac
done

echo "======================================================================"
echo "asr_otc_recip | SCHEDULE-SENSITIVITY SWEEPS (LibriSpeech 10 h, SSL)"
echo "  trainset : ${libri_trainset}"
echo "  modes    : ${modes}"
for _m in ${modes}; do
    case "${_m}" in
        ctc)       printf '    %-10s no sweep (1 cell)\n' "ctc" ;;
        otc_token) printf '    %-10s decay %s   (%d values)\n' "otc_token" "${decays}" "${#RUN_DECAYS[@]}" ;;
        otc_recip) printf '    %-10s k     %s   (%d values)\n' "otc_recip" "${ks}" "${#RUN_KS[@]}" ;;
    esac
done
echo "  seeds    : ${seeds}"
echo "  cells    : $(( n_cell * $(echo ${seeds} | wc -w) ))"
echo "  action   : ${action}"
if [ "${action}" = "decode" ]; then
    echo "  decoding : dev (+ test) sets, sequential, in the foreground"
elif [ "${max_decode_jobs}" -eq 0 ]; then
    echo "  decoding : dev (+ test) sets, inline after each training"
else
    echo "  decoding : dev (+ test) sets, backgrounded after each training," \
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
echo "Dev WERs: the Overall WER line of <expdir>/${OTCR_INFER_TAG}/org/dev/result_lid.txt;"
echo "plot with: python ../analysis/tables/plot_hparam_sweep.py"
