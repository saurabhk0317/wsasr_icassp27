# shellcheck shell=bash
# ============================================================================
# otc_recip/run_lib.sh -- shared driver for the asr_otc_recip run scripts.
#
# Sourced by run_ebf_otc_recip_*.sh.  Provides the resume logic, the dry-run
# mode, the deterministic experiment naming and the two asr_respin.sh calls,
# so the six master scripts hold only the per-corpus data layout.
#
# Part of ESPnet. Licensed under the Apache License, Version 2.0
# (see the LICENSE file in the ESPnet source root).
#
# ---------------------------------------------------------------------------
# EXPERIMENT NAMING
#   asr_tag = otcrecip_<corpus>_<lang>_<method>[_decay<D>]_seed<N>
#   exp dir = <expdir_root>/exp_<tag>_<suffix>/asr_<asr_tag>
#
# asr_respin.sh builds its path as "<expdir>/asr_<asr_tag>", so the full
# identity of a cell rides in asr_tag.  Nothing in asr_respin.sh is modified.
#
# RESUME
#   A cell is DONE when every test set has score_wer/result.txt.  Done cells
#   are skipped by DEFAULT (not behind a flag) and the reason is printed.
#   --force reruns them.  A cell that trained but did not decode resumes at
#   the decode stage.
#
# LOGS
#   ${OTCR_LOGROOT}/<corpus>/<asr_tag>.{train,decode}.log
#
# NO ABSOLUTE PATHS.  Corpus roots come from the *_DATA variables the master
# scripts define (overridable from the environment or --data_folder), so this
# tree can be moved or run on another machine unchanged.
# ============================================================================

set -euo pipefail

OTCR_LOGROOT="${OTCR_LOGROOT:-logs_otc_recip}"

# Common flags, consumed by utils/parse_options.sh in the master scripts.
dry_run=false
force=false
methods=""
langs=""
seeds=""

# Counters for the end-of-run summary.
OTCR_N_RUN=0
OTCR_N_SKIP=0
OTCR_N_PARTIAL=0

otcr_die() { echo "ERROR: $*" >&2; exit 1; }

# --------------------------------------------------------------------------
# otcr_expand_bool_flags "$@"
#   utils/parse_options.sh only understands "--flag value", so a bare
#   --dry-run would abort with 'expected "true" or "false"'.  Rewrite the bare
#   boolean flags into that form and leave everything else alone.  The caller
#   then does:  otcr_expand_bool_flags "$@"; set -- "${OTCR_ARGV[@]}"
# --------------------------------------------------------------------------
otcr_expand_bool_flags() {
    OTCR_ARGV=()
    local a
    for a in "$@"; do
        case "${a}" in
            --dry-run|--dry_run|--force) OTCR_ARGV+=("${a}" true) ;;
            *)                           OTCR_ARGV+=("${a}") ;;
        esac
    done
}

otcr_hr() { printf '%s\n' "----------------------------------------------------------------------"; }

# --------------------------------------------------------------------------
# otcr_require_config <path>
#   Fail loudly on a missing config.  Never fall back to another file: a
#   silent fallback would put a mislabelled result into the table.
# --------------------------------------------------------------------------
otcr_require_config() {
    local cfg="$1"
    [ -n "${cfg}" ] || otcr_die "empty config path"
    [ -f "${cfg}" ] || otcr_die "missing train config '${cfg}'.
  Configs are NOT generated on the fly. Expected one of:
    conf/{respin,libri,fleurs}_final/{ctc,otc_word,otc_token,otc_recip}[_conformer].yaml
    conf/cross_transfer/<corpus>_otc_token_{nonssl,ssl}.yaml
    conf/label_smoothing/libri_ls{0p005,0p01,0p02,0p05}.yaml"
}

# --------------------------------------------------------------------------
# otcr_filter <selected> <all...>
#   Sets the global array OTCR_FILTERED to the intersection, preserving the
#   order of <all...>.  Dies on a name that is not in <all...>, so a typo in
#   --methods/--langs stops the run instead of quietly doing nothing.
#
#   Deliberately NOT a subshell (no "$(...)" / process substitution): otcr_die
#   inside a subshell would kill only the subshell, and the caller would sail
#   on with an empty selection and report "0 cells run" as if that were fine.
#   Call it directly and read OTCR_FILTERED:
#       otcr_filter "${methods}" "${ALL_METHODS[@]}"
#       RUN_METHODS=("${OTCR_FILTERED[@]}")
# --------------------------------------------------------------------------
otcr_filter() {
    local selected="$1"; shift
    local all=("$@")
    OTCR_FILTERED=()
    if [ -z "${selected}" ]; then OTCR_FILTERED=("${all[@]}"); return; fi
    local want item found
    for want in ${selected}; do
        found=false
        for item in "${all[@]}"; do
            [ "${item}" = "${want}" ] && found=true && break
        done
        ${found} || otcr_die "'${want}' is not one of: ${all[*]}"
    done
    for item in "${all[@]}"; do
        for want in ${selected}; do
            [ "${item}" = "${want}" ] && OTCR_FILTERED+=("${item}") && break
        done
    done
    # MUST be here.  Without it the function inherits the status of the last
    # [ ... ] test, which is 1 whenever the LAST candidate was not selected --
    # and under `set -e` that killed the whole script before a single cell ran.
    # ("--corpora respin" hit this: "fleurs" is last, so the filter "failed".)
    return 0
}

# --------------------------------------------------------------------------
# otcr_cell_state <expdir> <infer_model> <infer_tag> <test_sets...>
#   -> "done"    every test set scored
#      "decode"  model trained, scoring incomplete
#      "train"   nothing usable yet
# --------------------------------------------------------------------------
otcr_cell_state() {
    local expdir="$1" infer_model="$2" infer_tag="$3"; shift 3
    local data
    if [ ! -f "${expdir}/${infer_model}" ]; then echo train; return; fi
    for data in "$@"; do
        if [ ! -f "${expdir}/${infer_tag}/${data}/score_wer/result.txt" ]; then
            echo decode; return
        fi
    done
    echo done
}

# --------------------------------------------------------------------------
# otcr_show <cmd...>   -- print a command the way it would be run
# --------------------------------------------------------------------------
otcr_show() {
    printf '  %s' "$1"; shift
    # Everything after the program name is a strict "--flag value" pair.  Pair
    # them up explicitly rather than splitting on a leading "--", because some
    # VALUES legitimately start with "--" (e.g. --local_data_opts "--stage 0").
    while [ "$#" -gt 0 ]; do
        printf ' \\\n      %s' "$1"
        if [ "$#" -ge 2 ]; then printf ' %q' "$2"; shift 2; else shift; fi
    done
    printf '\n'
}

# --------------------------------------------------------------------------
# otcr_run_cell
#   Runs (or skips, or prints) one experiment cell.
#
#   Required variables set by the caller BEFORE the call:
#     OTCR_CORPUS   corpus name (respin|libri|fleurs)
#     OTCR_LANG     language token, as it appears in the data layout
#     OTCR_METHOD   method name (ctc|otc_word|otc_token|otc_recip|...)
#     OTCR_SEED     integer seed
#     OTCR_ASR_ARGS optional extra training arguments, appended to --asr_args
#                   after --seed.  Used to override individual config keys
#                   without writing a config file per value, e.g.
#                     OTCR_ASR_ARGS="--ctc_conf bypass_weight_decay=0.8"
#                   espnet's NestedDictAction MERGES such a key into the block
#                   loaded from --config rather than replacing it, so only the
#                   named key changes.  Keep each token space-free: asr_respin.sh
#                   expands ${asr_args} unquoted, so the string is word-split.
#     OTCR_VARIANT  extra tag component, e.g. "decay0p7" or "" for none
#     OTCR_CONFIG   train config path
#     OTCR_EXPROOT  e.g. exp_respin
#     OTCR_EXPTAG   the per-language dir stem, e.g. "mt_s5"
#     OTCR_SUFFIX   feature suffix, e.g. stft_char / ssl_xlsr300m
#     OTCR_TESTSETS space-separated test set names
#     OTCR_INFER_MODEL / OTCR_INFER_TAG / OTCR_INFER_CONFIG
#     OTCR_TRAIN_STAGE
#     OTCR_COMMON   ARRAY of the corpus-specific asr_respin.sh flags
#     OTCR_SCRIPT   the recipe script (asr_respin.sh)
#     OTCR_GPU      CUDA device id
# --------------------------------------------------------------------------
otcr_run_cell() {
    local variant=""
    [ -n "${OTCR_VARIANT}" ] && variant="_${OTCR_VARIANT}"
    local asr_tag="otcrecip_${OTCR_CORPUS}_${OTCR_LANG}_${OTCR_METHOD}${variant}_seed${OTCR_SEED}"
    local expdir_root="${OTCR_EXPROOT}/exp_${OTCR_EXPTAG}_${OTCR_SUFFIX}"
    local expdir="${expdir_root}/asr_${asr_tag}"

    otcr_require_config "${OTCR_CONFIG}"

    local logdir="${OTCR_LOGROOT}/${OTCR_CORPUS}"
    local train_log="${logdir}/${asr_tag}.train.log"
    local decode_log="${logdir}/${asr_tag}.decode.log"

    # shellcheck disable=SC2086
    local state
    state=$(otcr_cell_state "${expdir}" "${OTCR_INFER_MODEL}" "${OTCR_INFER_TAG}" ${OTCR_TESTSETS})

    otcr_hr
    echo "CELL  ${asr_tag}"
    echo "  config : ${OTCR_CONFIG}"
    echo "  expdir : ${expdir}"
    echo "  logs   : ${train_log}"
    echo "             ${decode_log}"

    if [ "${state}" = "done" ] && ! ${force}; then
        echo "  SKIP   already complete -- score_wer/result.txt present for:"
        local d
        for d in ${OTCR_TESTSETS}; do echo "             ${OTCR_INFER_TAG}/${d}"; done
        echo "         (pass --force to rerun this cell)"
        OTCR_N_SKIP=$((OTCR_N_SKIP + 1))
        return 0
    fi

    if ${force}; then
        echo "  FORCE  rerunning regardless of existing results"
        state=train
    elif [ "${state}" = "decode" ]; then
        echo "  RESUME model exists, scoring incomplete -> decode + score only"
        OTCR_N_PARTIAL=$((OTCR_N_PARTIAL + 1))
    else
        echo "  RUN    no trained model -> collect-stats + train + decode + score"
    fi

    # ---- assemble the two invocations -----------------------------------
    local -a train_cmd decode_cmd
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

    decode_cmd=("./${OTCR_SCRIPT}"
        --stage 12 --stop_stage 13
        "${OTCR_COMMON[@]}"
        --expdir "${expdir_root}"
        --asr_config "${OTCR_CONFIG}"
        --inference_config "${OTCR_INFER_CONFIG}"
        --asr_tag "${asr_tag}"
        --inference_tag "${OTCR_INFER_TAG}"
        --inference_asr_model "${OTCR_INFER_MODEL}"
        --test_sets "${OTCR_TESTSETS}"
        --skip_train true
        --asr_task asr_otc_recip
        --gpu_inference false)

    if ${dry_run}; then
        echo "  DRY-RUN commands (not executed):"
        echo "    CUDA_VISIBLE_DEVICES=${OTCR_GPU}"
        [ "${state}" = "train" ] && otcr_show "${train_cmd[@]}"
        otcr_show "${decode_cmd[@]}"
        OTCR_N_RUN=$((OTCR_N_RUN + 1))
        return 0
    fi

    mkdir -p "${logdir}"
    export CUDA_VISIBLE_DEVICES="${OTCR_GPU}"

    if [ "${state}" = "train" ]; then
        echo "  ...training (tail -f ${train_log})"
        if ! "${train_cmd[@]}" > "${train_log}" 2>&1; then
            echo "  FAILED training -- see ${train_log}" >&2
            tail -n 25 "${train_log}" >&2 || true
            return 1
        fi
    fi

    if [ ! -f "${expdir}/${OTCR_INFER_MODEL}" ]; then
        echo "  FAILED ${expdir}/${OTCR_INFER_MODEL} was not produced" >&2
        return 1
    fi

    echo "  ...decoding + scoring (tail -f ${decode_log})"
    if ! "${decode_cmd[@]}" > "${decode_log}" 2>&1; then
        echo "  FAILED decoding -- see ${decode_log}" >&2
        tail -n 25 "${decode_log}" >&2 || true
        return 1
    fi

    echo "  DONE   ${expdir}/RESULTS.md"
    OTCR_N_RUN=$((OTCR_N_RUN + 1))
}

# --------------------------------------------------------------------------
# otcr_summary
#   End-of-run report.  Exits 1 if the selection matched NO cells: a run that
#   quietly does nothing and returns 0 is the worst possible outcome, because
#   it looks like success.  This is what "--corpora respin did not run
#   anything" was -- the selection filtered every cell away and said nothing.
# --------------------------------------------------------------------------
otcr_summary() {
    otcr_hr
    if [ $((OTCR_N_RUN + OTCR_N_SKIP)) -eq 0 ]; then
        echo "ERROR: no cells matched the selection -- nothing was run." >&2
        echo "  methods=${methods:-<all>}  langs=${langs:-<all>}" >&2
        [ -n "${corpora+x}" ] && echo "  corpora=${corpora:-<all>}" >&2
        echo "  seeds=${seeds:-<unset>}" >&2
        echo "Check the spelling of --methods / --langs / --corpora against" >&2
        echo "the defaults printed in the header above." >&2
        exit 1
    fi
    if ${dry_run}; then
        echo "DRY RUN: ${OTCR_N_RUN} cell(s) would run, ${OTCR_N_SKIP} already complete."
    else
        echo "Finished: ${OTCR_N_RUN} cell(s) run (${OTCR_N_PARTIAL} resumed at decode), \
${OTCR_N_SKIP} skipped as already complete."
    fi
    echo "Collect the table with:  python scripts/collect_otc_recip_results.py"
}
