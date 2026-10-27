#!/usr/bin/env python3
"""
GRANULARITY x SCHEDULE ablation -- LibriSpeech 10h clean + worderror30 (asr_otc_recip).

QUESTION.  otc_word wins on worderror30 and otc_token wins on clean data, but the
two configs differ in TWO things at once:

    granularity   one <star> arc per WORD            vs  per TOKEN
    arc weights   -19 / +3.75, decay 0.975 / 0.999   vs  -25 / -5, decay 0.775

so the gap cannot be attributed to either.  This crosses them:

                        word weights        token weights
    word granularity    otc_word (base)     word_tokw    <- new
    token granularity   token_wordw <- new  otc_token (base)

NO PER-CELL YAML.  Each swapped cell is its own base config plus command-line
overrides of the four schedule keys; espnet's NestedDictAction MERGES each
key=value into the ctc_conf block from --config, so every other key survives.
The four values are READ OUT of the other base config at startup (see
sched_args), so they can never drift from conf/libri_10h/*.yaml.  Only training
needs them: decode re-reads ${asr_exp}/config.yaml, already merged.

CELLS RUN HERE (--cells, default all five):
    clean:word          the missing clean-10h otc_word baseline
    clean:word_tokw     clean:token_wordw
    we30:word_tokw      we30:token_wordw

BASELINES NOT RUN HERE, only reported by --table, so that no two scripts ever
train into the same expdir:
    clean   ctc, otc_token   existing cross-transfer runs (ctc_ssl,
                             otc_token_ssl_decay0p775 -- identical settings to
                             conf/libri_10h/{ctc,otc_token}.yaml)
    we30    ctc, otc_word, otc_token   run_ebf_otc_recip_master_libri_error.sh
A cell whose asr_respin.sh is already running (pgrep on --asr_tag) is skipped,
so this can be started while that script is still working.

DATA.  Dump and collect-stats must already exist for the dataset; they are NOT
built here.  run_ebf_otc_recip_master_libri_error.sh prepares worderror30, and
the clean-10h dump/stats are already in place.  The BPE token list is the
clean-10h one for both datasets.

SCHEDULING.  One training per GPU slot (--gpus 0 1 runs two cells at once).
When a training finishes, its decode is launched in the BACKGROUND (CPU,
gpu_inference false) and the freed slot immediately takes the next cell.  A
decode is gated on the training subprocess's OWN exit code, so a crashed or
OOM-killed run never decodes a nonexistent checkpoint.

Examples
    ./run_ebf_otc_ablation_granularity_libri.py --dry-run
    ./run_ebf_otc_ablation_granularity_libri.py --gpus 0 1
    ./run_ebf_otc_ablation_granularity_libri.py --gpus 1 --cells we30:word_tokw we30:token_wordw
    ./run_ebf_otc_ablation_granularity_libri.py --table
"""
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================
# GLOBAL PROCESS REGISTRIES
# ============================================================
ALL_PROCS: List[subprocess.Popen] = []
ALL_DECODES: List[subprocess.Popen] = []
PROC_LOCK = threading.Lock()
DECODE_LOCK = threading.Lock()


def register_proc(p):
    with PROC_LOCK:
        ALL_PROCS.append(p)


def kill_all_procs():
    with PROC_LOCK:
        procs = list(ALL_PROCS)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in procs:
            try:
                if p.poll() is None:
                    os.killpg(os.getpgid(p.pid), sig)
            except Exception:
                pass
        time.sleep(1)


def handle_signal(signum, frame):
    print(f"\n{ts()} RECEIVED SIGNAL {signum}, TERMINATING ALL JOBS")
    kill_all_procs()
    sys.exit(1)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ============================================================
# USER CONFIG
# ============================================================
VENV_ACTIVATE = "/home3/Saurabh/tools/espnet_may22_2026/tools/venv/bin/activate"
ASR_SCRIPT = "./asr_respin.sh"
ASR_TASK = "asr_otc_recip"

CONF_DIR = "conf/libri_10h"
INFER_CONFIG = "conf/libri/decode_ctc.yaml"
INFER_MODEL = "valid.loss.ave.pth"
INFER_TAG = "decode_ctc_asr_model_valid.loss.ave"

DATA_ROOT = "data_libri"
CLEAN_TRAIN_SET = "train_clean_100_ran_10h"
WE30_TRAIN_SET = "train_clean_100_ran_10h_worderror30"
VALID_SET = "dev"
TEST_SETS = ["test_clean", "test_other"]

# Dump dir and BPE token list are SHARED by both datasets (the clean-10h ones);
# the expdir is per dataset so the two never mix.
DUMP_TAG = "libri100_ran_10h_en"
SUFFIX = "ssl_w2v2base960h_bpe200"

NBPE = 200
TOKEN_TYPE = "bpe"
SEED = 20261
TRAIN_NJ = 16
INFERENCE_NJ = 32
NGPU = 1
USE_LM = "false"
FEATS_TYPE = "raw"
AUDIO_FORMAT = "wav"
SPEED_PERTURB = "0.9 1.0 1.1"
FEATS_NORMALIZE = "utt_mvn"
BPE_NLSYMS = ""
LM_CONFIG = "conf/train_lm.yaml"
AUX_TAG = ""

GPUS = ["0", "1"]
LOG_DIR = Path("logs_otc_recip/libri_granularity")
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "clean": dict(train_set=CLEAN_TRAIN_SET, exptag="libri100_ran_10h_en", dstag="10h"),
    "we30": dict(train_set=WE30_TRAIN_SET,
                 exptag="libri100_ran_10h_worderror30_en", dstag="10h_we30"),
}

#: The four keys that make up a schedule; swapped as a block, never piecemeal.
SCHED_KEYS = ["initial_bypass_weight", "bypass_weight_decay",
              "initial_self_loop_weight", "self_loop_weight_decay"]


def ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def sched_args(config: str) -> str:
    """The SCHED_KEYS of `config` as "--ctc_conf key=value ..." overrides.

    Read from the yaml at startup rather than hard-coded, so the swapped cells
    always carry exactly what the base configs say.  A plain scan is enough:
    each key appears once, indented inside ctc_conf, as "key: value".
    """
    text = Path(config).read_text()
    out = []
    for key in SCHED_KEYS:
        m = re.search(rf"^\s+{key}:\s*(\S+)\s*$", text, re.M)
        if m is None:
            sys.exit(f"ERROR: key '{key}' not found in {config}")
        out.append(f"--ctc_conf {key}={m.group(1)}")
    return " ".join(out)


# ============================================================
# CELLS
# ============================================================
# name -> dataset, method, variant (the asr_tag suffix), base config, overrides.
# asr_tag = otcrecip_libri_en_<method>_<variant>_seed<SEED>, matching the bash
# master scripts, so baselines trained there are found by the table below.
def _cell(dataset: str, method: str, variant: str, config: str, args: str = "") -> Dict:
    ds = DATASETS[dataset]
    return dict(dataset=dataset, method=method,
                variant=f"{variant}{ds['dstag']}", config=config, args=args)


def build_cells() -> Dict[str, Dict]:
    token_sched = sched_args(f"{CONF_DIR}/otc_token.yaml")
    word_sched = sched_args(f"{CONF_DIR}/otc_word.yaml")
    cells = {}
    for ds in ("clean", "we30"):
        # The clean otc_word baseline is missing and is ours to run; the clean
        # ctc/otc_token baselines already exist under their cross-transfer tags
        # (see table_rows), so they are NOT cells -- re-running them under a new
        # name would just retrain the same settings.
        if ds == "clean":
            cells[f"{ds}:word"] = _cell(ds, "otc_word", "", f"{CONF_DIR}/otc_word.yaml")
        cells[f"{ds}:word_tokw"] = _cell(
            ds, "otc_word", "tokw_", f"{CONF_DIR}/otc_word.yaml", token_sched)
        cells[f"{ds}:token_wordw"] = _cell(
            ds, "otc_token", "wordw_", f"{CONF_DIR}/otc_token.yaml", word_sched)
    # The worderror30 baselines belong to run_ebf_otc_recip_master_libri_error.sh
    # and use the SAME asr_tags, so they are opt-in only (--cells): run them from
    # exactly one of the two scripts, never both at once.
    for name, method in (("ctc", "ctc"), ("word", "otc_word"), ("token", "otc_token")):
        cells[f"we30:{name}"] = _cell(
            "we30", method, "", f"{CONF_DIR}/{method}.yaml")
    return cells


CELLS = build_cells()
#: The ablation itself.  The opt-in worderror30 baselines are excluded.
DEFAULT_CELLS = [c for c in CELLS if c not in ("we30:ctc", "we30:word", "we30:token")]

#: Rows of the result table: (dataset, label, method, variant, granularity,
#: schedule).  Cells trained elsewhere are included so the 2x2 can be read off
#: in one place; "-" simply means "not there yet".
def table_rows():
    rows = []
    for ds, d in DATASETS.items():
        t = d["dstag"]
        base = ([("ctc", "ctc", "ssl", "-", "-"),
                 ("token", "otc_token", "ssl_decay0p775", "token", "token")]
                if ds == "clean" else
                [("ctc", "ctc", t, "-", "-"),
                 ("token", "otc_token", t, "token", "token")])
        rows += [(ds,) + r for r in base]
        rows += [(ds, "word", "otc_word", t, "word", "word"),
                 (ds, "word_tokw", "otc_word", f"tokw_{t}", "word", "token"),
                 (ds, "token_wordw", "otc_token", f"wordw_{t}", "token", "word")]
    return rows


# ============================================================
# PATHS / STATE
# ============================================================
def asr_tag(method: str, variant: str) -> str:
    return f"otcrecip_libri_en_{method}_{variant}_seed{SEED}"


def expdir_root(dataset: str) -> str:
    return f"exp_libri/exp_{DATASETS[dataset]['exptag']}_{SUFFIX}"


def expdir_for(dataset: str, method: str, variant: str) -> Path:
    return Path(expdir_root(dataset)) / f"asr_{asr_tag(method, variant)}"


def training_done(expdir: Path) -> bool:
    return (expdir / INFER_MODEL).is_file()


def decoding_done(expdir: Path, test_set: str) -> bool:
    d = expdir / INFER_TAG / test_set
    # result_lid.txt is written LAST by stage 13, so it marks a COMPLETE scoring
    # pass; score_wer/result.txt alone can be a run that died partway through.
    return (d / "result_lid.txt").is_file()


def already_running(tag: str) -> bool:
    """True if some other process is already running asr_respin.sh for this tag.

    Guards against racing run_ebf_otc_recip_master_libri_error.sh (or a second
    copy of this script) into the same expdir.  The pattern requires the recipe
    script itself, so an unrelated shell whose command line merely CONTAINS the
    tag (an ls, a grep, this script's own launcher) is not mistaken for a run.
    """
    return subprocess.run(["pgrep", "-f", f"asr_respin.sh .*asr_tag {tag}( |$)"],
                          stdout=subprocess.DEVNULL).returncode == 0


def check_data(dataset: str) -> None:
    ds = DATASETS[dataset]
    dump = Path(f"dumps/dump_{DUMP_TAG}/{FEATS_TYPE}/{ds['train_set']}_sp/wav.scp")
    stats = (Path(expdir_root(dataset))
             / f"asr_stats_{FEATS_TYPE}_en_{TOKEN_TYPE}{NBPE}_sp/train/text_shape.{TOKEN_TYPE}")
    for p, what in ((dump, "dump"), (stats, "collect-stats")):
        if not p.is_file():
            sys.exit(f"ERROR: missing {what} for '{dataset}': {p}\n"
                     "  run ./run_ebf_otc_recip_master_libri_error.sh first "
                     "(it prepares the worderror30 dump and stats)")


# ============================================================
# COMMANDS
# ============================================================
def run_bash_in_venv(cmd: str) -> subprocess.Popen:
    wrapped = f'bash -c \'source "{VENV_ACTIVATE}" && {cmd}\''
    proc = subprocess.Popen(wrapped, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    register_proc(proc)
    return proc


def pump_to_log(proc: subprocess.Popen, log_path: Path):
    with log_path.open("wb") as f:
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            f.write(chunk)
            f.flush()


def asr_cmd(gpu: str, cell: str, *, stage: int, stop_stage: int,
            test_sets: str, skip_train: str, asr_args: str) -> str:
    c = CELLS[cell]
    ds = DATASETS[c["dataset"]]
    # One line per flag, joined with " ": no line continuations, because a
    # trailing backslash would survive as an escaped space in the shell string.
    flags = [
        f"CUDA_VISIBLE_DEVICES={gpu} {ASR_SCRIPT}",
        f"--stage {stage} --stop_stage {stop_stage}",
        f'--tag "{DUMP_TAG}"',
        f'--data_folder "{DATA_ROOT}"',
        f'--dumpdir "dumps/dump_{DUMP_TAG}"',
        f'--expdir "{expdir_root(c["dataset"])}"',
        "--lang en",
        f'--auxiliary_data_tags "{AUX_TAG}"',
        f'--audio_format "{AUDIO_FORMAT}"',
        f"--use_lm {USE_LM}",
        f'--feats_normalize "{FEATS_NORMALIZE}"',
        f'--lm_config "{LM_CONFIG}"',
        f'--token_type "{TOKEN_TYPE}"',
        f'--nbpe "{NBPE}"',
        f'--bpe_nlsyms "{BPE_NLSYMS}"',
        f'--feats_type "{FEATS_TYPE}"',
        f'--speed_perturb_factors "{SPEED_PERTURB}"',
        f'--train_set "{ds["train_set"]}"',
        f'--valid_set "{VALID_SET}"',
        f'--test_sets "{test_sets}"',
        f'--bpe_train_text "{DATA_ROOT}/{CLEAN_TRAIN_SET}/text"',
        f'--lm_train_text "{DATA_ROOT}/{CLEAN_TRAIN_SET}/text"',
        '--local_score_opts "--score_lang_id false"',
        f"--nj {TRAIN_NJ}",
        f"--ngpu {NGPU}",
        f"--inference_nj {INFERENCE_NJ}",
        f'--asr_config "{c["config"]}"',
        f'--inference_config "{INFER_CONFIG}"',
        f'--asr_tag "{asr_tag(c["method"], c["variant"])}"',
        f'--inference_tag "{INFER_TAG}"',
        f'--inference_asr_model "{INFER_MODEL}"',
        f'--asr_args "{asr_args}"',
        f"--skip_train {skip_train}",
        f'--asr_task "{ASR_TASK}"',
        "--gpu_inference false",
    ]
    return " ".join(flags)


def train_cmd(gpu: str, cell: str) -> str:
    c = CELLS[cell]
    args = f"--seed {SEED}" + (f" {c['args']}" if c["args"] else "")
    return asr_cmd(gpu, cell, stage=11, stop_stage=11, test_sets=VALID_SET,
                   skip_train="false", asr_args=args)


def decode_cmd(gpu: str, cell: str, test_sets: List[str]) -> str:
    return asr_cmd(gpu, cell, stage=12, stop_stage=13,
                   test_sets=" ".join(test_sets), skip_train="true", asr_args="")


# ============================================================
# JOB LAUNCHER
# ============================================================
def launch_decode(gpu: str, cell: str) -> None:
    c = CELLS[cell]
    expdir = expdir_for(c["dataset"], c["method"], c["variant"])
    todo = [t for t in TEST_SETS if not decoding_done(expdir, t)]
    if not todo:
        print(f"{ts()} [GPU {gpu}] SKIP DECODE  {cell} (already scored)")
        return
    logf = LOG_DIR / f"{cell.replace(':', '_')}.decode.log"
    print(f"{ts()} [GPU {gpu}] LAUNCH DECODE {cell} -> {' '.join(todo)}  ({logf})")
    proc = run_bash_in_venv(decode_cmd(gpu, cell, todo))
    with DECODE_LOCK:
        ALL_DECODES.append(proc)
    threading.Thread(target=pump_to_log, args=(proc, logf), daemon=True).start()


def launch_job(gpu: str, cell: str) -> Optional[subprocess.Popen]:
    c = CELLS[cell]
    tag = asr_tag(c["method"], c["variant"])
    expdir = expdir_for(c["dataset"], c["method"], c["variant"])

    if already_running(tag):
        print(f"{ts()} [GPU {gpu}] SKIP {cell}: asr_respin.sh is already running for {tag}")
        return None

    need_train = not training_done(expdir)
    need_decode = any(not decoding_done(expdir, t) for t in TEST_SETS)

    if not need_train and not need_decode:
        print(f"{ts()} [GPU {gpu}] SKIP {cell} (trained and scored)")
        return None
    if not need_train:
        print(f"{ts()} [GPU {gpu}] SKIP TRAIN {cell}, model exists -> decode only")
        launch_decode(gpu, cell)
        return None

    check_data(c["dataset"])
    logf = LOG_DIR / f"{cell.replace(':', '_')}.train.log"
    print(f"{ts()} [GPU {gpu}] LAUNCH TRAIN {cell}  expdir={expdir}  ({logf})")
    proc = run_bash_in_venv(train_cmd(gpu, cell))
    threading.Thread(target=pump_to_log, args=(proc, logf), daemon=True).start()

    def _wait_then_decode(p=proc, gpu=gpu, cell=cell, expdir=expdir):
        rc = p.wait()
        if rc != 0:
            print(f"{ts()} [GPU {gpu}] TRAIN FAILED (exit {rc}) {cell} "
                  f"-- SKIPPING decode ({expdir})")
            return
        if not training_done(expdir):
            print(f"{ts()} [GPU {gpu}] TRAIN produced no {INFER_MODEL} for {cell} "
                  "-- SKIPPING decode")
            return
        launch_decode(gpu, cell)

    threading.Thread(target=_wait_then_decode, daemon=True).start()
    return proc


# ============================================================
# GPU SCHEDULER -- one training per GPU slot
# ============================================================
def run_all(jobs: List[str], gpus: List[str]) -> None:
    slots = {g: None for g in gpus}
    idx = 0
    while True:
        for g in gpus:
            p = slots[g]
            if p is None or p.poll() is not None:
                if idx >= len(jobs):
                    continue
                cell = jobs[idx]
                idx += 1
                slots[g] = launch_job(g, cell)
        if idx >= len(jobs) and all(p is None or p.poll() is not None
                                    for p in slots.values()):
            break
        time.sleep(5)

    print(f"{ts()} all trainings finished; waiting for background decodes")
    while True:
        with DECODE_LOCK:
            alive = [p for p in ALL_DECODES if p.poll() is None]
        if not alive:
            break
        time.sleep(5)


# ============================================================
# RESULT TABLE
# ============================================================
def find_expdir(dataset: str, method: str, variant: str) -> Path:
    """The expdir for a table row, tolerating a different seed.

    Cells are launched at --seed, but a baseline may have been trained at
    another one (the worderror30 ctc run was restarted at 20262), and a table
    that silently showed "-" for it would look like missing work.  Prefer the
    current seed; otherwise take the most recently written seed that exists.
    """
    exact = expdir_for(dataset, method, variant)
    if exact.is_dir():
        return exact
    others = sorted(Path(expdir_root(dataset)).glob(
        f"asr_otcrecip_libri_en_{method}_{variant}_seed*"),
        key=lambda p: p.stat().st_mtime, reverse=True)
    return others[0] if others else exact


def wer(expdir: Path, test_set: str) -> str:
    res = expdir / INFER_TAG / test_set / "score_wer" / "result.txt"
    if res.is_file():
        for line in res.read_text().splitlines():
            if "Sum/Avg" in line:
                return line.replace("|", " ").split()[7]
    if already_running(expdir.name[len("asr_"):]):
        return "running"
    return "trained" if training_done(expdir) else "-"


def print_table() -> None:
    print(f"{ts()} WER%  ({INFER_TAG}, seed {SEED})")
    head = f"  {'data':<6} {'cell':<12} {'gran':<6} {'sched':<6} "
    print(head + " ".join(f"{t:<11}" for t in TEST_SETS) + "asr_tag")
    for ds, label, method, variant, gran, sched in table_rows():
        expdir = find_expdir(ds, method, variant)
        cols = " ".join(f"{wer(expdir, t):<11}" for t in TEST_SETS)
        print(f"  {ds:<6} {label:<12} {gran:<6} {sched:<6} {cols}{expdir.name[4:]}")


# ============================================================
# MAIN
# ============================================================
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(
        description="Granularity x schedule ablation (LibriSpeech 10h clean + worderror30).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("examples:\n"
                "  %(prog)s --dry-run\n"
                "  %(prog)s --gpus 0 1\n"
                "  %(prog)s --gpus 1 --cells we30:word_tokw we30:token_wordw\n"
                "  %(prog)s --table\n"))
    ap.add_argument("--gpus", nargs="+", default=GPUS, metavar="ID",
                    help=f"GPU ids; one training per id (default: {' '.join(GPUS)})")
    ap.add_argument("--cells", nargs="+", default=None, metavar="CELL",
                    choices=list(CELLS),
                    help="cells to run (default: the five ablation cells). The "
                         "worderror30 baselines we30:{ctc,word,token} are opt-in "
                         "here because run_ebf_otc_recip_master_libri_error.sh "
                         "trains them under the same asr_tags.")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"training seed, part of the asr_tag (default: {SEED})")
    ap.add_argument("--dry-run", action="store_true",
                    help="print each cell's config, overrides and expdir, then exit")
    ap.add_argument("--table", action="store_true",
                    help="print the result table (including the baselines trained "
                         "by the other scripts) and exit")
    return ap.parse_args()


def main():
    a = parse_args()
    global SEED
    SEED = a.seed
    if a.table:
        print_table()
        return
    cells = a.cells or DEFAULT_CELLS

    print(f"{ts()} TOTAL CELLS = {len(cells)}   TASK = {ASR_TASK}   "
          f"gpus = {' '.join(a.gpus)}   seed = {SEED}")
    for i, cell in enumerate(cells):
        c = CELLS[cell]
        expdir = expdir_for(c["dataset"], c["method"], c["variant"])
        state = ("scored" if all(decoding_done(expdir, t) for t in TEST_SETS)
                 else "trained" if training_done(expdir) else "to train")
        print(f"  [{i + 1:02d}] {cell:<20} {state:<9} expdir={expdir}")
        if a.dry_run:
            print(f"       config   = {c['config']}")
            print(f"       asr_args = --seed {SEED} {c['args'] or '(no overrides)'}")
    if a.dry_run:
        print(f"\n{ts()} dry run -- nothing launched")
        print(train_cmd(a.gpus[0], cells[0]))
        return

    run_all(cells, a.gpus)
    print(f"{ts()} ALL JOBS COMPLETE")
    print_table()


if __name__ == "__main__":
    main()
