#!/usr/bin/env python3
#
# Part of ESPnet. Licensed under the Apache License, Version 2.0
# (see the LICENSE file in the ESPnet source root).
"""Within a corpus, the criteria may differ ONLY in the loss block.

The main WER table compares the seven criteria inside each corpus.
That comparison is only honest if the encoder, optimiser, LR schedule,
specaug, batching, seed and checkpoint-averaging policy are the same in all
four -- otherwise a WER difference could come from the training setup rather
than from the loss.

This test fails if ANYTHING outside ``ctc_conf`` differs between the four
configs of a corpus.  It also checks the parts of ``ctc_conf`` that are NOT
schedule parameters (the k2 decoding knobs), which must match too.

It exists because that invariant had already drifted in the source configs:
LibriSpeech ran CTC and OTC-WORD at ``accum_grad: 1`` while OTC-TOKEN and
OTC-RECIP ran at ``accum_grad: 4`` -- a 4x effective-batch difference inside
one column of the table.  All four LibriSpeech configs here use 4.

Run:  python -m pytest otc_recip/tests/test_config_within_corpus.py -v
"""

import pathlib
import sys

import pytest
import yaml

CONF = pathlib.Path(__file__).resolve().parents[2] / "conf"
CORPORA = ["respin", "libri", "fleurs"]
# otc_word_tuned / otc_token_word / otc_token_word_recip are the later
# additions (tuned OTC-word weights; dual token+word bypass on the epoch and on
# the entropy schedule), held to the same invariants.
METHODS = ["ctc", "otc_word", "otc_token", "otc_recip",
           "otc_word_tuned", "otc_token_word", "otc_token_word_recip"]

# (corpus, config-name suffix) pairs that form one self-contained comparison.
# RESPIN is filterbank-only (70 epochs); LibriSpeech and FLEURS are frozen-SSL
# only (30 epochs). The LibriSpeech "_conformer" set is the same seven methods
# with a conformer encoder (--encoder conformer in the LibriSpeech master
# script); each set has to be internally consistent on its own.
VARIANTS = [("respin", ""), ("libri", ""), ("fleurs", ""),
            ("libri", "_conformer")]

# ctc_conf keys that describe the LOSS and are therefore allowed to differ.
SCHEDULE_KEYS = {
    "arc_schedule",
    "otc_granularity",
    "allow_bypass_arc",
    "allow_self_loop_arc",
    "initial_bypass_weight",
    "initial_self_loop_weight",
    "bypass_weight_decay",
    "self_loop_weight_decay",
    "bypass_end",
    "self_loop_end",
    "entropy_mode",
    "entropy_source",
    "entropy_ema_alpha",
    "entropy_low",
    "entropy_shape_k",
    "entropy_eps",
    "ctc_label_smoothing",
    "word_boundary_symbol",
    # delayed star arcs, and the word bypass arc of otc_granularity="dual"
    "star_start_epoch",
    "initial_word_bypass_weight",
    "word_bypass_weight_decay",
    "word_bypass_end",
}


def _load(corpus, method, suffix=""):
    p = CONF / f"{corpus}_final" / f"{method}{suffix}.yaml"
    assert p.is_file(), f"missing config {p}"
    return yaml.safe_load(p.read_text())


def _flat(d, prefix=""):
    """Flatten nested mappings so a diff points at the exact key."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


@pytest.mark.parametrize("corpus,suffix", VARIANTS,
                         ids=[f"{c}{s or '_ebf'}" for c, s in VARIANTS])
def test_only_the_loss_block_differs(corpus, suffix):
    ref_method = METHODS[0]
    ref = _flat({k: v for k, v in _load(corpus, ref_method, suffix).items()
                 if k != "ctc_conf"})

    problems = []
    for method in METHODS[1:]:
        cur = _flat({k: v for k, v in _load(corpus, method, suffix).items()
                     if k != "ctc_conf"})
        for key in sorted(set(ref) | set(cur)):
            a, b = ref.get(key, "<absent>"), cur.get(key, "<absent>")
            if a != b:
                problems.append(f"  {key}: {ref_method}={a!r}  {method}={b!r}")

    assert not problems, (
        f"{corpus}{suffix}: these settings differ between methods but are OUTSIDE the "
        f"loss block, so a WER difference could not be attributed to the "
        f"loss:\n" + "\n".join(problems)
    )


@pytest.mark.parametrize("corpus,suffix", VARIANTS,
                         ids=[f"{c}{s or '_ebf'}" for c, s in VARIANTS])
def test_non_schedule_ctc_conf_keys_match(corpus, suffix):
    """k2 decoding knobs live in ctc_conf but are not part of the loss."""
    ref_method = METHODS[0]
    ref = _load(corpus, ref_method, suffix)["ctc_conf"]
    ref = {k: v for k, v in ref.items() if k not in SCHEDULE_KEYS}

    problems = []
    for method in METHODS[1:]:
        cur = _load(corpus, method, suffix)["ctc_conf"]
        cur = {k: v for k, v in cur.items() if k not in SCHEDULE_KEYS}
        for key in sorted(set(ref) | set(cur)):
            a, b = ref.get(key, "<absent>"), cur.get(key, "<absent>")
            if a != b:
                problems.append(f"  ctc_conf.{key}: {ref_method}={a!r}  {method}={b!r}")

    assert not problems, (
        f"{corpus}{suffix}: non-schedule ctc_conf keys differ between methods:\n"
        + "\n".join(problems)
    )


@pytest.mark.parametrize("corpus,suffix", VARIANTS,
                         ids=[f"{c}{s or '_ebf'}" for c, s in VARIANTS])
def test_ctc_only(corpus, suffix):
    """Every main-table cell is CTC-only: no attention decoder in the mix."""
    for method in METHODS:
        cfg = _load(corpus, method, suffix)
        assert cfg["decoder"] == "none", f"{corpus}{suffix}/{method}"
        assert cfg["model_conf"]["ctc_weight"] == 1.0, f"{corpus}{suffix}/{method}"


@pytest.mark.parametrize("corpus,suffix", VARIANTS,
                         ids=[f"{c}{s or '_ebf'}" for c, s in VARIANTS])
def test_baseline_is_starless(corpus, suffix):
    cc = _load(corpus, "ctc", suffix)["ctc_conf"]
    assert cc["arc_schedule"] == "none"
    assert cc["allow_bypass_arc"] is False
    assert cc["allow_self_loop_arc"] is False


# Conformer configs are their ebf counterparts with only these changed.
CONFORMER_DIFF = {"encoder", "encoder_conf", "patience", "keep_nbest_models"}


@pytest.mark.parametrize("corpus", ["libri"])
@pytest.mark.parametrize("method", METHODS)
def test_conformer_matches_ebf_counterpart(corpus, method):
    ebf = _load(corpus, method)
    cf = _load(corpus, method, "_conformer")
    assert ebf["encoder"] == "e_branchformer"
    assert cf["encoder"] == "conformer"
    assert cf["encoder_conf"]["num_blocks"] == 12, f"{corpus}/{method}"
    assert cf["encoder_conf"]["input_layer"] == ebf["encoder_conf"]["input_layer"]
    assert cf["patience"] == 5, f"{corpus}/{method}"
    assert cf["keep_nbest_models"] == 5, f"{corpus}/{method}"
    diff = {k for k in set(ebf) | set(cf) if ebf.get(k) != cf.get(k)}
    assert diff <= CONFORMER_DIFF, (
        f"{corpus}/{method}_conformer differs from the ebf config outside the "
        f"encoder / stopping keys: {sorted(diff - CONFORMER_DIFF)}"
    )


@pytest.mark.parametrize("corpus", CORPORA)
def test_libri_batching_regression(corpus):
    """The specific drift this test was written for: LibriSpeech accum_grad."""
    if corpus != "libri":
        pytest.skip("LibriSpeech only")
    for method in METHODS:
        cfg = _load(corpus, method)
        assert cfg["accum_grad"] == 4, f"libri/{method}: accum_grad={cfg['accum_grad']}"
        assert cfg["batch_bins"] == 16000000, f"libri/{method}"


# ---------------------------------------------------------------------------
# The schedule-sensitivity sweep (paper Fig. 2).
#
# conf/sweep/ holds one LibriSpeech SSL config per mode (ctc, otc_token,
# otc_recip).  The swept value is NOT baked into these files:
# run_ebf_otc_recip_sweep.sh overrides the decay (otc_token) or
# entropy_shape_k (otc_recip) per run via --ctc_conf.
#
# What must hold is that everything the sweep is NOT varying is identical
# across the three, so the curves differ only in the loss.
# ---------------------------------------------------------------------------
SW_DIR = CONF / "sweep"
SW_MODES = ["ctc", "otc_token", "otc_recip"]
SW_FIXED_TOP = {"batch_bins": 6000000, "accum_grad": 1, "max_epoch": 30,
                "frontend": "s3prl"}


def _sw(mode):
    p = SW_DIR / f"libri_{mode}_ssl.yaml"
    assert p.is_file(), f"missing {p}"
    return yaml.safe_load(p.read_text())


def test_sweep_dir_holds_only_the_fig2_configs():
    """The run script resolves conf/sweep/libri_<mode>_ssl.yaml and nothing else."""
    want = {f"libri_{m}_ssl.yaml" for m in SW_MODES}
    have = {p.name for p in SW_DIR.glob("*.yaml")}
    assert have == want, f"conf/sweep/ holds {sorted(have)}, expected {sorted(want)}"


@pytest.mark.parametrize("mode", SW_MODES)
def test_sweep_config_holds_the_controlled_values(mode):
    cfg = _sw(mode)
    for k, want in SW_FIXED_TOP.items():
        assert cfg.get(k) == want, f"{mode}: {k}={cfg.get(k)!r}, expected {want!r}"
    assert cfg["scheduler_conf"]["warmup_steps"] == 3000, mode


def test_sweep_token_and_recip_share_arc_start_weights():
    tok = _sw("otc_token")["ctc_conf"]
    rec = _sw("otc_recip")["ctc_conf"]
    for cc, name in ((tok, "otc_token"), (rec, "otc_recip")):
        assert cc["otc_granularity"] == "token", name
        assert cc["initial_bypass_weight"] == -25.0, name
        assert cc["initial_self_loop_weight"] == -5.0, name
    assert tok["arc_schedule"] == "epoch"
    assert rec["arc_schedule"] == "entropy_reciprocal"
    ctc = _sw("ctc")["ctc_conf"]
    assert ctc["allow_bypass_arc"] is False and ctc["allow_self_loop_arc"] is False


def test_sweep_configs_differ_only_in_the_loss():
    """Everything outside ctc_conf is identical across the three modes."""
    ref = {k: v for k, v in _flat(_sw("otc_token")).items()
           if not k.startswith("ctc_conf.")}
    for mode in ("ctc", "otc_recip"):
        cur = {k: v for k, v in _flat(_sw(mode)).items()
               if not k.startswith("ctc_conf.")}
        diffs = sorted(k for k in set(ref) | set(cur)
                       if ref.get(k, "<absent>") != cur.get(k, "<absent>"))
        assert not diffs, f"{mode} differs from otc_token outside ctc_conf: {diffs}"


def test_sweep_recip_entropy_block_matches_final():
    """Apart from the swept k, the entropy block equals the main-table one."""
    final = _load("libri", "otc_recip")["ctc_conf"]
    sweep = _sw("otc_recip")["ctc_conf"]
    keys = [k for k in final if k.startswith("entropy_") and k != "entropy_shape_k"]
    keys += ["bypass_end", "self_loop_end"]
    for k in keys:
        assert sweep.get(k) == final.get(k), f"{k}: sweep {sweep.get(k)!r} vs final {final.get(k)!r}"


def test_all_libri_configs_share_batching():
    """Every FULL-DATA LibriSpeech run uses batch_bins 16000000 / accum_grad 4.

    Effective batch size is not part of the loss, so letting it vary between
    LibriSpeech runs would confound any comparison that spans them.

    conf/sweep/ is deliberately excluded: those runs are on a 10h subset and
    use 6000000 / 1, which is checked by
    test_sweep_config_holds_the_controlled_values instead.
    """
    paths = sorted((CONF / "libri_final").glob("*.yaml"))
    assert paths, "no LibriSpeech configs found"
    bad = []
    for p in paths:
        c = yaml.safe_load(p.read_text())
        if (c.get("batch_bins"), c.get("accum_grad")) != (16000000, 4):
            bad.append(f"  {p.name}: batch_bins={c.get('batch_bins')} "
                       f"accum_grad={c.get('accum_grad')}")
    assert not bad, "LibriSpeech configs not on 16000000 / 4:\n" + "\n".join(bad)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
