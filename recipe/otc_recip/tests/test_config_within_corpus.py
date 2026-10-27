#!/usr/bin/env python3
#
# Part of ESPnet. Licensed under the Apache License, Version 2.0
# (see the LICENSE file in the ESPnet source root).
"""Within a corpus, the four methods may differ ONLY in the loss block.

Table 1 compares CTC / OTC-WORD / OTC-TOKEN / OTC-RECIP inside each corpus.
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
# only (30 epochs). The "_conformer" sets are the same four methods with a
# conformer encoder (--encoder conformer in the master scripts); each set has
# to be internally consistent on its own.
VARIANTS = [("respin", ""), ("libri", ""), ("fleurs", ""),
            ("respin", "_conformer"), ("libri", "_conformer"),
            ("fleurs", "_conformer")]

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
                         ids=[f"{c}{s or '_nonssl'}" for c, s in VARIANTS])
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
                         ids=[f"{c}{s or '_nonssl'}" for c, s in VARIANTS])
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
                         ids=[f"{c}{s or '_nonssl'}" for c, s in VARIANTS])
def test_ctc_only(corpus, suffix):
    """Every Table-1 cell is CTC-only: no attention decoder in the mix."""
    for method in METHODS:
        cfg = _load(corpus, method, suffix)
        assert cfg["decoder"] == "none", f"{corpus}{suffix}/{method}"
        assert cfg["model_conf"]["ctc_weight"] == 1.0, f"{corpus}{suffix}/{method}"


@pytest.mark.parametrize("corpus,suffix", VARIANTS,
                         ids=[f"{c}{s or '_nonssl'}" for c, s in VARIANTS])
def test_baseline_is_starless(corpus, suffix):
    cc = _load(corpus, "ctc", suffix)["ctc_conf"]
    assert cc["arc_schedule"] == "none"
    assert cc["allow_bypass_arc"] is False
    assert cc["allow_self_loop_arc"] is False


# Conformer configs are their ebf counterparts with only these changed.
CONFORMER_DIFF = {"encoder", "encoder_conf", "patience", "keep_nbest_models"}


@pytest.mark.parametrize("corpus", CORPORA)
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
# The cross-transfer decay sweep.
#
# conf/cross_transfer/ holds one otc_token base per (corpus, variant). The
# decay is NOT baked into these files: run_ebf_otc_recip_crosstransfer.sh
# overrides both decay keys per run via --ctc_conf, so there is no config per
# decay value any more.
#
# What must hold instead is that everything the sweep is NOT varying is
# identical across all six, so a shift in the optimal decay is attributable to
# the front-end and the epoch budget and nothing else.
# ---------------------------------------------------------------------------
XT_DIR = CONF / "cross_transfer"
XT_CORPORA = ["respin", "libri", "fleurs"]
XT_VARIANTS = ["nonssl", "ssl"]

# Held fixed across every sweep config.
XT_FIXED_CTC = {
    "initial_bypass_weight": -25.0,
    "initial_self_loop_weight": -5.0,
    "otc_granularity": "token",
    "arc_schedule": "epoch",
    "allow_bypass_arc": True,
    "allow_self_loop_arc": True,
}
XT_FIXED_TOP = {"batch_bins": 6000000, "accum_grad": 1}
XT_WARMUP = {"nonssl": 5000, "ssl": 3000}


def _xt(corpus, variant):
    p = XT_DIR / f"{corpus}_otc_token_{variant}.yaml"
    assert p.is_file(), f"missing {p}"
    return yaml.safe_load(p.read_text())


def test_every_sweep_config_exists():
    """The run script resolves conf/cross_transfer/<corpus>_otc_token_<variant>.yaml."""
    for corpus in XT_CORPORA:
        for variant in XT_VARIANTS:
            p = XT_DIR / f"{corpus}_otc_token_{variant}.yaml"
            assert p.is_file(), (
                f"missing {p} (needed by --corpora {corpus} --variants {variant})"
            )


def test_no_stale_per_decay_configs():
    """The old decay-in-the-filename configs must not linger.

    They encoded a single decay each and are superseded by the run-time
    override; leaving one behind invites a run against a stale setting.
    """
    stale = sorted(p.name for p in XT_DIR.glob("*decay*"))
    assert not stale, f"superseded per-decay configs still present: {stale}"


@pytest.mark.parametrize("corpus", XT_CORPORA)
@pytest.mark.parametrize("variant", XT_VARIANTS)
def test_sweep_config_holds_the_controlled_values(corpus, variant):
    cfg = _xt(corpus, variant)
    cc = cfg["ctc_conf"]
    for k, want in XT_FIXED_CTC.items():
        assert cc.get(k) == want, (
            f"{corpus}/{variant}: ctc_conf.{k}={cc.get(k)!r}, expected {want!r}. "
            f"The sweep varies only the decay; everything else must match across "
            f"all six configs or the comparison is confounded."
        )
    for k, want in XT_FIXED_TOP.items():
        assert cfg.get(k) == want, f"{corpus}/{variant}: {k}={cfg.get(k)!r}, expected {want!r}"
    assert cfg["scheduler_conf"]["warmup_steps"] == XT_WARMUP[variant], (
        f"{corpus}/{variant}: warmup_steps="
        f"{cfg['scheduler_conf']['warmup_steps']}, expected {XT_WARMUP[variant]}"
    )
    # the variant really is what the filename claims
    is_ssl = cfg.get("frontend") == "s3prl"
    assert is_ssl == (variant == "ssl"), (
        f"{corpus}/{variant}: frontend={cfg.get('frontend')!r} contradicts the "
        f"filename"
    )
    assert cfg["max_epoch"] == (30 if variant == "ssl" else 70), (
        f"{corpus}/{variant}: max_epoch={cfg['max_epoch']}"
    )


@pytest.mark.parametrize("corpus", XT_CORPORA)
def test_sweep_variants_differ_only_in_features(corpus):
    """Within a corpus, the two sweep configs differ only in feature-related keys."""
    allowed = {
        "encoder_conf.input_layer", "frontend", "preencoder", "freeze_param",
        "model_conf.extract_feats_in_collect_stats",
        "max_epoch", "keep_nbest_models", "scheduler_conf.warmup_steps",
        # Each file carries its variant's NATIVE decay so it is runnable on its
        # own; the sweep overrides both keys per run, so their value here does
        # not affect any swept cell.
        "ctc_conf.bypass_weight_decay", "ctc_conf.self_loop_weight_decay",
    }
    prefixes = ("frontend_conf.", "preencoder_conf.")
    a = _flat(_xt(corpus, "nonssl"))
    b = _flat(_xt(corpus, "ssl"))
    diffs = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    unexpected = {k for k in diffs if k not in allowed and not k.startswith(prefixes)}
    assert not unexpected, (
        f"{corpus}: the two sweep configs differ in keys that are not "
        f"feature-related: {sorted(unexpected)}"
    )


def test_sweep_configs_agree_across_corpora():
    """The loss block is identical in all six, so only the corpus differs."""
    ref = None
    for corpus in XT_CORPORA:
        for variant in XT_VARIANTS:
            cc = {k: v for k, v in _xt(corpus, variant)["ctc_conf"].items()
                  if k not in ("bypass_weight_decay", "self_loop_weight_decay")}
            if ref is None:
                ref, ref_name = cc, f"{corpus}/{variant}"
            else:
                assert cc == ref, (
                    f"{corpus}/{variant} ctc_conf differs from {ref_name} "
                    f"(ignoring the two swept decay keys): "
                    f"{ {k: (ref.get(k), cc.get(k)) for k in set(ref) | set(cc) if ref.get(k) != cc.get(k)} }"
                )


def test_all_libri_configs_share_batching():
    """Every FULL-DATA LibriSpeech run uses batch_bins 16000000 / accum_grad 4.

    Effective batch size is not part of the loss, so letting it vary between
    LibriSpeech runs would confound any comparison that spans them.

    conf/cross_transfer/ is deliberately excluded: those runs are on a 10h
    subset and use 6000000 / 1 across every corpus, which is checked by
    test_sweep_config_holds_the_controlled_values instead.
    """
    paths = sorted(
        list((CONF / "libri_final").glob("*.yaml"))
        + list((CONF / "label_smoothing").glob("libri_*.yaml"))
    )
    assert paths, "no LibriSpeech configs found"
    bad = []
    for p in paths:
        c = yaml.safe_load(p.read_text())
        if (c.get("batch_bins"), c.get("accum_grad")) != (16000000, 4):
            bad.append(f"  {p.name}: batch_bins={c.get('batch_bins')} "
                       f"accum_grad={c.get('accum_grad')}")
    assert not bad, "LibriSpeech configs not on 16000000 / 4:\n" + "\n".join(bad)


def test_label_smoothing_differs_only_by_the_eps_term():
    base = _load("libri", "ctc")
    for eps, fname in [
        (0.005, "libri_ls0p005.yaml"),
        (0.01, "libri_ls0p01.yaml"),
        (0.02, "libri_ls0p02.yaml"),
        (0.05, "libri_ls0p05.yaml"),
    ]:
        ls = yaml.safe_load((CONF / "label_smoothing" / fname).read_text())
        assert ls["ctc_conf"]["ctc_label_smoothing"] == eps, fname
        assert ls["ctc_conf"]["allow_bypass_arc"] is False, fname
        assert ls["ctc_conf"]["allow_self_loop_arc"] is False, fname

        a, b = _flat(base), _flat(ls)
        diffs = {
            k
            for k in set(a) | set(b)
            if a.get(k, "<absent>") != b.get(k, "<absent>")
        }
        assert diffs == {
            "ctc_conf.arc_schedule",
            "ctc_conf.ctc_label_smoothing",
        }, f"{fname} differs from libri_final/ctc.yaml in: {sorted(diffs)}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
