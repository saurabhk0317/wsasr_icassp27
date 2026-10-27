#!/usr/bin/env python3
#
# Part of ESPnet. Licensed under the Apache License, Version 2.0
# (see the LICENSE file in the ESPnet source root).
"""THIS TEST IS THE PAPER'S CLAIM.

The contribution is that the entropy-reciprocal schedule needs no per-setup
retuning: ONE parameter set drives RESPIN with filterbank features at 70
epochs, LibriSpeech (BPE-200, wav2vec2-base, 30 epochs) and FLEURS (char,
XLS-R, 30 epochs), with both the e_branchformer and the conformer encoder.

So the entire loss block of every ``otc_recip*.yaml`` must be byte-identical.
If someone ever "just tunes ``entropy_low`` a bit for FLEURS", the claim is
dead and this test says so loudly.

The comparison covers the WHOLE ``ctc_conf`` mapping, not only the
``entropy_*`` keys -- the initial arc weights, the endpoints, the granularity
and the k2 knobs all have to transfer as well, or the schedule is not really
parameter-free across setups.

The epoch-indexed baseline ``otc_token.yaml`` is pinned to its final values
too (-25 / -5, decay 0.775 in every corpus), so the comparison is against one
fixed otc_token setting rather than a per-corpus tuned one.

Run:  python -m pytest otc_recip/tests/test_config_entropy_keys.py -v
"""

import math
import pathlib
import sys

import pytest
import yaml

CONF = pathlib.Path(__file__).resolve().parents[2] / "conf"
CORPORA = ["respin", "libri", "fleurs"]

# Every otc_recip config in the study, as (corpus, filename suffix).
# RESPIN is filterbank-only; LibriSpeech and FLEURS are frozen-SSL only.  Each
# corpus ships an e_branchformer ("") and a conformer ("_conformer") config.
RECIP_CONFIGS = [("respin", ""), ("libri", ""), ("fleurs", ""),
                 ("respin", "_conformer"), ("libri", "_conformer"),
                 ("fleurs", "_conformer")]

# The parameter set claimed to transfer.  Written out literally so the test is
# a specification, not a tautology over whatever the files happen to contain.
EXPECTED = {
    "otc_granularity": "token",
    "allow_bypass_arc": True,
    "allow_self_loop_arc": True,
    "arc_schedule": "entropy_reciprocal",
    "entropy_mode": "soft",
    "entropy_source": "dev",
    "entropy_ema_alpha": 0.3,
    "entropy_low": 0.01,
    "entropy_shape_k": 1.3,
    "entropy_eps": 0.0001,
    "initial_bypass_weight": -25.0,
    "initial_self_loop_weight": -5.0,
    "bypass_end": -0.01,
    "self_loop_end": -0.001,
    "output_beam": 100.0,
    "use_double_scores": True,
}


def _ctc_conf(corpus, method="otc_recip", suffix=""):
    p = CONF / f"{corpus}_final" / f"{method}{suffix}.yaml"
    assert p.is_file(), f"missing config {p}"
    return yaml.safe_load(p.read_text())["ctc_conf"]


def test_otc_recip_loss_block_identical_everywhere():
    """One parameter set across corpora, feature types and epoch budgets."""
    ref_corpus, ref_suffix = RECIP_CONFIGS[0]
    ref_name = f"{ref_corpus}{ref_suffix}"
    ref = _ctc_conf(ref_corpus, suffix=ref_suffix)

    problems = []
    for corpus, suffix in RECIP_CONFIGS[1:]:
        name = f"{corpus}{suffix}"
        cur = _ctc_conf(corpus, suffix=suffix)
        for key in sorted(set(ref) | set(cur)):
            a, b = ref.get(key, "<absent>"), cur.get(key, "<absent>")
            if a != b:
                problems.append(f"  ctc_conf.{key}: {ref_name}={a!r}  {name}={b!r}")

    assert not problems, (
        "otc_recip's loss block is NOT identical everywhere. The paper's claim "
        "is that one parameter set transfers across every corpus, feature type "
        "and epoch budget; these keys break it:\n" + "\n".join(problems)
    )


@pytest.mark.parametrize("corpus,suffix", RECIP_CONFIGS,
                         ids=[f"{c}{s or '_ebf'}" for c, s in RECIP_CONFIGS])
def test_otc_recip_matches_the_specified_parameter_set(corpus, suffix):
    cur = _ctc_conf(corpus, suffix=suffix)
    assert cur == EXPECTED, (
        f"{corpus}_final/otc_recip{suffix}.yaml ctc_conf does not match the "
        f"parameter set the paper reports.\n"
        f"  extra   : {sorted(set(cur) - set(EXPECTED))}\n"
        f"  missing : {sorted(set(EXPECTED) - set(cur))}\n"
        f"  changed : "
        + str(
            {k: (EXPECTED[k], cur[k]) for k in set(EXPECTED) & set(cur) if EXPECTED[k] != cur[k]}
        )
    )


@pytest.mark.parametrize("corpus,suffix", RECIP_CONFIGS,
                         ids=[f"{c}{s or '_ebf'}" for c, s in RECIP_CONFIGS])
def test_no_dead_epoch_decay_keys(corpus, suffix):
    """``*_weight_decay`` is read only by arc_schedule="epoch".

    The source configs carried them under entropy_reciprocal as ignored
    leftovers -- with DIFFERENT values per corpus (0.9 vs 0.70), which made the
    file look setup-specific even though nothing read them.  They are gone, so
    that "one parameter set" is true of the whole block and not just of the
    keys that happen to be live.
    """
    cur = _ctc_conf(corpus, suffix=suffix)
    for dead in ("bypass_weight_decay", "self_loop_weight_decay"):
        assert dead not in cur, (
            f"{corpus}_final/otc_recip{suffix}.yaml still carries {dead}; it is never "
            f"read under entropy_reciprocal and differs between corpora, so it "
            f"undermines the transfer claim on sight."
        )


# Final otc_token loss block, identical in every corpus and encoder.
OTC_TOKEN_EXPECTED = {
    "otc_granularity": "token",
    "allow_bypass_arc": True,
    "allow_self_loop_arc": True,
    "arc_schedule": "epoch",
    "initial_bypass_weight": -25.0,
    "bypass_weight_decay": 0.775,
    "initial_self_loop_weight": -5.0,
    "self_loop_weight_decay": 0.775,
    "output_beam": 100.0,
    "use_double_scores": True,
}


@pytest.mark.parametrize("corpus,suffix", RECIP_CONFIGS,
                         ids=[f"{c}{s or '_ebf'}" for c, s in RECIP_CONFIGS])
def test_otc_token_matches_the_final_parameter_set(corpus, suffix):
    cur = _ctc_conf(corpus, "otc_token", suffix)
    assert cur == OTC_TOKEN_EXPECTED, (
        f"{corpus}_final/otc_token{suffix}.yaml ctc_conf: "
        + str({k: (OTC_TOKEN_EXPECTED.get(k), cur.get(k))
               for k in set(OTC_TOKEN_EXPECTED) | set(cur)
               if OTC_TOKEN_EXPECTED.get(k) != cur.get(k)})
    )


# Final otc_word_tuned / otc_token_word loss blocks, identical in every corpus
# and encoder.  Tuned on the LibriSpeech 10h SSL cross-transfer sweep; both
# start their star arcs at epoch 3.
OTC_WORD_TUNED_EXPECTED = {
    "otc_granularity": "word",
    "allow_bypass_arc": True,
    "allow_self_loop_arc": True,
    "arc_schedule": "epoch",
    "star_start_epoch": 3,
    "initial_bypass_weight": -19.0,
    "bypass_weight_decay": 0.925,
    "initial_self_loop_weight": -1.0,
    "self_loop_weight_decay": 0.975,
    "output_beam": 100.0,
    "use_double_scores": True,
}
OTC_TOKEN_WORD_EXPECTED = {
    "otc_granularity": "dual",
    "allow_bypass_arc": True,
    "allow_self_loop_arc": True,
    "arc_schedule": "epoch",
    "star_start_epoch": 3,
    "initial_bypass_weight": -25.0,
    "bypass_weight_decay": 0.775,
    "initial_word_bypass_weight": -19.0,
    "word_bypass_weight_decay": 0.925,
    "initial_self_loop_weight": -5.0,
    "self_loop_weight_decay": 0.775,
    "output_beam": 100.0,
    "use_double_scores": True,
}


# otc_token_word_recip: EXACTLY the otc_recip block (EXPECTED above) on the dual
# graph, plus the word bypass arc and the delayed star arcs.  Written out as a
# delta so it cannot drift from otc_recip unnoticed.
OTC_TOKEN_WORD_RECIP_EXPECTED = {
    **EXPECTED,
    "otc_granularity": "dual",
    "star_start_epoch": 3,
    "initial_word_bypass_weight": -19.0,
    "word_bypass_end": -1.0,
}


@pytest.mark.parametrize("method,expected", [
    ("otc_word_tuned", OTC_WORD_TUNED_EXPECTED),
    ("otc_token_word", OTC_TOKEN_WORD_EXPECTED),
    ("otc_token_word_recip", OTC_TOKEN_WORD_RECIP_EXPECTED),
])
@pytest.mark.parametrize("corpus,suffix", RECIP_CONFIGS,
                         ids=[f"{c}{s or '_ebf'}" for c, s in RECIP_CONFIGS])
def test_tuned_word_methods_match_the_final_parameter_set(corpus, suffix, method, expected):
    cur = _ctc_conf(corpus, method, suffix)
    assert cur == expected, (
        f"{corpus}_final/{method}{suffix}.yaml ctc_conf: "
        + str({k: (expected.get(k), cur.get(k))
               for k in set(expected) | set(cur)
               if expected.get(k) != cur.get(k)})
    )


@pytest.mark.parametrize("method", ["otc_word_tuned", "otc_token_word", "otc_token_word_recip"])
@pytest.mark.parametrize("corpus,suffix", RECIP_CONFIGS,
                         ids=[f"{c}{s or '_ebf'}" for c, s in RECIP_CONFIGS])
def test_tuned_word_configs_build(corpus, suffix, method):
    """The ctc_conf is accepted by the CTC class the task builds."""
    from espnet2.asr.ctc_otc_recip import CTC

    token_list = ["<blank>", "<unk>", "<space>", "\u2581a", "b", "c"]
    ctc = CTC(odim=len(token_list) + 1, encoder_output_size=4,
              token_list=token_list, **_ctc_conf(corpus, method, suffix))
    assert not ctc.stars_on          # epochs 1-2: plain CTC graph
    ctc.step_epoch(3)
    assert ctc.stars_on
    # First star epoch uses the initial weights (entropy progress is still 0 --
    # the entropy schedule's log-geometric form rounds -25 to -24.999999999999996,
    # hence the tolerance).
    cc = _ctc_conf(corpus, method, suffix)
    assert math.isclose(ctc.bypass_weight, cc["initial_bypass_weight"], rel_tol=1e-12)
    assert math.isclose(ctc.self_loop_weight, cc["initial_self_loop_weight"], rel_tol=1e-12)
    if "initial_word_bypass_weight" in cc:
        assert math.isclose(ctc.word_bypass_weight, cc["initial_word_bypass_weight"], rel_tol=1e-12)


def test_epoch_budgets_really_do_differ():
    """The transfer claim is only interesting because the budgets differ."""
    import yaml as _y

    epochs = {
        f"{c}{s}": _y.safe_load(
            (CONF / f"{c}_final" / f"otc_recip{s}.yaml").read_text()
        )["max_epoch"]
        for c, s in RECIP_CONFIGS
    }
    for s in ("", "_conformer"):
        assert epochs[f"respin{s}"] == 70, epochs
        assert epochs[f"libri{s}"] == 30, epochs
        assert epochs[f"fleurs{s}"] == 30, epochs


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
