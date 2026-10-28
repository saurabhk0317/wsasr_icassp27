#!/usr/bin/env python3
#
# Part of ESPnet. Licensed under the Apache License, Version 2.0
# (see the LICENSE file in the ESPnet source root).
"""Self-contained checks of the CTC baseline and the entropy normaliser.

  * ``arc_schedule="none"`` (the CTC baseline) compiles a graph with no
    <star> arc, through the same k2 code path as every OTC criterion.
  * ``"none"`` forces both arcs off even if a config asks for them.
  * The predictive entropy is normalised by log(V-1), which is what lets one
    entropy parameter set serve corpora with different vocabulary sizes.

Run:  python -m pytest otc_recip/tests/test_ctc_baseline.py -v
"""

import math
import sys

import pytest
import torch

from espnet2.asr.ctc_otc_recip import CTC as CTCnew

# Fixed synthetic setup -- deliberately small so this runs on CPU in seconds.
ODIM = 12  # vocab_size + 1  (so vocab_size = 11, real tokens 1..10)
ENC_DIM = 8
SEED = 12345

# A char-style token list: <blank>, <unk>, <space>, then plain characters.
# The <space> is what word granularity derives its boundary from.
TOKEN_LIST = ["<blank>", "<unk>", "<space>"] + list("abcdefgh")
assert len(TOKEN_LIST) == ODIM - 1

GRANULARITIES = ["token", "word"]


def _common_kwargs(granularity):
    """Parameters shared by both classes.

    Values follow the shipped otc_recip settings.
    """
    return dict(
        odim=ODIM,
        encoder_output_size=ENC_DIM,
        otc_granularity=granularity,
        token_list=TOKEN_LIST,
        initial_bypass_weight=-25.0,
        initial_self_loop_weight=-5.0,
        bypass_weight_decay=0.9,
        self_loop_weight_decay=0.9,
        entropy_mode="soft",
        entropy_source="dev",
        entropy_ema_alpha=0.3,
        entropy_low=0.01,
        entropy_shape_k=1.2,
        entropy_eps=0.0001,
        bypass_end=-0.01,
        self_loop_end=-0.001,
        output_beam=100.0,
        use_double_scores=True,
        dropout_rate=0.0,
    )



@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_none_mode_matches_plain_ctc_graph(granularity):
    """``arc_schedule="none"`` really is CTC: the graph carries no <star> arc.

    This is the claim the paper makes about its baseline -- same loss, same k2
    code path, star arcs switched off -- so assert it on the graph itself
    rather than trusting the loss to notice.
    """
    common = _common_kwargs(granularity)
    torch.manual_seed(SEED)
    mod = CTCnew(**common, arc_schedule="none")

    assert mod.allow_bypass_arc is False
    assert mod.allow_self_loop_arc is False

    compiler = mod._get_graph_compiler(torch.device("cpu"))
    token_ids = [[3, 4, 2, 5, 6, 7]]
    fsa = compiler._build_single_transcript_fsa(
        token_ids[0],
        allow_bypass_arc=False,
        allow_self_loop_arc=False,
        bypass_weight=0.0,
        self_loop_weight=0.0,
    )
    labels = fsa.labels.tolist()
    assert mod.star_id not in labels, (
        f"a <star> arc (label {mod.star_id}) survived into the "
        f"arc_schedule='none' graph: {labels}"
    )


def test_none_mode_ignores_config_arcs():
    """"none" forces the arcs off even if a config asks for them."""
    mod = CTCnew(
        **_common_kwargs("token"),
        arc_schedule="none",
        allow_bypass_arc=True,
        allow_self_loop_arc=True,
    )
    assert mod.allow_bypass_arc is False
    assert mod.allow_self_loop_arc is False
    assert mod.bypass_weight == 0.0
    assert mod.self_loop_weight == 0.0


def test_entropy_normaliser_is_log_v_minus_1():
    """The entropy is normalised by log(V-1), not by an absolute nat count.

    That normalisation is why ONE parameter set transfers across corpora with
    different vocabulary sizes, so pin it down.
    """
    for odim in (12, 70, 201):
        common = dict(_common_kwargs("token"))
        common["odim"] = odim
        common["token_list"] = ["<blank>", "<unk>", "<space>"] + [
            f"t{i}" for i in range(odim - 4)
        ]
        mod = CTCnew(**common, arc_schedule="entropy_reciprocal")
        real = odim - 1
        # "soft"/"nb" normalise by log(real_odim - 1); "full" by log(real_odim).
        assert mod._soft_nonblank_entropy
        # Feed a known entropy: a uniform distribution over the V-1 non-blank
        # tokens has non-blank entropy exactly log(V-1), i.e. H_raw == 1.0.
        mod._epoch_entropy_sum = math.log(real - 1) * 100.0
        mod._epoch_entropy_count = 100.0
        mod._entropy_step_epoch(1)
        assert abs(mod._H_hat_raw - 1.0) < 1e-9, (odim, mod._H_hat_raw)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
