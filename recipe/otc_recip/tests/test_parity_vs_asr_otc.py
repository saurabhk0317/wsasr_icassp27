#!/usr/bin/env python3
#
# Part of ESPnet. Licensed under the Apache License, Version 2.0
# (see the LICENSE file in the ESPnet source root).
"""Parity: asr_otc_recip must reproduce asr_otc exactly on the shared modes.

``espnet2/asr/ctc_otc_recip.py`` is a strip-down of ``ctc_otc_clean.py`` (the
asr_otc task).  Everything it kept must be BIT-FOR-BIT the same, otherwise the
already-completed asr_otc runs cannot be reported alongside new asr_otc_recip
runs in the same table.

Three shared configurations are checked, each at both graph granularities:

  ``none``               <-> old CTCotc with allow_bypass_arc=False and
                             allow_self_loop_arc=False.  The old task had no
                             "none" schedule; the CTC baseline was expressed by
                             switching the arcs off, which is exactly what the
                             new "none" mode does internally.
  ``epoch``              <-> old ``arc_schedule="epoch"``.
  ``entropy_reciprocal`` <-> old ``arc_schedule="entropy_reciprocal"``.

For each we assert equality of
  * the loss on a fixed synthetic batch, at several epochs;
  * ``bypass_weight`` and ``self_loop_weight`` at every epoch;
  * the entropy state (H_raw, H_ema, H_start, progress) that drives them.

Run:  python -m pytest otc_recip/tests/test_parity_vs_asr_otc.py -v
      (or just: python otc_recip/tests/test_parity_vs_asr_otc.py)
"""

import math
import sys

import pytest
import torch

from espnet2.asr.ctc_otc_clean import CTC as CTCold  # asr_otc (reference)
from espnet2.asr.ctc_otc_recip import CTC as CTCnew  # asr_otc_recip (new)

# Fixed synthetic setup -- deliberately small so this runs on CPU in seconds.
ODIM = 12  # vocab_size + 1  (so vocab_size = 11, real tokens 1..10)
ENC_DIM = 8
BATCH = 4
T_MAX = 30
N_EPOCHS = 6
SEED = 12345

# A char-style token list: <blank>, <unk>, <space>, then plain characters.
# The <space> is what word granularity derives its boundary from.
TOKEN_LIST = ["<blank>", "<unk>", "<space>"] + list("abcdefgh")
assert len(TOKEN_LIST) == ODIM - 1

# (arc_schedule, extra kwargs for the NEW class, extra kwargs for the OLD class)
SHARED_MODES = [
    (
        "none",
        dict(arc_schedule="none"),
        # The old task expressed "stars off" by disabling both arcs; the
        # schedule name was then irrelevant to the graph.
        dict(
            arc_schedule="epoch",
            allow_bypass_arc=False,
            allow_self_loop_arc=False,
        ),
    ),
    (
        "epoch",
        dict(arc_schedule="epoch"),
        dict(arc_schedule="epoch"),
    ),
    (
        "entropy_reciprocal",
        dict(arc_schedule="entropy_reciprocal"),
        dict(arc_schedule="entropy_reciprocal"),
    ),
]

GRANULARITIES = ["token", "word"]


def _fixed_batch():
    """A deterministic batch: encoder output, lengths, targets, target lengths."""
    g = torch.Generator().manual_seed(SEED)
    hs = torch.randn(BATCH, T_MAX, ENC_DIM, generator=g)
    hlens = torch.tensor([T_MAX, 26, 21, 15], dtype=torch.long)
    # Targets include the <space> id (2) so "word" granularity actually groups.
    ys = torch.tensor(
        [
            [3, 4, 2, 5, 6, 7],
            [5, 2, 6, 7, -1, -1],
            [3, 3, 4, -1, -1, -1],
            [8, 2, 9, 10, -1, -1],
        ],
        dtype=torch.long,
    )
    ylens = torch.tensor([6, 4, 3, 4], dtype=torch.long)
    return hs, hlens, ys, ylens


def _common_kwargs(granularity):
    """Parameters shared by both classes.

    Values are the shipped otc_recip settings, so the parity check exercises
    the configuration the paper actually runs.
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


def _build_pair(granularity, new_kw, old_kw):
    """Two modules with byte-identical ctc_lo parameters."""
    common = _common_kwargs(granularity)

    torch.manual_seed(SEED)
    new = CTCnew(**{**common, **new_kw})

    torch.manual_seed(SEED)
    old = CTCold(**{**common, **old_kw})

    # Belt and braces: force the projections to be literally the same tensors,
    # so any loss difference can only come from the graph or the schedule.
    old.load_state_dict(
        {k: v.clone() for k, v in new.state_dict().items() if k.startswith("ctc_lo")},
        strict=False,
    )
    torch.testing.assert_close(new.ctc_lo.weight, old.ctc_lo.weight)
    torch.testing.assert_close(new.ctc_lo.bias, old.ctc_lo.bias)
    return new, old


@pytest.mark.parametrize("granularity", GRANULARITIES)
@pytest.mark.parametrize(
    "name,new_kw,old_kw", SHARED_MODES, ids=[m[0] for m in SHARED_MODES]
)
def test_parity(granularity, name, new_kw, old_kw):
    new, old = _build_pair(granularity, new_kw, old_kw)
    hs, hlens, ys, ylens = _fixed_batch()

    starless = name == "none"

    for epoch in range(1, N_EPOCHS + 1):
        new.step_epoch(epoch)
        old.step_epoch(epoch)

        # --- arc weights -------------------------------------------------
        if starless:
            # The two classes disagree on what to REPORT for an arc that is
            # not in the graph, and only on that.  The old class leaves its
            # "epoch" schedule running and reports -25 * 0.9^(e-1); the new
            # one reports 0.0 because under arc_schedule="none" the arc does
            # not exist.  Neither number is ever read: the compiler is called
            # with allow_*_arc=False, so no <star> arc is emitted and the
            # weight is never consulted.  What must match is that both have
            # the arcs off -- and the loss, asserted below.
            assert new.allow_bypass_arc is False
            assert new.allow_self_loop_arc is False
            assert old.allow_bypass_arc is False
            assert old.allow_self_loop_arc is False
        else:
            assert new.bypass_weight == old.bypass_weight, (
                f"{name}/{granularity} epoch {epoch}: bypass_weight "
                f"{new.bypass_weight!r} != {old.bypass_weight!r}"
            )
            assert new.self_loop_weight == old.self_loop_weight, (
                f"{name}/{granularity} epoch {epoch}: self_loop_weight "
                f"{new.self_loop_weight!r} != {old.self_loop_weight!r}"
            )

        # --- entropy state ------------------------------------------------
        assert new._H_hat_raw == old._H_hat_raw
        assert new._H_hat_ema == old._H_hat_ema
        assert new._H_start == old._H_start
        assert new._ent_p == old._ent_p

        # --- loss, training pass ------------------------------------------
        new.train()
        old.train()
        loss_new = new(hs, hlens, ys, ylens)
        loss_old = old(hs, hlens, ys, ylens)
        assert torch.equal(loss_new, loss_old), (
            f"{name}/{granularity} epoch {epoch} (train): loss "
            f"{loss_new.item()!r} != {loss_old.item()!r}"
        )

        # --- loss, validation pass ----------------------------------------
        # entropy_source="dev" means THIS pass is what fills the accumulator
        # that the next step_epoch() consumes, so it must match too.
        new.eval()
        old.eval()
        with torch.no_grad():
            loss_new = new(hs, hlens, ys, ylens)
            loss_old = old(hs, hlens, ys, ylens)
        assert torch.equal(loss_new, loss_old), (
            f"{name}/{granularity} epoch {epoch} (valid): loss "
            f"{loss_new.item()!r} != {loss_old.item()!r}"
        )
        assert new._epoch_entropy_sum == old._epoch_entropy_sum
        assert new._epoch_entropy_count == old._epoch_entropy_count


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


def test_none_mode_equals_ctc_k2():
    """``arc_schedule="none"`` == ``ctc_k2``, to the last bit.

    The completed CTC baselines were run under the old task with
    ``ctc_variant: k2`` (``espnet2/asr/ctc_k2.py``), while the new task always
    goes through the OTC graph.  If those two disagreed, the old CTC numbers
    could not be reported in the same table as new OTC numbers.  They do not.
    """
    from espnet2.asr.ctc_k2 import CTC as CTCk2

    common = _common_kwargs("token")

    torch.manual_seed(SEED)
    otc = CTCnew(
        **common,
        arc_schedule="none",
        allow_bypass_arc=False,
        allow_self_loop_arc=False,
    )
    torch.manual_seed(SEED)
    # ctc_k2 has no <star> slot, so its odim is the real vocabulary size --
    # which is exactly the width of the OTC module's ctc_lo.
    k2ctc = CTCk2(
        odim=ODIM - 1,
        encoder_output_size=ENC_DIM,
        output_beam=100.0,
        use_double_scores=True,
        dropout_rate=0.0,
    )
    k2ctc.load_state_dict(
        {k: v.clone() for k, v in otc.state_dict().items() if k.startswith("ctc_lo")},
        strict=False,
    )

    hs, hlens, ys, ylens = _fixed_batch()
    loss_otc = otc(hs, hlens, ys, ylens)
    loss_k2 = k2ctc(hs, hlens, ys, ylens)
    assert torch.equal(loss_otc, loss_k2), (
        f"arc_schedule='none' loss {loss_otc.item()!r} != ctc_k2 loss "
        f"{loss_k2.item()!r}"
    )


def test_starless_modes_ignore_config_arcs():
    """"none"/"label_smoothing" force the arcs off even if a config asks for them."""
    common = _common_kwargs("token")
    for sched, extra in (
        ("none", {}),
        ("label_smoothing", {"ctc_label_smoothing": 0.01}),
    ):
        mod = CTCnew(
            **common,
            arc_schedule=sched,
            allow_bypass_arc=True,
            allow_self_loop_arc=True,
            **extra,
        )
        assert mod.allow_bypass_arc is False, sched
        assert mod.allow_self_loop_arc is False, sched
        assert mod.bypass_weight == 0.0, sched
        assert mod.self_loop_weight == 0.0, sched


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
