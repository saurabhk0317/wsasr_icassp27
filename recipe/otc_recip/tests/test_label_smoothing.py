#!/usr/bin/env python3
#
# Part of ESPnet. Licensed under the Apache License, Version 2.0
# (see the LICENSE file in the ESPnet source root).
"""CTC label smoothing (Table 3 control).

ESPnet's ``lsm_weight`` smooths the ATTENTION DECODER's cross-entropy; there is
no CTC label smoothing in ESPnet, so ``arc_schedule="label_smoothing"``
implements it:

    L         = (1 - eps) * L_CTC + eps * L_uniform
    L_uniform = - mean over VALID frames of [ (1/V) * sum_v log p_t(v) ]

with V = ``real_odim`` -- blank INCLUDED, the ``<star>`` column EXCLUDED.

These tests pin down each part of that formula independently, because a
reviewer will ask which variant we used.

Run:  python -m pytest otc_recip/tests/test_label_smoothing.py -v
"""

import sys

import pytest
import torch

from espnet2.asr.ctc_otc_recip import CTC

ODIM = 12  # vocab_size + 1
ENC_DIM = 8
SEED = 4242
TOKEN_LIST = ["<blank>", "<unk>", "<space>"] + list("abcdefgh")

EPS_SWEEP = [0.005, 0.01, 0.02, 0.05]  # the Table 3 sweep


def _batch():
    g = torch.Generator().manual_seed(SEED)
    hs = torch.randn(4, 30, ENC_DIM, generator=g)
    hlens = torch.tensor([30, 26, 21, 15], dtype=torch.long)
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


def _build(arc_schedule, **kw):
    torch.manual_seed(SEED)
    return CTC(
        odim=ODIM,
        encoder_output_size=ENC_DIM,
        token_list=TOKEN_LIST,
        dropout_rate=0.0,
        arc_schedule=arc_schedule,
        allow_bypass_arc=False,
        allow_self_loop_arc=False,
        **kw,
    )


@pytest.mark.parametrize("eps", EPS_SWEEP)
def test_matches_the_formula(eps):
    """L == (1 - eps) * L_CTC + eps * L_uniform, computed independently here."""
    plain = _build("none")
    smooth = _build("label_smoothing", ctc_label_smoothing=eps)
    # Same projection in both, so L_CTC is shared.
    smooth.load_state_dict(plain.state_dict())

    hs, hlens, ys, ylens = _batch()
    l_ctc = plain(hs, hlens, ys, ylens)
    l_total = smooth(hs, hlens, ys, ylens)

    # --- L_uniform, recomputed from scratch --------------------------------
    logits = smooth.ctc_lo(hs)
    log_probs = logits.log_softmax(dim=-1)  # (B, T, real_odim)
    assert log_probs.size(-1) == ODIM - 1, "the <star> column must be excluded"

    T = log_probs.size(1)
    valid = torch.arange(T).unsqueeze(0) < hlens.unsqueeze(1)  # (B, T)
    # -(1/V) * sum_v log p_t(v), averaged over valid frames only
    per_frame = -log_probs.mean(dim=-1)
    l_uniform = (per_frame * valid.float()).sum() / valid.float().sum()

    expected = (1.0 - eps) * l_ctc + eps * l_uniform
    torch.testing.assert_close(l_total, expected, rtol=0, atol=1e-6)


def test_padding_frames_are_excluded():
    """L_uniform averages over VALID frames; padding must not dilute it."""
    mod = _build("label_smoothing", ctc_label_smoothing=0.02)
    hs, hlens, _, _ = _batch()
    log_probs = mod.ctc_lo(hs).log_softmax(dim=-1)
    T = log_probs.size(1)
    valid = torch.arange(T).unsqueeze(0) < hlens.unsqueeze(1)

    got = mod._uniform_loss(log_probs, valid)

    # Same thing, computed by explicitly gathering only the valid frames.
    kept = torch.cat([log_probs[b, : int(hlens[b])] for b in range(len(hlens))])
    want = -kept.mean(dim=-1).mean()
    torch.testing.assert_close(got, want, rtol=0, atol=1e-6)

    # And it must differ from the naive all-frames average, or the test is vacuous.
    naive = -log_probs.mean(dim=-1).mean()
    assert not torch.allclose(got, naive, atol=1e-6)


def test_uniform_term_is_minimised_by_a_uniform_posterior():
    """L_uniform is the CE against uniform: minimal exactly at the uniform posterior."""
    mod = _build("label_smoothing", ctc_label_smoothing=0.01)
    V = ODIM - 1
    valid = torch.ones(1, 5, dtype=torch.bool)

    uniform = torch.full((1, 5, V), -torch.tensor(float(V)).log())
    peaked = torch.full((1, 5, V), -20.0)
    peaked[..., 3] = 0.0
    peaked = peaked.log_softmax(dim=-1)

    l_uniform = mod._uniform_loss(uniform, valid)
    l_peaked = mod._uniform_loss(peaked, valid)

    torch.testing.assert_close(
        l_uniform, torch.tensor(float(V)).log(), rtol=0, atol=1e-5
    )
    assert l_peaked > l_uniform


def test_eps_zero_is_exactly_plain_ctc():
    plain = _build("none")
    hs, hlens, ys, ylens = _batch()
    # eps must be > 0 under the label_smoothing schedule, so check the guard
    # and that "none" carries no smoothing term at all.
    assert plain._ctc_label_smoothing == 0.0
    with pytest.raises(ValueError, match="needs ctc_label_smoothing > 0"):
        _build("label_smoothing", ctc_label_smoothing=0.0)


def test_smoothing_rejected_outside_its_schedule():
    """Silently ignoring eps under another schedule would be a trap."""
    with pytest.raises(ValueError, match="arc_schedule"):
        _build("epoch", ctc_label_smoothing=0.02)
    with pytest.raises(ValueError, match="arc_schedule"):
        _build("entropy_reciprocal", ctc_label_smoothing=0.02)


def test_stars_are_off():
    mod = _build("label_smoothing", ctc_label_smoothing=0.02)
    assert mod.allow_bypass_arc is False
    assert mod.allow_self_loop_arc is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
