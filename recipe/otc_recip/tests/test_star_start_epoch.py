#!/usr/bin/env python3
#
# Part of ESPnet. Licensed under the Apache License, Version 2.0
# (see the LICENSE file in the ESPnet source root).
"""ctc_conf.star_start_epoch in espnet2/asr/ctc_otc_recip.py (delayed star arcs).

  * default 1 changes nothing: loss, gradient and weights match a run that never
    set the key, for every schedule and granularity, over several epochs;
  * before star_start_epoch the loss IS the plain-CTC ("none") loss;
  * from star_start_epoch on, "epoch" weights restart their clock, so epoch e
    of a delayed run equals epoch e - (S - 1) of an undelayed one;
  * entropy_reciprocal keeps its own clock;
  * the key is validated, survives a checkpoint round trip, and only delayed
    runs gain the stars_on field on the [OTCRECIP EPOCH] line.

Run:  python -m pytest otc_recip/tests/test_star_start_epoch.py -v
"""

import logging
import math

import pytest
import torch

from espnet2.asr.ctc_otc_recip import CTC

ODIM = 12
ENC_DIM = 8
TOKEN_LIST = ["<blank>", "<unk>", "<space>"] + list("abcdefgh")
SP = TOKEN_LIST.index("<space>")

EPOCH_KW = dict(arc_schedule="epoch", initial_bypass_weight=-8.0, bypass_weight_decay=0.8,
                initial_self_loop_weight=-2.0, self_loop_weight_decay=0.7)
DUAL_KW = dict(initial_word_bypass_weight=-4.0, word_bypass_weight_decay=0.9)
ENT_KW = dict(arc_schedule="entropy_reciprocal", initial_bypass_weight=-8.0,
              initial_self_loop_weight=-2.0)

CONFIGS = [
    ("token", EPOCH_KW), ("word", EPOCH_KW), ("dual", {**EPOCH_KW, **DUAL_KW}),
    ("token", ENT_KW), ("word", ENT_KW),
    ("dual", {**ENT_KW, "initial_word_bypass_weight": -4.0, "word_bypass_end": -0.05}),
]


def _ctc(gran, **kw):
    torch.manual_seed(0)
    return CTC(odim=ODIM, encoder_output_size=ENC_DIM, token_list=TOKEN_LIST,
               otc_granularity=gran, **kw)


def _batch():
    torch.manual_seed(1)
    hs = torch.randn(3, 30, ENC_DIM)
    hl = torch.tensor([30, 26, 20])
    refs = [[3, 4, SP, 5, 6, 7, SP, 8], [SP, 3, 4, 5], [9, SP, 10]]
    ys = torch.full((3, 8), -1)
    for i, r in enumerate(refs):
        ys[i, : len(r)] = torch.tensor(r)
    return hs, hl, ys, torch.tensor([len(r) for r in refs])


def _loss_grad(ctc):
    ctc.train()
    hs, hl, ys, yl = _batch()
    hs = hs.clone().requires_grad_(True)
    loss = ctc(hs, hl, ys, yl)
    loss.backward()
    return loss.item(), hs.grad.clone()


def _weights(ctc):
    return (ctc.bypass_weight, ctc.self_loop_weight, ctc.word_bypass_weight)


@pytest.mark.parametrize("gran,kw", CONFIGS)
def test_default_is_unchanged(gran, kw):
    a = _ctc(gran, **kw)                       # key not given
    b = _ctc(gran, star_start_epoch=1, **kw)   # key given as the default
    b.load_state_dict(a.state_dict())
    for e in range(1, 6):
        a.step_epoch(e); b.step_epoch(e)
        assert a.stars_on and b.stars_on
        assert _weights(a) == _weights(b)
        la, ga = _loss_grad(a); lb, gb = _loss_grad(b)
        assert la == lb and torch.equal(ga, gb)
    assert list(a.state_dict()) == list(b.state_dict())


@pytest.mark.parametrize("gran,kw", CONFIGS)
@pytest.mark.parametrize("start", [2, 3])
def test_star_free_epochs_are_plain_ctc(gran, kw, start):
    delayed = _ctc(gran, star_start_epoch=start, **kw)
    plain = _ctc(gran, arc_schedule="none")
    plain.load_state_dict(delayed.state_dict())
    for e in range(1, start):
        delayed.step_epoch(e); plain.step_epoch(e)
        assert not delayed.stars_on
        ld, gd = _loss_grad(delayed); lp, gp = _loss_grad(plain)
        assert ld == lp and torch.equal(gd, gp)
    delayed.step_epoch(start)
    assert delayed.stars_on
    assert _loss_grad(delayed)[0] < _loss_grad(plain)[0]   # arcs now add paths


@pytest.mark.parametrize("gran", ["token", "word", "dual"])
@pytest.mark.parametrize("start", [2, 3])
def test_epoch_clock_restarts_at_star_start(gran, start):
    kw = {**EPOCH_KW, **(DUAL_KW if gran == "dual" else {})}
    delayed = _ctc(gran, star_start_epoch=start, **kw)
    ref = _ctc(gran, **kw)
    ref.load_state_dict(delayed.state_dict())
    for e in range(1, start):                    # weights held at initial_*
        delayed.step_epoch(e)
        assert delayed.bypass_weight == kw["initial_bypass_weight"]
        assert delayed.self_loop_weight == kw["initial_self_loop_weight"]
    for e in range(start, start + 4):
        delayed.step_epoch(e)
        ref.step_epoch(e - start + 1)
        assert _weights(delayed) == _weights(ref)
        assert _loss_grad(delayed)[0] == _loss_grad(ref)[0]
    assert math.isclose(delayed.bypass_weight, -8.0 * 0.8 ** 3)


def test_entropy_schedule_keeps_its_own_clock():
    delayed = _ctc("token", star_start_epoch=2, **ENT_KW)
    ref = _ctc("token", **ENT_KW)
    for p in (0.0, 0.4, 1.0):
        delayed._ent_p = ref._ent_p = p
        for e in (1, 2, 5):
            delayed._cur_epoch = ref._cur_epoch = e
            assert _weights(delayed) == _weights(ref)


def test_star_share_not_measured_before_start():
    ctc = _ctc("token", star_start_epoch=2, **EPOCH_KW)
    ctc.step_epoch(1)
    ctc.eval()
    with torch.no_grad():
        ctc(*_batch())
    assert ctc._star_share_mass == 0.0 and ctc._n_frames_measured > 0
    ctc.step_epoch(2)
    with torch.no_grad():
        ctc(*_batch())
    assert ctc._star_share_mass > 0.0


def test_checkpoint_round_trip_keeps_the_switch():
    a = _ctc("word", star_start_epoch=2, **EPOCH_KW)
    a.step_epoch(2)
    b = _ctc("word", star_start_epoch=2, **EPOCH_KW)
    assert not b.stars_on
    b.load_state_dict(a.state_dict())
    assert b.stars_on and _weights(a) == _weights(b)


@pytest.mark.parametrize("kw", [
    dict(star_start_epoch=0, **EPOCH_KW),
    dict(star_start_epoch=2, arc_schedule="none"),
    dict(star_start_epoch=2, arc_schedule="label_smoothing", ctc_label_smoothing=0.01),
    dict(star_start_epoch=2, allow_bypass_arc=False, allow_self_loop_arc=False, **EPOCH_KW),
])
def test_invalid_settings_rejected(kw):
    with pytest.raises(ValueError, match="star_start_epoch"):
        _ctc("token", **kw)


def test_one_arc_enough_for_delay():
    ctc = _ctc("word", star_start_epoch=2, allow_bypass_arc=False, **EPOCH_KW)
    assert not ctc.stars_on


def test_epoch_line_field_only_when_delayed(caplog):
    caplog.set_level(logging.INFO)
    for start, expect in ((1, False), (2, True)):
        caplog.clear()
        ctc = _ctc("token", star_start_epoch=start, **EPOCH_KW)
        ctc.eval()
        with torch.no_grad():
            ctc(*_batch())
        ctc.step_epoch(2)
        lines = [r.getMessage() for r in caplog.records if "[OTCRECIP EPOCH]" in r.getMessage()]
        assert lines and (("stars_on=" in lines[-1]) == expect), lines
