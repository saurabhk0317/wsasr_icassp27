#!/usr/bin/env python3
#
# Part of ESPnet. Licensed under the Apache License, Version 2.0
# (see the LICENSE file in the ESPnet source root).
"""otc_granularity="dual" (mode otc_token_word) in espnet2/asr/ctc_otc_recip.py.

"dual" is the "token" graph plus one word-level bypass arc per word of >= 2
tokens, with its own penalty and schedule.  Checked here:

  * the transcript FSA is EXACTLY token-mode arcs + the expected word arcs;
  * one-token words get no word arc;
  * with the bypass switched off, dual == token;
  * token bypass / self-loop weights are unchanged by dual, and the word
    bypass weight follows its own epoch / entropy_reciprocal schedule;
  * the extra paths can only raise the likelihood, and a prohibitive word
    penalty gives back the token-mode loss;
  * the word-bypass keys are required for dual and rejected elsewhere;
  * the [OTCRECIP EPOCH] line gains word_bypass_w for dual only.

Run:  python -m pytest otc_recip/tests/test_dual_granularity.py -v
"""

import logging
import math

import k2
import pytest
import torch

from espnet2.asr.ctc_otc_recip import CTC, OtcGraphCompiler

ODIM = 12  # vocab_size + 1
ENC_DIM = 8
TOKEN_LIST = ["<blank>", "<unk>", "<space>"] + list("abcdefgh")
SP = TOKEN_LIST.index("<space>")
STAR = ODIM - 1
WORD_STARTS = frozenset({SP})

EPOCH_KW = dict(
    arc_schedule="epoch",
    initial_bypass_weight=-25.0,
    bypass_weight_decay=0.775,
    initial_self_loop_weight=-5.0,
    self_loop_weight_decay=0.775,
)
DUAL_KW = dict(initial_word_bypass_weight=-19.0, word_bypass_weight_decay=0.95)


def _arcs(fsa):
    """Transcript FSA as a sorted list of (src, dst, label, rounded weight)."""
    out = []
    for line in k2.to_str_simple(fsa).strip().split("\n"):
        p = line.split()
        if len(p) == 4:
            out.append((int(p[0]), int(p[1]), int(p[2]), round(float(p[3]), 4)))
    return sorted(out)


def _single(granularity, ids, bypass=True, self_loop=True, word_w=-19.0):
    comp = OtcGraphCompiler(
        vocab_size=ODIM, star_id=STAR, granularity=granularity,
        word_start_ids=WORD_STARTS,
    )
    return comp._build_single_transcript_fsa(
        ids, allow_bypass_arc=bypass, allow_self_loop_arc=self_loop,
        bypass_weight=-25.0, self_loop_weight=-5.0,
        word_bypass_weight=word_w if granularity == "dual" else None,
    )


# Words: [a b] [SP c d e] [SP f] [SP g]  -> spans (0,2) (2,6) (6,8) (8,10)
IDS = [3, 4, SP, 5, 6, 7, SP, 8, SP, 9]
# Leading / duplicated separators and one-token words
IDS_EDGE = [SP, 3, SP, SP, 4, 5, 6]  # words [SP a] [SP] [SP b c d]


@pytest.mark.parametrize("ids", [IDS, IDS_EDGE, [3], [3, 4, 5], [SP]])
def test_dual_is_token_graph_plus_word_arcs(ids):
    token = _arcs(_single("token", ids))
    dual = _arcs(_single("dual", ids, word_w=-19.0))
    comp = OtcGraphCompiler(
        vocab_size=ODIM, star_id=STAR, granularity="dual", word_start_ids=WORD_STARTS
    )
    expected_word, start = [], 0
    for word in comp._word_groups(ids):
        end = start + len(word)
        if len(word) >= 2:
            expected_word.append((start, end, STAR, -19.0))
        start = end
    assert dual == sorted(token + expected_word)


def test_one_token_words_get_no_word_arc():
    # [a] alone, and [SP] alone in IDS_EDGE, are one-token words.
    for ids in ([3], IDS_EDGE):
        extra = set(_arcs(_single("dual", ids))) - set(_arcs(_single("token", ids)))
        assert all(dst - src >= 2 for src, dst, _, _ in extra), extra
    assert len(set(_arcs(_single("dual", [3]))) - set(_arcs(_single("token", [3])))) == 0


def test_word_groups_match_word_granularity():
    word = OtcGraphCompiler(ODIM, STAR, granularity="word", word_start_ids=WORD_STARTS)
    dual = OtcGraphCompiler(ODIM, STAR, granularity="dual", word_start_ids=WORD_STARTS)
    for ids in (IDS, IDS_EDGE, [3, 4], [SP, SP]):
        assert dual._word_groups(ids) == word.group_into_words(ids)
        # dual POSITIONS are tokens
        assert dual.group_into_words(ids) == [[t] for t in ids]


@pytest.mark.parametrize("self_loop", [True, False])
def test_dual_without_bypass_equals_token(self_loop):
    for ids in (IDS, IDS_EDGE):
        assert _arcs(_single("dual", ids, bypass=False, self_loop=self_loop, word_w=None)) == \
            _arcs(_single("token", ids, bypass=False, self_loop=self_loop))


def test_compiler_rejects_dual_without_word_weight_or_boundaries():
    comp = OtcGraphCompiler(ODIM, STAR, granularity="dual", word_start_ids=WORD_STARTS)
    with pytest.raises(ValueError):
        comp._build_single_transcript_fsa(IDS, True, True, -25.0, -5.0, word_bypass_weight=None)
    with pytest.raises(ValueError):
        OtcGraphCompiler(ODIM, STAR, granularity="dual", word_start_ids=None)


# ---------------------------------------------------------------------------
# CTC module
# ---------------------------------------------------------------------------
def _ctc(granularity, **kw):
    torch.manual_seed(0)
    return CTC(odim=ODIM, encoder_output_size=ENC_DIM, token_list=TOKEN_LIST,
               otc_granularity=granularity, **kw)


def test_token_weights_unchanged_and_word_weight_schedule_epoch():
    tok = _ctc("token", **EPOCH_KW)
    dual = _ctc("dual", **EPOCH_KW, **DUAL_KW)
    for e in range(1, 8):
        tok.step_epoch(e)
        dual.step_epoch(e)
        assert dual.bypass_weight == tok.bypass_weight
        assert dual.self_loop_weight == tok.self_loop_weight
        assert math.isclose(dual.word_bypass_weight, -19.0 * 0.95 ** (e - 1))
        assert tok.word_bypass_weight == 0.0


def test_word_weight_schedule_entropy_reciprocal():
    kw = dict(arc_schedule="entropy_reciprocal", initial_bypass_weight=-25.0,
              initial_self_loop_weight=-5.0, bypass_end=-0.01, self_loop_end=-0.001,
              entropy_shape_k=1.3)
    dual = _ctc("dual", **kw, initial_word_bypass_weight=-19.0, word_bypass_end=-0.05)
    tok = _ctc("token", **kw)
    for p in (0.0, 0.3, 0.7, 1.0):
        dual._ent_p = tok._ent_p = p
        assert dual.bypass_weight == tok.bypass_weight
        assert math.isclose(
            dual.word_bypass_weight, CTC._geo_interp(-19.0, -0.05, p ** 1.3)
        )


def _batch():
    torch.manual_seed(1)
    B, T = 3, 30
    hs = torch.randn(B, T, ENC_DIM)
    hl = torch.tensor([30, 26, 20])
    refs = [IDS, IDS_EDGE, [3, 4, SP, 5]]
    U = max(map(len, refs))
    ys = torch.full((B, U), -1)
    for i, r in enumerate(refs):
        ys[i, : len(r)] = torch.tensor(r)
    yl = torch.tensor([len(r) for r in refs])
    return hs, hl, ys, yl


def _loss(ctc):
    ctc.eval()
    with torch.no_grad():
        return float(ctc(*_batch()))


def test_extra_word_arcs_only_add_likelihood():
    # A mild word penalty so the extra paths are visible above float32 noise
    # (at -19 they add ~exp(-19) of probability to a loss of ~40 nats).
    tok = _ctc("token", **EPOCH_KW)
    losses = []
    for w in (-8.0, -4.0, -2.0, 0.0):
        dual = _ctc("dual", **EPOCH_KW, initial_word_bypass_weight=w,
                    word_bypass_weight_decay=1.0)
        dual.load_state_dict(tok.state_dict())
        losses.append(_loss(dual))
    assert losses[0] < _loss(tok)
    # a cheaper word arc can only lower the loss further
    assert all(a > b for a, b in zip(losses, losses[1:])), losses


def test_prohibitive_word_weight_recovers_token_loss():
    tok = _ctc("token", **EPOCH_KW)
    dual = _ctc("dual", **EPOCH_KW, initial_word_bypass_weight=-1.0e4,
                word_bypass_weight_decay=1.0)
    dual.load_state_dict(tok.state_dict())
    assert math.isclose(_loss(dual), _loss(tok), rel_tol=0, abs_tol=1e-4)


def test_dual_trains_backward():
    dual = _ctc("dual", **EPOCH_KW, **DUAL_KW).train()
    hs, hl, ys, yl = _batch()
    hs.requires_grad_(True)
    loss = dual(hs, hl, ys, yl)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(hs.grad).all()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("granularity", ["token", "word"])
@pytest.mark.parametrize("key,val", [("initial_word_bypass_weight", -19.0),
                                     ("word_bypass_weight_decay", 0.95),
                                     ("word_bypass_end", -0.01)])
def test_word_keys_rejected_outside_dual(granularity, key, val):
    with pytest.raises(ValueError, match="only apply to otc_granularity='dual'"):
        _ctc(granularity, **EPOCH_KW, **{key: val})


@pytest.mark.parametrize("missing", ["initial_word_bypass_weight", "word_bypass_weight_decay"])
def test_dual_epoch_requires_word_keys(missing):
    kw = dict(DUAL_KW)
    kw.pop(missing)
    with pytest.raises(ValueError, match="needs"):
        _ctc("dual", **EPOCH_KW, **kw)


def test_dual_entropy_requires_word_end():
    with pytest.raises(ValueError, match="word_bypass_end"):
        _ctc("dual", arc_schedule="entropy_reciprocal", initial_word_bypass_weight=-19.0)


def test_dual_without_bypass_needs_no_word_keys():
    ctc = _ctc("dual", **EPOCH_KW, allow_bypass_arc=False)
    assert ctc.word_bypass_weight == 0.0
    tok = _ctc("token", **EPOCH_KW, allow_bypass_arc=False)
    ctc.load_state_dict(tok.state_dict())
    assert _loss(ctc) == _loss(tok)


def test_state_dict_keys_unchanged_by_dual():
    assert set(_ctc("dual", **EPOCH_KW, **DUAL_KW).state_dict()) == \
        set(_ctc("token", **EPOCH_KW).state_dict())


def test_epoch_line_field_only_for_dual(caplog):
    caplog.set_level(logging.INFO)
    for gran, kw, expect in (("token", {}, False), ("dual", DUAL_KW, True)):
        caplog.clear()
        ctc = _ctc(gran, **EPOCH_KW, **kw)
        _loss(ctc)
        ctc.step_epoch(2)
        line = [r.getMessage() for r in caplog.records if "[OTCRECIP EPOCH]" in r.getMessage()]
        assert line, gran
        assert ("word_bypass_w=" in line[-1]) == expect, line[-1]
