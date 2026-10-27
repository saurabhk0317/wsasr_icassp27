# espnet2/asr/ctc_otc_recip.py
#
# Part of ESPnet. Licensed under the Apache License, Version 2.0
# (see the LICENSE file in the ESPnet source root).
"""
Omni-temporal Classification (OTC) with the entropy-reciprocal arc schedule.

This is the open-source build behind the ICASSP submission.  It is a
deliberate strip-down of ``ctc_otc_clean.py`` (the asr_otc task) exposing
EXACTLY FOUR arc-weight schedules and nothing else.

WHAT OTC IS
-----------
Standard CTC composes a ctc_topo with a linear transcript FSA, so every
training path must emit the reference token sequence exactly.  OTC (Gao et
al. 2023) adds two ``<star>`` wildcard arcs per reference position:

  * a BYPASS arc     (substitution) -- advance past the position without
                     emitting its token, at cost ``bypass_weight``;
  * a SELF-LOOP arc  (insertion)    -- stay at the position while consuming a
                     frame, at cost ``self_loop_weight``.

Both weights are negative (a penalty in the log domain) and are RELAXED over
training, so the model is free to ignore transcript errors early and is forced
back onto the reference as it becomes confident.  The wildcard emission is the
log-AVERAGE of the non-blank posteriors:

    log P(*|t) = logsumexp_{v != blank} log P(v|t) - log(V - 1)

The ``-log(V-1)`` tax is load-bearing: it keeps the emission at or below the
mean non-blank posterior, a provable lower bound, so the k2 lattice cannot
manufacture probability mass and the loss cannot go negative.  ``<star>`` is
appended as the last output slot, so ``odim = vocab_size + 1``; the extra
column is an ANALYTIC function of the other logits, never a learned parameter.

THE FOUR SCHEDULES (``arc_schedule``)
-------------------------------------
"none"
    Star arcs OFF (``allow_bypass_arc = allow_self_loop_arc = False``).  This
    is the CTC baseline, and it deliberately runs through the SAME k2 graph
    compiler, the same ctc_topo and the same ``k2.ctc_loss`` call as every OTC
    mode -- it is the identical loss with the wildcard arcs switched off, not
    a different CTC implementation.  The ``<star>`` column is still appended
    but is unreferenced by the transcript FSA, so it contributes nothing.

"epoch"
    The OTC paper's epoch-indexed geometric decay:

        w(e) = w_initial * decay^(e-1)

    Indexed on an EPOCH CLOCK, which is why ``decay`` has to be retuned
    whenever the epoch budget changes (this work: 0.9 at 70 epochs, 0.7 at 30).

    With ``star_start_epoch = S`` (see DELAYED STAR ARCS) the clock starts at
    the first epoch that has star arcs: ``w(e) = w_initial * decay^(e-S)``.

"entropy_reciprocal"
    The contribution.  Indexed on the MODEL'S OWN ENTROPY rather than on a
    clock, so it carries no epoch budget and needs no per-setup retuning.
    Once per epoch the non-blank conditional entropy is measured on held-out
    data (see below), normalised by ``log(V-1)``, EMA-smoothed to ``H_ema``,
    and turned into progress by the reciprocal map

        num = 1/(H_ema + eps) - 1/(H_start + eps)
        den = 1/(low    + eps) - 1/(H_start + eps)
        p   = clamp(num / den, 0, 1)
        w   = geo_interp(w_initial -> w_end, p ** shape_k)

    ``H_start`` is the anchor captured at the first measured epoch.  Each arc
    weight then glides log-geometrically from its initial penalty to its
    endpoint (``bypass_end`` / ``self_loop_end``).

    NOT MONOTONE.  ``H_ema`` tracks the EMA in both directions and progress is
    recomputed from it every epoch, so the arc weights TIGHTEN AGAIN if the
    entropy rises.  This is intentional; both the old H_eff ratchet and the old
    progress ratchet were removed because they made the schedule one-way.

    Because the entropy is normalised by ``log(V-1)``, ``H`` is a fraction of
    maximum entropy rather than an absolute nat count.  That is what lets ONE
    parameter set transfer across corpora whose vocabularies differ by a
    factor of three (RESPIN char ~70, FLEURS char ~135, LibriSpeech BPE 200).

"label_smoothing"
    Star arcs OFF, plus a uniform-distribution regulariser on the CTC
    posteriors.  See ``ctc_label_smoothing`` below.  Included as the control
    that answers "is OTC just doing label smoothing?".

DELAYED STAR ARCS (``star_start_epoch``)
----------------------------------------
Star arcs (bypass, self-loop, and "dual" word bypass) exist only from epoch
``star_start_epoch`` on; earlier epochs train -- and validate -- on the plain
CTC graph, exactly the ``arc_schedule="none"`` graph.  An untrained model puts
nearly all lattice mass on the wildcard paths, so switching the arcs on from
the first step can stop some runs from ever converging; one star-free epoch
lets the model learn the transcripts first.  DEFAULT 1 = arcs from the start,
bit-identical to the behaviour without this key.
  * "epoch": the decay clock starts at the first epoch WITH arcs, so that
    epoch uses exactly ``initial_*`` (see above).
  * "entropy_reciprocal": unchanged.  Entropy is still measured every epoch
    (the model is the same, only the loss graph differs), and the schedule
    anchors H_start at the start of epoch 2 as before -- so with
    ``star_start_epoch = 2`` the arcs also start from ``initial_*``.
  * Rejected (> 1) when no star arc can ever exist: the starless schedules, or
    both allow_* switches off.
  * The validation loss steps DOWN when the arcs come on (the extra paths add
    likelihood to the same model), so the star-free epochs are never the
    lowest-loss checkpoints.

THE ENTROPY SIGNAL (``entropy_mode``; entropy_reciprocal only)
--------------------------------------------------------------
"full"  full-distribution entropy over all ``real_odim`` tokens (blank
        included), uniformly averaged over ALL valid frames, normalised by
        ``log(real_odim)``.
"nb"    non-blank (V-1) renormalised entropy (drop blank, renormalise the
        rest), averaged uniformly over NON-BLANK frames only -- a frame counts
        only if its full-distribution argmax is not blank.  Normalised by
        ``log(real_odim - 1)``.
"soft"  (DEFAULT) non-blank (V-1) renormalised entropy, weighted over ALL
        valid frames by the non-blank posterior mass ``w_t = 1 - P(blank|t)``:
        ``H = sum_t w_t H_t / sum_t w_t``.  A thresholdless "nb".  Normalised
        by ``log(real_odim - 1)``.

WHICH SPLIT IT IS MEASURED ON (``entropy_source``)
--------------------------------------------------
"dev"   (DEFAULT) accumulate during the VALIDATION pass.  The trainer runs
        ``set_epoch(e) -> train(e) -> valid(e)``, so ``valid(e-1)`` fills the
        accumulator that ``set_epoch(e)`` consumes: the schedule is driven by
        HELD-OUT entropy, which cannot be lowered by memorising the (possibly
        erroneous) training transcripts.
"train" accumulate during the training pass.
Either way the measurement is ``no_grad`` and never enters the objective.

GRAPH GRANULARITY (``otc_granularity``)
---------------------------------------
"token" (DEFAULT) A bypass arc spans a SINGLE token and a self-loop sits at
        EVERY token position.  Substituting an n-token word costs n star arcs,
        and a star may land mid-word.
"word"  (icefall / Gao-2023 parity) A bypass arc spans a WHOLE WORD and
        self-loops sit only at word boundaries, so substituting a word costs
        exactly ONE star arc regardless of its length and stars cannot be
        inserted mid-word.
"dual"  (mode otc_token_word) The "token" graph -- a self-loop at EVERY token
        position and a bypass arc per token -- PLUS one word-level bypass arc
        spanning each word of two or more tokens.  A substitution can therefore
        be taken token by token (n * token bypass weight) or as one wildcard
        for the whole word (word bypass weight), whichever the lattice favours.
        The two bypass kinds carry SEPARATE penalties and schedules:
        ``initial_bypass_weight`` / ``bypass_weight_decay`` / ``bypass_end``
        for the token arcs, ``initial_word_bypass_weight`` /
        ``word_bypass_weight_decay`` / ``word_bypass_end`` for the word arcs.
        Self-loops use ``initial_self_loop_weight`` etc. exactly as in "token".
        A one-token word gets NO word arc: it would duplicate its token bypass
        arc (same states, same label) and count that path twice.

Word boundaries are derived from ``token_list`` (the task passes it in
automatically): an explicit ``word_boundary_symbol`` if given, else every
sentencepiece/BPE piece starting with "▁", else the char tokenizer's
``<space>`` id.  The separator LEADS its word, matching icefall's "▁word".

WEIGHTS ARE NOT ON THE SAME SCALE ACROSS GRANULARITIES.  In "token" mode the
bypass weight is charged per token, so with char tokenisation a 6-character
word costs 6 * initial_bypass_weight; in "word" mode it is charged once.
"dual" keeps both scales side by side, hence its separate word-arc penalty.  The
two must be tuned separately (icefall's word recipe uses -19 / +3.75).  For
the same reason our TOKEN-level penalties are NOT comparable to the
WORD-level penalties printed in Gao et al. 2023.

Reference:
    "Learning from Flawed Data: Weakly Supervised Automatic Speech
    Recognition", Gao et al., 2023.  https://arxiv.org/abs/2309.15796
    (icefall egs/librispeech/WSASR)
"""

import logging
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import k2

try:
    from typeguard import typechecked
except ImportError:

    def typechecked(fn):
        return fn


_NAN = float("nan")

ARC_SCHEDULES: Tuple[str, ...] = (
    "none",
    "epoch",
    "entropy_reciprocal",
    "label_smoothing",
)

# Schedules for which the <star> arcs are unconditionally disabled.
_STARLESS_SCHEDULES: Tuple[str, ...] = ("none", "label_smoothing")


def _fmt(v, prec: int = 6) -> str:
    """Fixed-width float formatting for the machine-parseable log line."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "nan"
    return "nan" if math.isnan(f) else f"{f:.{prec}f}"


# ---------------------------------------------------------------------------
# Graph compiler
# ---------------------------------------------------------------------------


class OtcGraphCompiler:
    """Builds OTC training graphs from lists of token-ID sequences.

    Per reference POSITION the transcript FSA carries:
      * the correct token arc(s) (weight 0),
      * a <star> SELF-LOOP arc (insertion   -- stay put, self_loop_weight),
      * a <star> BYPASS arc    (substitution -- advance, bypass_weight),
    plus a trailing <star> self-loop at the final token state.  Composed with
    ``k2.ctc_topo`` this is the OTC training graph.

    What a "position" is depends on ``granularity``:
      * "token" -- one position per token; the bypass arc spans one token.
      * "word"  -- one position per WORD; the bypass arc spans the whole word,
                   exactly as icefall's OtcTrainingGraphCompiler does.
      * "dual"  -- the "token" graph plus one extra bypass arc per word of two
                   or more tokens, weighted by ``word_bypass_weight``.

    With ``allow_bypass_arc = allow_self_loop_arc = False`` no <star> arc is
    emitted at all and the graph reduces to the standard CTC training graph --
    that is the ``arc_schedule="none"`` baseline, and it is byte-identical to
    what plain CTC would build.
    """

    def __init__(
        self,
        vocab_size: int,
        star_id: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        granularity: str = "token",
        word_start_ids=None,
    ) -> None:
        self.device = torch.device(device)
        self.star_id = star_id if star_id is not None else vocab_size - 1
        max_token_id = vocab_size - 1
        ctc_topo = k2.ctc_topo(max_token_id, modified=False)
        self.ctc_topo = ctc_topo.to(self.device)

        if granularity not in ("token", "word", "dual"):
            raise ValueError(
                f"granularity must be 'token', 'word' or 'dual', got {granularity!r}"
            )
        if granularity in ("word", "dual") and not word_start_ids:
            raise ValueError(
                f"granularity={granularity!r} requires a non-empty word_start_ids set"
            )
        self.granularity = granularity
        self.word_start_ids = frozenset(word_start_ids or ())

    def group_into_words(self, token_ids: List[int]) -> List[List[int]]:
        """Split a token-id sequence into the groups one <star> arc may span.

        "token": one group per token -- the graph built from this is
                 byte-identical to a per-token builder.
        "word" : a new group opens at every word-start token (sentencepiece
                 "▁..." piece, or the char tokenizer's <space>), so the
                 separator LEADS its word exactly as icefall's "▁" does:

                     [c1,c2,SP,c3,c4]  ->  [[c1,c2], [SP,c3,c4]]
                     [▁HE,LLO,▁WORLD]  ->  [[▁HE,LLO], [▁WORLD]]

                 A group can never come out empty: a word-start token arriving
                 while the current group is empty is appended to it instead of
                 opening a new one, which absorbs leading and duplicated
                 separators without producing a degenerate i->i bypass arc.
        """
        # "dual" positions are tokens, like "token"; its word arcs come from
        # _word_groups() separately.
        if self.granularity != "word" or not self.word_start_ids:
            return [[t] for t in token_ids]
        return self._word_groups(token_ids)

    def _word_groups(self, token_ids: List[int]) -> List[List[int]]:
        """The "word" grouping rule of group_into_words, for any granularity."""
        groups: List[List[int]] = []
        cur: List[int] = []
        for tok in token_ids:
            if cur and tok in self.word_start_ids:
                groups.append(cur)
                cur = []
            cur.append(tok)
        if cur:
            groups.append(cur)
        return groups

    def compile(
        self,
        token_ids_list: List[List[int]],
        allow_bypass_arc: bool = True,
        allow_self_loop_arc: bool = True,
        bypass_weight=0.0,
        self_loop_weight=0.0,
        word_bypass_weight=None,
    ) -> k2.Fsa:
        """``word_bypass_weight`` is read only by granularity="dual"."""
        transcript_fsa = self._build_transcript_fsa_vec(
            token_ids_list,
            allow_bypass_arc=allow_bypass_arc,
            allow_self_loop_arc=allow_self_loop_arc,
            bypass_weight=bypass_weight,
            self_loop_weight=self_loop_weight,
            word_bypass_weight=word_bypass_weight,
        )
        transcript_fsa = transcript_fsa.to(self.device)
        fsa_with_self_loops = k2.remove_epsilon_and_add_self_loops(transcript_fsa)
        fsa_with_self_loops = k2.arc_sort(fsa_with_self_loops)
        graph = k2.compose(
            self.ctc_topo,
            fsa_with_self_loops,
            treat_epsilons_specially=False,
        )
        assert graph.requires_grad is False
        return graph

    def _make_arc(self, from_state, to_state, label, weight) -> str:
        return f"{from_state} {to_state} {label} {weight}"

    def _build_single_transcript_fsa(
        self,
        token_ids: List[int],
        allow_bypass_arc: bool,
        allow_self_loop_arc: bool,
        bypass_weight,
        self_loop_weight,
        word_bypass_weight=None,
    ) -> k2.Fsa:
        arcs: List[str] = []
        star_id = self.star_id
        dual_word_arcs = self.granularity == "dual" and allow_bypass_arc
        if dual_word_arcs and word_bypass_weight is None:
            raise ValueError(
                "granularity='dual' with allow_bypass_arc needs word_bypass_weight"
            )

        def _w(weight, i):
            if isinstance(weight, (list, tuple)):
                return weight[i] if 0 <= i < len(weight) else weight[-1]
            return weight

        if len(token_ids) == 0:
            arcs.append(self._make_arc(0, 1, -1, 0.0))
            arcs.append("1")
            fsa = k2.Fsa.from_str("\n".join(arcs))
            return k2.arc_sort(fsa)

        num_tokens = len(token_ids)
        final_state = num_tokens + 1

        # One group per <star> position.  In "token" granularity every group
        # holds exactly one token, which reproduces the per-token graph arc for
        # arc; in "word" granularity a group is a whole word and the star arcs
        # span it (icefall parity).  State numbering is unaffected either way:
        # there is still one state per token, 0..num_tokens.
        # NOTE: _w() is indexed by GROUP index (== token index in "token" mode).
        groups = self.group_into_words(token_ids)

        cur_state = 0
        next_state = 1
        for g, group in enumerate(groups):
            bypass_from = cur_state
            # The single flat mean-star column, routed to BOTH the self-loop
            # and the bypass arc at this position.
            star_label = star_id
            if allow_self_loop_arc:
                arcs.append(
                    self._make_arc(
                        cur_state, cur_state, star_label, _w(self_loop_weight, g)
                    )
                )
            for tok in group:
                arcs.append(self._make_arc(cur_state, next_state, tok, 0.0))
                cur_state = next_state
                next_state += 1
            # Spans the WHOLE group: one token in "token" mode, one word in
            # "word" mode (icefall's bypass_cur_state -> bypass_next_state).
            if allow_bypass_arc:
                arcs.append(
                    self._make_arc(
                        bypass_from, cur_state, star_label, _w(bypass_weight, g)
                    )
                )

        assert cur_state == num_tokens, (cur_state, num_tokens)

        # "dual": one word-level bypass arc per word of >= 2 tokens, from the
        # state before the word's first token to the state after its last.  A
        # one-token word is skipped -- its arc would duplicate the token
        # bypass arc added above.  _w() is indexed by WORD index here.
        if dual_word_arcs:
            word_start = 0
            for wi, word in enumerate(self._word_groups(token_ids)):
                word_end = word_start + len(word)
                if len(word) >= 2:
                    arcs.append(
                        self._make_arc(
                            word_start,
                            word_end,
                            star_id,
                            _w(word_bypass_weight, wi),
                        )
                    )
                word_start = word_end
            assert word_start == num_tokens, (word_start, num_tokens)

        if allow_self_loop_arc:
            arcs.append(
                self._make_arc(
                    num_tokens,
                    num_tokens,
                    star_id,
                    _w(self_loop_weight, len(groups) - 1),
                )
            )

        arcs.append(self._make_arc(num_tokens, final_state, -1, 0.0))
        arcs.append(f"{final_state}")

        arc_lines = [a for a in arcs if len(a.split()) == 4]
        final_marker = [a for a in arcs if len(a.split()) == 1]
        sorted_arcs = (
            sorted(arc_lines, key=lambda a: int(a.split()[0])) + final_marker
        )

        fsa = k2.Fsa.from_str("\n".join(sorted_arcs))
        return k2.arc_sort(fsa)

    def _build_transcript_fsa_vec(
        self,
        token_ids_list,
        allow_bypass_arc,
        allow_self_loop_arc,
        bypass_weight,
        self_loop_weight,
        word_bypass_weight=None,
    ) -> k2.Fsa:
        def _per_utt(weight, b):
            return weight[b] if isinstance(weight, (list, tuple)) else weight

        fsa_list = [
            self._build_single_transcript_fsa(
                ids,
                allow_bypass_arc=allow_bypass_arc,
                allow_self_loop_arc=allow_self_loop_arc,
                bypass_weight=_per_utt(bypass_weight, b),
                self_loop_weight=_per_utt(self_loop_weight, b),
                word_bypass_weight=_per_utt(word_bypass_weight, b),
            )
            for b, ids in enumerate(token_ids_list)
        ]
        return k2.create_fsa_vec(fsa_list)


# ---------------------------------------------------------------------------
# OTC module
# ---------------------------------------------------------------------------


class CTC(nn.Module):
    """OTC CTC with four arc schedules: none | epoch | entropy_reciprocal |
    label_smoothing.

    Drop-in for ESPnet's CTC: same ``forward(hs_pad, hlens, ys_pad, ys_lens)``
    contract, same ``softmax`` / ``log_softmax`` / ``argmax`` / ``forced_align``
    helpers.  ``odim`` must be ``vocab_size + 1``; the extra slot is the
    analytic ``<star>`` column.

    Args:
        odim: output dimension INCLUDING the appended <star> slot, i.e.
            ``vocab_size + 1``.  The task builds this for you.
        encoder_output_size: encoder projection width.
        dropout_rate: dropout applied to the encoder output before ``ctc_lo``.
        blank_id: blank index (0 in every ESPnet token list).
        arc_schedule: one of ``ARC_SCHEDULES``.
        allow_bypass_arc / allow_self_loop_arc: per-arc switches.  Forced to
            False (with a warning if explicitly set True) under the starless
            schedules "none" and "label_smoothing".
        star_start_epoch: first epoch with star arcs (default 1 = from the
            start).  Earlier epochs use the plain CTC graph; see DELAYED STAR
            ARCS in the module docstring.
        otc_granularity: "token" | "word" | "dual".  "word" and "dual" need
            ``token_list``.
        token_list: vocabulary, used to derive word boundaries.  The task
            passes it automatically.
        word_boundary_symbol: override the automatic boundary derivation.
        initial_bypass_weight / initial_self_loop_weight: arc penalties at
            epoch 1 / progress 0.
        bypass_weight_decay / self_loop_weight_decay: geometric per-epoch decay
            factors, read ONLY by ``arc_schedule="epoch"``.
        initial_word_bypass_weight / word_bypass_weight_decay /
        word_bypass_end: the word-level bypass arc of ``otc_granularity="dual"``
            (its token bypass arcs keep the plain bypass_* keys).  Required
            there -- the initial weight always, the decay under "epoch", the
            end under "entropy_reciprocal" -- and REJECTED for any other
            granularity so a run can never carry a silently ignored penalty.
        entropy_mode / entropy_source / entropy_ema_alpha / entropy_low /
        entropy_shape_k / entropy_eps / bypass_end / self_loop_end: the
            entropy_reciprocal schedule; see the module docstring.
        ctc_label_smoothing: epsilon for ``arc_schedule="label_smoothing"``.
        reduce / zero_infinity / output_beam / use_double_scores: k2 loss
            parameters.
        log_epoch_line: emit the ``[OTCRECIP EPOCH]`` diagnostic (default on).
        ctc_type / ignore_nan_grad: accepted for ESPnet CTC signature
            compatibility and unused.
    """

    @typechecked
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        blank_id: int = 0,
        # ── Schedule selector ──────────────────────────────────────────────
        arc_schedule: str = "epoch",
        # ── Star arcs ──────────────────────────────────────────────────────
        allow_bypass_arc: bool = True,
        allow_self_loop_arc: bool = True,
        star_start_epoch: int = 1,  # first epoch WITH star arcs (1 = from start)
        # ── Graph granularity ──────────────────────────────────────────────
        otc_granularity: str = "token",
        token_list: Optional[List[str]] = None,
        word_boundary_symbol: Optional[str] = None,
        # ── Arc weights ────────────────────────────────────────────────────
        initial_bypass_weight: float = -25.0,
        initial_self_loop_weight: float = -5.0,
        # Epoch-schedule decays (arc_schedule="epoch" only)
        bypass_weight_decay: float = 0.900,
        self_loop_weight_decay: float = 0.900,
        # Word-level bypass arc (otc_granularity="dual" only; see docstring)
        initial_word_bypass_weight: Optional[float] = None,
        word_bypass_weight_decay: Optional[float] = None,
        word_bypass_end: Optional[float] = None,
        # ── entropy_reciprocal schedule ────────────────────────────────────
        entropy_mode: str = "soft",  # "full" | "nb" | "soft"
        entropy_source: str = "dev",  # "dev" (validation) | "train"
        entropy_ema_alpha: float = 0.3,
        entropy_low: float = 0.01,  # H_ema at which progress reaches 1.0
        entropy_shape_k: float = 1.2,  # exponent on progress (convexity)
        entropy_eps: float = 0.0001,  # epsilon in the 1/H reciprocal mapping
        bypass_end: float = -0.01,  # bypass weight at progress = 1
        self_loop_end: float = -0.001,  # self-loop weight at progress = 1
        # ── CTC label smoothing (arc_schedule="label_smoothing") ───────────
        ctc_label_smoothing: float = 0.0,
        # ── Diagnostics ────────────────────────────────────────────────────
        log_epoch_line: bool = True,
        # ── k2 loss params ─────────────────────────────────────────────────
        reduce: bool = True,
        zero_infinity: bool = True,
        output_beam: float = 100.0,
        use_double_scores: bool = True,
        # ── ESPnet CTC signature compat (accepted, unused) ─────────────────
        ctc_type: str = "builtin",
        ignore_nan_grad: Optional[bool] = None,
    ):
        super().__init__()

        if odim < 3:
            raise ValueError(
                f"odim must be >= 3 (blank + >=1 token + <star>), got {odim}"
            )

        self.odim = odim
        self.real_odim = odim - 1  # == vocab_size (the <star> slot is appended)
        self.blank_id = blank_id
        self.star_id = odim - 1
        self.dropout_rate = dropout_rate

        # ── Schedule ───────────────────────────────────────────────────────
        if arc_schedule not in ARC_SCHEDULES:
            raise ValueError(
                f"arc_schedule must be one of {ARC_SCHEDULES}, got {arc_schedule!r}"
            )
        self._arc_schedule = arc_schedule

        # ── Star arcs.  "none" and "label_smoothing" are starless by
        #    definition; disable the arcs rather than trusting the config.
        if arc_schedule in _STARLESS_SCHEDULES:
            if allow_bypass_arc or allow_self_loop_arc:
                logging.warning(
                    "OTC arc_schedule=%r is a starless mode; forcing "
                    "allow_bypass_arc=allow_self_loop_arc=False (config asked "
                    "for bypass=%s self_loop=%s).",
                    arc_schedule,
                    allow_bypass_arc,
                    allow_self_loop_arc,
                )
            allow_bypass_arc = False
            allow_self_loop_arc = False
        self.allow_bypass_arc = allow_bypass_arc
        self.allow_self_loop_arc = allow_self_loop_arc

        self.initial_bypass_weight = initial_bypass_weight
        self.initial_self_loop_weight = initial_self_loop_weight
        self.bypass_weight_decay = bypass_weight_decay
        self.self_loop_weight_decay = self_loop_weight_decay

        # ── Delayed star arcs ──────────────────────────────────────────────
        if star_start_epoch < 1:
            raise ValueError(
                f"star_start_epoch must be >= 1, got {star_start_epoch!r}"
            )
        if star_start_epoch > 1 and not (allow_bypass_arc or allow_self_loop_arc):
            raise ValueError(
                f"star_start_epoch={star_start_epoch} delays the star arcs, but "
                f"this run has none (arc_schedule={arc_schedule!r}, "
                f"allow_bypass_arc={allow_bypass_arc}, "
                f"allow_self_loop_arc={allow_self_loop_arc})."
            )
        self._star_start_epoch = int(star_start_epoch)

        # ── CTC label smoothing ────────────────────────────────────────────
        if not 0.0 <= ctc_label_smoothing < 1.0:
            raise ValueError(
                f"ctc_label_smoothing must be in [0, 1), got {ctc_label_smoothing!r}"
            )
        if arc_schedule == "label_smoothing" and ctc_label_smoothing <= 0.0:
            raise ValueError(
                "arc_schedule='label_smoothing' needs ctc_label_smoothing > 0"
            )
        if arc_schedule != "label_smoothing" and ctc_label_smoothing > 0.0:
            raise ValueError(
                f"ctc_label_smoothing={ctc_label_smoothing} was set but "
                f"arc_schedule={arc_schedule!r}; the smoothing term is only "
                f"applied under arc_schedule='label_smoothing'."
            )
        self._ctc_label_smoothing = float(ctc_label_smoothing)

        # ── Graph granularity ──────────────────────────────────────────────
        if otc_granularity not in ("token", "word", "dual"):
            raise ValueError(
                "otc_granularity must be 'token', 'word' or 'dual', got "
                f"{otc_granularity!r}"
            )
        self._otc_granularity = otc_granularity
        self._word_start_ids = self._derive_word_start_ids(
            token_list, word_boundary_symbol
        )
        if otc_granularity in ("word", "dual") and not self._word_start_ids:
            raise ValueError(
                f"otc_granularity={otc_granularity!r} needs word-boundary information, but "
                "none could be derived from token_list. The task passes "
                "token_list automatically; make sure it either uses "
                "sentencepiece pieces prefixed with '▁' or contains an "
                "explicit space symbol, or name the symbol via "
                "ctc_conf.word_boundary_symbol."
            )
        if otc_granularity in ("word", "dual"):
            logging.info(
                "OTC graph granularity=%r: %d word-start token id(s) detected.",
                otc_granularity,
                len(self._word_start_ids),
            )

        # ── Word-level bypass arc ("dual" only) ────────────────────────────
        word_keys = {
            "initial_word_bypass_weight": initial_word_bypass_weight,
            "word_bypass_weight_decay": word_bypass_weight_decay,
            "word_bypass_end": word_bypass_end,
        }
        if otc_granularity != "dual":
            given = sorted(k for k, v in word_keys.items() if v is not None)
            if given:
                raise ValueError(
                    f"{given} only apply to otc_granularity='dual', but "
                    f"otc_granularity={otc_granularity!r}"
                )
        elif self.allow_bypass_arc:
            need = ["initial_word_bypass_weight"]
            if arc_schedule == "epoch":
                need.append("word_bypass_weight_decay")
            elif arc_schedule == "entropy_reciprocal":
                need.append("word_bypass_end")
            missing = [k for k in need if word_keys[k] is None]
            if missing:
                raise ValueError(
                    f"otc_granularity='dual' with arc_schedule={arc_schedule!r} "
                    f"needs {missing}"
                )
        self.initial_word_bypass_weight = initial_word_bypass_weight
        self.word_bypass_weight_decay = word_bypass_weight_decay
        self._word_bypass_end = word_bypass_end

        # ── entropy_reciprocal state ───────────────────────────────────────
        if entropy_mode not in ("full", "nb", "soft"):
            raise ValueError(
                f"entropy_mode must be 'full', 'nb' or 'soft', got {entropy_mode!r}"
            )
        self._entropy_mode = entropy_mode
        self._full_entropy = entropy_mode == "full"
        self._skip_blank_frames_entropy = entropy_mode == "nb"
        self._soft_nonblank_entropy = entropy_mode == "soft"

        if entropy_source not in ("dev", "train"):
            raise ValueError(
                f"entropy_source must be 'dev' or 'train', got {entropy_source!r}"
            )
        self._entropy_source = entropy_source
        # True -> accumulate in eval() (the validation pass); False -> train().
        self._entropy_on_dev = entropy_source == "dev"

        if not 0.0 < entropy_ema_alpha <= 1.0:
            raise ValueError(
                f"entropy_ema_alpha must be in (0, 1], got {entropy_ema_alpha!r}"
            )
        if not entropy_low > 0.0:
            raise ValueError(f"entropy_low must be > 0, got {entropy_low!r}")
        if not entropy_shape_k > 0.0:
            raise ValueError(f"entropy_shape_k must be > 0, got {entropy_shape_k!r}")
        if not entropy_eps >= 0.0:
            raise ValueError(f"entropy_eps must be >= 0, got {entropy_eps!r}")

        self._entropy_ema_alpha = entropy_ema_alpha
        self._entropy_low = entropy_low
        self._entropy_shape_k = entropy_shape_k
        self._entropy_eps = entropy_eps
        self._bypass_end = bypass_end
        self._self_loop_end = self_loop_end

        self._entropy_driven = arc_schedule == "entropy_reciprocal"

        self._H_hat_raw: float = 1.0  # latest raw normalised entropy
        self._H_hat_ema: float = 1.0  # EMA-smoothed (cold start = max entropy)
        self._H_start: Optional[float] = None  # anchor at first measured epoch
        self._ent_p: float = 0.0  # progress in [0, 1] (NOT ratcheted)
        self._epoch_entropy_sum: float = 0.0
        # Frame count (hard modes: integer-valued) or frame-weight mass
        # ("soft": fractional) -- kept as float uniformly.
        self._epoch_entropy_count: float = 0.0

        # ── k2 loss params ─────────────────────────────────────────────────
        self.reduce = reduce
        self.zero_infinity = zero_infinity
        self.output_beam = output_beam
        self.use_double_scores = use_double_scores

        self._cur_epoch: int = 1

        self.ctc_lo = nn.Linear(encoder_output_size, self.real_odim)
        self._graph_compiler: Optional[OtcGraphCompiler] = None

        # ── [OTCRECIP EPOCH] diagnostic accumulators ───────────────────────
        self._log_epoch_line = bool(log_epoch_line)
        self._seen_forward = False
        self._star_share_sum: float = 0.0
        self._star_share_mass: float = 0.0
        self._n_frames_measured: float = 0.0

        logging.info(
            "[OTCRECIP CONFIG] arc_schedule=%s granularity=%s bypass=%d "
            "self_loop=%d initial_bypass_weight=%s initial_self_loop_weight=%s "
            "bypass_weight_decay=%s self_loop_weight_decay=%s bypass_end=%s "
            "self_loop_end=%s entropy_mode=%s entropy_source=%s "
            "entropy_ema_alpha=%s entropy_low=%s entropy_shape_k=%s "
            "entropy_eps=%s ctc_label_smoothing=%s vocab_size=%d odim=%d",
            arc_schedule,
            otc_granularity,
            int(self.allow_bypass_arc),
            int(self.allow_self_loop_arc),
            initial_bypass_weight,
            initial_self_loop_weight,
            bypass_weight_decay,
            self_loop_weight_decay,
            bypass_end,
            self_loop_end,
            entropy_mode,
            entropy_source,
            entropy_ema_alpha,
            entropy_low,
            entropy_shape_k,
            entropy_eps,
            ctc_label_smoothing,
            self.real_odim,
            self.odim,
        )
        if self._star_start_epoch != 1:
            logging.info(
                "[OTCRECIP CONFIG] star_start_epoch=%d (epochs before it train on "
                "the plain CTC graph)",
                self._star_start_epoch,
            )
        if otc_granularity == "dual":
            logging.info(
                "[OTCRECIP CONFIG] dual word-bypass initial_word_bypass_weight=%s "
                "word_bypass_weight_decay=%s word_bypass_end=%s",
                initial_word_bypass_weight,
                word_bypass_weight_decay,
                word_bypass_end,
            )

    # ------------------------------------------------------------------
    # Word-boundary derivation (graph granularity)
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_word_start_ids(token_list, word_boundary_symbol):
        """Token ids that START a new word, or None when undeterminable.

        Priority:
          1. explicit ``word_boundary_symbol`` (e.g. a non-default space);
          2. sentencepiece/BPE -- every piece whose string starts with "▁"
             (exactly icefall's word boundary);
          3. ESPnet char tokenizer -- the single ``<space>`` id.

        Config only; deliberately NOT part of ``_PERSIST_ATTRS``/state_dict.
        """
        if not token_list:
            return None
        token_list = list(token_list)

        if word_boundary_symbol is not None:
            if word_boundary_symbol not in token_list:
                raise ValueError(
                    f"word_boundary_symbol={word_boundary_symbol!r} is not in "
                    f"token_list"
                )
            return frozenset({token_list.index(word_boundary_symbol)})

        bpe_ids = frozenset(
            i
            for i, t in enumerate(token_list)
            if isinstance(t, str) and t.startswith("▁")
        )
        if bpe_ids:
            return bpe_ids

        if "<space>" in token_list:
            return frozenset({token_list.index("<space>")})

        return None

    # ------------------------------------------------------------------
    # Checkpoint persistence of the schedule state
    # ------------------------------------------------------------------
    # Path-dependent scalar state (progress, entropy anchor + EMA,
    # accumulators) held as plain Python attrs -- absent from state_dict() by
    # default -- would reset on resume and snap the arc weights back to
    # initial_*.  Persist them under the "sched." prefix.
    _PERSIST_ATTRS = (
        "_cur_epoch",
        "_ent_p",
        "_H_start",
        "_H_hat_raw",
        "_H_hat_ema",
        "_epoch_entropy_sum",
        "_epoch_entropy_count",
    )
    _INT_ATTRS = ("_cur_epoch",)
    _NONE_ATTRS = ("_H_start",)  # legitimately None before the anchor epoch

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        for name in self._PERSIST_ATTRS:
            val = getattr(self, name)
            destination[prefix + "sched." + name] = torch.tensor(
                float("nan") if val is None else float(val), dtype=torch.float64
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name in self._PERSIST_ATTRS:
            key = prefix + "sched." + name
            if key in state_dict:  # absent in foreign ckpts -> keep defaults
                v = state_dict.pop(key)  # pop so super() won't flag "unexpected"
                v = float(v.item() if hasattr(v, "item") else v)
                if name in self._NONE_ATTRS:
                    setattr(self, name, None if math.isnan(v) else v)
                elif name in self._INT_ATTRS:
                    setattr(self, name, int(v))
                else:
                    setattr(self, name, v)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    # ------------------------------------------------------------------
    # Epoch stepping
    # ------------------------------------------------------------------

    def step_epoch(self, epoch: int) -> None:
        """Advance the epoch counter and update arc weights.

        Called by the trainer at the START of each epoch, via the model's
        ``set_epoch``.  The entropy accumulator filled during the window that
        just closed is consumed here, so the arc weights emitted on the
        ``[OTCRECIP EPOCH]`` line are the ones in force for ``epoch``.
        """
        self._cur_epoch = epoch
        if self._entropy_driven:
            self._entropy_step_epoch(epoch)
        if self._star_start_epoch != 1 and epoch == self._star_start_epoch:
            logging.info(
                "OTC star arcs ON from epoch %d (epochs 1-%d used the plain CTC graph)",
                epoch,
                epoch - 1,
            )
        self._emit_epoch_line(epoch)
        self._reset_epoch_stats()

    def _entropy_progress(self, H: float) -> float:
        """Reciprocal-entropy progress p in [0, 1], anchored at H_start.

        Clamped to [0, 1] per call, but NOT ratcheted across epochs: p falls
        again if H rises, so the arc weights can tighten as well as relax.
        """
        H0 = self._H_start
        low = self._entropy_low
        eps = self._entropy_eps
        num = 1.0 / (H + eps) - 1.0 / (H0 + eps)
        den = 1.0 / (low + eps) - 1.0 / (H0 + eps)
        p = num / den if abs(den) > 1e-12 else 0.0
        return max(0.0, min(1.0, p))

    def _entropy_step_epoch(self, epoch: int) -> None:
        """Finalise the epoch entropy accumulator and update H_ema / progress."""
        if self._epoch_entropy_count <= 0.0:
            logging.info(
                "OTC entropy_reciprocal epoch %d: no data accumulated, "
                "H_ema=%.4f held  bypass_w=%.4f  self_loop_w=%.4f",
                epoch,
                self._H_hat_ema,
                self.bypass_weight,
                self.self_loop_weight,
            )
            return

        # Normaliser: log(real_odim) for the full-distribution signal ("full"),
        # else log(V-1) for the non-blank signal ("nb"/"soft").  Normalising is
        # what makes ONE parameter set transfer across vocabulary sizes.
        vocab = self.real_odim if self._full_entropy else self.real_odim - 1
        H_epoch = self._epoch_entropy_sum / self._epoch_entropy_count
        H_hat_raw = max(0.0, min(1.0, H_epoch / math.log(vocab)))
        self._H_hat_raw = H_hat_raw

        # EMA (cold start: the first real observation replaces the init of 1.0)
        if self._H_hat_ema == 1.0 and epoch <= 2:
            self._H_hat_ema = H_hat_raw
        else:
            self._H_hat_ema = (
                self._entropy_ema_alpha * H_hat_raw
                + (1.0 - self._entropy_ema_alpha) * self._H_hat_ema
            )

        # Capture the high anchor once, at the first measured epoch.
        if self._H_start is None:
            self._H_start = self._H_hat_ema

        self._ent_p = self._entropy_progress(self._H_hat_ema)

        self._epoch_entropy_sum = 0.0
        self._epoch_entropy_count = 0.0

    # ------------------------------------------------------------------
    # Arc weight properties
    # ------------------------------------------------------------------

    @property
    def stars_on(self) -> bool:
        """True from epoch ``star_start_epoch`` on (always, by default)."""
        return self._cur_epoch >= self._star_start_epoch

    def _epoch_decay_steps(self) -> int:
        """Exponent of the "epoch" schedule: epochs since the arcs came on.

        ``_cur_epoch - 1`` by default.  Clamped at 0 so the weights reported
        before the arcs exist are the initial ones.
        """
        return max(0, self._cur_epoch - self._star_start_epoch)

    @staticmethod
    def _geo_interp(w_start: float, w_end: float, p: float) -> float:
        """Geometric (log-magnitude) interpolation w_start (p=0) -> w_end (p=1)."""
        if w_start == 0.0:
            return w_end * p
        sign = -1.0 if w_start < 0 else 1.0
        a = math.log(abs(w_start))
        b = math.log(abs(w_end))
        return sign * math.exp((1.0 - p) * a + p * b)

    @property
    def bypass_weight(self) -> float:
        if self._arc_schedule == "epoch":
            return self.initial_bypass_weight * (
                self.bypass_weight_decay ** self._epoch_decay_steps()
            )
        if self._arc_schedule == "entropy_reciprocal":
            p = self._ent_p ** self._entropy_shape_k
            return self._geo_interp(self.initial_bypass_weight, self._bypass_end, p)
        return 0.0  # starless modes: the arc does not exist

    @property
    def word_bypass_weight(self) -> float:
        """Word-level bypass weight of granularity "dual" (0.0 when unused)."""
        if self._otc_granularity != "dual" or not self.allow_bypass_arc:
            return 0.0
        if self._arc_schedule == "epoch":
            return self.initial_word_bypass_weight * (
                self.word_bypass_weight_decay ** self._epoch_decay_steps()
            )
        if self._arc_schedule == "entropy_reciprocal":
            p = self._ent_p ** self._entropy_shape_k
            return self._geo_interp(
                self.initial_word_bypass_weight, self._word_bypass_end, p
            )
        return 0.0  # starless modes: the arc does not exist

    @property
    def self_loop_weight(self) -> float:
        if self._arc_schedule == "epoch":
            return self.initial_self_loop_weight * (
                self.self_loop_weight_decay ** self._epoch_decay_steps()
            )
        if self._arc_schedule == "entropy_reciprocal":
            p = self._ent_p ** self._entropy_shape_k
            return self._geo_interp(
                self.initial_self_loop_weight, self._self_loop_end, p
            )
        return 0.0  # starless modes: the arc does not exist

    # ------------------------------------------------------------------
    # Graph compiler -- lazy init
    # ------------------------------------------------------------------

    def _get_graph_compiler(self, device: torch.device) -> OtcGraphCompiler:
        if self._graph_compiler is None or self._graph_compiler.device != device:
            self._graph_compiler = OtcGraphCompiler(
                vocab_size=self.odim,
                star_id=self.star_id,
                device=device,
                granularity=self._otc_granularity,
                word_start_ids=self._word_start_ids,
            )
        return self._graph_compiler

    # ------------------------------------------------------------------
    # <star> log-prob helpers
    # ------------------------------------------------------------------

    def _flat_star_logprob(self, log_probs_base: torch.Tensor) -> torch.Tensor:
        """Flat <star> column (B, T, 1): the log-average of the non-blank mass.

            log P(*|t) = logsumexp_{v != blank} log P(v|t) - log(V-1)

        (blank is at index 0).  The ``-log(V-1)`` tax keeps the emission a
        provable lower bound so the lattice cannot manufacture mass.
        """
        V_minus_1 = self.real_odim - 1
        return torch.logsumexp(
            log_probs_base[:, :, 1:], dim=-1, keepdim=True
        ) - math.log(V_minus_1)

    def _append_star_logprob(self, log_probs: torch.Tensor) -> torch.Tensor:
        """(B, T, real_odim) -> (B, T, odim); appends the flat <star> column."""
        return torch.cat([log_probs, self._flat_star_logprob(log_probs)], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_lens: torch.Tensor,
    ) -> torch.Tensor:
        ys_hat = self.ctc_lo(
            F.dropout(hs_pad, p=self.dropout_rate, training=self.training)
        )
        loss = self.loss_fn(ys_hat, ys_pad, hlens, ys_lens)
        return loss.to(device=hs_pad.device, dtype=hs_pad.dtype)

    # ------------------------------------------------------------------
    # Label smoothing
    # ------------------------------------------------------------------

    def _uniform_loss(
        self, log_probs_base: torch.Tensor, frame_valid: torch.Tensor
    ) -> torch.Tensor:
        """The uniform-distribution term of CTC label smoothing.

            L_uniform = - mean_{valid t} [ (1/V) * sum_v log p_t(v) ]

        V is ``real_odim``: the blank IS included and the ``<star>`` column is
        EXCLUDED (it is an analytic function of the others, not a vocabulary
        entry, so including it would double-count).  This is the
        cross-entropy of the model's posterior against the uniform
        distribution over the real vocabulary, averaged over valid frames only
        -- padding frames contribute nothing.

        NOTE on which variant this is: ESPnet's ``lsm_weight`` smooths the
        ATTENTION DECODER's per-token cross-entropy.  There is no CTC label
        smoothing in ESPnet, so this term is ours.  The total objective is

            L = (1 - eps) * L_CTC + eps * L_uniform

        where ``L_CTC`` is the batch-mean per-utterance NLL and ``L_uniform``
        is a per-FRAME mean.  The two are deliberately on different scales
        (that is the formula as specified); ``eps`` therefore has to be small,
        which is why the sweep runs over {0.005, 0.01, 0.02, 0.05}.
        """
        # -(1/V) * sum_v log p_t(v), per frame
        per_frame = -log_probs_base.mean(dim=-1)  # (B, T)
        w = frame_valid.to(per_frame.dtype)
        denom = w.sum().clamp(min=1.0)
        return (per_frame * w).sum() / denom

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _ctc_occupancy(self, log_probs_s, graphs, supervision_segments):
        """CTC posterior occupancy gamma(t, v) = d log P / d log_probs[t, v].

        One DETACHED re-forward: the input is a fresh leaf, so the returned
        gradient never reaches the real objective, no parameter grad is
        touched, and DDP's reducer sees nothing.  ``torch.enable_grad`` is
        required because the trainer's validation pass runs under
        ``@torch.no_grad()``.

        gamma is non-negative and sums to 1 over v at every valid frame, which
        is exactly the arc occupancy the star-share statistic needs -- most of
        the rectangle is off-path, so uniform averaging would measure noise.
        """
        with torch.enable_grad():
            lp = log_probs_s.detach().clone().requires_grad_(True)
            dense = k2.DenseFsaVec(lp, supervision_segments=supervision_segments)
            loss = k2.ctc_loss(
                decoding_graph=graphs,
                dense_fsa_vec=dense,
                output_beam=self.output_beam,
                reduction="none",
                use_double_scores=self.use_double_scores,
            )
            ok = torch.isfinite(loss)
            if not bool(ok.any()):
                return None
            total_logp = (-loss[ok]).sum()
            (g,) = torch.autograd.grad(total_logp, lp, allow_unused=False)
        g = torch.nan_to_num(g.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        return g.clamp_(min=0.0)

    @torch.no_grad()
    def _accum_star_share(
        self,
        log_probs_base_s: torch.Tensor,  # (B, T, real_odim)
        log_probs_s: torch.Tensor,  # (B, T, odim)  incl. <star>
        flat_star: torch.Tensor,  # (B, T, 1)
        graphs,
        supervision_segments: torch.Tensor,
        th_ilen_s: torch.Tensor,
        frame_valid: torch.Tensor,  # (B, T) bool
    ) -> None:
        """Accumulate ``star_share``: how much of the reference occupancy the
        <star> bypass arc would take.  (Under "dual" this uses the TOKEN bypass
        weight; the word arcs are not reflected in it.)

            Delta_t = log p_t(reference token) - log P(*|t)
            share_t = sigmoid(bypass_weight - Delta_t)

        occupancy-weighted over frames.  A share near 0 means the star arc is
        no longer competitive (the model has learned the reference); near 1
        means the star is winning and the transcript is being bypassed.  Both
        terms come from the SAME log-softmax that builds the <star> column, so
        nothing needs renormalising and no log(V-1) constant appears.

        VALIDATION ONLY, and only when a star arc actually exists -- it costs
        one extra detached k2 forward + backward per batch.
        """
        gamma = self._ctc_occupancy(log_probs_s, graphs, supervision_segments)
        if gamma is None:
            return

        base = log_probs_base_s.detach().float()
        star_t = flat_star.detach().squeeze(-1).float()  # (B, T)

        # Non-blank REAL tokens only (index 0 is blank, the last column is
        # <star>): the star arc substitutes tokens, never blank.
        lab_w = gamma[..., 1 : self.real_odim] * frame_valid.unsqueeze(-1).float()
        m_label = lab_w.sum(-1)  # (B, T)
        ref_logp = (lab_w * base[..., 1 : self.real_odim]).sum(-1) / m_label.clamp(
            min=1e-12
        )

        delta = ref_logp - star_t
        w = torch.where(m_label > 0, m_label, torch.zeros_like(m_label))
        finite = torch.isfinite(delta) & torch.isfinite(w)
        w = torch.where(finite, w, torch.zeros_like(w)).double()
        d = torch.where(finite, delta, torch.zeros_like(delta)).double()

        mass = float(w.sum().item())
        if mass <= 0.0:
            return
        lam = float(self.bypass_weight)
        self._star_share_sum += float((w * torch.sigmoid(lam - d)).sum().item())
        self._star_share_mass += mass

    def _reset_epoch_stats(self) -> None:
        self._star_share_sum = 0.0
        self._star_share_mass = 0.0
        self._n_frames_measured = 0.0

    def _emit_epoch_line(self, epoch: int) -> None:
        """One machine-parseable line per epoch, fixed key order.

        Consumed by ``scripts/plot_arc_weights.py``.  Emitted from
        ``step_epoch`` (i.e. at the START of ``epoch``), so ``bypass_w`` /
        ``self_loop_w`` are the weights that will be IN FORCE for ``epoch``
        while ``H_*`` / ``star_share`` are what was measured in the window that
        just closed.
        """
        if not (self._log_epoch_line and self._seen_forward):
            return
        share = (
            self._star_share_sum / self._star_share_mass
            if self._star_share_mass > 0
            else _NAN
        )
        ent = self._entropy_driven
        fields = (
            ("epoch", str(int(epoch))),
            ("arc_schedule", self._arc_schedule),
            ("granularity", self._otc_granularity),
            ("bypass_w", _fmt(self.bypass_weight)),
            ("self_loop_w", _fmt(self.self_loop_weight)),
            ("H_raw", _fmt(self._H_hat_raw if ent else _NAN)),
            ("H_ema", _fmt(self._H_hat_ema if ent else _NAN)),
            (
                "H_start",
                _fmt(self._H_start if (ent and self._H_start is not None) else _NAN),
            ),
            ("progress", _fmt(self._ent_p if ent else _NAN)),
            ("star_share", _fmt(share)),
            ("n_frames_measured", str(int(self._n_frames_measured))),
        )
        # Appended only for "dual" / a delayed start, so every other run's line
        # is unchanged.
        if self._otc_granularity == "dual":
            fields = fields + (("word_bypass_w", _fmt(self.word_bypass_weight)),)
        if self._star_start_epoch != 1:
            fields = fields + (("stars_on", str(int(self.stars_on))),)
        logging.info(
            "[OTCRECIP EPOCH] %s", " ".join(f"{k}={v}" for k, v in fields)
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss_fn(
        self,
        th_pred: torch.Tensor,
        th_target: torch.Tensor,
        th_ilen: torch.Tensor,
        th_olen: torch.Tensor,
    ) -> torch.Tensor:
        device = th_pred.device
        B = th_pred.size(0)
        self._seen_forward = True

        # 1. log-softmax over the REAL vocabulary
        log_probs_base = th_pred.log_softmax(dim=-1).float()

        # 2. Sort by descending encoder length (k2 CUDA requirement)
        th_ilen_s, order = torch.sort(th_ilen, descending=True)
        inv_order = torch.argsort(order)
        log_probs_base_s = log_probs_base[order]
        th_target_s = th_target[order]
        th_olen_s = th_olen[order]

        # 3. Token lists (sorted order)
        token_ids_list = [
            [t for t in th_target_s[i, : int(th_olen_s[i])].tolist() if t >= 0]
            for i in range(B)
        ]

        # 3b. <star> emission column at index real_odim, shared by the
        #     self-loop and bypass arcs.  Grad-carrying by design.
        flat_star = self._flat_star_logprob(log_probs_base_s)  # (B, T, 1)
        log_probs_s = torch.cat([log_probs_base_s, flat_star], dim=-1)

        # 3c. supervision segments
        supervision_segments = torch.stack(
            [
                torch.tensor([i, 0, int(th_ilen_s[i])], dtype=torch.int32)
                for i in range(B)
            ]
        )
        dense = k2.DenseFsaVec(log_probs_s, supervision_segments=supervision_segments)

        T_max = log_probs_base_s.size(1)
        frame_valid = torch.arange(T_max, device=device).unsqueeze(
            0
        ) < th_ilen_s.to(device).unsqueeze(1)  # (B, T)

        # Which split carries the entropy measurement.
        measure_entropy = (
            (not self.training) if self._entropy_on_dev else self.training
        )
        # The star-share diagnostic is validation-only regardless: it costs an
        # extra k2 forward + backward, which is unaffordable per training batch.
        measure_diag = not self.training

        # 4. Entropy accumulation (entropy_reciprocal only).
        if self._entropy_driven and measure_entropy:
            with torch.no_grad():
                if self._full_entropy:
                    # full-distribution entropy over all real_odim (incl. blank)
                    p = log_probs_base_s.exp()
                    H_frame = (
                        -(p * log_probs_base_s).masked_fill(p == 0, 0.0).sum(dim=-1)
                    )
                    w_frame = frame_valid.float()  # all valid frames
                else:
                    # per-frame non-blank (V-1) renormalised entropy
                    lp = log_probs_base_s.clone()
                    lp[..., self.blank_id] = float("-inf")
                    log_q = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
                    q = log_q.exp()
                    H_frame = -(q * log_q).masked_fill(q == 0, 0.0).sum(dim=-1)
                    if self._soft_nonblank_entropy:
                        # soft weight = non-blank posterior mass
                        w_frame = (
                            1.0 - log_probs_base_s[..., self.blank_id].exp()
                        ).clamp(min=0.0) * frame_valid.float()
                    else:  # "nb": keep frames whose FULL-dist argmax is non-blank
                        argmax_full = log_probs_base_s.argmax(dim=-1)
                        keep = frame_valid & (argmax_full != self.blank_id)
                        w_frame = keep.float()
                w_mass = float(w_frame.sum().item())
            if w_mass > 0.0:
                self._epoch_entropy_sum += float((H_frame * w_frame).sum().item())
                self._epoch_entropy_count += w_mass

        compiler = self._get_graph_compiler(device)

        # 5. Build the OTC graphs and take the k2 CTC loss.  One global scalar
        #    weight per arc type for the whole batch (the schedule is the only
        #    source of variation), so train and eval see the same lattice.
        #    With both arcs off this is exactly the standard CTC graph; before
        #    star_start_epoch the arcs are off for every run.
        stars_on = self.stars_on
        graphs = compiler.compile(
            token_ids_list=token_ids_list,
            allow_bypass_arc=self.allow_bypass_arc and stars_on,
            allow_self_loop_arc=self.allow_self_loop_arc and stars_on,
            bypass_weight=self.bypass_weight,
            self_loop_weight=self.self_loop_weight,
            word_bypass_weight=(
                self.word_bypass_weight if self._otc_granularity == "dual" else None
            ),
        )

        loss_per_utt = k2.ctc_loss(
            decoding_graph=graphs,
            dense_fsa_vec=dense,
            output_beam=self.output_beam,
            reduction="none",
            use_double_scores=self.use_double_scores,
        )

        # 6. Diagnostics.  Detached, validation-only, never touches the loss.
        if measure_diag:
            self._n_frames_measured += float(frame_valid.sum().item())
            if stars_on and (self.allow_bypass_arc or self.allow_self_loop_arc):
                self._accum_star_share(
                    log_probs_base_s=log_probs_base_s,
                    log_probs_s=log_probs_s,
                    flat_star=flat_star,
                    graphs=graphs,
                    supervision_segments=supervision_segments,
                    th_ilen_s=th_ilen_s,
                    frame_valid=frame_valid,
                )

        # 7. Restore the original batch order
        loss_per_utt = loss_per_utt[inv_order]

        # 8. zero_infinity
        if self.zero_infinity:
            bad = ~torch.isfinite(loss_per_utt)
            if bad.any():
                logging.warning(
                    "OTC: %d/%d utterances had inf/nan loss; zeroed out.",
                    int(bad.sum().item()),
                    B,
                )
                loss_per_utt = loss_per_utt.masked_fill(bad, 0.0)

        # 9. Reduction
        loss = loss_per_utt.sum() / B if self.reduce else loss_per_utt / B

        # 10. CTC label smoothing (arc_schedule="label_smoothing" only):
        #     L = (1 - eps) * L_CTC + eps * L_uniform
        if self._ctc_label_smoothing > 0.0:
            eps = self._ctc_label_smoothing
            l_uniform = self._uniform_loss(log_probs_base_s, frame_valid)
            loss = (1.0 - eps) * loss + eps * l_uniform

        return loss

    # ------------------------------------------------------------------
    # ESPnet compat helpers
    # ------------------------------------------------------------------

    def softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(self.ctc_lo(hs_pad), dim=2)
        return self._append_star_logprob(log_p).exp()

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """Decoding-time posteriors over the REAL vocabulary.

        The ``<star>`` column is deliberately NOT appended: it is a training
        construct, and the decoder's token inventory is ``real_odim``.
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)

    def forced_align(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_lens: torch.Tensor,
        blank_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        import torchaudio

        log_probs = F.log_softmax(self.ctc_lo(hs_pad), dim=2)
        assert log_probs.size(0) == 1, "forced_align requires batch size 1"
        assert not (ys_pad == blank_idx).any(), "Target contains blank tokens."
        return torchaudio.functional.forced_align(
            log_probs, ys_pad, hlens, ys_lens, blank=blank_idx
        )
