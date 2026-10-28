#!/usr/bin/env python3
#
# Part of ESPnet. Licensed under the Apache License, Version 2.0
# (see the LICENSE file in the ESPnet source root).
"""``--seed`` must reach BOTH parameter init AND data ordering.

Runs with different ``--seed`` values are only distinct if the seed actually
re-randomises the run.  A seed that only reached
``torch.manual_seed`` would re-initialise the weights but replay the identical
batch order, which is not an independent sample.

This pins down the two mechanisms in ESPnet that carry it:

  1. ``AbsTask.main_worker`` calls ``set_all_random_seed(args.seed)`` BEFORE
     ``build_model``, and that seeds ``random``, ``numpy`` and ``torch``
     -- so parameter initialisation moves with the seed.
  2. ``AbsTask.build_sequence_iter_factory`` passes ``seed=args.seed`` into
     ``SequenceIterFactory``, which derives every per-epoch ordering decision
     from ``epoch + self.seed``: ``sampler.generate()``, the batch shuffle and
     the DataLoader ``worker_init_fn``.

CONCLUSION (checked, not assumed): no fix was needed -- ``--seed`` already
reaches both.  These tests exist so that a future refactor cannot quietly
break it.

CAVEAT, deliberately asserted below: because the offset is ``epoch + seed``,
seed s at epoch e draws the same ordering as seed s+1 at epoch e-1.  Different
seeds therefore give SHIFTED orderings rather than independent ones.  Parameter
init does differ properly, so the runs are genuinely distinct, but their
data orderings are not independent.

Run:  python -m pytest otc_recip/tests/test_seed_reaches_sampler.py -v
"""

import ast
import inspect
import random
import sys

import numpy as np
import pytest
import torch

from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
from espnet2.tasks import abs_task
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed


class _DummyDataset:
    def __len__(self):
        return 200

    def __getitem__(self, uid):
        return uid, {"x": np.zeros(2, dtype=np.float32)}


def _batches():
    """40 batches of 5 utterance ids."""
    return [tuple(f"utt{i * 5 + j}" for j in range(5)) for i in range(40)]


def _order(seed, epoch=1):
    f = SequenceIterFactory(
        dataset=_DummyDataset(),
        batches=_batches(),
        seed=seed,
        shuffle=True,
        num_workers=0,
    )
    return list(f.build_iter(epoch, shuffle=True).batch_sampler)


# ---------------------------------------------------------------------------
# 1. Parameter initialisation
# ---------------------------------------------------------------------------


def test_set_all_random_seed_covers_all_three_rngs():
    """torch, numpy AND python-random -- ESPnet inits draw from all of them."""
    set_all_random_seed(20261)
    a = (torch.randn(4).tolist(), np.random.rand(4).tolist(), random.random())
    set_all_random_seed(20261)
    b = (torch.randn(4).tolist(), np.random.rand(4).tolist(), random.random())
    set_all_random_seed(20262)
    c = (torch.randn(4).tolist(), np.random.rand(4).tolist(), random.random())

    assert a == b, "same seed must reproduce"
    assert a != c, "different seed must diverge"


def test_seed_is_applied_before_build_model():
    """Order matters: seeding AFTER build_model would not affect init."""
    src = inspect.getsource(abs_task.AbsTask.main_worker)
    i_seed = src.find("set_all_random_seed(args.seed)")
    i_build = src.find("build_model")
    assert i_seed != -1, "AbsTask.main_worker no longer seeds from args.seed"
    assert i_build != -1, "AbsTask.main_worker no longer calls build_model"
    assert i_seed < i_build, (
        "set_all_random_seed(args.seed) must run BEFORE build_model, or the "
        "seed does not reach parameter initialisation"
    )


def test_model_init_moves_with_the_seed():
    """End-to-end on the real module the task builds."""
    from espnet2.asr.ctc_otc_recip import CTC

    def build(seed):
        set_all_random_seed(seed)
        return CTC(odim=12, encoder_output_size=8, arc_schedule="epoch")

    w1 = build(20261).ctc_lo.weight.clone()
    w1b = build(20261).ctc_lo.weight.clone()
    w2 = build(20262).ctc_lo.weight.clone()

    assert torch.equal(w1, w1b)
    assert not torch.equal(w1, w2)


# ---------------------------------------------------------------------------
# 2. Data ordering
# ---------------------------------------------------------------------------


def test_task_passes_seed_to_the_iterator_factory():
    """``build_sequence_iter_factory`` must hand args.seed to SequenceIterFactory."""
    src = inspect.getsource(abs_task.AbsTask.build_sequence_iter_factory)
    tree = ast.parse(inspect.cleandoc(src))
    seeded = [
        kw
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "SequenceIterFactory"
        for kw in node.keywords
        if kw.arg == "seed"
    ]
    assert seeded, "SequenceIterFactory is no longer constructed with seed=..."
    (kw,) = seeded
    assert ast.unparse(kw.value) == "args.seed", (
        f"SequenceIterFactory gets seed={ast.unparse(kw.value)}, not args.seed"
    )


def test_batch_order_moves_with_the_seed():
    """The functional claim: a different --seed reorders the data."""
    o1 = _order(20261)
    o1b = _order(20261)
    o2 = _order(20262)

    assert o1 == o1b, "same seed must reproduce the batch order"
    assert o1 != o2, (
        "different seeds produced the SAME batch order -- --seed is not "
        "reaching the sampler"
    )
    assert sorted(o1) == sorted(o2), "reordering only; no data may be dropped"


def test_worker_seed_moves_with_the_seed():
    """DataLoader workers must not all replay one augmentation stream."""
    f1 = SequenceIterFactory(
        dataset=_DummyDataset(), batches=_batches(), seed=20261, shuffle=True
    )
    f2 = SequenceIterFactory(
        dataset=_DummyDataset(), batches=_batches(), seed=20262, shuffle=True
    )
    w1 = f1.build_iter(1, shuffle=True).worker_init_fn
    w2 = f2.build_iter(1, shuffle=True).worker_init_fn
    assert w1.keywords["base_seed"] != w2.keywords["base_seed"]


def test_known_caveat_seed_offset_is_shared_with_epoch():
    """Document the ``epoch + seed`` collision rather than pretend it away.

    Seed s at epoch e draws the ordering of seed s+1 at epoch e-1.  This is
    upstream ESPnet behaviour, not something this task introduces.
    """
    assert _order(20262, epoch=1) == _order(20263, epoch=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
