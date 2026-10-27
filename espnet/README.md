# Installing the criterion into an ESPnet tree

The results in the paper were produced against ESPnet commit
`29a4df6492` (recorded in `ESPNET_COMMIT`). Later commits will probably work,
but the `trainer.py` patch below may need rebasing.

## 1. Copy the five new files

Each file sits at the path shown, relative to the root of your ESPnet clone:

```
espnet2/asr/ctc_otc_recip.py            graph compiler + loss; all four arc schedules
espnet2/asr/espnet_model_otc_recip.py   model wrapper (owns set_epoch)
espnet2/tasks/asr_otc_recip.py          task definition and config surface
espnet2/bin/asr_otc_recip_train.py      training entry point
espnet2/bin/asr_otc_recip_inference.py  inference entry point
```

```bash
ESPNET=/path/to/espnet
cp -r espnet2 "$ESPNET/"
```

Nothing in ESPnet is overwritten: all five are new files.

## 2. Apply the trainer patch (required)

```bash
cd "$ESPNET" && git apply /path/to/this/repo/espnet/patches/trainer_set_epoch.patch
```

It adds seven lines to `espnet2/train/trainer.py` that call `set_epoch()` on
the model once per epoch:

```python
_m = model.module if hasattr(model, "module") else model
if hasattr(_m, "set_epoch"):
    _m.set_epoch(iepoch)
```

**This patch is not optional.** The wildcard-arc weights are a function of the
epoch index (or, for the entropy-indexed schedules, of the entropy measured at
each epoch boundary). Without the call the weights stay at their initial
values, the arcs remain effectively suppressed, and every OTC system trains as
CTC with no error or warning. It is a no-op for models that do not define
`set_epoch`, so it is safe to carry in a shared tree.

## 3. Arc schedules

`arc_schedule` in the config selects one of four behaviours, documented at
length in the module docstring of `ctc_otc_recip.py`:

| value | behaviour |
|---|---|
| `none` | star arcs off; plain CTC through the same k2 graph |
| `epoch` | geometric per-epoch decay of the arc weights |
| `entropy_reciprocal` | relaxation indexed by held-out non-blank predictive entropy |
| `label_smoothing` | star arcs off, plus a uniform-KL regulariser on the frame posteriors |

`otc_granularity` selects `word`, `token` or `dual` (token arcs plus one word
bypass per word).

## 4. Check the install

The tests in `../recipe/otc_recip/tests/` exercise the config surface, the dual
graph, the `star_start_epoch` gate, the label-smoothing branch, the seed path,
and parity against the predecessor implementation. Run them from the recipe
directory with the ESPnet environment active.
