# `asr_otc_recip` — OTC with token-level arcs and an entropy-indexed schedule

An ESPnet2 task implementing **Omni-temporal Classification (OTC)** for CTC ASR,
with wildcard arcs at word, token or both granularities and a choice of
arc-weight schedules. It is the implementation behind the paper; the top-level
`README.md` maps each table and figure onto it.

---

## What OTC is

Standard CTC composes a `ctc_topo` with a linear transcript FSA, so every
training path must emit the reference token sequence exactly, even where the
acoustics do not support it.

OTC (Gao et al., 2023) adds `<star>` wildcard arcs to the transcript graph:

| arc | meaning | effect |
|---|---|---|
| **BYPASS** | substitution | advance past a unit without emitting it, at cost `bypass_weight` |
| **SELF-LOOP** | insertion | stay at a position while consuming a frame, at cost `self_loop_weight` |

Both weights are negative — a penalty in the log domain — and start strongly
negative, so the transcript is trusted early, before being relaxed over
training.

The wildcard emission is the log-average of the non-blank posteriors:

```
log P(*|t) = logsumexp_{v != blank} log P(v|t) - log(V-1)
```

The `-log(V-1)` term holds the emission at the mean non-blank posterior, so the
k2 lattice cannot manufacture probability mass and the loss cannot go negative.
`<star>` occupies the last output slot (`odim = vocab_size + 1`) and is an
**analytic** function of the other logits, never a learned parameter.

---

## Arc schedules (`ctc_conf.arc_schedule`)

### `none` — the CTC baseline

Star arcs off (`allow_bypass_arc = allow_self_loop_arc = false`). It runs
through the **identical** k2 graph compiler, `ctc_topo` and `k2.ctc_loss` call
as every OTC mode; the only difference is that no `<star>` arc is emitted into
the transcript FSA. The baseline is therefore *the same loss with the wildcard
arcs switched off*, not a different CTC implementation.

### `epoch` — the OTC paper's schedule

```
w(e) = w_init * decay^(e-1)
```

Indexed on an **epoch clock**, so a decay tuned for one epoch budget need not
reach the same stage of training under another. The paper uses `decay = 0.775`
for every corpus, selected on the 10 h LibriSpeech sweep.

### `entropy_reciprocal` — entropy-indexed relaxation

Once per epoch the model's non-blank conditional entropy is measured on
**held-out** data, normalised by `log(V-1)`, EMA-smoothed to `H_ema`, and mapped
to progress:

```
num = 1/(H_ema + eps) - 1/(H_start + eps)
den = 1/(low    + eps) - 1/(H_start + eps)
p   = clamp(num/den, 0, 1)
w   = geo_interp(w_init -> w_end, p ** shape_k)
```

`H_start` is the anchor captured at the first measured epoch. Each arc weight
then glides log-geometrically from its initial penalty to its endpoint.

**Not monotone.** `H_ema` tracks in both directions and progress is recomputed
each epoch, so the arc weights *tighten again* if entropy rises.

**Why one parameter set transfers.** The entropy is normalised by `log(V-1)`,
so `H` is a *fraction of maximum entropy* rather than an absolute nat count.
A vocabulary of 70 characters and one of 200 BPE pieces both start near `H = 1`
and fall toward the same floor, so `entropy_low = 0.01` means the same thing in
both. That is what lets the identical block below drive RESPIN (char, ~70),
FLEURS (char, ~135) and LibriSpeech (BPE-200) unchanged:

```yaml
ctc_conf:
    arc_schedule: entropy_reciprocal
    entropy_mode: soft
    entropy_source: dev
    entropy_ema_alpha: 0.3
    entropy_low: 0.01
    entropy_shape_k: 1.3
    entropy_eps: 0.0001
    initial_bypass_weight: -25.0
    initial_self_loop_weight: -5.0
    bypass_end: -0.01
    self_loop_end: -0.001
```

The hybrid (`otc_token_word_recip`) adds `initial_word_bypass_weight: -19.0`
and `word_bypass_end: -1.0`, driven by the same progress `p`.
`tests/test_config_entropy_keys.py` checks that this block is identical in all
three corpora's configs.

The entropy signal is selected by `entropy_mode`:

| mode | signal | frames averaged | normaliser |
|---|---|---|---|
| `full` | full distribution, blank included | all valid | `log(real_odim)` |
| `nb` | non-blank, renormalised | non-blank frames only (hard cut) | `log(real_odim - 1)` |
| `soft` *(default, used in the paper)* | non-blank, renormalised | all valid, weighted by `1 - P(blank\|t)` | `log(real_odim - 1)` |

`entropy_source: dev` (default) accumulates during the **validation** pass. The
trainer runs `set_epoch(e) → train(e) → valid(e)`, so `valid(e-1)` fills the
accumulator that `set_epoch(e)` consumes: the schedule is driven by held-out
entropy, which cannot be lowered by memorising the training transcripts.
The measurement is `no_grad` and never enters the objective.

### Delayed arcs (`star_start_epoch`)

With `star_start_epoch: s`, epochs `1 .. s-1` train and validate on the plain
CTC graph and the star arcs appear from epoch `s`. An untrained model puts
nearly all lattice mass on the wildcard paths, so switching the arcs on from
the first step can stop some runs from converging. Under `epoch` the decay
clock starts at the first epoch with arcs; under `entropy_reciprocal` the
entropy is still measured every epoch. The paper's `otc_word_tuned`,
`otc_token_word` and `otc_token_word_recip` use `s = 3`; the other criteria use
the default `s = 1`.

---

## Graph granularity (`ctc_conf.otc_granularity`)

| | `word` (icefall / Gao-2023 parity) | `token` | `dual` |
|---|---|---|---|
| bypass arc spans | a whole word | one token | one token, plus one per word |
| self-loops sit at | word boundaries only | every token position | every token position |
| substituting an *n*-token word costs | exactly 1 star arc | *n* star arcs | either |

`dual` is the `token` graph plus one extra bypass arc for every word of two or
more tokens; a one-token word gets no word arc, since it would duplicate its
token bypass. Its word arcs carry their own weight (`word_bypass_weight`).

Word boundaries are derived from `token_list`, which the task passes in
automatically: an explicit `ctc_conf.word_boundary_symbol` if given, else every
sentencepiece piece starting with `▁`, else the char tokenizer's `<space>` id.
RESPIN and FLEURS use the `<space>` branch; LibriSpeech the `▁` branch.

> ### ⚠ Penalty weights are not comparable across granularities
>
> In `token` mode the bypass weight is charged *per token*, so with character
> tokenisation a 6-character word costs `6 × initial_bypass_weight`; in `word`
> mode it is charged once. The absolute penalty values are therefore not
> comparable between granularities, nor with values printed for word-level OTC
> elsewhere. The `otc_word` configs use icefall's word-scale values
> (`-19.0 / +3.75`, decays `0.975 / 0.999`) for that reason.

---

## Relationship to stock ESPnet

Both new ESPnet files that mirror stock files are **stock files for this ESPnet
version with additions only — nothing is deleted**, so they can be diffed
against upstream and every stock code path that inspects them keeps working:

| file | diff vs stock |
|---|---|
| `espnet2/asr/espnet_model_otc_recip.py` | `espnet_model.py` + `set_epoch()`; the `ctc` argument retyped `torch.nn.Module` (our CTC is not `espnet2.asr.ctc.CTC`, and `__init__` is `@typechecked`) |
| `espnet2/tasks/asr_otc_recip.py` | `tasks/asr.py` with the CTC swapped for `ctc_otc_recip.CTC` at `odim = vocab_size + 1`, and the matching model imported |

The transducer branch, the joint network, multi-blank, the linear/classification
decoder, interCTC and every decoder choice are kept exactly as upstream has
them. This matters concretely: `espnet2/bin/asr_inference.py` reads
`asr_model.use_transducer_decoder` and `asr_model.joint_network` off the model,
so a model missing those attributes crashes at decode time with

```
AttributeError: 'ESPnetASRModel' object has no attribute 'use_transducer_decoder'
```

`espnet2/asr/ctc_otc_recip.py` is a new file, not a stock one.

Installation, including the required `trainer.py` patch, is described in
`../../espnet/README.md`. The task has no other non-stock ESPnet dependency;
verify an install with:

```bash
python -c "from espnet2.tasks.asr_otc_recip import ASRTask; print('ok')"
```

---

## Layout

```
run_ebf_otc_recip_master_{respin,libri,fleurs}.sh   main WER table (libri: also the Conformer table)
run_ebf_otc_recip_sweep.sh                          LibriSpeech 10 h sweeps (schedule-sensitivity figure)
otc_recip/run_lib.sh                                shared resume / dry-run driver
otc_recip/tests/                                    config, graph and seed tests

conf/{respin,libri,fleurs}_final/                   the seven criteria (libri: + _conformer variants)
conf/sweep/                                         base configs for the sweeps
conf/{respin,libri,fleurs}/decode_ctc.yaml          CTC prefix-search decoding, no LM
```

Within a corpus, **everything outside the `ctc_conf` loss block is identical
across the seven criteria** — encoder, optimiser, LR schedule, SpecAugment,
batching, seed, `keep_nbest_models`, averaging and decode config.
`tests/test_config_within_corpus.py` fails if that ever drifts.

---

## Running

All run scripts **skip any cell whose `score_wer/result.txt` already exists**
(that is the default, not a flag) and print what they skipped and why. Add
`--dry-run` to print every command without executing, `--force` to rerun a
completed cell, and `--methods` / `--langs` to run a subset. Commands for each
table are in `../README.md`.

**Nothing about the experiment grid is baked into the scripts.** Every corpus,
language and seed is a flag with a default, and **language tokens are used
verbatim** — they are never translated or checked against an internal list, so
whatever names the dump on your machine uses (`hi`, `hindi`, `hi_in`, `ru_ru`)
are what you pass. Corpus roots are variables too: pass `--data_root` to point
at wherever the dumps live.

A selection that matches **no** cell is an error, not a silent success:

```
ERROR: no cells matched the selection -- nothing was run.
  methods=<all>  langs=nosuchlang  corpora=respin
```

### Setups

| | RESPIN | LibriSpeech | FLEURS |
|---|---|---|---|
| data | 9 Indian languages | `train_clean_100` | 14 languages |
| frontend | STFT filterbank | frozen `wav2vec2_base_960` (s3prl) | frozen XLS-R 300M `xlsr2_300m.pt` (s3prl) |
| tokens | char (~68–76) | BPE-200 | char (~135) |
| epochs | 70 | 30 | 30 |
| `batch_bins` / `accum_grad` | 6M / 1 | 16M / 4 | 6M / 1 |
| `warmup_steps` | 15000 | 3000 | 2000 |
| `keep_nbest_models` | 10 | 5 | 5 |
| epoch-schedule decay | 0.775 | 0.775 | 0.775 |

A frozen SSL frontend gave no benefit on RESPIN in our earlier work, so RESPIN
is reported with filterbank features. SSL checkpoints download to `./hub` on
first use.

### Seeds

`--seed` reaches **both** parameter initialisation (`set_all_random_seed(args.seed)`
runs before `build_model`) and data ordering (`SequenceIterFactory` derives the
sampler, the batch shuffle and the DataLoader worker seeds from `epoch + seed`).
This was checked rather than assumed — see `tests/test_seed_reaches_sampler.py`.

One caveat that test also asserts: because the offset is `epoch + seed`, seed
*s* at epoch *e* replays the ordering of seed *s+1* at epoch *e-1*. Different
seeds give **shifted** data orderings rather than independent ones. Parameter
init does differ properly, so the runs are genuinely distinct, but the
seed-to-seed spread slightly understates true run-to-run variance.

Every cell in the paper is a single training run. Within a language all
criteria share one seed, so the comparisons the paper draws are never
confounded by it. The seed used for each corpus and language is listed in
`../README.md`.

---

## Tests

```bash
cd recipe && PYTHONPATH=/path/to/espnet python -m pytest otc_recip/tests/ -q
```

| file | what it guarantees |
|---|---|
| `test_config_within_corpus.py` | nothing outside the loss block differs between the criteria of a corpus |
| `test_config_entropy_keys.py` | the entropy-indexed loss block is identical across all three corpora, and every config holds the reported parameter set |
| `test_dual_granularity.py` | the `dual` graph: token arcs plus one word bypass per multi-token word |
| `test_star_start_epoch.py` | arcs are absent before `star_start_epoch` and the schedules start correctly after it |
| `test_seed_reaches_sampler.py` | `--seed` reaches parameter init *and* data ordering |
| `test_parity_vs_asr_otc.py` | identical losses against an earlier in-house implementation, and `none` equal to `ctc_k2` bit-for-bit; skipped when that implementation is absent, as it is in this release |

---

## Diagnostics

One machine-parseable line per epoch, fixed key order:

```
[OTCRECIP EPOCH] epoch=3 arc_schedule=entropy_reciprocal granularity=token \
  bypass_w=-12.481 self_loop_w=-3.104 H_raw=0.2788 H_ema=0.2788 \
  H_start=0.3612 progress=0.2284 star_share=0.0417 n_frames_measured=184320
```

`star_share` is the occupancy-weighted `sigmoid(bypass_w - Δ)` where
`Δ_t = log p_t(reference) - log P(*|t)`: near 0 the star arc is no longer
competitive, near 1 the transcript is being bypassed. It costs one extra
detached k2 forward/backward and is computed on **validation batches only**.
Set `ctc_conf.log_epoch_line: false` to turn the line off.

---

## Environment

Recorded from the machine these were developed on:

| | |
|---|---|
| ESPnet commit | `29a4df6492ff4ffef7b00e27eda9f4cffca22b4b` (master, 2026-05-15) |
| PyTorch | 2.9.1+cu128 |
| k2 | 1.24.4.dev20260525+cuda12.8.torch2.9.1 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 3090, 24 GB (single GPU per run, `ngpu=1`) |
| CPU / RAM | 64 cores / 251 GB |

k2 is a hard requirement: the OTC graph is built with `k2.ctc_topo`,
`k2.compose` and `k2.ctc_loss`. The SSL frontends additionally need `s3prl`.

LibriSpeech `train_clean_100` at 30 epochs takes roughly a day on one RTX 3090;
a 70-epoch RESPIN language is comparable. `entropy_reciprocal` and `epoch` cost
the same to train — the entropy measurement is a no-grad reduction on
validation batches — and the only measurable overhead is the optional
`star_share` diagnostic.

---

## Citation

For this work, see `../../CITATION.bib`. The OTC method, the `<star>` wildcard
arcs and the word-level training graph are due to:

```bibtex
@inproceedings{gao2023otc,
  title     = {Learning from Flawed Data: Weakly Supervised Automatic Speech Recognition},
  author    = {Gao, Dongji and Xu, Hainan and Raj, Desh and Garcia, Paola and
               Povey, Daniel and Khudanpur, Sanjeev},
  booktitle = {IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  year      = {2023},
  eprint    = {2309.15796},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2309.15796}
}
```

Reference implementation: icefall, `egs/librispeech/WSASR`.

---

## Licence

Apache License 2.0, as ESPnet — see the `LICENSE` file in the ESPnet source
root. All files added by this task carry the ESPnet Apache-2.0 header.
