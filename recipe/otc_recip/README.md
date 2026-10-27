# `asr_otc_recip` — OTC with an entropy-indexed arc schedule

An ESPnet2 task implementing **Omni-temporal Classification (OTC)** for CTC ASR,
with four selectable arc-weight schedules. The contribution is
`entropy_reciprocal`: a schedule indexed on the model's own entropy rather than
on an epoch clock, which lets **one parameter set** drive every corpus, feature
type and epoch budget we tried.

---

## What OTC is

Standard CTC composes a `ctc_topo` with a linear transcript FSA, so every
training path must emit the reference token sequence exactly. When the
transcript is wrong, CTC has no choice but to fit the error.

OTC (Gao et al., 2023) adds two `<star>` wildcard arcs at every reference
position:

| arc | meaning | effect |
|---|---|---|
| **BYPASS** | substitution | advance past the position without emitting its token, at cost `bypass_weight` |
| **SELF-LOOP** | insertion | stay at the position while consuming a frame, at cost `self_loop_weight` |

Both weights are negative — a penalty in the log domain — and are **relaxed
over training**, so the model may ignore transcript errors early and is pushed
back onto the reference as it grows confident.

The wildcard emission is the log-average of the non-blank posteriors:

```
log P(*|t) = logsumexp_{v != blank} log P(v|t) - log(V-1)
```

The `-log(V-1)` tax is load-bearing: it holds the emission at or below the mean
non-blank posterior, a provable lower bound, so the k2 lattice cannot
manufacture probability mass and the loss cannot go negative. `<star>` occupies
the last output slot (`odim = vocab_size + 1`) and is an **analytic** function
of the other logits — never a learned parameter.

---

## The four modes (`ctc_conf.arc_schedule`)

### `none` — the CTC baseline

Star arcs off (`allow_bypass_arc = allow_self_loop_arc = false`). It runs
through the **identical** k2 graph compiler, `ctc_topo` and `k2.ctc_loss` call
as every OTC mode; the only difference is that no `<star>` arc is emitted into
the transcript FSA. The baseline is therefore *the same loss with the wildcard
arcs switched off*, not a different CTC implementation.

This is verified, not asserted: `tests/test_parity_vs_asr_otc.py` checks that
the resulting graph contains no `<star>`-labelled arc, and that the loss is
**bit-for-bit equal** to `espnet2/asr/ctc_k2.py` on a fixed batch.

### `epoch` — the OTC paper's schedule

```
w(e) = w_init * decay^(e-1)
```

Indexed on an **epoch clock**. The decay must be retuned whenever the epoch
budget changes: our 70-epoch RESPIN runs need `0.9`, our 30-epoch SSL runs need
`0.70`. Table 2 measures what transplanting one onto the other costs.

### `entropy_reciprocal` — the contribution

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
each epoch, so the arc weights *tighten again* if entropy rises. This is
deliberate — both the old `H_eff` ratchet and the old progress ratchet were
removed because they made the schedule one-way.

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
    entropy_shape_k: 1.2
    entropy_eps: 0.0001
    initial_bypass_weight: -25.0
    initial_self_loop_weight: -5.0
    bypass_end: -0.01
    self_loop_end: -0.001
```

`tests/test_config_entropy_keys.py` enforces that this block is byte-identical
in all three `otc_recip.yaml` files. **That test is the paper's claim.**

The entropy signal is selected by `entropy_mode`:

| mode | signal | frames averaged | normaliser |
|---|---|---|---|
| `full` | full distribution, blank included | all valid | `log(real_odim)` |
| `nb` | non-blank, renormalised | non-blank frames only (hard cut) | `log(real_odim - 1)` |
| `soft` *(default)* | non-blank, renormalised | all valid, weighted by `1 - P(blank\|t)` | `log(real_odim - 1)` |

`entropy_source: dev` (default) accumulates during the **validation** pass. The
trainer runs `set_epoch(e) → train(e) → valid(e)`, so `valid(e-1)` fills the
accumulator that `set_epoch(e)` consumes: the schedule is driven by held-out
entropy, which cannot be lowered by memorising erroneous training transcripts.
The measurement is `no_grad` and never enters the objective.

### `label_smoothing` — the Table-3 control

Star arcs off, plus a uniform-distribution regulariser. Answers "isn't OTC just
label smoothing?".

```
L         = (1 - eps) * L_CTC + eps * L_uniform
L_uniform = - mean over VALID frames of [ (1/V) * sum_v log p_t(v) ]
```

`V = real_odim`: blank **included**, the `<star>` column **excluded**.

> **Note.** ESPnet's `model_conf.lsm_weight` smooths the *attention decoder's*
> cross-entropy and does not touch CTC. There is no CTC label smoothing in
> ESPnet, so this term is implemented here
> (`espnet2/asr/ctc_otc_recip.py::_uniform_loss`) and pinned down by
> `tests/test_label_smoothing.py`. `L_CTC` is a batch-mean per-utterance NLL
> while `L_uniform` is a per-frame mean, so the two are on different scales and
> `eps` is correspondingly small.

---

## Graph granularity (`ctc_conf.otc_granularity`)

| | `token` (default) | `word` (icefall / Gao-2023 parity) |
|---|---|---|
| bypass arc spans | one token | a whole word |
| self-loops sit at | every token position | word boundaries only |
| substituting an *n*-token word costs | *n* star arcs | exactly 1 |
| mid-word stars | possible | impossible |

Word boundaries are derived from `token_list`, which the task passes in
automatically: an explicit `ctc_conf.word_boundary_symbol` if given, else every
sentencepiece piece starting with `▁`, else the char tokenizer's `<space>` id.
In this study RESPIN and FLEURS use the `<space>` branch (1 boundary id) and
LibriSpeech the `▁` branch (118 of 200 pieces).

> ### ⚠ Penalty weights are not comparable across papers
>
> **Our star arcs are TOKEN-level; the original OTC paper's are WORD-level.**
> In `token` mode the bypass weight is charged *per token*, so with character
> tokenisation a 6-character word costs `6 × initial_bypass_weight`; in `word`
> mode it is charged once. The absolute penalty values in this repository are
> therefore **not** comparable with those printed in Gao et al. 2023, and the
> two granularities are not comparable with each other either. Our `otc_word`
> configs use icefall's word-scale values (`-19.0 / +3.75`, decays
> `0.975 / 0.999`) precisely for that reason.

---

## Relationship to stock ESPnet

Both new ESPnet files are **stock files for this ESPnet version with additions
only — nothing is deleted**, so they can be diffed against upstream and so that
every stock code path that inspects them keeps working:

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

Only `espnet2/asr/ctc_otc_recip.py` is a genuine strip-down — it is a new file,
not a stock one, and it deliberately exposes only the four arc schedules.

## Layout

```
espnet2/asr/ctc_otc_recip.py           the OTC loss, graph compiler, schedules
espnet2/asr/espnet_model_otc_recip.py  stock espnet_model.py + set_epoch (nothing removed)
espnet2/tasks/asr_otc_recip.py         stock tasks/asr.py with the CTC swapped
espnet2/bin/asr_otc_recip_train.py     entry points
espnet2/bin/asr_otc_recip_inference.py

conf/{respin,libri,fleurs}_final/{ctc,otc_word,otc_token,otc_recip}.yaml
conf/cross_transfer/<corpus>_otc_token_decay{0p7,0p9}.yaml
conf/label_smoothing/libri_ls{0p005,0p01,0p02,0p05}.yaml

run_ebf_otc_recip_master_{respin,libri,fleurs}.sh
run_ebf_otc_recip_{crosstransfer,labelsmooth,seeds}.sh
otc_recip/run_lib.sh                   shared resume / dry-run driver

scripts/collect_otc_recip_results.py   exp trees -> one CSV + coverage report
scripts/plot_arc_weights.py            the arc-weight figure

otc_recip/tests/                       parity, config and label-smoothing tests
```

Within a corpus, **everything outside the `ctc_conf` loss block is identical
across the four methods** — encoder, optimiser, LR schedule, specaug, batching,
seed, `keep_nbest_models`, averaging and decode config.
`tests/test_config_within_corpus.py` fails if that ever drifts.

---

## Porting to another machine

### ESPnet files — `list_files_changed_asr_otc_recip`

The manifest in the recipe root lists the ESPnet files that differ from a stock
checkout: one absolute path per line, no comments, so it feeds straight into a
copy loop. All six go under the target ESPnet root.

```bash
SRC_ESPNET=<this machine's espnet root>
DST_ESPNET=<target espnet root>

while read -r f; do
  rel=${f#"$SRC_ESPNET"/}
  mkdir -p "$DST_ESPNET/$(dirname "$rel")"
  cp -p "$f" "$DST_ESPNET/$rel"
done < list_files_changed_asr_otc_recip
```

| file | |
|---|---|
| `espnet2/asr/ctc_otc_recip.py` | new — the OTC loss, graph compiler and schedules |
| `espnet2/asr/espnet_model_otc_recip.py` | new — stock `espnet_model.py` + `set_epoch`; nothing removed |
| `espnet2/tasks/asr_otc_recip.py` | new — stock `tasks/asr.py` with the CTC swapped |
| `espnet2/bin/asr_otc_recip_train.py` | new |
| `espnet2/bin/asr_otc_recip_inference.py` | new |
| `espnet2/train/trainer.py` | **MODIFIED, not new** |

> `trainer.py` carries the `set_epoch` hook that drives the arc schedule.
> Without it the arc weights never move off `initial_*` and every OTC run
> silently degenerates into a fixed-penalty run. It is **shared with
> `asr_otc`**, which patches it identically, so overwriting is safe on a
> machine that already runs `asr_otc` — diff it first otherwise.

The task has **no other non-stock ESPnet dependency**: the `espnet2/beats/`
import inherited from `asr_otc` was removed so this list is the complete
closure. Verify on the target with:

```bash
python -c "from espnet2.tasks.asr_otc_recip import ASRTask; print('ok')"
```

### Recipe files

Not in the manifest — they live in the recipe directory and travel with it:

| | |
|---|---|
| `conf/{respin,libri,fleurs}_final/`, `conf/cross_transfer/`, `conf/label_smoothing/` | the 19 shipped configs |
| `run_ebf_otc_recip_*.sh`, `otc_recip/run_lib.sh` | the six drivers and their shared library |
| `scripts/collect_otc_recip_results.py`, `scripts/plot_arc_weights.py` | analysis |
| `otc_recip/tests/`, `otc_recip/README.md` | tests and this file |

The run scripts also require `conf/{respin,libri,fleurs}/decode_ctc.yaml`,
`asr_respin.sh`, `utils/`, `path.sh` and `cmd.sh` — all unchanged and already
present wherever `asr_otc` runs. They abort loudly if a config is absent rather
than falling back.

### Running the tests on the target

`otc_recip/tests/test_parity_vs_asr_otc.py` compares against the **`asr_otc`**
implementation, so it additionally needs `espnet2/asr/{ctc_otc_clean,
otc_sched_affine,ctc_k2}.py` — copy those with the existing
`list_files_changed_asr_otc` manifest. Nothing in the `asr_otc_recip` runtime
imports them; the other four test files need only the six files above.

```bash
python -m pytest otc_recip/tests/ -q
```

---

## Reproducing the tables

All run scripts **skip any cell whose `score_wer/result.txt` already exists**
(that is the default, not a flag) and print what they skipped and why. Add
`--dry-run` to print every command without executing, `--force` to rerun a
completed cell, and `--methods` / `--langs` / `--corpora` to run a subset.

**Nothing about the experiment grid is baked into the scripts.** Every corpus,
language, seed, decay and eps is a flag with a default, and **language tokens
are used verbatim** — they are never translated or checked against an internal
list, so whatever names the dump on your machine uses (`hi`, `hindi`, `hi_in`,
`ru_ru`) are what you pass. Corpus roots are variables too: pass `--data_root`
(or the per-corpus `--*_data_root`) to point at wherever the dumps live, or set
them from `db.sh` if your recipe has one.

A selection that matches **no** cell is an error, not a silent success:

```
ERROR: no cells matched the selection -- nothing was run.
  methods=<all>  langs=nosuchlang  corpora=respin
```

```bash
# --- Table 1: 4 methods x {9 RESPIN langs, LibriSpeech, 14 FLEURS langs} ---
./run_ebf_otc_recip_master_respin.sh
./run_ebf_otc_recip_master_libri.sh   --data_root <librispeech dump>
./run_ebf_otc_recip_master_fleurs.sh  --data_root <fleurs dump>
#   subsets:  --langs "hi kn"   --methods "ctc otc_recip"

# --- Table 2: cross-transfer, 7 runs (otc_token with the WRONG decay)
#     as reported: RESPIN non-SSL 0.70; LibriSpeech + FLEURS SSL 0.90
./run_ebf_otc_recip_crosstransfer.sh
#   corpora, VARIANT, languages and decay are all flags:
#     --corpora respin --respin_variant ssl
#     --corpora libri  --libri_variant nonssl
#     --corpora fleurs --fleurs_langs "hi_in el_gr"
#   the config is conf/cross_transfer/<corpus>_otc_token_decay<D>[_ssl].yaml
#   and the run aborts if it is absent -- configs are never generated on the fly


### Cross-transfer: which decay pairs with which variant

The decay to transplant follows the **epoch budget**, not the corpus. In this
study every non-SSL setup is 70 epochs with native decay 0.9, and every SSL
setup is 30 epochs with native decay 0.7:

| variant | epochs | native decay | decay transplanted | config |
|---|---|---|---|---|
| `nonssl` | 70 | 0.9 | **0.7** | `<corpus>_otc_token_decay0p7.yaml` |
| `ssl` | 30 | 0.7 | **0.9** | `<corpus>_otc_token_decay0p9_ssl.yaml` |

Both variants exist for all three corpora, so any of the six is runnable via
`--<corpus>_variant nonssl|ssl`. `--<corpus>_decay` defaults from the variant
and rarely needs setting; picking a variant's own native decay would make the
run a no-op, which `tests/test_config_within_corpus.py` asserts against.

The variant is part of the run's identity — it selects the experiment
directory suffix (`stft_char` vs `ssl_xlsr300m`, `stft_bpe500` vs
`ssl_w2v2base960h_bpe200`), the training stage (SSL redoes collect-stats) and,
for LibriSpeech, the BPE size (500 non-SSL, 200 SSL) — so the same corpus and
language at two feature types cannot collide in one exp dir.

# --- Table 3: CTC label smoothing on LibriSpeech, eps in {0.005,0.01,0.02,0.05}
./run_ebf_otc_recip_labelsmooth.sh    --data_root <librispeech dump>
#   subset:  --eps "0.01 0.02"

# --- Table 4: seed variance, otc_recip on LibriSpeech + RESPIN hindi
./run_ebf_otc_recip_seeds.sh
#   subsets:  --corpora respin  --seeds "20262 20263"  --respin_lang hi

# --- collect everything into one CSV and list the still-empty cells
python scripts/collect_otc_recip_results.py

# --- the arc-weight figure
python scripts/plot_arc_weights.py -o arc_weights.pdf
```

### Setups

| | RESPIN | LibriSpeech | FLEURS |
|---|---|---|---|
| data | 9 Indian languages | `train_clean_100` | 14 languages |
| frontend | STFT filterbank | frozen `wav2vec2_base_960` (s3prl) | frozen XLS-R 300M `xlsr2_300m.pt` (s3prl) |
| tokens | char (~68–76) | BPE-200 | char (~135) |
| epochs | 70 | 30 | 30 |
| `batch_bins` / `accum_grad` | 6M / 1 | 16M / 4 | 6M / 1 |
| `warmup_steps` | 15000 | 3000 | 1000 |
| epoch-schedule decay | 0.9 | 0.70 | 0.70 |

A frozen SSL frontend gives no benefit on RESPIN in our prior work, so RESPIN
is reported with STFT features. SSL checkpoints download to `./hub` on first
use.

### Seeds

`--seed` reaches **both** parameter initialisation (`set_all_random_seed(args.seed)`
runs before `build_model`) and data ordering (`SequenceIterFactory` derives the
sampler, the batch shuffle and the DataLoader worker seeds from `epoch + seed`).
This was checked rather than assumed — see `tests/test_seed_reaches_sampler.py`
— and needed no change.

One caveat that test also asserts: because the offset is `epoch + seed`, seed
*s* at epoch *e* replays the ordering of seed *s+1* at epoch *e-1*. Different
seeds give **shifted** data orderings rather than independent ones. Parameter
init does differ properly, so the runs are genuinely distinct, but the
seed-to-seed spread slightly understates true run-to-run variance. Widen the
seed spacing if that matters for the claim you are making.

---

## Tests

```bash
python -m pytest otc_recip/tests/ -v
```

| file | what it guarantees |
|---|---|
| `test_parity_vs_asr_otc.py` | identical losses and arc weights vs the original `asr_otc` task on the shared modes, at both granularities; `none` == `ctc_k2` bit-for-bit |
| `test_config_within_corpus.py` | nothing outside the loss block differs between the four methods of a corpus |
| `test_config_entropy_keys.py` | the `otc_recip` loss block is byte-identical across all three corpora — **the paper's claim** |
| `test_label_smoothing.py` | the smoothing term matches the stated formula, excludes padding frames and the `<star>` column |
| `test_seed_reaches_sampler.py` | `--seed` reaches parameter init *and* data ordering |

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

`scripts/plot_arc_weights.py` reads these lines, and also the older
`[OTCSCHED EPOCH]` and `OTC <sched> epoch N` formats emitted by the previous
task, so existing runs can be plotted without rerunning.

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

The OTC method, the `<star>` wildcard arcs and the word-level training graph:

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
