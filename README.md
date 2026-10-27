# Token-level tolerance to transcription ambiguity for ASR

Code, configurations, analysis scripts and validator transcripts for
*A Training Criterion with Token-Level Tolerance to Transcription Ambiguity
for Automatic Speech Recognition* (ICASSP 2027 submission).

The paper moves the wildcard (`<star>`) arcs of Omni-temporal Classification
(OTC) from word to token granularity, adds a hybrid graph carrying both, and
replaces the epoch-indexed relaxation of the arc weights with a schedule
indexed by held-out predictive entropy.

## Layout

| path | contents |
|---|---|
| `espnet/` | the five ESPnet files that implement the criterion, plus a required one-hunk patch to `trainer.py` |
| `recipe/` | the master run scripts and every training config used in the paper |
| `analysis/disagreement/` | the validator-disagreement pipeline (labelling, probes, statistics) |
| `analysis/tables/` | scripts that generate the paper's tables and figure |
| `analysis/results/` | the analysis outputs the paper quotes |
| `transcripts/` | the three validator transcriptions of RESPIN dev/test (pending licence clearance; see its README) |
| `docs/` | supplementary note on the bypass-probability probe |

## The seven training criteria

Every system shares one graph compiler and one WFST forward--backward
implementation; they differ only in where the wildcard arcs sit and how their
weights are relaxed.

| paper | config stem | `otc_granularity` | `arc_schedule` | `star_start_epoch` |
|---|---|---|---|---|
| CTC | `ctc` | — (arcs off) | `none` | — |
| OTC-W<sub>e</sub> | `otc_word` | `word` | `epoch` | 1 |
| OTC-W'<sub>e</sub> | `otc_word_tuned` | `word` | `epoch` | 3 |
| OTC-T<sub>e</sub> | `otc_token` | `token` | `epoch` | 1 |
| OTC-T<sub>H</sub> | `otc_recip` | `token` | `entropy_reciprocal` | 1 |
| OTC-TW<sub>e</sub> | `otc_token_word` | `dual` | `epoch` | 3 |
| OTC-TW<sub>H</sub> | `otc_token_word_recip` | `dual` | `entropy_reciprocal` | 3 |

`star_start_epoch: 3` means epochs 1--2 train on the plain CTC graph and the
arc-weight clock starts at epoch 3. Configs with `_conformer` appended differ
from their counterparts only in the encoder block and in
`patience: 5` / `keep_nbest_models: 5`.

## From the paper to this repository

| paper artifact | how to regenerate |
|---|---|
| WER table (`tab:main`), all 25 tasks x 7 criteria | `analysis/tables/gen_table1_v2.py`, reading `analysis/tables/wer_data.json` |
| `wer_data.json` itself | `analysis/tables/extract_wer.py`, scanning the ESPnet experiment directories |
| arithmetic check on the WER table | `analysis/tables/verify_table.py` |
| Conformer table (`tab:conformer`) | the same runs with `--encoder conformer` |
| wildcard-arc settings (`tab:criteria`) | the `ctc_conf` blocks of `recipe/conf/*_final/*.yaml` |
| schedule-sensitivity figure (`fig:sweep`) | `recipe/run_ebf_otc_ablation_granularity_libri.py` to train, `analysis/tables/plot_hparam_sweep.py` to plot |
| disagreement table (`tab:disagree`) | `analysis/disagreement/run_mutual.sh`, then `analysis/tables/make_table2.py` |
| significance tests quoted in the text | `analysis/disagreement/bootstrap_pvalues.py`; output in `analysis/results/bootstrap_pvalues.txt` |

The pooled figures the paper quotes are in
`analysis/results/results_mutual_drop/` (`test_report.md` and `dev_report.md`
carry the per-language and pooled tables).

## Environment

The reported results were produced with:

| | |
|---|---|
| Python | 3.10.18 |
| PyTorch | 2.9.1+cu128 |
| ESPnet | commit `29a4df6492` (see `espnet/ESPNET_COMMIT`) |
| k2 | used for all WFST losses; see `espnet/README.md` |
| sentencepiece | 0.2.1 |
| numpy | 2.2.6 |

Start from `espnet/README.md`: the criterion needs five new files **and** a
one-hunk patch to `espnet2/train/trainer.py`. Without the patch the arc
weights never advance and every OTC system silently degenerates toward CTC.

## Data

- **LibriSpeech** and **FLEURS** are public and downloaded by the ESPnet recipes.
- **RESPIN** audio and reference transcripts come from the official corpus
  release and are **not** redistributed here.
- The **three independent validator transcriptions** that the disagreement
  analysis is built on are not in this repository yet: their redistribution
  terms are still being confirmed with the RESPIN project and the vendors.
  `transcripts/README.md` records their format, coverage and labelling
  convention, and they will be added there once cleared. Everything else in
  this repository, including the analysis outputs the paper quotes, is
  present.

## Not included

- Model checkpoints.
- The cached frame posteriors used by the disagreement probes (~136 MB
  intermediate; regenerable with `analysis/disagreement/run_star_posteriors.sh`
  from the checkpoints).
- Exploratory variants of the criterion that the paper does not report.

## Citation

See `CITATION.bib`.
