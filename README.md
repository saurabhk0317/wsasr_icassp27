# A Training Criterion with Token-Level Tolerance to Transcription Ambiguity for Automatic Speech Recognition

Code, configurations and analysis scripts for
*A Training Criterion with Token-Level Tolerance to Transcription Ambiguity
for Automatic Speech Recognition*.

**Paper:** [arXiv:2609.30160](https://arxiv.org/abs/2609.30160) · submitted to ICASSP 2027

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
| `transcripts/` | format and coverage of the three validator transcriptions of RESPIN dev/test (available on request) |

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
arc-weight clock starts at epoch 3.

**Divergent OTC-W<sub>e</sub> runs.** With the published bypass weight of −19,
OTC-W<sub>e</sub> diverged on some languages and was rerun with a larger
penalty, as stated in the paper. The reported cells use:

| corpus | language | `initial_bypass_weight` | penalties that also diverged |
|---|---|---|---|
| FLEURS | `ml_in` | −30 | −19, −25 |
| FLEURS | `ta_in` | −30 | −19 |
| RESPIN | `kn` | −35 | −19, −25, −30 |
| RESPIN | `mr`, `te` | −30 | −19 |
| RESPIN | `mg` | −30 | — |

Every other language uses −19 from `otc_word.yaml`. To rerun one of these
cells, export the override before calling the master script; `run_lib.sh`
merges it into the `ctc_conf` block loaded from the config:

```bash
OTCR_ASR_ARGS="--ctc_conf initial_bypass_weight=-35" \
  ./run_ebf_otc_recip_master_respin.sh --methods otc_word --langs kn
```

**Conformer.** The LibriSpeech configs with `_conformer` appended
(`recipe/conf/libri_final/*_conformer.yaml`) differ from their E-Branchformer
counterparts only in the encoder block and in `patience: 5` /
`keep_nbest_models: 5`. As in the paper, the Conformer comparison covers
LibriSpeech only.

## From the paper to this repository

| paper artifact | how to regenerate |
|---|---|
| WER table (`tab:main`), all 25 tasks x 7 criteria | `analysis/tables/gen_table1_v2.py`, reading `analysis/tables/wer_data.json` |
| `wer_data.json` itself | `analysis/tables/extract_wer.py`, scanning the ESPnet experiment directories |
| arithmetic check on the WER table | `analysis/tables/verify_table.py` |
| Conformer table (`tab:conformer`) | the same runs with `--encoder conformer` |
| wildcard-arc settings (`tab:criteria`) | the `ctc_conf` blocks of `recipe/conf/*_final/*.yaml` |
| schedule-sensitivity figure (`fig:sweep`) | `recipe/run_ebf_otc_recip_sweep.sh` (CTC, the OTC-T<sub>e</sub> τ sweep and the OTC-T<sub>H</sub> κ sweep on a 10 h LibriSpeech subset with the frozen SSL front-end, scored on dev-clean + dev-other) to train, `analysis/tables/plot_hparam_sweep.py` to plot |
| disagreement table (`tab:disagree`) | `analysis/disagreement/run_mutual.sh`, then `analysis/tables/make_table2.py` |
| significance tests quoted in the text | `analysis/disagreement/bootstrap_pvalues.py`; output in `analysis/results/bootstrap_pvalues.txt` |

The pooled and per-language disagreement statistics the paper quotes are in
`analysis/results/` (see `analysis/README.md` for which file backs which
statement).

## Environment

The reported results were produced with:

| | |
|---|---|
| Python | 3.10.18 |
| PyTorch | 2.9.1+cu128 |
| ESPnet | commit `29a4df6492` (see `espnet/ESPNET_COMMIT`) |
| k2 | 1.24.4.dev20260525+cuda12.8.torch2.9.1 (all WFST losses) |
| sentencepiece | 0.2.1 |
| numpy | 2.2.6 |
| jiwer | 4.0.0 (WER scoring, `recipe/local/score_lang_id.py`) |

Start from `espnet/README.md`: the criterion needs five new files **and** a
one-hunk patch to `espnet2/train/trainer.py`. Without the patch the arc
weights never advance and every OTC system silently degenerates toward CTC.

## Data

- **LibriSpeech** and **FLEURS** are public and downloaded by the ESPnet recipes.
- **RESPIN** audio and reference transcripts come from the official corpus
  release and are **not** redistributed here.
- The **three independent validator transcriptions** that the disagreement
  analysis is built on are not included. They may be used for research
  purposes and are available from the authors on request;
  `transcripts/README.md` records their format, coverage and labelling
  convention. The analysis outputs the paper quotes are in `analysis/results/`.

## Not included

- Model checkpoints.
- The cached frame posteriors used by the disagreement probes (~136 MB
  intermediate; regenerable with `analysis/disagreement/run_star_posteriors.sh`
  from the checkpoints).
- Experiments that the paper does not report, together with their configs
  and run scripts. Only the runs behind the paper's tables and figure are
  covered here. The loss module also implements a `label_smoothing` schedule
  that no shipped config selects.

## Licence

Code: Apache License 2.0 (`LICENSE`). See `LICENSE.md` for the data.

## Citation

See `CITATION.bib`.
