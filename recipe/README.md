# Running the experiments

These are the master scripts as used for the paper. They are ESPnet2 recipe
drivers and expect to be run from inside an ESPnet egs2-style recipe directory
(the one holding `asr_respin.sh` / `asr.sh`, `utils/`, `dumps/` and `data_*`),
with the criterion installed as described in `../espnet/README.md`.

```
run_ebf_otc_recip_master_respin.sh   RESPIN   9 languages x 7 criteria   (main WER table)
run_ebf_otc_recip_master_libri.sh    LibriSpeech train-clean-100, 7 criteria  (main WER + Conformer tables)
run_ebf_otc_recip_master_fleurs.sh   FLEURS  14 languages x 7 criteria   (main WER table)
run_ebf_otc_recip_sweep.sh           LibriSpeech 10 h tau / kappa sweeps   (schedule-sensitivity figure)
otc_recip/run_lib.sh                 shared cell runner used by all of the above
otc_recip/tests/                     unit tests for the config surface and graphs
```

All of them resolve a *cell* (language x criterion x seed), skip it when its
`score_wer/result.txt` already exists, and print why. `--dry-run` prints the
commands without running them; `--force` reruns completed cells.

## Examples

```bash
./run_ebf_otc_recip_master_respin.sh --dry-run
./run_ebf_otc_recip_master_respin.sh --methods otc_recip
./run_ebf_otc_recip_master_respin.sh --langs "hi kn" --gpu 1
./run_ebf_otc_recip_master_libri.sh  --encoder conformer --dry-run
./run_ebf_otc_recip_master_libri.sh  --methods "ctc otc_token otc_token_word_recip"
```

The LibriSpeech script's `--encoder` takes `ebf` (default, E-Branchformer) or
`conformer`, which swaps in the `*_conformer.yaml` configs and appends
`_conformer` to the experiment tag; this is the Conformer table, which the
paper reports for LibriSpeech only. `--langs` is taken verbatim and must match
the directory names in your dump; nothing is translated against a baked-in
list.

## Seeds

Every cell in the paper is one training run, and within a language every
criterion uses the same seed. The scripts default to `--seeds 20261`; to
reproduce the paper exactly, pass the seed its runs used:

| corpus | seed |
|---|---|
| LibriSpeech | 20261 |
| FLEURS (all 14 languages) | 2026 |
| RESPIN `bh`, `bn`, `mg`, `mr`, `mt`, `te` | 20261 |
| RESPIN `ch`, `hi`, `kn` | 2026 |

```bash
./run_ebf_otc_recip_master_fleurs.sh --seeds 2026
./run_ebf_otc_recip_master_respin.sh --seeds 2026  --langs "ch hi kn"
./run_ebf_otc_recip_master_respin.sh --seeds 20261 --langs "bh bn mg mr mt te"
```

## Front-ends

| corpus | features | epochs |
|---|---|---|
| RESPIN | 80-dim filterbank, no SSL | 70 |
| LibriSpeech | frozen wav2vec 2.0 Base via s3prl | 30 |
| FLEURS | frozen XLS-R 300M via s3prl | 30 |

The SSL checkpoints download to `./hub` on first use. A frozen SSL front-end
gave no benefit on RESPIN in our earlier work, so the filterbank setup is the
one reported.

## Configs

```
conf/respin_final/   conf/libri_final/   conf/fleurs_final/
    ctc · otc_word · otc_word_tuned · otc_token · otc_recip
    otc_token_word · otc_token_word_recip
    (conf/libri_final/ also holds a _conformer variant of each)
conf/sweep/              base configs for run_ebf_otc_recip_sweep.sh
conf/{respin,libri,fleurs}/decode_ctc.yaml   CTC prefix-search decoding, no LM
```

Only the loss block (`ctc_conf`) varies between criteria within a corpus;
batching, epochs, warmup and SpecAugment are shared. The paper's wildcard-arc
settings table is taken from the `ctc_conf` blocks of these files, which also
record `star_start_epoch` (see the top-level README).

A note on one dead default: the scripts set `lm_config=conf/train_lm.yaml`, but
every run uses `use_lm=false`, so that file is never read and is not shipped.

## The schedule-sensitivity figure

`run_ebf_otc_recip_sweep.sh` drives the sweeps on a random 10 h subset of
LibriSpeech `train-clean-100` (`train_clean_100_ran_10h`) with the frozen
wav2vec 2.0 front-end. It has three modes: `ctc` (the reference line),
`otc_token` with the decay τ swept from 1.0 to 0.6 in steps of 0.025, and
`otc_recip` with the exponent κ swept from 0.8 to 1.4 in steps of 0.1. Each
cell is decoded on the combined dev-clean and dev-other set, which is what the
figure reports. The swept value is passed at run time through `--ctc_conf`, so
the three base configs in `conf/sweep/` are the only files involved; they
differ only in the loss block. Everything else (initial weights -25 / -5,
batch bins 6M, warmup 3000, 30 epochs) is held fixed.

```bash
./run_ebf_otc_recip_sweep.sh --dry-run
./run_ebf_otc_recip_sweep.sh                    # train; dev decodes are queued
./run_ebf_otc_recip_sweep.sh --action decode    # score anything left unscored
python ../analysis/tables/plot_hparam_sweep.py figures
```

The selected values, τ = 0.775 and κ = 1.3, are the ones in every
`conf/*_final/` config. The OTC-W′<sub>e</sub> word-level constants (bypass
-19, decay 0.925; self-loop -1, decay 0.975) were selected on the same subset,
as the paper states; only the selected values are shipped, in
`otc_word_tuned.yaml`.
