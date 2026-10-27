# Running the experiments

These are the master scripts as used for the paper. They are ESPnet2 recipe
drivers and expect to be run from inside an ESPnet egs2-style recipe directory
(the one holding `asr_respin.sh` / `asr.sh`, `utils/`, `dumps/` and `data_*`),
with the criterion installed as described in `../espnet/README.md`.

```
run_ebf_otc_recip_master_respin.sh   RESPIN   9 languages x 7 criteria   (main WER table)
run_ebf_otc_recip_master_libri.sh    LibriSpeech train-clean-100, 7 criteria  (main WER + Conformer tables)
run_ebf_otc_recip_master_fleurs.sh   FLEURS  14 languages x 7 criteria   (main WER table)
run_ebf_otc_recip_crosstransfer.sh   10 h sweeps behind the schedule-sensitivity figure
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

`--encoder` takes `ebf` (default, E-Branchformer) or `conformer`, which swaps
in the `*_conformer.yaml` configs and appends `_conformer` to the experiment
tag. The paper reports Conformer results for LibriSpeech only. `--langs` is taken verbatim and must match the directory names in your
dump; nothing is translated against a baked-in list.

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
    otc_token_word · otc_token_word_recip      (+ a _conformer variant of each)
conf/cross_transfer/     base configs for run_ebf_otc_recip_crosstransfer.sh
conf/{respin,libri,fleurs}/decode_ctc.yaml   CTC prefix-search decoding, no LM
```

Only the loss block (`ctc_conf`) varies between criteria within a corpus;
batching, epochs, warmup and SpecAugment are shared. The paper's wildcard-arc
settings table is taken from the `ctc_conf` blocks of these files, which also
record `star_start_epoch` (see the top-level README).

A note on one dead default: the scripts set `lm_config=conf/train_lm.yaml`, but
every run uses `use_lm=false`, so that file is never read and is not shipped.

## The schedule-sensitivity figure

`run_ebf_otc_recip_crosstransfer.sh` drives the sweeps on a random 10 h subset
of LibriSpeech `train-clean-100` with the frozen wav2vec 2.0 front-end. The
figure uses three of its modes on that subset: `ctc` (the reference line),
`otc_token` with the decay swept from 1.0 to 0.6, and `otc_recip` with the
exponent κ swept from 0.8 to 1.4, each scored on the combined dev-clean and
dev-other set. The base configs are `conf/cross_transfer/libri_*_ssl.yaml`.

The script can also run the same sweeps on RESPIN and FLEURS and transplant a
decay tuned for one epoch budget onto another; those runs are not reported in
the paper. Its configs for all three corpora are included because the config
tests check them.
