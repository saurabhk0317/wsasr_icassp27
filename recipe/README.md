# Running the experiments

These are the master scripts as used for the paper. They are ESPnet2 recipe
drivers, run from inside an ESPnet egs2-style recipe directory, with the
criterion installed as described in `../espnet/README.md`. To set one up:

```bash
cd /path/to/espnet/egs2
TEMPLATE/asr1/setup.sh wsasr/asr1     # utils/ steps/ scripts/ pyscripts/ path.sh cmd.sh local/
cp -r /path/to/this/repo/recipe/. wsasr/asr1/
cd wsasr/asr1
```

The data directories (`data_respin/`, `data_libri/`, `data_fleurs/`) and the
dumps are prepared with the standard ESPnet recipes for each corpus; the
master scripts start at stage 11 (training).

`asr_respin.sh` is ESPnet's `egs2/TEMPLATE/asr1/asr.sh` at the pinned commit
with three changes: the data directory is a `--data_folder` option rather than
a fixed `data/`, with a `--tag` that names the token list; the collect-stats
job uses one GPU; and stage 13 scores with `local/score_lang_id.py` instead of
`local/score.sh`. That script writes the `result_lid.txt` whose
`Overall WER` line is the number reported in the paper's tables and figure.
It removes `, ? । .` from reference and hypothesis, computes the
corpus-level WER with jiwer, and scores only the utterances listed in
`<data_folder>/<set>/utt2dialect` (`<utt-id> <label>`). RESPIN ships dialect
labels; for LibriSpeech and FLEURS every utterance carries the language code as
its label (e.g. `en`), so the file just lists every utterance of the set.

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
criterion uses the same seed. The scripts' defaults are the seeds the paper's
runs used, and the `seed:` key in each config agrees (the scripts always pass
`--seed` explicitly, so it is the script value that takes effect):

| corpus | seed |
|---|---|
| LibriSpeech (incl. Conformer and the 10 h sweeps) | 20261 |
| FLEURS (all 14 languages) | 2026 |
| RESPIN `bh`, `bn`, `mg`, `mr`, `mt`, `te` | 20261 |
| RESPIN `ch`, `hi`, `kn` | 2026 |

The RESPIN script picks the seed per language when `--seeds` is not given;
passing `--seeds N` applies `N` to every selected language.

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
