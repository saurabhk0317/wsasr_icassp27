# Running the experiments

These are the master scripts as used for the paper. They are ESPnet2 recipe
drivers and expect to be run from inside an ESPnet egs2-style recipe directory
(the one holding `asr_respin.sh` / `asr.sh`, `utils/`, `dumps/` and `data_*`),
with the criterion installed as described in `../espnet/README.md`.

```
run_ebf_otc_recip_master_respin.sh        RESPIN   9 languages x 7 criteria
run_ebf_otc_recip_master_libri.sh         LibriSpeech train-clean-100, 7 criteria
run_ebf_otc_recip_master_fleurs.sh        FLEURS  14 languages x 7 criteria
run_ebf_otc_recip_labelsmooth.sh          uniform-KL regulariser sweep (LibriSpeech)
run_ebf_otc_ablation_granularity_libri.py tau / kappa sweep on a 10 h subset
otc_recip/run_lib.sh                      shared cell runner used by all of the above
otc_recip/tests/                          unit tests for the config surface and graphs
```

All of them resolve a *cell* (language x criterion x seed), skip it when its
`score_wer/result.txt` already exists, and print why. `--dry-run` prints the
commands without running them; `--force` reruns completed cells.

## Examples

```bash
./run_ebf_otc_recip_master_respin.sh --dry-run
./run_ebf_otc_recip_master_respin.sh --methods otc_recip
./run_ebf_otc_recip_master_respin.sh --langs "hi kn" --gpu 1
./run_ebf_otc_recip_master_respin.sh --encoder conformer --dry-run
./run_ebf_otc_recip_master_libri.sh  --methods "ctc otc_token otc_token_word_recip"
```

`--encoder` takes `ebf` (default, E-Branchformer) or `conformer`, which swaps
in the `*_conformer.yaml` configs and appends `_conformer` to the experiment
tag. `--langs` is taken verbatim and must match the directory names in your
dump; nothing is translated against a baked-in list.

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
conf/libri_10h/          configs for the tau / kappa sweep
conf/label_smoothing/    the four epsilon values
conf/{respin,libri,fleurs}/decode_ctc.yaml   CTC prefix-search decoding, no LM
```

Only the loss block (`ctc_conf`) varies between criteria within a corpus;
batching, epochs, warmup and SpecAugment are shared. The wildcard-arc settings
table in the paper is exactly the `ctc_conf` blocks of these files.

A note on one dead default: the scripts set `lm_config=conf/train_lm.yaml`, but
every run uses `use_lm=false`, so that file is never read and is not shipped.
