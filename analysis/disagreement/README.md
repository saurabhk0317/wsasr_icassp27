# Does `<star>` land where validators disagree?

RESPIN dev/test utterances have four transcripts: `text` (the base
transcript every model was trained and scored on) and three independent
validator transcripts (`text_gts`, `text_nt`, `text_artpark`). This study asks
whether a trained model, given the chance, routes the characters/words that
validators disagree on through `<star>` arcs, and whether star-trained (OTC)
models do this more than the CTC control.

Everything here is **read-only** on `WSASR/exp_final` and
`CR_CTC/data_respin`; all outputs are written in this directory.

## Run

```bash
./run_all.sh --gpus "0 1"          # all 9 languages, dev + test (~35 min on 2 GPUs)
```

## Files

| file | step | what |
|---|---|---|
| `label_disagreement.py` | 1 | per-char / per-word / per-boundary validator disagreement labels -> `labels/` |
| `star_posteriors.py` | 2 | loads one model's averaged checkpoint, runs the audio, builds the OTC graph for `text`, exact `<star>` posteriors per position -> `posteriors/` |
| `run_star_posteriors.sh` | 2 | driver for all (language, split, model); resumable; multi-GPU |
| `sanity_checks.py` | - | tagged-lattice posteriors vs brute force (must print PASS) |
| `analyze.py` | 3-5 | descriptives, AUC, OTC-vs-CTC comparison, pooled model -> `results/` |
| `run_all.sh` | 1-5 | everything |

## Method, briefly

* **Labels.** Character-level Levenshtein alignment of `text` against each
  validator. A char is *disputed* if a validator substitutes or deletes it;
  a boundary is disputed if a validator inserts there; a word is disputed if
  any of its chars is, or a validator inserts inside it. Punctuation is
  excluded by default (silent; no model can star it).
* **Star measures.** For each model, the same probe graphs are built on its
  posteriors (the CTC model included -- its `<star>` column is computed from
  its logits exactly as in training):
  token bypass at weights 0 / -2 / -5, token self-loop at 0 / -2, word bypass
  at 0 / -2 / -6, and each OTC model's own granularity at its final training
  weights. Posteriors are exact (tagged arcs through k2), verified against a
  brute force. A weight-free critical bypass weight (`lam`) comes from the
  stars-off Viterbi path.
* **Control.** CTC never trained with `<star>`; its association between star
  placement and disagreement measures how much of the effect is plain
  acoustics. The OTC claim is about the difference from CTC.
* **Statistics.** AUC per model; paired cluster-bootstrap dAUC (OTC - CTC);
  logistic regression of star (p > 0.5) on disputed x model with covariates
  (char class, word length, position, char frequency, utterance CER) and
  cluster-robust SEs by utterance; pooled across languages with language
  fixed effects.

## Models

`exp_final/respin/<lang>_s5/asr_otcrecip_respin_<lang>_{ctc,otc_word,otc_token,otc_recip}`
(averaged `valid.loss.ave.pth`). For kn, `otc_word` = `otc_word_bpn35`
(bpn19/25/30 never converged); change with `run_star_posteriors.sh --kn_word`.
