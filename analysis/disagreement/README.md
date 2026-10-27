# Wildcard-bypass probability vs validator disagreement

The pipeline behind the paper's disagreement table (`tab:disagree`) and the
statistics quoted beside it.

RESPIN dev/test utterances have the reference transcript `text`, on which every
model was trained and scored, and three independent validator transcripts
(`text_gts`, `text_nt`, `text_artpark`; see `../../transcripts/README.md`).

## Definitions (as in the paper)

* **Disputed.** Each validator transcript is aligned to the reference by a
  character-level Levenshtein alignment. A character or word is *disputed* when
  at least one validator changes it **and** the validators do not all propose
  the same alternative. A unit that all three validators correct identically is
  a unanimous correction (an erroneous rather than ambiguous reference) and is
  dropped from both pools (`--drop_consensus`). Every other unit is *agreed*.
  A word is changed if any of its characters is, or a validator inserts inside
  it. Punctuation is excluded.
* **Bypass probability** `m_k`. Each model's frame posteriors are composed with
  the wildcard graph of the reference and the posterior probability that an
  alignment takes the bypass arc at unit `k` is computed exactly by
  forward--backward (k2, tagged arcs). Every system is probed at the same fixed
  bypass weight `w = 0`, for characters (`tok_bp_w0`) and words (`word_bp_w0`).
  CTC is scored through the same graph as a control.
* **Statistics.** Means over units, pooled over the nine languages; paired
  cluster bootstrap over utterances for the differences from CTC.

## Steps

| step | script | output |
|---|---|---|
| 1 | `label_disagreement_mutual.py` (uses the alignment in `label_disagreement.py`) | `labels_mutual/<lang>_<split>.pkl.gz` |
| 2 | `run_star_posteriors.sh` -> `star_posteriors.py` | `posteriors/` (needs the checkpoints, GPU; ~136 MB, not shipped) |
| 3 | `analyze.py --drop_consensus` | per-language means (`*_descriptives.csv`) and the split by number of dissenting validators (`*_by_ndisp.csv`) |
| 4 | `mean_diff_bootstrap.py --drop_consensus` | pooled agreed/disputed means with 95% CIs (`mean_diff_bootstrap.txt`, 1000 resamples) |
| 5 | `bootstrap_pvalues.py` | the p-values quoted in the paper (10000 resamples) |
| - | `sanity_checks.py` | checks the bypass posteriors against a brute-force recomputation (must print PASS) |

`run_mutual.sh` runs steps 1, 3, 4 and 5 once step 2 has produced the
posteriors. `analyze.py` also writes further diagnostics that the paper does
not use; only the outputs listed above are shipped, in `../results/`.

For RESPIN `kn`, the OTC-W<sub>e</sub> model is the converged rerun at bypass
weight -35 (`otc_word_bpn35`), as in the main WER table; this is the default
of `run_star_posteriors.sh --kn_word`.

Paths to the RESPIN data, the experiment root and the virtualenv are set at the
top of the scripts (see `../README.md`, "Paths to edit before running").
