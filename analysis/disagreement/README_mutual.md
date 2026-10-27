# Mutual-dissent variant of the star-disagreement study

A re-run of the validator-disagreement study under a stricter definition of
"disputed", with a uniform probe weight at both granularities. Nothing in
`labels/`, `results/` or `posteriors/` is modified.

## What changed

**1. Definition of a disputed unit.** The original run marks a unit disputed
when at least one validator substitutes or deletes it. That also flags units
where the reference is simply wrong and all three validators independently
supply the *same* correction, which is a consensus correction rather than an
ambiguous unit. Here a unit is disputed only when

* at least one validator differs from the reference, **and**
* the validators do not all give the identical realisation.

The second condition can only fail when *every* validator differs from the
reference: if one validator sides with the reference while another does not,
the validators already disagree among themselves. So the rule removes exactly
the `n_disp == n_validators` units on which the validators mutually agree, and
it leaves every other unit untouched. Consensus needs at least two annotators
to be observable, so utterances with fewer keep the original labelling.

The realisation compared across validators is the aligned validator character
for a character (or `None` when that validator deletes it), the aligned
substring including interior insertions for a word, and the inserted string
for a boundary.

**2. Probe weight.** The word-level probe is `w=0`, matching the character
probe, instead of the `w=-2` used before. Both were already computed in
`posteriors/`, so no GPU work and no model access was needed.

**3. Consensus units** are counted as *agreed* in `results_mutual/`, following
"should not be considered as disputed". `results_mutual_drop/` drops them
instead, as a sensitivity check: they carry high bypass mass, so folding them
into the agreed group slightly raises every agreed mean.

## Files

| Path | Contents |
|---|---|
| `label_disagreement_mutual.py` | builds `labels_mutual/`; same schema as `labels/` plus `n_disp_raw`, `word_n_disp_raw`, `n_ins_raw`, `cons_char`, `cons_word`, `cons_bound` |
| `run_mutual.sh` | relabels, runs both analyses and both bootstraps |
| `labels_mutual/` | 9 languages x {dev,test} |
| `results_mutual/` | primary: consensus counted as agreed |
| `results_mutual_drop/` | sensitivity: consensus units dropped |
| `*/mean_diff_bootstrap.txt` | group means, differences from CTC and hybrid-minus-token-only, with paired bootstrap CIs |

`analyze.py` and `mean_diff_bootstrap.py` gained `--labels_dir`,
`--results_dir`, `--word_measure`/`--word_probe` and `--drop_consensus`.
Their defaults reproduce the original run byte for byte (verified: all 156
descriptive rows for kn and mr match `results/test_descriptives.csv` exactly).

## Reproduce

```bash
./run_mutual.sh                 # ~30 min on CPU, no GPU needed
```
