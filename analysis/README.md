# Analysis

## `tables/` — the paper's tables and figure

| script | produces |
|---|---|
| `extract_wer.py` | `wer_data.json`, by scanning `result_lid.txt` under the ESPnet experiment directories |
| `gen_table1_v2.py` | the main WER table (25 tasks x 7 criteria, per-corpus means, relative reductions) |
| `verify_table.py` | arithmetic cross-check of that table against `wer_data.json` |
| `make_table2.py` | the disagreement table, from a `mean_diff_bootstrap.txt` run |
| `plot_hparam_sweep.py` | the schedule-sensitivity figure, from the dev WERs written into the script (needs matplotlib; output directory as the first argument, default `./figures`) |

`wer_data.json` holds every WER quoted in the paper, so the WER table can be
regenerated without retraining anything.

The script names use the numbering of an earlier draft: `gen_table1_v2.py` and
`verify_table.py` produce and check the main WER table (`tab:main`), and
`make_table2.py` produces the disagreement table (`tab:disagree`).

## `disagreement/` — the validator-disagreement pipeline

Run in this order:

| step | script | what it does |
|---|---|---|
| 1 | `label_disagreement_mutual.py` | aligns each validator transcript to the reference and labels every character and word as agreed, disputed or unanimously corrected |
| 2 | `star_posteriors.py` | for each trained model, composes its frame posteriors with the wildcard transcript graph and records the bypass posterior of every unit by forward--backward |
| 3 | `analyze.py` | per-language agreed/disputed means and the split by number of dissenting validators |
| 4 | `mean_diff_bootstrap.py` | paired cluster bootstrap over utterances for the pooled agreed/disputed means |
| 5 | `bootstrap_pvalues.py` | the same comparisons at 10000 resamples, which is what the p-values in the paper come from |
| - | `sanity_checks.py` | verifies the bypass posteriors against a brute-force recomputation |

`run_star_posteriors.sh` drives step 2 and `run_mutual.sh` chains steps 1 and
3-5. `disagreement/README.md` states the definitions.

Step 2 needs the trained checkpoints and writes ~136 MB of cached posteriors,
which are not shipped here.

## `results/` — the outputs the paper quotes

All of these use the paper's convention: unanimous corrections are excluded,
and characters and words are probed at `w = 0`.

| file | supports |
|---|---|
| `results_mutual_drop/mean_diff_bootstrap.txt` | the Agree / Disp. columns of `tab:disagree` (test), the same means on dev, the 11x disputed/agreed ratio for CTC and the 27--36% increase for the token-level criteria, with 95% CIs |
| `results_mutual_drop/{test,dev}_by_ndisp.csv` | the 1 / 2 / 3 dissenting-validator columns (pooled with unit-count weights) |
| `results_mutual_drop/{test,dev}_descriptives.csv` | per-language agreed and disputed means, behind "the gain holds in all nine languages on both splits" |
| `bootstrap_pvalues.txt` | the paired-bootstrap p-values (10000 resamples, p < 0.001). It covers disputed units only, which are the same whether or not unanimous corrections are dropped, so it was run without `--drop_consensus` |

## Paths to edit before running

Several scripts carry absolute paths from the machine the experiments ran on:

| file | variable | points at |
|---|---|---|
| `disagreement/label_disagreement.py` | `DATA_ROOT` | the RESPIN data root holding `<lang>/IISc_RESPIN_{dev,test}_<lang>/` |
| `disagreement/star_posteriors.py` | `DATA_ROOT` | the same RESPIN data root |
| `disagreement/run_mutual.sh`, `run_all.sh`, `run_star_posteriors.sh` | `EXP`, `VENV` | the experiment root and the ESPnet virtualenv activate script |
| `tables/extract_wer.py` | `ROOT` | the directory holding the `{libri,fleurs,respin}` experiment trees |

The disagreement pipeline additionally needs the three validator transcripts
beside each `text` in the RESPIN tree that `DATA_ROOT` points at. Those are not
yet in this repository (see `../transcripts/README.md`); until they are, step 1
cannot be rerun, but the outputs it produced are in `results/` and the tables
can be regenerated from them.
