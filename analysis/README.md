# Analysis

## `tables/` — the paper's tables and figure

| script | produces |
|---|---|
| `extract_wer.py` | `wer_data.json`, by scanning `result_lid.txt` under the ESPnet experiment directories |
| `gen_table1_v2.py` | the main WER table (25 tasks x 7 criteria, per-corpus means, relative reductions) |
| `verify_table.py` | arithmetic cross-check of that table against `wer_data.json` |
| `make_table2.py` | the disagreement table, from a `mean_diff_bootstrap.txt` run |
| `plot_hparam_sweep.py` | the schedule-sensitivity figure (needs matplotlib; `OUT_DIR` at the top is a local path and must be edited) |

`wer_data.json` holds every WER quoted in the paper, so the WER table can be
regenerated without retraining anything.

The script names use the numbering of an earlier draft: `gen_table1_v2.py` and
`verify_table.py` produce and check the main WER table (`tab:main`), and
`make_table2.py` produces the disagreement table (`tab:disagree`).

## `disagreement/` — the validator-disagreement pipeline

Run in this order:

| step | script | what it does |
|---|---|---|
| 1 | `label_disagreement_mutual.py` | aligns each validator transcript to the reference and labels every character, word and boundary as agreed, disputed or unanimously corrected |
| 2 | `star_posteriors.py` | for each trained model, composes its frame posteriors with the wildcard transcript graph and records the bypass posterior of every unit by forward--backward |
| 3 | `analyze.py` | group means, dose--response by number of dissenting validators, AUCs, and the covariate-adjusted logistic comparison against the CTC control |
| 4 | `mean_diff_bootstrap.py` | paired cluster bootstrap over utterances for the agreed/disputed means |
| 5 | `bootstrap_pvalues.py` | the same comparisons at 10000 resamples, which is what the p-values in the paper come from |
| — | `sanity_checks.py` | verifies the bypass posteriors against a remove-one-arc recomputation |

`run_mutual.sh` chains steps 1, 3, 4; `run_star_posteriors.sh` drives step 2;
`run_all.sh` is the older variant that does not drop unanimous corrections.
`README.md` and `README_mutual.md` in that directory document the two label
conventions.

Step 2 needs the trained checkpoints and writes ~136 MB of cached posteriors,
which are not shipped here.

## `results/` — the outputs the paper quotes

`results_mutual_drop/` is the run that excludes unanimous corrections, which is
the convention the paper uses. `test_report.md` and `dev_report.md` carry the
per-language and the pooled tables, including the adjusted odds ratios;
`*_by_ndisp.csv` holds the dose--response split; `mean_diff_bootstrap.txt` holds
the agreed/disputed means with confidence intervals.

`bootstrap_pvalues.txt` is the 10000-resample run. It is computed on disputed
units only, and the disputed pool is identical whether or not unanimous
corrections are dropped, so it applies to both label conventions.

### Two conventions, one difference

`results_mutual/` keeps a unanimously corrected unit in the **agreed** pool;
`results_mutual_drop/` removes it from both pools. The disputed pools are
identical (17526 characters, 18897 words on test). The difference touches three
cells of the agreed columns in the third decimal and nothing else.

## Paths to edit before running

Several scripts carry absolute paths from the machine the experiments ran on:

| file | variable | points at |
|---|---|---|
| `disagreement/label_disagreement.py` | `DATA_ROOT` | the RESPIN data root holding `<lang>/IISc_RESPIN_{dev,test}_<lang>/` |
| `disagreement/star_posteriors.py` | `DATA_ROOT` | the same RESPIN data root |
| `disagreement/run_mutual.sh`, `run_all.sh`, `run_star_posteriors.sh` | `EXP`, `VENV` | the experiment root and the ESPnet virtualenv activate script |
| `tables/extract_wer.py` | `ROOT` | the directory holding the `{libri,fleurs,respin}` experiment trees |
| `tables/plot_hparam_sweep.py` | `OUT_DIR` | where the figure is written |

The disagreement pipeline additionally needs the three validator transcripts
beside each `text` in the RESPIN tree that `DATA_ROOT` points at. Those are not
yet in this repository (see `../transcripts/README.md`); until they are, step 1
cannot be rerun, but the outputs it produced are in `results/` and the tables
can be regenerated from them.
