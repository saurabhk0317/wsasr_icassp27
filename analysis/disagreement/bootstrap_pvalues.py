#!/usr/bin/env python3
"""
Paired cluster-bootstrap p-values for the differences in mean bypass
probability quoted in the paper (Sec. 4.5).

mean_diff_bootstrap.py reports 95% CIs from 1000 resamples; a two-sided
bootstrap p-value from N resamples cannot fall below 2/N, so 1000 resamples
can only support p<0.01.  This script uses 10000 by default so that the
p<0.001 claim in the paper is attainable, and reports, for each comparison,
the point estimate, the percentile CI, the bootstrap SE and the two-sided
bootstrap p-value (the fraction of resamples on the wrong side of zero,
doubled).

Utterances are resampled with replacement jointly across languages, and both
systems in a comparison are always scored on the SAME resampled utterances.

Note: the normal-approximation p-value is printed only as a sanity check.
It is not reported in the paper -- extrapolating a normal tail to p~1e-200
is meaningless; the bootstrap p is floored at 2/n_boot and reported as
"p < 0.001".

Usage:
  bootstrap_pvalues.py --labels_dir labels_mutual --word_probe word_bp_w0 \
      --out results_mutual/bootstrap_pvalues.txt
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import norm

from mean_diff_bootstrap import HERE, LANGS, METHODS, NAME, PAIRS, collect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", default="labels_mutual")
    ap.add_argument("--tok_probe", default="tok_bp_w0")
    ap.add_argument("--word_probe", default="word_bp_w0")
    ap.add_argument("--drop_consensus", action="store_true")
    ap.add_argument("--exclude_classes", default="punct")
    ap.add_argument("--langs", default=" ".join(LANGS))
    ap.add_argument("--splits", default="test dev")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    langs = a.langs.split()
    exclude = set(a.exclude_classes.split())
    floor = 2.0 / a.n_boot
    lines = [f"labels_dir={a.labels_dir}  tok_probe={a.tok_probe}  word_probe={a.word_probe}  "
             f"drop_consensus={a.drop_consensus}  n_boot={a.n_boot}",
             f"p_boot is two-sided and floored at 2/n_boot = {floor:.0e}; "
             f"'<{floor:.0e}' means no resample crossed zero.", ""]

    for split in a.splits.split():
        o = collect(split, langs, METHODS, a.labels_dir, a.tok_probe, a.word_probe,
                    a.drop_consensus, exclude)
        for unit in ("char", "word"):
            keys = sorted(o["ctc"][unit])
            N = len(keys)
            X = {m: np.array([o[m][unit][k] for k in keys], float) for m in METHODS if o[m][unit]}
            rng = np.random.default_rng(a.seed)
            idxs = [rng.integers(0, N, N) for _ in range(a.n_boot)]
            full = np.arange(N)

            def mean_disp(m, idx):
                s = X[m][idx].sum(0)
                return s[0] / s[1]

            comps = [(m, "ctc") for m in METHODS if m != "ctc" and m in X]
            comps += [(h, t) for h, t in PAIRS if h in X and t in X]
            lines += [f"===== {split} / {unit} (disputed units, {N} utterances) =====",
                      f"{'comparison':<24}{'diff':>9}{'95% CI':>22}{'SE':>9}{'p_boot':>10}{'p_normal':>11}"]
            for x, y in comps:
                pt = mean_disp(x, full) - mean_disp(y, full)
                d = np.array([mean_disp(x, i) - mean_disp(y, i) for i in idxs])
                se = d.std(ddof=1)
                frac = float((d <= 0).mean())
                p_boot = 2 * min(frac, 1 - frac)
                p_norm = 2 * norm.sf(abs(pt / se)) if se > 0 else float("nan")
                lo, hi = np.percentile(d, [2.5, 97.5])
                ps = f"<{floor:.0e}" if p_boot == 0 else f"{p_boot:.1e}"
                lines.append(f"{NAME[x] + ' - ' + NAME[y]:<24}{pt:+9.4f}"
                             f"{f'[{lo:+.4f},{hi:+.4f}]':>22}{se:9.4f}{ps:>10}{p_norm:11.1e}")
            lines.append("")
        print(f"{split}: done", file=sys.stderr)

    txt = "\n".join(lines)
    print(txt)
    if a.out:
        out = a.out if os.path.isabs(a.out) else f"{HERE}/{a.out}"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w").write(txt + "\n")
        print(f"wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
