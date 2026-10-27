#!/usr/bin/env python3
"""
Paper Table 2 numbers: mean bypass probability on agreed vs disputed units,
with paired cluster-bootstrap CIs on the differences between systems.

Each cell is the mean of m_k over the units in that group, where m_k is the
posterior probability that the alignment takes the bypass arc at unit k under
a fixed probe weight (identical for every system).  The bootstrap resamples
UTTERANCES with replacement, jointly across languages, and recomputes the
ratio-of-sums statistic; both systems are always scored on the same resampled
utterances, so shared utterance difficulty cancels in the difference.

Reported per split and unit:
  disp/agree means, each system's difference from CTC with a 95% CI, the
  disputed-to-agreed ratio, and each hybrid minus its same-schedule
  token-only counterpart.

Defaults reproduce the original run (labels/, char w=0, word w=-2).  Use
--labels_dir labels_mutual --word_probe word_bp_w0 for the mutual-dissent
definition with a uniform w=0 probe.

Usage:
  mean_diff_bootstrap.py                                   # original setup
  mean_diff_bootstrap.py --labels_dir labels_mutual \
      --word_probe word_bp_w0 --out results_mutual/mean_diff_bootstrap.txt
"""
import argparse
import gzip
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANGS = ["bh", "bn", "ch", "hi", "kn", "mg", "mr", "mt", "te"]
METHODS = ["ctc", "otc_word", "otc_word_tuned", "otc_token", "otc_recip",
           "otc_token_word", "otc_token_word_recip"]
NAME = {"ctc": "CTC", "otc_word": "OTC-W_e", "otc_word_tuned": "OTC-W_e'",
        "otc_token": "OTC-T_e", "otc_recip": "OTC-T_H",
        "otc_token_word": "OTC-TW_e", "otc_token_word_recip": "OTC-TW_H"}
PAIRS = [("otc_token_word", "otc_token"), ("otc_token_word_recip", "otc_recip")]


def collect(split, langs, methods, labels_dir, tok_probe, word_probe, drop_cons, exclude):
    """Per-utterance (sum, count) of the probe over disputed and agreed units."""
    out = {m: {"char": {}, "word": {}} for m in methods}
    for lang in langs:
        lab_path = f"{HERE}/{labels_dir}/{lang}_{split}.pkl.gz"
        labels = {r["uid"]: r for r in pickle.load(gzip.open(lab_path))}
        for m in methods:
            p = f"{HERE}/posteriors/{lang}_{split}_{m}.pkl.gz"
            if not os.path.exists(p):
                print(f"  missing {p}, skipped", file=sys.stderr)
                continue
            for u in pickle.load(gzip.open(p))["utts"]:
                lab = labels[u["uid"]]
                n = len(lab["text"])
                pr = u["probes"]
                cons_c = lab.get("cons_char") if drop_cons else None
                cons_w = lab.get("cons_word") if drop_cons else None

                keep = np.array([(not u["oov"][i]) and lab["char_class"][i] not in exclude
                                 and not (cons_c and cons_c[i]) for i in range(n)])
                d = np.array(lab["n_disp"]) > 0
                x = np.asarray(pr[tok_probe]["bypass"], float)
                out[m]["char"][(lang, u["uid"])] = (x[keep & d].sum(), int((keep & d).sum()),
                                                    x[keep & ~d].sum(), int((keep & ~d).sum()))

                nw = lab["n_words"]
                kw = np.array([not (cons_w and cons_w[w]) for w in range(nw)])
                dw = np.array(lab["word_n_disp"]) > 0
                xw = np.asarray(pr[word_probe]["bypass"], float)
                out[m]["word"][(lang, u["uid"])] = (xw[kw & dw].sum(), int((kw & dw).sum()),
                                                    xw[kw & ~dw].sum(), int((kw & ~dw).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", default="labels")
    ap.add_argument("--tok_probe", default="tok_bp_w0")
    ap.add_argument("--word_probe", default="word_bp_w-2")
    ap.add_argument("--drop_consensus", action="store_true")
    ap.add_argument("--exclude_classes", default="punct")
    ap.add_argument("--langs", default=" ".join(LANGS))
    ap.add_argument("--splits", default="test dev")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    langs = a.langs.split()
    exclude = set(a.exclude_classes.split())
    methods = [m for m in METHODS]
    rng = np.random.default_rng(a.seed)
    lines = [f"labels_dir={a.labels_dir}  tok_probe={a.tok_probe}  word_probe={a.word_probe}  "
             f"drop_consensus={a.drop_consensus}  n_boot={a.n_boot}", ""]

    for split in a.splits.split():
        o = collect(split, langs, methods, a.labels_dir, a.tok_probe, a.word_probe,
                    a.drop_consensus, exclude)
        for unit in ("char", "word"):
            keys = sorted(o["ctc"][unit])
            N = len(keys)
            X = {m: np.array([o[m][unit][k] for k in keys], float) for m in methods if o[m][unit]}
            idxs = [rng.integers(0, N, N) for _ in range(a.n_boot)]

            def stat(m, idx):
                s = X[m][idx].sum(0)
                return (s[0] / s[1] if s[1] else np.nan, s[2] / s[3] if s[3] else np.nan)

            def ci(f):
                v = np.array([f(i) for i in idxs])
                return np.percentile(v, [2.5, 97.5])

            full = np.arange(N)
            nd, na = X["ctc"][:, 1].sum(), X["ctc"][:, 3].sum()
            lines += [f"===== {split} / {unit}  (utterances={N}, disputed={int(nd)}, agreed={int(na)}) =====",
                      f"{'system':<10} {'disp':>7} {'agree':>8} | {'disp - CTC [95% CI]':>30} | "
                      f"{'agree - CTC [95% CI]':>30} | disp/agree"]
            dc, ac = stat("ctc", full)
            for m in methods:
                if m not in X:
                    continue
                d, ag = stat(m, full)
                r = d / ag if ag else float("nan")
                if m == "ctc":
                    lines.append(f"{NAME[m]:<10} {d:7.3f} {ag:8.4f} | {'--':>30} | {'--':>30} | {r:5.1f}")
                    continue
                l1, h1 = ci(lambda i: stat(m, i)[0] - stat("ctc", i)[0])
                l2, h2 = ci(lambda i: stat(m, i)[1] - stat("ctc", i)[1])
                sig = "*" if (l1 > 0 or h1 < 0) else " "
                lines.append(f"{NAME[m]:<10} {d:7.3f} {ag:8.4f} | {d - dc:+.3f} [{l1:+.3f},{h1:+.3f}]{sig:>7} | "
                             f"{ag - ac:+.4f} [{l2:+.4f},{h2:+.4f}] | {r:5.1f}")
            lines.append("-- hybrid minus token-only (same schedule):")
            for hyb, tok in PAIRS:
                if hyb not in X or tok not in X:
                    continue
                d1, a1 = stat(hyb, full)
                d0, a0 = stat(tok, full)
                l1, h1 = ci(lambda i: stat(hyb, i)[0] - stat(tok, i)[0])
                l2, h2 = ci(lambda i: stat(hyb, i)[1] - stat(tok, i)[1])
                sig = "*" if (l1 > 0 or h1 < 0) else " "
                lines.append(f"  {NAME[hyb]} - {NAME[tok]}: disp {d1 - d0:+.3f} [{l1:+.3f},{h1:+.3f}]{sig} "
                             f"  agree {a1 - a0:+.4f} [{l2:+.4f},{h2:+.4f}]")
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
