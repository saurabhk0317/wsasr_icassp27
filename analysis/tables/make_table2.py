#!/usr/bin/env python3
"""Emit the paper's Table 2 (tab:disagree) from a mean_diff_bootstrap.txt run.

Parses the bootstrap text output and writes a LaTeX tabular with the agreed
and disputed means for characters and words, bolding the column maxima.

Usage: make_table2.py results_mutual/mean_diff_bootstrap.txt [--split test]
"""
import argparse
import re

ORDER = ["CTC", "OTC-W_e", "OTC-W_e'", "OTC-T_e", "OTC-T_H", "OTC-TW_e", "OTC-TW_H"]
MACRO = {"CTC": "CTC", "OTC-W_e": r"\OTCw{}", "OTC-W_e'": r"\OTCwt{}",
         "OTC-T_e": r"\OTCt{}", "OTC-T_H": r"\OTCr{}",
         "OTC-TW_e": r"\OTCtw{}", "OTC-TW_H": r"\OTCtwr{}"}


def parse(path, split):
    vals, counts, unit = {}, {}, None
    for ln in open(path):
        h = re.match(r"===== (\w+) / (\w+)\s+\(utterances=(\d+), disputed=(\d+), agreed=(\d+)\)", ln)
        if h:
            unit = h.group(2) if h.group(1) == split else None
            if unit:
                counts[unit] = (int(h.group(5)), int(h.group(4)))
            continue
        if not unit:
            continue
        m = re.match(r"(\S+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+\|", ln)
        if m and m.group(1) in MACRO:
            vals.setdefault(m.group(1), {})[unit] = (float(m.group(3)), float(m.group(2)))
    return vals, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    vals, counts = parse(a.path, a.split)
    # bold the largest DISPUTED value per granularity; a high agreed value is
    # not a better result, so the agreed columns are left plain
    best = {u: max(vals[s][u][1] for s in vals if u in vals[s]) for u in ("char", "word")}
    fmt = lambda v, u, c: (rf"\textbf{{{v:.3f}}}" if c == 1 and abs(v - best[u]) < 1e-12 else f"{v:.3f}")
    rows = []
    for s in ORDER:
        if s not in vals:
            continue
        ch, wd = vals[s]["char"], vals[s]["word"]
        rows.append(f"{MACRO[s]:<10} & {fmt(ch[0],'char',0)} & {fmt(ch[1],'char',1)} & "
                    f"{fmt(wd[0],'word',0)} & {fmt(wd[1],'word',1)} \\\\")
    ca, cd = counts["char"]
    wa, wd_ = counts["word"]
    print(rf"""\begin{{table}}[t]
\caption{{Mean wildcard-bypass probability at a fixed probe weight ($w=0$ for
both characters and words), pooled over 9 RESPIN languages ({a.split} set), for
units all validators agree on vs.\ units they dispute. A unit is disputed when
at least one validator differs from the reference and the validators do not all
supply the same alternative. The CTC row applies the same probe to CTC
posteriors; CTC was not trained with wildcard arcs. Char.: {ca/1000:.0f}k agree /
{cd/1000:.1f}k disputed; Word: {wa/1000:.0f}k agree / {wd_/1000:.1f}k disputed.}}
\label{{tab:disagree}}
\centering
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrr}}
\toprule
& \multicolumn{{2}}{{c}}{{\textbf{{Character}}}} & \multicolumn{{2}}{{c}}{{\textbf{{Word}}}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
& Agree & Disputed & Agree & Disputed \\
\midrule""")
    print("\n".join(rows))
    print(r"""\bottomrule
\end{tabular}\vspace{-3mm}
\end{table}""")


if __name__ == "__main__":
    main()
