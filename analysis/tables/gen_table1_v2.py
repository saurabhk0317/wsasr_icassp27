#!/usr/bin/env python3
"""Generate the row data for tables/table1_v2.tex (wsasr_v2.tex): all seven
systems (CTC, OTC-W_e, OTC-W_e', OTC-T_e, OTC-T_H, OTC-TW_e, OTC-TW_H),
LibriSpeech with the E-Branchformer encoder only (no Conformer rows in this
version), corpus sub-averages (Lavg/Favg/Ravg) and a grand Avg over all 25
rows, with a single Delta column (relative reduction of OTC-TW_H over CTC).

OTC-TW_H (otc_token_word_recip) is still training for RESPIN ch/hi/kn on
another machine: those three rows, and any average that would otherwise mix
complete and incomplete rows (Ravg, Avg), print "--" for that column and for
Delta rather than a partial/misleading number.

Usage:
    python3 extract_wer.py > wer_data.json
    python3 gen_table1_v2.py wer_data.json

The printed block is meant to be pasted into tables/table1_v2.tex between
\\midrule and \\bottomrule -- do NOT \\input it from a separate file: a
\\input right before \\bottomrule breaks booktabs ("Misplaced \\noalign")
because of how TeX handles the file-boundary token right there.
"""
import json
import sys
from decimal import Decimal, ROUND_HALF_UP

# Column order: word-only pair, token-only pair, then the token+word hybrids.
COLS = ["ctc", "word", "wtuned", "token", "recip", "tw", "twr"]
DAGGER_COL = "word"

FLEURS_ORDER = ["ar_eg", "hy_am", "bn_in", "en_us", "ka_ge", "el_gr", "gu_in",
                "hi_in", "kn_in", "ml_in", "pa_in", "ru_ru", "ta_in", "te_in"]
RESPIN_ORDER = ["bh", "bn", "ch", "hi", "kn", "mg", "mr", "mt", "te"]
FLEURS_DAGGER = {"ml_in", "ta_in"}
RESPIN_DAGGER = {"kn", "mr", "te"}


def D(x):
    """Exact decimal from a float/JSON number: relies on Python's
    shortest-round-trip float repr (str(5.96) == '5.96'), so this recovers
    the original 2-decimal value exactly instead of its binary float noise."""
    return Decimal(str(x))


def fmt(x):
    return str(D(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def delta(c, x):
    return (D(c) - D(x)) / D(c) * 100


def mean(values):
    values = [D(v) for v in values]
    return sum(values) / len(values)


def build_row(label, v, dagger=False):
    vals = {c: v[c] for c in COLS if v.get(c) is not None}
    best = min(vals, key=lambda k: vals[k])
    cells = []
    for c in COLS:
        x = v.get(c)
        if x is None:
            cells.append("--")
            continue
        s = fmt(x)
        if c == DAGGER_COL and dagger:
            s += r"$^{\dag}$"
        if c == best:
            s = r"\textbf{" + s + "}"
        cells.append(s)
    dstr = fmt(delta(v["ctc"], v["twr"])) if v.get("twr") is not None else "--"
    return f"{label} & " + " & ".join(cells) + f" & {dstr} \\\\\n"


def corpus_avg(rows):
    avg = {}
    for c in COLS:
        present = [r[c] for r in rows if r.get(c) is not None]
        avg[c] = mean(present) if len(present) == len(rows) else None
        if len(present) not in (0, len(rows)):
            print(f"WARN: partial data for {c}: {len(present)}/{len(rows)}", file=sys.stderr)
    return avg


def main():
    d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "wer_data.json"))

    lines = []

    libri_rows = [d["libri"]["test_clean"], d["libri"]["test_other"]]
    lines.append(r"\multicolumn{9}{l}{\textit{LibriSpeech}} \\" + "\n")
    lines.append(build_row("test-clean", libri_rows[0]))
    lines.append(build_row("test-other", libri_rows[1]))
    lines.append("\n" + r"\midrule" + "\n\n")

    fleurs_rows = [d["fleurs"][c] for c in FLEURS_ORDER]
    lines.append(r"\multicolumn{9}{l}{\textit{FLEURS}} \\" + "\n")
    for code in FLEURS_ORDER:
        lines.append(build_row(code.replace("_", r"\_"), d["fleurs"][code],
                                dagger=(code in FLEURS_DAGGER)))
    lines.append("\n" + r"\midrule" + "\n\n")

    respin_rows = [d["respin"][c] for c in RESPIN_ORDER]
    lines.append(r"\multicolumn{9}{l}{\textit{RESPIN}} \\" + "\n")
    for code in RESPIN_ORDER:
        lines.append(build_row(code, d["respin"][code], dagger=(code in RESPIN_DAGGER)))
    lines.append("\n" + r"\midrule" + "\n\n")

    all_rows = libri_rows + fleurs_rows + respin_rows
    lines.append(build_row(r"\textbf{Lavg}", corpus_avg(libri_rows)))
    lines.append(build_row(r"\textbf{Favg}", corpus_avg(fleurs_rows)))
    lines.append(build_row(r"\textbf{Ravg}", corpus_avg(respin_rows)))
    lines.append(build_row(r"\textbf{Avg}", corpus_avg(all_rows)))

    print("".join(lines), end="")


if __name__ == "__main__":
    main()
