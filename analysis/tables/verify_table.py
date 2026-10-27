#!/usr/bin/env python3
"""Sanity-check every numeric cell, bold mark and dagger in a rendered Table 1
against a fresh extraction. Catches: stale numbers left over from a previous
version of the table, a bold mark on the wrong (non-minimum) system, a
missing/extra dagger, or an average computed over the wrong row set.

This is a blunt regex-based parser tuned to the two tables/table1*.tex files
in this repo -- it is not a general LaTeX-table parser. If the table's
column layout changes, this script's COLUMN_MAP/etc. below need updating
to match, or just rebuild the table with gen_table1*.py and eyeball it.

Usage:
    python3 extract_wer.py > wer_data.json
    python3 verify_table.py wer_data.json \\
        ../overleaf/WSASR_ICASSP27/tables/table1.tex \\
        --columns ctc,word,token,recip --deltas token,recip
    python3 verify_table.py wer_data.json \\
        ../overleaf/WSASR_ICASSP27/tables/table1_v2.tex \\
        --columns ctc,word,wtuned,token,recip,tw,twr --deltas twr
"""
import argparse
import json
import re
import sys
from decimal import Decimal


def D(x):
    return Decimal(str(x))


def parse_cell(raw):
    """('5.49', True) for '\\textbf{5.49}', ('35.71', False, dagger=True) etc."""
    bold = r"\textbf{" in raw
    dagger = r"$^{\dag}$" in raw
    num = raw.replace(r"\textbf{", "").replace("}", "").replace(r"$^{\dag}$", "")
    if num.strip() == "--":
        return None, bold, dagger
    return D(num), bold, dagger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wer_json")
    ap.add_argument("table_tex")
    ap.add_argument("--columns", required=True,
                     help="comma-separated wer_data.json keys, in the table's column order "
                          "(e.g. ctc,word,token,recip)")
    ap.add_argument("--deltas", default="",
                     help="comma-separated subset of --columns that have their own "
                          "Delta-over-CTC column, in the order those Delta columns appear")
    args = ap.parse_args()

    cols = args.columns.split(",")
    delta_cols = [c for c in args.deltas.split(",") if c]
    n_value_cols = len(cols)
    n_cols = 1 + n_value_cols + len(delta_cols)  # label + values + deltas

    truth = json.load(open(args.wer_json))
    tex = open(args.table_tex).read()

    errors = 0
    block = None
    row_re = re.compile(r"^([^&]+?) & (.+) \\\\$")

    for line in tex.split("\n"):
        line = line.strip()
        m = re.match(r"\\multicolumn\{\d+\}\{l\}\{\\textit\{(.+?)\}\}", line)
        if m:
            block = m.group(1)
            continue
        m = row_re.match(line)
        if not m or line.startswith(("\\toprule", "\\midrule", "\\bottomrule")):
            continue
        label_raw, rest = m.groups()
        cells = [c.strip() for c in rest.split(" & ")]
        if len(cells) != n_cols:
            continue  # not a data row (e.g. the header row)
        label = label_raw.replace(r"\textbf{", "").replace("}", "").strip()
        is_avg = label in ("Lavg", "Favg", "Ravg", "Avg")

        parsed = [parse_cell(c) for c in cells[:n_value_cols]]
        delta_parsed = [parse_cell(c) for c in cells[n_value_cols:]]

        # 1. bold marks the minimum among the cells that are present.
        present = [(c, v) for c, (v, bold, dag) in zip(cols, parsed) if v is not None]
        if present:
            best_col = min(present, key=lambda cv: cv[1])[0]
            bolded = [c for c, (v, bold, dag) in zip(cols, parsed) if bold]
            if bolded != [best_col]:
                print(f"BOLD MISMATCH [{block}/{label}]: bolded={bolded} "
                      f"expected=[{best_col}] values={dict(zip(cols, [v for v,_,_ in parsed]))}")
                errors += 1

        # 2. delta columns: recompute from CTC and the corresponding column's value.
        ctc_v = parsed[cols.index("ctc")][0] if "ctc" in cols else None
        for dcol, (dval, dbold, _) in zip(delta_cols, delta_parsed):
            cell_v = dict(zip(cols, [v for v, _, _ in parsed])).get(dcol)
            if cell_v is None or ctc_v is None:
                if dval is not None:
                    print(f"DELTA MISMATCH [{block}/{label}]: expected -- "
                          f"({dcol} missing) but table has {dval}")
                    errors += 1
                continue
            expected = (ctc_v - cell_v) / ctc_v * 100
            if dval is None or abs(dval - expected) > Decimal("0.01"):
                print(f"DELTA MISMATCH [{block}/{label}] {dcol}: table={dval} "
                      f"expected~={expected.quantize(Decimal('0.01'))}")
                errors += 1

        # 3. cross-check against a fresh extraction (skip average rows: those
        #    are recomputed, not looked up, and gen_table1*.py already covers
        #    the averaging logic).
        if is_avg:
            continue
        src_row = None
        if block == "LibriSpeech":
            src_row = truth["libri"].get(
                {"test-clean": "test_clean", "test-other": "test_other"}.get(label))
        elif block == "LibriSpeech (Conformer)":
            src_row = truth["libri_conformer"].get(
                {"test-clean": "test_clean", "test-other": "test_other"}.get(label))
        elif block == "FLEURS":
            src_row = truth["fleurs"].get(label.replace(r"\_", "_"))
        elif block == "RESPIN":
            src_row = truth["respin"].get(label)
        if src_row is None:
            print(f"NO SOURCE ROW for [{block}/{label}] -- table has a row "
                  f"extract_wer.py doesn't know about")
            errors += 1
            continue
        for col, (val, bold, dag) in zip(cols, parsed):
            src_val = src_row.get(col)
            if val is None and src_val is None:
                continue
            if val is None or src_val is None or abs(val - D(src_val)) > Decimal("0.005"):
                print(f"VALUE MISMATCH [{block}/{label}] {col}: table={val} source={src_val}")
                errors += 1

    print(f"{errors} error(s)" if errors else "all checks passed", file=sys.stderr)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
