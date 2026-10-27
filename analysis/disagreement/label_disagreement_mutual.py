#!/usr/bin/env python3
"""
Step 1 (variant) -- validator-disagreement labels under the MUTUAL-DISSENT
definition of "disputed".

Difference from label_disagreement.py
-------------------------------------
The original definition marks a unit disputed when AT LEAST ONE validator
substitutes/deletes it.  That also flags units where the reference is simply
wrong and all three validators independently supply the SAME correction --
a consensus correction, which is not an ambiguous unit at all.

Here a unit is disputed only if
    (a) at least one validator differs from the reference, AND
    (b) the validators do not all give the identical realisation.

Because (b) can only fail when EVERY validator differs from the reference
(if one validator sides with the reference while another does not, the
validators already disagree among themselves), this rule removes exactly the
n_disp == n_validators units on which the validators mutually agree.
Consensus units are folded into the AGREED group (n_disp set to 0), matching
"should not be considered as disputed"; they are also flagged so that
analyze.py --drop_consensus can drop them instead.

Consensus needs at least two validators to be observable, so utterances with
a single (or no) annotator keep the original labelling; n_validators records
how many actually annotated.

The realisation compared across validators is
  char      the aligned validator character, or None if that validator
            deletes the reference character
  word      the aligned validator substring for the whole word, including
            characters inserted inside it (same insertion-to-word assignment
            the original script uses: an insertion at boundary k belongs to
            the word of char k-1, and boundary 0 to word 0)
  boundary  the inserted string at that boundary

Schema matches labels/, so analyze.py runs on it unchanged, plus:
  n_disp_raw, word_n_disp_raw, n_ins_raw   the original counts
  cons_char, cons_word, cons_bound         1 where a consensus unit was
                                           reclassified as agreed

Output: labels_mutual/<lang>_<split>.pkl.gz
Usage:  label_disagreement_mutual.py --lang kn --split test
"""
import argparse
import gzip
import os
import pickle

from label_disagreement import DATA_ROOT, VALIDATORS, char_class, read_kaldi_text, word_index

OUT = os.path.dirname(os.path.abspath(__file__))


def align_full(a, b):
    """Levenshtein alignment of a (reference) against b.

    Same DP and tie-breaking as label_disagreement.align, so the ops agree
    exactly, but it also returns what b puts at each position of a.

    ops  per a-position, "=", "S" or "D"
    sym  per a-position, the aligned b character, or None when deleted
    ins  per boundary k (len(a)+1), the string b inserts before a[k]
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j - 1] + (ai != b[j - 1]), prev[j] + 1, row[j - 1] + 1)
    ops = [None] * n
    sym = [None] * n
    ins = [[] for _ in range(n + 1)]
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops[i - 1] = "=" if a[i - 1] == b[j - 1] else "S"
            sym[i - 1] = b[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops[i - 1] = "D"
            sym[i - 1] = None
            i -= 1
        else:
            ins[i].append(b[j - 1])
            j -= 1
    return ops, sym, ["".join(reversed(x)) for x in ins]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    a = ap.parse_args()

    d = f"{DATA_ROOT}/{a.lang}/IISc_RESPIN_{a.split}_{a.lang}"
    base = read_kaldi_text(f"{d}/text")
    vals = {v: read_kaldi_text(f"{d}/{v}") for v in VALIDATORS}

    recs = []
    n_cons_c = n_cons_w = n_cons_b = 0
    for uid, text in base.items():
        chars = list(text)
        n = len(chars)
        word_of = word_index(chars)
        n_words = (word_of[-1] + 1) if word_of else 0

        present = []                      # (validator index, ops, sym, ins)
        space_only = {}
        for vi, v in enumerate(VALIDATORS):
            other = vals[v].get(uid)
            if other is None or not other.strip():
                continue                  # empty transcript = missing annotation
            ops, sym, ins = align_full(chars, list(other))
            present.append((vi, ops, sym, ins))
            space_only[v] = (other != text) and other.replace(" ", "") == text.replace(" ", "")
        n_val = len(present)
        can_cons = n_val >= 2             # consensus is unobservable below this

        # ---- characters -----------------------------------------------------
        n_disp_raw = [0] * n
        disp_by = [0] * n
        for vi, ops, _s, _i in present:
            for i, op in enumerate(ops):
                if op in ("S", "D"):
                    n_disp_raw[i] += 1
                    disp_by[i] |= 1 << vi
        cons_char = [0] * n
        n_disp = list(n_disp_raw)
        for i in range(n):
            if n_disp_raw[i] and can_cons and len({s[i] for _v, _o, s, _i in present}) == 1:
                cons_char[i] = 1          # every validator supplies the same correction
                n_disp[i] = 0
        n_cons_c += sum(cons_char)

        # ---- boundaries (validator insertions) -------------------------------
        n_ins_raw = [0] * (n + 1)
        for _vi, _o, _s, ins in present:
            for k, s in enumerate(ins):
                if s:
                    n_ins_raw[k] += 1
        cons_bound = [0] * (n + 1)
        n_ins = list(n_ins_raw)
        for k in range(n + 1):
            if n_ins_raw[k] and can_cons and len({ins[k] for _v, _o, _s, ins in present}) == 1:
                cons_bound[k] = 1
                n_ins[k] = 0
        n_cons_b += sum(cons_bound)

        # ---- words -----------------------------------------------------------
        # realisation of a word = aligned characters plus insertions inside it,
        # with an insertion at boundary k assigned to the word of char k-1
        word_n_disp_raw = [0] * n_words
        word_real = [[] for _ in range(n_words)]
        for _vi, ops, sym, ins in present:
            hit = [False] * n_words
            parts = [[] for _ in range(n_words)]
            if n_words and ins[0]:
                parts[0].append(ins[0])
                hit[0] = True
            for i in range(n):
                w = word_of[i]
                if ops[i] in ("S", "D"):
                    hit[w] = True
                if sym[i] is not None:
                    parts[w].append(sym[i])
                if ins[i + 1]:
                    parts[w].append(ins[i + 1])
                    hit[w] = True
            for w in range(n_words):
                word_n_disp_raw[w] += hit[w]
                word_real[w].append("".join(parts[w]))
        cons_word = [0] * n_words
        word_n_disp = list(word_n_disp_raw)
        for w in range(n_words):
            if word_n_disp_raw[w] and can_cons and len(set(word_real[w])) == 1:
                cons_word[w] = 1
                word_n_disp[w] = 0
        n_cons_w += sum(cons_word)

        recs.append(dict(
            uid=uid, text=text, n_disp=n_disp, disp_by=disp_by, n_ins=n_ins,
            word_of=word_of, n_words=n_words, word_n_disp=word_n_disp,
            char_class=[char_class(c) for c in chars], space_only=space_only,
            n_validators=n_val,
            validators={v: vals[v].get(uid) for v in VALIDATORS},
            n_disp_raw=n_disp_raw, word_n_disp_raw=word_n_disp_raw, n_ins_raw=n_ins_raw,
            cons_char=cons_char, cons_word=cons_word, cons_bound=cons_bound,
        ))

    os.makedirs(f"{OUT}/labels_mutual", exist_ok=True)
    out = f"{OUT}/labels_mutual/{a.lang}_{a.split}.pkl.gz"
    with gzip.open(out, "wb") as f:
        pickle.dump(recs, f)
    n_chars = sum(len(r["n_disp"]) for r in recs)
    n_old = sum(1 for r in recs for x in r["n_disp_raw"] if x > 0)
    n_new = sum(1 for r in recs for x in r["n_disp"] if x > 0)
    n_wo = sum(1 for r in recs for x in r["word_n_disp_raw"] if x > 0)
    n_wn = sum(1 for r in recs for x in r["word_n_disp"] if x > 0)
    print(f"{a.lang} {a.split}: {len(recs)} utts, {n_chars} chars | "
          f"char disputed {n_old} -> {n_new} (-{n_old - n_new}, {100 * n_cons_c / max(1, n_old):.1f}% consensus) | "
          f"word disputed {n_wo} -> {n_wn} (-{n_wo - n_wn}) | "
          f"boundary consensus {n_cons_b} -> {out}")


if __name__ == "__main__":
    main()
