#!/usr/bin/env python3
"""
Step 1 -- per-character validator-disagreement labels for RESPIN dev/test.

Each RESPIN dev/test utterance has 4 transcripts: "text" (the base transcript,
the one every model was trained and scored on) and three independent validator
transcripts (text_gts, text_nt, text_artpark). For every CHARACTER of "text"
this records how many validators disagree with it, found by a character-level
Levenshtein alignment of "text" against each validator:

  n_disp[i]  number of validators (0-3) that SUBSTITUTE or DELETE text char i
  n_ins[k]   number of validators that INSERT something before text char k
             (k = 0..len(text); k = len(text) is the end of the utterance)

Word-level labels follow the OTC word grouping the models use (a word opens at
a <space>, which LEADS its word):

  word_of[i]    word index of text char i
  word_n_disp   per word: number of validators disputing any char of the word
                or inserting inside it

Plus per-char unicode class (consonant, vowel_sign, virama, independent vowel,
other mark, space, digit, punct, other), used as a covariate, and per
validator whether its whole difference from "text" is spacing only.

An empty validator transcript counts as MISSING (that validator casts no
vote), not as a deletion of every character; n_validators records how many
validators actually annotated the utterance.

Characters are unicode code points, exactly how ESPnet's char tokenizer splits
the text, so position i here is token i of the training reference.

Output: labels/<lang>_<split>.pkl.gz  (list of per-utterance dicts)
Usage:  label_disagreement.py --lang kn --split test
"""
import argparse
import gzip
import os
import pickle
import unicodedata

DATA_ROOT = "/home3/Saurabh/exp/espnet/RESPIN/CR_CTC/data_respin"
OUT = os.path.dirname(os.path.abspath(__file__))
VALIDATORS = ["text_gts", "text_nt", "text_artpark"]

_VOWEL_SUFFIXES = (
    "LETTER A", "LETTER AA", "LETTER I", "LETTER II", "LETTER U", "LETTER UU",
    "LETTER VOCALIC R", "LETTER VOCALIC RR", "LETTER VOCALIC L",
    "LETTER VOCALIC LL", "LETTER E", "LETTER EE", "LETTER AI", "LETTER O",
    "LETTER OO", "LETTER AU", "LETTER CANDRA E", "LETTER CANDRA O",
    "LETTER SHORT E", "LETTER SHORT O",
)


def char_class(c):
    if c == " ":
        return "space"
    name = unicodedata.name(c, "")
    cat = unicodedata.category(c)
    if "VIRAMA" in name:
        return "virama"
    if "VOWEL SIGN" in name:
        return "vowel_sign"
    if cat in ("Mn", "Mc", "Me"):
        return "other_mark"          # anusvara, visarga, nukta, candrabindu ...
    if cat == "Nd":
        return "digit"
    if cat.startswith("P") or cat.startswith("S"):
        return "punct"
    if cat.startswith("L"):
        return "indep_vowel" if name.endswith(_VOWEL_SUFFIXES) else "consonant"
    return "other"


def align(a, b):
    """Levenshtein alignment of sequences a (reference) and b.

    Returns per-a-position op in {"=", "S", "D"} and a list ins[k] (len(a)+1)
    counting b-symbols inserted before a[k]. Ties prefer match/substitution,
    then deletion, then insertion, so the alignment is deterministic.
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
    ins = [0] * (n + 1)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops[i - 1] = "=" if a[i - 1] == b[j - 1] else "S"
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops[i - 1] = "D"
            i -= 1
        else:
            ins[i] += 1
            j -= 1
    return ops, ins


def word_index(chars):
    """OTC word grouping: a word opens at a space unless the current word is empty."""
    out, w, cur = [], -1, 0
    for c in chars:
        if w < 0 or (cur > 0 and c == " "):
            w += 1
            cur = 0
        out.append(w)
        cur += 1
    return out


def read_kaldi_text(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t", 1)
            if len(p) == 1:                      # tolerate "id text" with a space
                p = line.rstrip("\n").split(" ", 1)
            d[p[0]] = p[1] if len(p) == 2 else ""
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    a = ap.parse_args()

    d = f"{DATA_ROOT}/{a.lang}/IISc_RESPIN_{a.split}_{a.lang}"
    base = read_kaldi_text(f"{d}/text")
    vals = {v: read_kaldi_text(f"{d}/{v}") for v in VALIDATORS}

    recs = []
    for uid, text in base.items():
        chars = list(text)
        n = len(chars)
        n_disp = [0] * n
        disp_by = [0] * n                  # bit v set if validator v disputes char i
        n_ins = [0] * (n + 1)
        space_only = {}
        n_val = 0
        for vi, v in enumerate(VALIDATORS):
            other = vals[v].get(uid)
            # An EMPTY validator transcript is a missing annotation, not a
            # validator deleting every character (mr dev has 243 of these in
            # text_nt, bn/ch test 2 each).  It casts no vote.
            if other is None or not other.strip():
                continue
            n_val += 1
            ops, ins = align(chars, list(other))
            for i, op in enumerate(ops):
                if op in ("S", "D"):
                    n_disp[i] += 1
                    disp_by[i] |= 1 << vi
            for k, c in enumerate(ins):
                if c:
                    n_ins[k] += 1
            space_only[v] = (other != text) and other.replace(" ", "") == text.replace(" ", "")

        word_of = word_index(chars)
        n_words = (word_of[-1] + 1) if word_of else 0
        word_n_disp = [0] * n_words
        for vi in range(len(VALIDATORS)):
            hit = [False] * n_words
            for i in range(n):
                if disp_by[i] >> vi & 1:
                    hit[word_of[i]] = True
            other = vals[VALIDATORS[vi]].get(uid)
            if other is not None and other.strip():
                _, ins = align(chars, list(other))
                for k in range(1, n + 1):
                    if ins[k]:
                        hit[word_of[k - 1]] = True        # insertion at k belongs to the word before it
                if ins[0] and n_words:
                    hit[0] = True
            for w in range(n_words):
                word_n_disp[w] += hit[w]

        recs.append(dict(
            uid=uid, text=text, n_disp=n_disp, disp_by=disp_by, n_ins=n_ins,
            word_of=word_of, n_words=n_words, word_n_disp=word_n_disp,
            char_class=[char_class(c) for c in chars], space_only=space_only, n_validators=n_val,
            validators={v: vals[v].get(uid) for v in VALIDATORS},
        ))

    out = f"{OUT}/labels/{a.lang}_{a.split}.pkl.gz"
    with gzip.open(out, "wb") as f:
        pickle.dump(recs, f)
    n_chars = sum(len(r["n_disp"]) for r in recs)
    n_d = sum(1 for r in recs for x in r["n_disp"] if x > 0)
    n_utt_d = sum(1 for r in recs if any(r["n_disp"]) or any(r["n_ins"]))
    print(f"{a.lang} {a.split}: {len(recs)} utts ({n_utt_d} with any disagreement), "
          f"{n_chars} chars, {n_d} disputed ({100 * n_d / max(1, n_chars):.2f}%) -> {out}")


if __name__ == "__main__":
    main()
