# Independent validator transcriptions of the RESPIN dev and test sets

**Not in this repository yet.** The transcriptions exist and the analysis in the
paper is built on them, but their redistribution terms have to be confirmed with
the RESPIN project and the three validation vendors before they can be
published. This file records what will be added here and in what form, so that
the analysis pipeline and the paper can be read in the meantime.

## What will be added

The RESPIN corpus releases a single reference transcript per utterance. For the
disagreement analysis we additionally obtained **three independent
transcriptions** of the dev and test sets of nine languages from three external
validation vendors.

```
transcripts/<split>/<lang>/
    text          the official RESPIN reference (for alignment convenience)
    text_gts      validator 1
    text_nt       validator 2
    text_artpark  validator 3
```

Kaldi-style, one utterance per line, `<utt-id>\t<transcript>`, UTF-8, sorted by
utterance id. Utterance ids match the official RESPIN release, so the files drop
straight into a RESPIN data directory beside the distributed `text`.

| lang | test | dev |
|---|---|---|
| bh (Bhojpuri) | 2220 | 1500 |
| bn (Bengali) | 2174 | 1500 |
| ch (Chhattisgarhi) | 2234 | 1413 |
| hi (Hindi) | 2288 | 1539 |
| kn (Kannada) | 2161 | 1430 |
| mg (Magahi) | 2193 | 1431 |
| mr (Marathi) | 2170 | 1386 |
| mt (Maithili) | 2172 | 1409 |
| te (Telugu) | 2226 | 1438 |
| **total** | **19838** | **13046** |

A small number of validator transcripts are empty; these are treated as a
missing annotation (that validator casts no vote) rather than as a deletion of
every character. The handling is in `../analysis/disagreement/label_disagreement.py`.

## How "disputed" is defined in the paper

Each validator transcript is aligned to `text` by character-level Levenshtein
alignment with unit costs. A character is **disputed** when at least one
validator substitutes or deletes it *and* the validators do not all propose the
same alternative; a unit that all validators change identically is a unanimous
correction of the reference and is excluded from both the agreed and the
disputed pools. A word is disputed when a validator disputes any of its
characters or inserts inside it. Punctuation is excluded throughout, because it
is silent. Spacing differences fall out of the same mechanism, since the space
is a character: splitting a compound is an insertion, merging two words is a
deleted space.

`../analysis/disagreement/label_disagreement_mutual.py` implements this;
`../analysis/disagreement/README_mutual.md` describes it in more detail.

## What the disagreement looks like

It is overwhelmingly orthographic rather than lexical. Validators disagree on
the spelling of colloquial forms and on the spacing of compounds, not usually on
which word was spoken. Across the test sets a disputed word contains on average
0.9 disputed characters out of 4.8: half of all disputed words differ in a
single character, and a third contain no disputed character at all and are
disputed only because a validator inserted something inside them, usually a
space. The paper therefore speaks of transcription *variation*, not
transcription *errors*.
