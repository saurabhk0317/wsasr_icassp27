#!/usr/bin/env python3
"""
Steps 3-5 -- is <star> placed preferentially where validators disagree, and
more so for star-trained (OTC) models than for the CTC control?

Inputs (from steps 1-2): labels/<lang>_<split>.pkl.gz and
posteriors/<lang>_<split>_<method>.pkl.gz.

UNITS
  char   every character of the base reference "text" (OOV chars dropped).
         disputed = at least one validator substitutes or deletes it.
         n_disp   = how many of the 3 validators do (0-3).
  word   every word (OTC word grouping). disputed = some validator disputes
         a char of it or inserts inside it.
  bound  every boundary before char i (self-loop analysis): a validator
         INSERTS something there.

STAR MEASURES (from step 2)
  p_tok_w{0,-2,-5}   P(char bypassed by <star>), token probe, bypass weight W
  p_word_w{0,-2,-6}  P(word bypassed by <star>), word probe
  ins_w{0,-2}        E[#<star> self-loop frames inserted at the boundary]
  lam / lam_word     weight-free critical bypass weight; LOWER = more star-prone
                     (score used below is -lam)
  own_*              the model's own granularity at its final training weights

ANALYSES, per (lang, split)
  1. Descriptives per model: mean star measure on disputed vs undisputed
     units, the ratio, and the same by n_disp (0/1/2/3) and by char class.
  2. AUC per model: how well the star measure separates disputed from
     undisputed units (0.5 = no association).
  3. OTC vs CTC (the control): for each OTC model,
       dAUC  = AUC(OTC) - AUC(CTC), paired over the same units, 95% CI from a
               cluster bootstrap over utterances.
       logit interaction: on the stacked (OTC, CTC) rows of the same units,
         logit P(star) = b0 + b1*disp + b2*otc + b3*disp*otc
                         + covariates + covariates*otc
         covariates: char class, word length, relative position, log char
         frequency, the model's utterance greedy CER.
         exp(b1) = adjusted OR(disputed) for CTC, exp(b1+b3) for OTC, and
         exp(b3) = how much MORE the OTC model's stars prefer disputed chars.
         95% CI and p from cluster-robust (sandwich) SEs, clustered by
         utterance -- rows of one utterance, and the paired OTC/CTC rows of
         one char, are not independent.
       star outcome = p > 0.5 at the probe weight (default tok w=0, word w=-2).
  4. Pooled across languages (when several are present): the same logit
     interaction with language fixed effects (and language x otc).

EXCLUDED CHARACTER CLASSES (--exclude_classes, default "punct"): punctuation
is silent, so a validator adding/removing/changing it ("H.P" vs "H_P") can
never be heard and no model can put a star there; keeping it would only dilute
every rate. Excluded chars are dropped from the char AND boundary tables.

Outputs: results/<tag>_*.csv and results/<tag>_report.md, tag = the split
(or --tag).

Usage:
  analyze.py --split test                      # every lang with posteriors
  analyze.py --split test --langs "kn te"
"""
import argparse
import glob
import gzip
import math
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CTRL = "ctc"
WORD_M = "p_word_w-2"   # headline word probe; overridden by --word_measure


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def load_tables(lang, split, exclude=("punct",), labels_dir="labels", drop_cons=False):
    labels = {r["uid"]: r for r in pickle.load(gzip.open(f"{HERE}/{labels_dir}/{lang}_{split}.pkl.gz"))}
    freq = Counter(c for r in labels.values() for c in r["text"])
    chars, words, bounds = [], [], []
    for path in sorted(glob.glob(f"{HERE}/posteriors/{lang}_{split}_*.pkl.gz")):
        method = re.match(rf"{lang}_{split}_(.+)\.pkl\.gz$", os.path.basename(path)).group(1)
        blob = pickle.load(gzip.open(path))
        own = blob["meta"]["own_regime"]
        for u in blob["utts"]:
            lab = labels.get(u["uid"])
            if lab is None:
                continue
            text, n = lab["text"], len(lab["text"])
            pr = u["probes"]
            # --drop_consensus: leave out units the validators mutually agree on
            cons_c = lab.get("cons_char") if drop_cons else None
            cons_w = lab.get("cons_word") if drop_cons else None
            cons_b = lab.get("cons_bound") if drop_cons else None
            wl = Counter(lab["word_of"][i] for i in range(n) if text[i] != " ")
            base = dict(lang=lang, split=split, method=method, uid=u["uid"], greedy_cer=u["greedy_cer"])
            for i in range(n):
                if u["oov"][i] or lab["char_class"][i] in exclude:
                    continue
                if cons_c and cons_c[i]:
                    continue
                row = dict(base, i=i, n_disp=lab["n_disp"][i], cls=lab["char_class"][i],
                           word_len=wl.get(lab["word_of"][i], 1), rel_pos=i / max(1, n - 1),
                           logfreq=math.log(freq[text[i]]), lam=float(u["lam_char"][i]))
                for w in (0, -2, -5):
                    row[f"p_tok_w{w}"] = float(pr[f"tok_bp_w{w}"]["bypass"][i])
                # "dual" (OTC-TW_e/OTC-TW_H) has token-level arcs at every char
                # exactly like "token", so its char-level own_p comes from the
                # same "bypass" array.
                if own and own[0] in ("token", "dual") and pr["own"]["bypass"] is not None:
                    row["own_p"] = float(pr["own"]["bypass"][i])
                chars.append(row)
            for k in range(n + 1):
                if k < n and (u["oov"][k] or lab["char_class"][k] in exclude):
                    continue
                if cons_b and cons_b[k]:
                    continue
                row = dict(base, k=k, n_ins=lab["n_ins"][k], rel_pos=k / max(1, n),
                           cls=lab["char_class"][k] if k < n else "end")
                for w in (0, -2):
                    row[f"ins_w{w}"] = float(pr[f"tok_sl_w{w}"]["ins"][k])
                bounds.append(row)
            for w_ in range(lab["n_words"]):
                if cons_w and cons_w[w_]:
                    continue
                row = dict(base, w=w_, n_disp=lab["word_n_disp"][w_], word_len=wl.get(w_, 1),
                           rel_pos=w_ / max(1, lab["n_words"] - 1), lam=float(u["lam_word"][w_]))
                for w in (0, -2, -6):
                    row[f"p_word_w{w}"] = float(pr[f"word_bp_w{w}"]["bypass"][w_])
                if own and own[0] == "word" and pr["own"]["bypass"] is not None:
                    row["own_p"] = float(pr["own"]["bypass"][w_])
                    if pr["own"]["ins"] is not None:
                        row["own_ins"] = float(pr["own"]["ins"][w_])
                # "dual"'s word-level arc is the SEPARATE word_bypass array
                # (its "bypass" array is the token-level component, used above
                # for the char table); no word-boundary self-loop of its own
                # (self-loops in "dual" are token-level, handled by ins_w*).
                elif own and own[0] == "dual" and pr["own"].get("word_bypass") is not None:
                    row["own_p"] = float(pr["own"]["word_bypass"][w_])
                words.append(row)
    to_df = lambda rows: pd.DataFrame(rows) if rows else pd.DataFrame()
    C, W, Bd = to_df(chars), to_df(words), to_df(bounds)
    for df in (C, W):
        if len(df):
            df["disp"] = (df["n_disp"] > 0).astype(int)
    if len(Bd):
        Bd["disp"] = (Bd["n_ins"] > 0).astype(int)
    return C, W, Bd


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def auc(y, s):
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def boot_dauc(df_a, df_b, key, score, n_boot, rng):
    """Paired cluster bootstrap of AUC(a) - AUC(b) over utterances."""
    m = df_a.merge(df_b, on=key + ["disp"], suffixes=("_a", "_b"))
    m = m[np.isfinite(m[f"{score}_a"]) & np.isfinite(m[f"{score}_b"])]
    y = m["disp"].to_numpy()
    sa, sb = m[f"{score}_a"].to_numpy(), m[f"{score}_b"].to_numpy()
    point = auc(y, sa) - auc(y, sb)
    groups = m["uid"].to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[g] for g in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        stats.append(roc_auc_score(yy, sa[idx]) - roc_auc_score(yy, sb[idx]))
    lo, hi = (np.percentile(stats, [2.5, 97.5]) if stats else (np.nan, np.nan))
    return point, float(lo), float(hi), len(m)


def design(df, cont, cats, with_otc):
    cols, names = [np.ones(len(df))], ["const"]
    cols.append(df["disp"].to_numpy(float)); names.append("disp")
    if with_otc:
        otc = df["otc"].to_numpy(float)
        cols += [otc, otc * df["disp"].to_numpy(float)]
        names += ["otc", "disp:otc"]
    for c in cont:
        v = df[c].to_numpy(float)
        v = (v - v.mean()) / (v.std() + 1e-12)
        cols.append(v); names.append(c)
        if with_otc:
            cols.append(v * otc); names.append(f"{c}:otc")
    for cat in cats:
        levels = sorted(df[cat].unique())[1:]
        for lv in levels:
            v = (df[cat] == lv).to_numpy(float)
            cols.append(v); names.append(f"{cat}={lv}")
            if with_otc:
                cols.append(v * otc); names.append(f"{cat}={lv}:otc")
    return np.column_stack(cols), names


def logit_cluster(X, y, groups):
    """Logistic fit + cluster-robust (sandwich) covariance, clustered by `groups`."""
    lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=5000, fit_intercept=False)
    lr.fit(X, y)
    b = lr.coef_.ravel()
    p = 1.0 / (1.0 + np.exp(-X @ b))
    H = (X * (p * (1 - p))[:, None]).T @ X
    Hinv = np.linalg.pinv(H)
    _, inv = np.unique(groups, return_inverse=True)
    S = np.zeros((inv.max() + 1, X.shape[1]))
    np.add.at(S, inv, X * (y - p)[:, None])
    G = S.shape[0]
    V = (G / max(1, G - 1)) * Hinv @ (S.T @ S) @ Hinv
    return b, np.sqrt(np.clip(np.diag(V), 0, None)), V


def or_row(b, V, names, terms):
    """exp(sum of coefficients) with a Wald 95% CI, for a linear combination."""
    c = np.zeros(len(b))
    for t in terms:
        c[names.index(t)] = 1.0
    est = float(c @ b)
    se = float(math.sqrt(max(0.0, c @ V @ c)))
    z = est / se if se > 0 else float("nan")
    p = float(2 * norm.sf(abs(z))) if se > 0 else float("nan")
    return math.exp(est), math.exp(est - 1.96 * se), math.exp(est + 1.96 * se), p


def interaction(df_otc, df_ctl, outcome, thr, cont, cats, extra_fe=None):
    a, b = df_otc.copy(), df_ctl.copy()
    a["otc"], b["otc"] = 1, 0
    d = pd.concat([a, b], ignore_index=True)
    d = d[np.isfinite(d[outcome])]
    y = (d[outcome] > thr).astype(int).to_numpy()
    if y.sum() < 20 or (len(y) - y.sum()) < 20:
        return None
    cats = list(cats) + (extra_fe or [])
    X, names = design(d, cont, cats, with_otc=True)
    bcoef, _, V = logit_cluster(X, y, d["uid"].to_numpy())
    return dict(
        n=len(d), n_star=int(y.sum()),
        or_ctc=or_row(bcoef, V, names, ["disp"]),
        or_otc=or_row(bcoef, V, names, ["disp", "disp:otc"]),
        ratio=or_row(bcoef, V, names, ["disp:otc"]),
    )


def per_model_or(df, outcome, thr, cont, cats):
    d = df[np.isfinite(df[outcome])]
    y = (d[outcome] > thr).astype(int).to_numpy()
    if y.sum() < 20 or (len(y) - y.sum()) < 20:
        return None
    X, names = design(d, cont, cats, with_otc=False)
    b, _, V = logit_cluster(X, y, d["uid"].to_numpy())
    return or_row(b, V, names, ["disp"])


# ---------------------------------------------------------------------------
def fmt_or(t):
    return "-" if t is None else f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}] p={t[3]:.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--langs", default="")
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude_classes", default="punct",
                    help='space-separated char classes to drop ("" keeps all)')
    ap.add_argument("--labels_dir", default="labels",
                    help='label directory ("labels_mutual" for the mutual-dissent definition)')
    ap.add_argument("--results_dir", default="results", help="output directory")
    ap.add_argument("--word_measure", default="p_word_w-2",
                    help='word-level probe used for the headline/report columns and the '
                         'OR/dAUC comparisons (e.g. "p_word_w0")')
    ap.add_argument("--drop_consensus", action="store_true",
                    help="drop units the validators mutually agree on instead of "
                         "treating them as agreed (needs labels_mutual)")
    a = ap.parse_args()
    global WORD_M
    WORD_M = a.word_measure
    rng = np.random.default_rng(a.seed)
    tag = a.tag or a.split
    RES = a.results_dir if os.path.isabs(a.results_dir) else f"{HERE}/{a.results_dir}"
    os.makedirs(RES, exist_ok=True)

    langs = a.langs.split() or sorted({re.match(r"([a-z]+)_", os.path.basename(p)).group(1)
                                       for p in glob.glob(f"{HERE}/posteriors/*_{a.split}_*.pkl.gz")})
    desc, bylevel, bycls, comp = [], [], [], []
    allC, allW = [], []
    lines = [f"# <star> placement vs validator disagreement ({a.split})", "",
             f"Excluded char classes: {a.exclude_classes or 'none'}. "
             "Star measures: see analyze.py docstring.", ""]
    rates, clsrates = [], []
    char_cont, char_cats = ["word_len", "rel_pos", "logfreq", "greedy_cer"], ["cls"]
    word_cont = ["word_len", "rel_pos", "greedy_cer"]

    for lang in langs:
        C, W, Bd = load_tables(lang, a.split, exclude=tuple(a.exclude_classes.split()),
                               labels_dir=a.labels_dir, drop_cons=a.drop_consensus)
        if not len(C):
            print(f"{lang}: no posteriors, skipped", file=sys.stderr)
            continue
        methods = sorted(C["method"].unique(), key=lambda m: (m != CTRL, m))
        allC.append(C); allW.append(W)
        n_c, n_d = C[C.method == methods[0]].shape[0], int(C[C.method == methods[0]]["disp"].sum())
        lines += [f"## {lang} ({n_c} chars, {n_d} disputed = {100 * n_d / n_c:.2f}%)", ""]

        # ---- 1-2. descriptives + AUC per model -------------------------------
        lines += ["| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |",
                  "|---|---|---|---|---|---|---|"]
        for m in methods:
            for unit, df, measures in (
                ("char", C[C.method == m], ["p_tok_w0", "p_tok_w-2", "p_tok_w-5", "neg_lam", "own_p"]),
                ("word", W[W.method == m], ["p_word_w0", "p_word_w-2", "p_word_w-6", "neg_lam", "own_p"]),
                ("bound", Bd[Bd.method == m], ["ins_w0", "ins_w-2"]),
            ):
                df = df.copy()
                df["neg_lam"] = -df["lam"] if "lam" in df else np.nan
                for meas in measures:
                    if meas not in df or df[meas].isna().all():
                        continue
                    s = df[meas].to_numpy(float)
                    y = df["disp"].to_numpy()
                    ok = np.isfinite(s)
                    md, mu = np.mean(s[ok & (y == 1)]), np.mean(s[ok & (y == 0)])
                    A = auc(y, s)
                    desc.append(dict(lang=lang, split=a.split, method=m, unit=unit, measure=meas,
                                     n=int(ok.sum()), n_disp=int((ok & (y == 1)).sum()),
                                     mean_disp=md, mean_undisp=mu,
                                     ratio=md / mu if mu > 0 else np.nan, auc=A))
                    if meas in ("p_tok_w0", WORD_M, "neg_lam", "ins_w0", "own_p"):
                        ratio = f"{md / mu:.1f}" if mu > 0 and meas != "neg_lam" else "-"
                        lines.append(f"| {m} | {unit} | {meas} | {md:.3f} | {mu:.3f} | {ratio} | {A:.3f} |")
                if unit in ("char", "word"):
                    meas = "p_tok_w0" if unit == "char" else WORD_M
                    for lv in range(4):
                        sub = df[df["n_disp"] == lv]
                        if len(sub):
                            bylevel.append(dict(lang=lang, split=a.split, method=m, unit=unit, measure=meas,
                                                n_disp=lv, n=len(sub), mean=float(sub[meas].mean())))
                if unit == "char":
                    for cl, sub in df.groupby("cls"):
                        for dv, ss in sub.groupby("disp"):
                            bycls.append(dict(lang=lang, split=a.split, method=m, cls=cl, disp=int(dv),
                                              n=len(ss), mean_p_tok_w0=float(ss["p_tok_w0"].mean()),
                                              mean_neg_lam=float((-ss["lam"]).mean())))
        lines.append("")

        # ---- share of units with star > 0.5 (easiest to read) ----------------
        lines += ["**Share of units with star > 0.5** (disputed / undisputed; and how many of "
                  "all starred units are disputed)", "",
                  "| model | char: disputed | char: undisputed | char: precision | "
                  f"word ({WORD_M}): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in methods:
            c, w_, bd = C[C.method == m], W[W.method == m], Bd[Bd.method == m]
            r = lambda df, col, dv: float((df[df.disp == dv][col] > 0.5).mean())
            n_sd, n_su = int((c[c.disp == 1].p_tok_w0 > 0.5).sum()), int((c[c.disp == 0].p_tok_w0 > 0.5).sum())
            prec = n_sd / max(1, n_sd + n_su)
            rates.append(dict(lang=lang, split=a.split, method=m,
                              char_disp=r(c, "p_tok_w0", 1), char_undisp=r(c, "p_tok_w0", 0),
                              char_n_star_disp=n_sd, char_n_star_undisp=n_su, char_precision=prec,
                              word_disp=r(w_, WORD_M, 1), word_undisp=r(w_, WORD_M, 0),
                              bound_disp=r(bd, "ins_w0", 1), bound_undisp=r(bd, "ins_w0", 0)))
            lines.append(f"| {m} | {r(c, 'p_tok_w0', 1):.3f} ({n_sd}) | {r(c, 'p_tok_w0', 0):.4f} ({n_su}) | "
                         f"{prec:.3f} | {r(w_, WORD_M, 1):.3f} | {r(w_, WORD_M, 0):.4f} | "
                         f"{r(bd, 'ins_w0', 1):.3f} | {r(bd, 'ins_w0', 0):.4f} |")
        lines.append("")
        dcls = C[(C.method == methods[0]) & (C.disp == 1)]["cls"].value_counts()
        lines += ["**Disputed chars with star > 0.5, by character class** (token probe, w=0)", "",
                  "| class | n disputed | " + " | ".join(methods) + " |",
                  "|---|---|" + "---|" * len(methods)]
        for cl, n_cl in dcls.items():
            vals = []
            for m in methods:
                s_ = C[(C.method == m) & (C.disp == 1) & (C.cls == cl)]
                v = float((s_.p_tok_w0 > 0.5).mean())
                vals.append(v)
                clsrates.append(dict(lang=lang, split=a.split, method=m, cls=cl, n_disputed=int(n_cl),
                                     share_star=v))
            lines.append(f"| {cl} | {n_cl} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
        lines.append("")

        # ---- 3. OTC vs CTC ---------------------------------------------------
        if CTRL in methods:
            lines += [f"**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; "
                      f"dAUC with {a.n_boot}-sample cluster bootstrap)", "",
                      "| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |",
                      "|---|---|---|---|---|---|---|---|"]
            for m in methods:
                if m == CTRL:
                    continue
                for unit, df, outcome, thr, key, cont, cats in (
                    ("char", C, "p_tok_w0", 0.5, ["uid", "i"], char_cont, char_cats),
                    ("word", W, WORD_M, 0.5, ["uid", "w"], word_cont, []),
                    ("bound", Bd, "ins_w0", 0.5, ["uid", "k"], ["rel_pos", "greedy_cer"], ["cls"]),
                ):
                    a_df, c_df = df[df.method == m].copy(), df[df.method == CTRL].copy()
                    res = interaction(a_df, c_df, outcome, thr, cont, cats)
                    dp = boot_dauc(a_df[key + ["disp", outcome]], c_df[key + ["disp", outcome]],
                                   key, outcome, a.n_boot, rng)
                    if "lam" in a_df:
                        a_df["neg_lam"], c_df["neg_lam"] = -a_df["lam"], -c_df["lam"]
                        dl = boot_dauc(a_df[key + ["disp", "neg_lam"]], c_df[key + ["disp", "neg_lam"]],
                                       key, "neg_lam", a.n_boot, rng)
                    else:
                        dl = (float("nan"),) * 4
                    row = dict(lang=lang, split=a.split, method=m, unit=unit, outcome=f"{outcome}>{thr}",
                               dauc_p=dp[0], dauc_p_lo=dp[1], dauc_p_hi=dp[2],
                               dauc_lam=dl[0], dauc_lam_lo=dl[1], dauc_lam_hi=dl[2])
                    if res:
                        for nm in ("or_ctc", "or_otc", "ratio"):
                            row.update({f"{nm}": res[nm][0], f"{nm}_lo": res[nm][1],
                                        f"{nm}_hi": res[nm][2], f"{nm}_p": res[nm][3]})
                        row.update(n=res["n"], n_star=res["n_star"])
                    comp.append(row)
                    lines.append(
                        f"| {m} | {unit} | {outcome}>{thr} | {fmt_or(res and res['or_ctc'])} | "
                        f"{fmt_or(res and res['or_otc'])} | {fmt_or(res and res['ratio'])} | "
                        f"{dp[0]:+.3f} [{dp[1]:+.3f}, {dp[2]:+.3f}] | "
                        + ("-" if math.isnan(dl[0]) else f"{dl[0]:+.3f} [{dl[1]:+.3f}, {dl[2]:+.3f}]") + " |")
            lines.append("")
        print(f"{lang}: done", file=sys.stderr)

    # ---- 4. pooled across languages -------------------------------------------
    if len(allC) > 1:
        C, W = pd.concat(allC, ignore_index=True), pd.concat(allW, ignore_index=True)
        lines += ["## Pooled across languages (language fixed effects)", "",
                  "| model | unit | OR CTC | OR model | ratio (model/CTC) | langs |", "|---|---|---|---|---|---|"]
        for m in sorted(set(C["method"]) - {CTRL}):
            for unit, df, outcome, cont, cats in (
                ("char", C, "p_tok_w0", char_cont, char_cats),
                ("word", W, WORD_M, word_cont, []),
            ):
                langs_m = sorted(set(df[df.method == m]["lang"]) & set(df[df.method == CTRL]["lang"]))
                if len(langs_m) < 2:
                    continue
                sub = df[df["lang"].isin(langs_m)]
                res = interaction(sub[sub.method == m], sub[sub.method == CTRL], outcome, 0.5,
                                  cont, cats, extra_fe=["lang"])
                if res:
                    comp.append(dict(lang="pooled", split=a.split, method=m, unit=unit,
                                     outcome=f"{outcome}>0.5", or_ctc=res["or_ctc"][0], or_otc=res["or_otc"][0],
                                     ratio=res["ratio"][0], ratio_lo=res["ratio"][1], ratio_hi=res["ratio"][2],
                                     ratio_p=res["ratio"][3], n=res["n"], n_star=res["n_star"]))
                    lines.append(f"| {m} | {unit} | {fmt_or(res['or_ctc'])} | {fmt_or(res['or_otc'])} | "
                                 f"{fmt_or(res['ratio'])} | {' '.join(langs_m)} |")
        lines.append("")

    for name, rows in (("descriptives", desc), ("by_ndisp", bylevel), ("by_class", bycls), ("vs_ctc", comp),
                       ("star_rates", rates), ("class_star_rates", clsrates)):
        pd.DataFrame(rows).to_csv(f"{RES}/{tag}_{name}.csv", index=False)
    with open(f"{RES}/{tag}_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {a.results_dir}/{tag}_{{descriptives,by_ndisp,by_class,vs_ctc,star_rates,class_star_rates}}.csv "
          f"and {a.results_dir}/{tag}_report.md")


if __name__ == "__main__":
    main()
