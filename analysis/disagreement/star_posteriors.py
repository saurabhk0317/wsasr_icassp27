#!/usr/bin/env python3
"""
Step 2 -- where would a trained model put <star> instead of the reference?

Loads one trained model (its averaged checkpoint, valid.loss.ave.pth), runs the
held-out dev/test audio through it, and for every utterance builds the OTC
training graph for its "text" reference -- the reference the model was trained
and scored on -- then measures, per reference position, how much posterior
mass the lattice routes through a <star> arc instead of the real token.

The graph is a PROBE built here, independently of the model's own training
granularity, so every model (the CTC control included) is measured with the
same instrument:

  tok_bp_w{W}   token graph, bypass arcs only, weight W  -> P(char i bypassed)
  tok_sl_w{W}   token graph, self-loop arcs only, weight W
                                                        -> E[#star frames inserted before char i]
  word_bp_w{W}  word graph (OTC word grouping), bypass only, weight W
                                                        -> P(word w bypassed)
  own           OTC models only: the model's OWN granularity and its final
                arc weights (the regime training ended in), all its arcs.
                For otc_granularity="dual" (OTC-TW_e / OTC-TW_H: token bypass
                + self-loop at every token, PLUS a separate word-level bypass
                for words of >=2 tokens -- see ctc_otc_recip.py), "own" builds
                that exact three-weight graph, and its result is split into a
                CHAR-indexed bypass array (the token-bypass component, same
                shape as the token probes above) and a WORD-indexed
                word_bypass array (the word-bypass component, same shape as
                the word probes above).

Posteriors are exact: every transcript arc carries an integer tag (arc kind,
position) that k2 propagates through compose/intersect, and arc posteriors are
summed by tag. (Checked against a remove-one-arc brute force; see
sanity_checks.py.)

Weight-free per-position statistic, from the Viterbi path of the stars-OFF
(plain CTC) graph:
  lam_char[i] / lam_word[w]  critical bypass weight: over the position's frame
      span, R = score of the real path, B = best one-run-of-<star> path with
      blanks around it; lam = R - B. The bypass beats the reference iff the
      bypass weight exceeds lam, so LOWER lam = more star-prone.

<star> emission for every model, incl. CTC (whose training had no star):
      log P(*|t) = logsumexp_{v != blank} log p(v|t) - log(V - 1)
exactly as in ctc_otc_recip.py / ctc_otc_clean.py.

READ-ONLY on the model directory. Output:
  posteriors/<lang>_<split>_<method>.pkl.gz

Usage:
  star_posteriors.py --model_dir <exp_final/respin/<lang>_s5/asr_...> \
      --lang kn --split test --method otc_token [--device cuda] [--max_utts N]
"""
import argparse
import gzip
import math
import os
import pickle
import re
import sys
import time

import numpy as np
import soundfile as sf
import torch
import yaml
import k2

DATA_ROOT = "/home3/Saurabh/exp/espnet/RESPIN/CR_CTC/data_respin"
OUT = os.path.dirname(os.path.abspath(__file__))

KIND_REAL, KIND_BYPASS, KIND_SELFLOOP, KIND_WORD_BYPASS = 1, 2, 3, 4
# Each entry: (name, gran, token_bypass_w, self_loop_w, word_bypass_w).
# word_bypass_w is only meaningful for gran="dual"; every fixed probe below
# leaves it None (fixed probes are always plain token or word graphs).
PROBES = (
    [("tok_bp_w%g" % w, "token", w, None, None) for w in (0.0, -2.0, -5.0)]
    + [("tok_sl_w%g" % w, "token", None, w, None) for w in (0.0, -2.0)]
    + [("word_bp_w%g" % w, "word", w, None, None) for w in (0.0, -2.0, -6.0)]
)


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
def load_model(model_dir, device):
    run_sh = open(f"{model_dir}/run.sh").read()
    m = re.search(r"--asr_task\s+(\S+)", run_sh)
    task = (m.group(1).strip("'\"") if m else "asr")
    mod = __import__(f"espnet2.tasks.{task}", fromlist=["ASRTask"])
    model, args = mod.ASRTask.build_model_from_file(
        f"{model_dir}/config.yaml", f"{model_dir}/valid.loss.ave.pth", str(device))
    model.eval()
    return model, args, task


def own_regime(model_dir):
    """(granularity, token_bypass_w, self_loop_w, word_bypass_w) the model ended
    training with, or None. word_bypass_w is None unless granularity="dual"
    (OTC-TW_e / OTC-TW_H), which is the only mode that has a separate
    word-level bypass alongside the token-level arcs.
    """
    cfg = yaml.safe_load(open(f"{model_dir}/config.yaml"))
    cc = cfg.get("ctc_conf") or {}
    if cfg.get("ctc_variant") in ("builtin", "k2"):
        return None
    sched = cc.get("arc_schedule", "epoch")
    if sched in ("none", "label_smoothing"):
        return None
    if not (cc.get("allow_bypass_arc", True) or cc.get("allow_self_loop_arc", True)):
        return None
    gran = cc.get("otc_granularity", "token")
    if gran not in ("token", "word", "dual"):
        return None
    byp = sl = word_byp = None
    log = f"{model_dir}/train.log"
    if os.path.isfile(log):
        last = None
        for line in open(log, errors="replace"):
            if "[OTCRECIP EPOCH]" in line or "[OTCSCHED EPOCH]" in line:
                last = line
        if last:
            mb = re.search(r"bypass_w=(-?[\d.]+)", last)
            ms = re.search(r"self_loop_w=(-?[\d.]+)", last)
            mw = re.search(r"word_bypass_w=(-?[\d.]+)", last)
            byp = float(mb.group(1)) if mb else None
            sl = float(ms.group(1)) if ms else None
            word_byp = float(mw.group(1)) if mw else None
    if byp is None and sched == "epoch":
        # Fallback only (real models are read from train.log above): does NOT
        # account for star_start_epoch, since the log line above always does.
        e = int(cfg.get("max_epoch", 1))
        byp = cc.get("initial_bypass_weight", -25.0) * cc.get("bypass_weight_decay", 0.9) ** (e - 1)
        sl = cc.get("initial_self_loop_weight", -5.0) * cc.get("self_loop_weight_decay", 0.9) ** (e - 1)
        if gran == "dual" and "initial_word_bypass_weight" in cc:
            word_byp = cc["initial_word_bypass_weight"] * cc.get("word_bypass_weight_decay", 0.9) ** (e - 1)
    if byp is None:
        return None
    if not cc.get("allow_bypass_arc", True):
        byp = None
        word_byp = None
    if not cc.get("allow_self_loop_arc", True):
        sl = None
    if gran == "dual" and word_byp is None:
        return None                        # dual needs its word arc to be meaningful
    return gran, byp, sl, word_byp


# ---------------------------------------------------------------------------
# probe graphs
# ---------------------------------------------------------------------------
class Probe:
    def __init__(self, vocab_size, space_id, device):
        self.star = vocab_size                     # <star> is appended after the real vocab
        self.space = space_id
        self.device = device
        self.topo = k2.ctc_topo(vocab_size, modified=False).to(device)

    def groups(self, ids, gran):
        if gran == "token":
            return [[t] for t in ids]
        out, cur = [], []
        for t in ids:
            if cur and t == self.space:
                out.append(cur)
                cur = []
            cur.append(t)
        if cur:
            out.append(cur)
        return out

    def fsa(self, ids, gran, byp, sl, word_byp=None):
        """Transcript FSA with diag_kind / diag_pos tags (pos is 1-based; 0 = untagged).

        gran="dual" (OTC-TW_e / OTC-TW_H): token-level self-loop + bypass at
        EVERY token (exactly gran="token" below), PLUS one extra word-level
        bypass arc per word of >=2 tokens, spanning the same states the word's
        tokens occupy -- byte-for-byte the same construction as
        OtcGraphCompiler(granularity="dual") in ctc_otc_recip.py. diag_pos for
        the word-bypass arcs is the WORD index (1-based), which overlaps
        numerically with the token diag_pos range; that's fine since kind is
        always filtered first (KIND_WORD_BYPASS never collides with
        KIND_BYPASS/KIND_SELFLOOP/KIND_REAL when selecting arcs by kind).
        """
        if not ids:
            return k2.Fsa.from_str("0 1 -1 0 0 0.0\n1", aux_label_names=["diag_kind", "diag_pos"])
        arcs = []
        state = 0
        token_groups = self.groups(ids, "token" if gran == "dual" else gran)
        for g, grp in enumerate(token_groups):
            start = state
            if sl is not None:
                arcs.append((state, f"{state} {state} {self.star} {KIND_SELFLOOP} {g + 1} {sl}"))
            for tok in grp:
                arcs.append((state, f"{state} {state + 1} {tok} {KIND_REAL} {g + 1} 0.0"))
                state += 1
            if byp is not None:
                arcs.append((start, f"{start} {state} {self.star} {KIND_BYPASS} {g + 1} {byp}"))
        if sl is not None:
            arcs.append((state, f"{state} {state} {self.star} {KIND_SELFLOOP} {len(token_groups) + 1} {sl}"))
        if gran == "dual" and word_byp is not None:
            word_start = 0
            for w, word in enumerate(self.groups(ids, "word")):
                word_end = word_start + len(word)
                if len(word) >= 2:
                    arcs.append((word_start,
                                f"{word_start} {word_end} {self.star} {KIND_WORD_BYPASS} {w + 1} {word_byp}"))
                word_start = word_end
            assert word_start == state, (word_start, state)
        arcs.append((state, f"{state} {state + 1} -1 0 0 0.0"))
        lines = [a[1] for a in sorted(arcs, key=lambda a: a[0])] + [str(state + 1)]
        return k2.arc_sort(k2.Fsa.from_str("\n".join(lines), aux_label_names=["diag_kind", "diag_pos"]))

    def lattice(self, fsas, dense):
        vec = k2.create_fsa_vec(fsas).to(self.device)
        vec = k2.arc_sort(k2.remove_epsilon_and_add_self_loops(vec))
        g = k2.compose(self.topo, vec, treat_epsilons_specially=False)
        return k2.intersect_dense(g, dense, 100.0)


def arc_utt(fsa_vec):
    shape = fsa_vec.arcs.shape()
    return shape.row_ids(1).long()[shape.row_ids(2).long()]


def by_pos(post, utt, kind, pos, want_kind, B, P):
    """(B, P) sum of arc posteriors of the given kind, indexed by utterance and tag pos."""
    m = kind == want_kind
    out = torch.zeros(B * P, dtype=torch.float64, device=post.device)
    out.index_add_(0, (utt * P + pos)[m], post[m].double())
    return out.view(B, P).cpu().numpy()


def max_contiguous_sum(d):
    c = np.concatenate([[0.0], np.cumsum(d, dtype=np.float64)])
    return float(np.max(c[1:] - np.minimum.accumulate(c[:-1])))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--method", required=True, help="label for outputs, e.g. ctc / otc_token")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_sec", type=float, default=120.0, help="audio seconds per batch")
    ap.add_argument("--max_utts", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out_path = a.out or f"{OUT}/posteriors/{a.lang}_{a.split}_{a.method}.pkl.gz"
    t0 = time.time()
    device = torch.device(a.device)
    model, args, task = load_model(a.model_dir, device)
    tl = list(args.token_list)
    V = len(tl)
    space_id = tl.index("<space>")
    unk_id = tl.index("<unk>") if "<unk>" in tl else None
    probe = Probe(V, space_id, device)
    own = own_regime(a.model_dir)
    probes = list(PROBES)
    if own is not None:
        probes.append(("own", own[0], own[1], own[2], own[3]))
    print(f"[{a.method}] task={task} V={V} own_regime={own} device={device}", flush=True)

    labels = pickle.load(gzip.open(f"{OUT}/labels/{a.lang}_{a.split}.pkl.gz"))
    d = f"{DATA_ROOT}/{a.lang}/IISc_RESPIN_{a.split}_{a.lang}"
    wav = dict(l.split(None, 1) for l in open(f"{d}/wav.scp") if l.strip())
    dur = {}
    if os.path.isfile(f"{d}/utt2dur"):
        dur = {k: float(v) for k, v in (l.split() for l in open(f"{d}/utt2dur") if l.strip())}
    utts = [r for r in labels if r["uid"] in wav]
    if a.max_utts:
        utts = utts[: a.max_utts]
    utts.sort(key=lambda r: -dur.get(r["uid"], 0.0))

    # batches of ~batch_sec audio, longest first
    batches, cur, cur_s = [], [], 0.0
    for r in utts:
        s = dur.get(r["uid"], 5.0)
        if cur and cur_s + s > a.batch_sec:
            batches.append(cur)
            cur, cur_s = [], 0.0
        cur.append(r)
        cur_s += s
    if cur:
        batches.append(cur)

    results = []
    n_done = 0
    for batch in batches:
        waves = [sf.read(wav[r["uid"]].strip())[0].astype(np.float32) for r in batch]
        lens = torch.tensor([len(w) for w in waves])
        x = torch.zeros(len(waves), int(lens.max()))
        for i, w in enumerate(waves):
            x[i, : len(w)] = torch.from_numpy(w)
        with torch.no_grad():
            enc, enc_lens = model.encode(x.to(device), lens.to(device))
            lp = model.ctc.log_softmax(enc).float()                     # (B, T, V)
            star = torch.logsumexp(lp[..., 1:], dim=-1, keepdim=True) - math.log(V - 1)
            lps = torch.cat([lp, star], dim=-1)                        # (B, T, V+1)

        # k2 wants supervisions sorted by length, longest first
        order = torch.argsort(enc_lens.cpu(), descending=True).tolist()
        B = len(batch)
        T = enc_lens.cpu().tolist()
        ids_all, oov_all = [], []
        for r in batch:
            ids, oov = [], []
            for c in r["text"]:
                tok = "<space>" if c == " " else c
                if tok in tl:
                    ids.append(tl.index(tok)); oov.append(False)
                else:
                    ids.append(unk_id if unk_id is not None else 1); oov.append(True)
            ids_all.append(ids)
            oov_all.append(oov)
        sup = torch.tensor([[j, 0, T[order[j]]] for j in range(B)], dtype=torch.int32)
        dense = k2.DenseFsaVec(lps[order], sup)
        ord_ids = [ids_all[i] for i in order]

        per = [dict(probes={}) for _ in range(B)]
        for name, gran, byp, sl, word_byp in probes:
            try:
                lat = probe.lattice([probe.fsa(ids, gran, byp, sl, word_byp) for ids in ord_ids], dense)
            except Exception as e:                                    # pragma: no cover
                print(f"  lattice failed ({name}): {e}", file=sys.stderr)
                continue
            post = torch.nan_to_num(lat.get_arc_post(use_double_scores=True, log_semiring=True).exp(), nan=0.0)
            utt = arc_utt(lat).to(post.device)
            kind = lat.diag_kind.long().to(post.device)
            pos = lat.diag_pos.long().to(post.device)
            P = max(len(i) for i in ord_ids) + 2
            byp_m = by_pos(post, utt, kind, pos, KIND_BYPASS, B, P)
            sl_m = by_pos(post, utt, kind, pos, KIND_SELFLOOP, B, P)
            wbyp_m = by_pos(post, utt, kind, pos, KIND_WORD_BYPASS, B, P) if word_byp is not None else None
            for j, oi in enumerate(order):
                n_tok = len(ord_ids[j]) if gran == "dual" else len(probe.groups(ord_ids[j], gran))
                n_word = len(probe.groups(ord_ids[j], "word"))
                per[oi]["probes"][name] = dict(
                    gran=gran,
                    bypass=None if byp is None else np.round(byp_m[j, 1:n_tok + 1], 4).astype(np.float32),
                    ins=None if sl is None else np.round(sl_m[j, 1:n_tok + 2], 4).astype(np.float32),
                    word_bypass=(None if wbyp_m is None
                                else np.round(wbyp_m[j, 1:n_word + 1], 4).astype(np.float32)),
                )

        # weight-free critical weights from the stars-off Viterbi path
        lat0 = probe.lattice([probe.fsa(ids, "token", None, None) for ids in ord_ids], dense)
        best = k2.shortest_path(lat0, use_double_scores=True)
        b_utt = arc_utt(best).cpu().numpy()
        b_lab = best.labels.long().cpu().numpy()
        b_sc = best.scores.float().cpu().numpy()
        b_pos = best.diag_pos.long().cpu().numpy()
        blank = lps[order][..., 0].cpu().numpy()
        starc = lps[order][..., V].cpu().numpy()
        for j, oi in enumerate(order):
            sel = b_utt == j
            lab, sc, ps = b_lab[sel], b_sc[sel], b_pos[sel]
            n = len(ord_ids[j])
            start, end, cur = {}, {}, 0
            for t in range(min(T[oi], len(lab))):
                if ps[t] > 0:
                    start.setdefault(ps[t], t)
                    cur = ps[t]
                if lab[t] > 0 and cur > 0:
                    end[cur] = t + 1
            lam_c = np.full(n, np.nan, dtype=np.float32)
            spans = {}
            for i in range(1, n + 1):
                if i in start and i in end and end[i] > start[i]:
                    s, e = start[i], end[i]
                    spans[i - 1] = (s, e)
                    lb = blank[j, s:e]
                    lam_c[i - 1] = float(sc[s:e].sum()) - (float(lb.sum()) + max_contiguous_sum(starc[j, s:e] - lb))
            word_of = batch[oi]["word_of"]
            n_w = batch[oi]["n_words"]
            lam_w = np.full(n_w, np.nan, dtype=np.float32)
            for w in range(n_w):
                cs = [i for i in range(n) if word_of[i] == w and i in spans]
                if not cs:
                    continue
                s, e = min(spans[i][0] for i in cs), max(spans[i][1] for i in cs)
                lb = blank[j, s:e]
                lam_w[w] = float(sc[s:e].sum()) - (float(lb.sum()) + max_contiguous_sum(starc[j, s:e] - lb))
            per[oi]["lam_char"] = np.round(lam_c, 3)
            per[oi]["lam_word"] = np.round(lam_w, 3)

        # greedy CER against "text" (utterance-level difficulty covariate)
        hyp = lp.argmax(-1).cpu().numpy()
        for i, r in enumerate(batch):
            h = hyp[i, : T[i]]
            col, prev = [], -1
            for v in h:
                if v != prev and v != 0:
                    col.append(int(v))
                prev = v
            ref = ids_all[i]
            dp = list(range(len(col) + 1))
            for a_ in ref:
                nd = [dp[0] + 1]
                for jj, b_ in enumerate(col, 1):
                    nd.append(min(dp[jj] + 1, nd[jj - 1] + 1, dp[jj - 1] + (a_ != b_)))
                dp = nd
            per[i].update(uid=r["uid"], T=T[i], oov=np.array(oov_all[i]),
                          greedy_cer=dp[-1] / max(1, len(ref)))
            results.append(per[i])
        n_done += B
        print(f"  {n_done}/{len(utts)} utts  ({time.time() - t0:.0f}s)", flush=True)

    meta = dict(model_dir=a.model_dir, task=task, method=a.method, lang=a.lang, split=a.split,
                vocab_size=V, own_regime=own, probes=[p[0] for p in probes])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path + ".part", "wb") as f:
        pickle.dump(dict(meta=meta, utts=results), f)
    os.replace(out_path + ".part", out_path)
    print(f"[{a.method}] wrote {out_path}  ({len(results)} utts, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
