#!/usr/bin/env python3
"""
Sanity checks for star_posteriors.py (run before trusting any numbers).

 1. Tagged-lattice posteriors == brute force.  For a few real dev utterances
    and one model, P(position i bypassed) from the tagged lattice must equal
    1 - exp(logZ(graph without bypass arc i) - logZ(full graph)), for both the
    token and the word probe.
 2. Self-loop expectation matches a finite-difference check on logZ
    (d logZ / d self_loop_weight at boundary i == E[#self-loop traversals]).
 3. The worked example (kn dev ...519320_50240649) puts the token bypass on
    the two characters GTS disagrees on.

Usage: sanity_checks.py --model_dir <kn otc_token dir> [--device cpu]
"""
import argparse
import gzip
import math
import pickle
import sys

import numpy as np
import soundfile as sf
import torch
import k2

sys.path.insert(0, __import__("os").path.dirname(__file__))
import star_posteriors as sp  # noqa: E402

EXAMPLE = "IISc_RESPIN_kn_D1_50506_583101_M_AGRI_519320_50240649"


def logZ(probe, fsa, lps):
    """Total log-likelihood in DOUBLE precision (k2.ctc_loss returns float32,
    which is too coarse for a finite difference on a ~150-nat quantity)."""
    g = k2.compose(probe.topo, k2.arc_sort(k2.remove_epsilon_and_add_self_loops(fsa)),
                   treat_epsilons_specially=False)
    sup = torch.tensor([[0, 0, lps.shape[1]]], dtype=torch.int32)
    lat = k2.intersect_dense(k2.create_fsa_vec([g]), k2.DenseFsaVec(lps, sup), 100.0)
    return float(lat.get_tot_scores(use_double_scores=True, log_semiring=True)[0])


def drop_arc(fsa, pred):
    lines = [l.split() for l in k2.to_str_simple(fsa).strip().split("\n")]
    keep = [" ".join(l[:4]) if len(l) >= 4 else l[0] for l in lines if not (len(l) >= 4 and pred(l))]
    return k2.arc_sort(k2.Fsa.from_str("\n".join(keep)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, args, task = sp.load_model(a.model_dir, dev)
    tl = list(args.token_list)
    V = len(tl)
    probe = sp.Probe(V, tl.index("<space>"), dev)
    labels = pickle.load(gzip.open(f"{sp.OUT}/labels/kn_dev.pkl.gz"))
    wav = dict(l.split(None, 1) for l in open(f"{sp.DATA_ROOT}/kn/IISc_RESPIN_dev_kn/wav.scp"))
    picks = [r for r in labels if r["uid"] == EXAMPLE] + [r for r in labels if r["uid"] != EXAMPLE][:2]

    worst = 0.0
    for r in picks:
        w = sf.read(wav[r["uid"]].strip())[0].astype(np.float32)
        with torch.no_grad():
            enc, el = model.encode(torch.from_numpy(w)[None].to(dev), torch.tensor([len(w)]).to(dev))
            lp = model.ctc.log_softmax(enc).float()
            lps = torch.cat([lp, torch.logsumexp(lp[..., 1:], -1, keepdim=True) - math.log(V - 1)], -1).cpu()
        ids = [tl.index("<space>" if c == " " else c) for c in r["text"]]
        dense = k2.DenseFsaVec(lps.to(dev), torch.tensor([[0, 0, lps.shape[1]]], dtype=torch.int32))
        for gran, byp, sl in (("token", 0.0, None), ("word", -2.0, None), ("token", -1.0, -2.0)):
            lat = probe.lattice([probe.fsa(ids, gran, byp, sl)], dense)
            post = lat.get_arc_post(True, True).exp()
            kind, pos = lat.diag_kind.long(), lat.diag_pos.long()
            n_pos = len(probe.groups(ids, gran))
            tag_byp = [float(post[(kind == 2) & (pos == p)].sum()) for p in range(1, n_pos + 1)]
            full = probe.fsa(ids, gran, byp, sl)
            Z = logZ(probe, full, lps)
            # brute force on every bypass arc: remove it, compare mass
            fsa_arcs = [l.split() for l in k2.to_str_simple(full).strip().split("\n") if len(l.split()) >= 4]
            byp_arcs = [l for l in fsa_arcs if int(l[2]) == probe.star and int(l[0]) != int(l[1])]
            brute = []
            for l in byp_arcs:
                src, dst = int(l[0]), int(l[1])
                f2 = drop_arc(full, lambda x, s=src, d=dst: int(x[0]) == s and int(x[1]) == d and int(x[2]) == probe.star)
                brute.append(1 - math.exp(logZ(probe, f2, lps) - Z))
            err = max(abs(x - y) for x, y in zip(tag_byp, brute)) if brute else 0.0
            worst = max(worst, err)
            msg = f"  {r['uid'][-24:]} {gran:<5} byp={byp} sl={sl}: bypass max|tag-brute|={err:.2e}"
            if sl is not None:
                tag_ins = [float(post[(kind == 3) & (pos == p)].sum()) for p in range(1, n_pos + 2)]
                eps = 1e-3                                 # central difference, double precision
                Zp = logZ(probe, probe.fsa(ids, gran, byp, sl + eps), lps)
                Zm = logZ(probe, probe.fsa(ids, gran, byp, sl - eps), lps)
                fd = (Zp - Zm) / (2 * eps)                 # d logZ / d w_sl = E[#self-loop arcs]
                msg += f"   sum E[ins]={sum(tag_ins):.4f} vs finite-diff {fd:.4f}"
                worst = max(worst, abs(sum(tag_ins) - fd) / max(1.0, fd))
            print(msg)
        if r["uid"] == EXAMPLE:
            lat = probe.lattice([probe.fsa(ids, "token", 0.0, None)], dense)
            post = lat.get_arc_post(True, True).exp()
            kind, pos = lat.diag_kind.long(), lat.diag_pos.long()
            pb = [float(post[(kind == 2) & (pos == p)].sum()) for p in range(1, len(ids) + 1)]
            marked = "".join(f"[{c}]" if p > 0.2 else c for c, p in zip(r["text"], pb))
            disp = "".join(f"[{c}]" if n else c for c, n in zip(r["text"], r["n_disp"]))
            print(f"  example, star (tok_bp_w0 > 0.2): {marked}")
            print(f"  example, disputed by validators : {disp}")
    print(f"WORST discrepancy: {worst:.2e}  ->  {'PASS' if worst < 1e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
