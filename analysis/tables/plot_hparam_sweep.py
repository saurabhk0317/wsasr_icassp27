#!/usr/bin/env python3
"""Hyper-parameter sweeps for OTC-T_e (epoch decay tau) and OTC-T_H
(shaping exponent kappa) on a random 10h subset of LibriSpeech
train-clean-100, SSL front-end, WER on dev-clean + dev-other.

Writes figures/hparam_sweep.{eps,png} into the ICASSP-27 Overleaf folder.

Run with the ESPnet venv (system python has no matplotlib):
  /home3/Saurabh/tools/espnet_may22_2026/tools/venv/bin/python plot_hparam_sweep.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = "/home3/Saurabh/exp/espnet/RESPIN/WSASR/overleaf/WSASR_ICASSP27/figures"
OUT_NAME = "hparam_sweep"

CTC_WER = 13.92

# OTC-TOKEN: geometric decay constant tau of the bypass weight.
TOKEN_TAU = [1.000, 0.975, 0.950, 0.925, 0.900, 0.875, 0.850, 0.825, 0.800,
             0.775, 0.750, 0.725, 0.700, 0.675, 0.650, 0.625, 0.600]
TOKEN_WER = [14.18, 13.69, 13.20, 13.30, 12.82, 12.87, 12.61, 12.51, 12.53,
             12.42, 12.44, 12.57, 12.61, 12.51, 12.59, 12.70, 12.54]

# OTC-RECIP: shaping exponent kappa on the entropy-derived progress.
RECIP_KAPPA = [1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8]
RECIP_WER = [12.56, 12.40, 12.41, 12.47, 12.52, 12.57, 12.60]

LINE = "#2a5caa"
BEST = "#c0392b"
GRID = "#d9d9d9"
MUTED = "#555555"


def panel(ax, x, y, xlabel, title, xticks, invert=False):
    ax.plot(x, y, color=LINE, lw=1.4, marker="o", ms=3.2, zorder=3)
    ax.axhline(CTC_WER, color=MUTED, lw=1.0, ls="--", zorder=2)
    i = min(range(len(y)), key=y.__getitem__)
    ax.plot(x[i], y[i], marker="o", ms=6.5, mfc="none", mec=BEST, mew=1.3,
            zorder=4)
    ax.annotate(f"{y[i]:.2f}", (x[i], y[i]), xytext=(0, -6),
                textcoords="offset points", ha="center", va="top",
                fontsize=7, color="black")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.set_xticks(xticks)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if invert:
        ax.invert_xaxis()


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "ps.fonttype": 42,
        "pdf.fonttype": 42,
    })
    # Single IEEE column (3.39 in).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.39, 1.75), sharey=True,
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    panel(ax1, TOKEN_TAU, TOKEN_WER, r"decay $\tau$",
          r"(a) OTC-$\mathrm{T}_e$", [1.0, 0.9, 0.8, 0.7, 0.6], invert=True)
    panel(ax2, RECIP_KAPPA, RECIP_WER, r"exponent $\kappa$",
          r"(b) OTC-$\mathrm{T}_H$", [0.8, 1.0, 1.2, 1.4])

    ax1.set_ylabel("WER (%)")
    ax1.set_ylim(11.8, 14.4)
    ax1.text(0.60, CTC_WER + 0.06, "CTC", ha="right", va="bottom",
             fontsize=7, color=MUTED)

    fig.tight_layout(pad=0.3, w_pad=0.8)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("eps", "png"):
        path = os.path.join(OUT_DIR, f"{OUT_NAME}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print("wrote", path)


if __name__ == "__main__":
    main()
