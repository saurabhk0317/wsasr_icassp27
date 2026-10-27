#!/usr/bin/env python3
"""Extract WER for every (corpus, language, system) cell used in Table 1
from exp_final/*/**/decode_*/**/result_lid.txt ("Overall WER: X.XX%").

NB: use result_lid.txt, not RESULTS.md -- the two are scored differently
(RESULTS.md rounds to one decimal and does not match result_lid.txt).

Usage:
    python3 extract_wer.py > wer_data.json

Output JSON schema:
    {
      "libri":  {"test_clean": {...}, "test_other": {...}},
      "fleurs": {"<fleurs_locale_code>": {...}, ...},
      "respin": {"<respin_two_letter_code>": {...}, ...},
      "libri_conformer":   {"test_clean": {...}, "test_other": {...}},
      "libri_labelsmooth": {"test_clean": {"0p005": WER, ...}, "test_other": {...}},
    }
  where each {...} under libri/fleurs/respin/libri_conformer has keys
  ctc/word/token/recip/tw/twr/wtuned (WER, float or null if that run doesn't
  exist yet) plus a "<key>_path" pointing at the result_lid.txt each number
  was read from, for traceability. libri_labelsmooth instead has one key per
  epsilon in LABELSMOOTH_EPS (Sec. 4.4's regularisation control, not one of
  the seven Table 1 systems).

System key -> model-dir suffix (asr_otcrecip_<corpus>_<lang>_<suffix>):
    ctc     otc_word (or the converged-bypass variant, see *_word_override)
    word    otc_word (or the converged-bypass variant, see *_word_override)
    token   otc_token          (OTC-T_e:  bypass-weight decay 0.775)
    recip   otc_recip          (OTC-T_H:  entropy shaping exponent k=1.3)
    tw      otc_token_word     (OTC-TW_e: token+word hybrid, epoch-indexed)
    twr     otc_token_word_recip (OTC-TW_H: token+word hybrid, entropy-indexed)
    wtuned  otc_word_tuned     (OTC-W_e': OTC-W_e with a re-tuned self-loop/decay)
"""
import datetime
import json
import os
import re
import sys

ROOT = "/home3/Saurabh/exp/espnet/RESPIN/WSASR/exp_final"

SYS = {
    "ctc": "ctc",
    "word": "otc_word",
    "token": "otc_token",
    "recip": "otc_recip",
    "tw": "otc_token_word",
    "twr": "otc_token_word_recip",
    "wtuned": "otc_word_tuned",
}

RESPIN_LANGS = {
    "bh_s5": "bh", "bn_s5": "bn", "ch_s5": "ch", "hi_s5": "hi", "kn_s5": "kn",
    "mg_s5": "mg", "mr_s5": "mr", "mt_s5": "mt", "te_s5": "te",
}
FLEURS_LANGS = ["ar_eg", "hy_am", "bn_in", "en_us", "ka_ge", "el_gr", "gu_in",
                "hi_in", "kn_in", "ml_in", "pa_in", "ru_ru", "ta_in", "te_in"]

# Per-language override for the converged OTC-W_e run: the published
# bypass weight (-19) diverged on these five languages (Sec. 3.1 of the
# paper); RESULTS.md/result_lid.txt only exist under the rerun's dir name.
RESPIN_WORD_OVERRIDE = {"kn": "otc_word_bpn35"}
FLEURS_WORD_OVERRIDE = {"ml_in": "otc_word_bp30", "ta_in": "otc_word_bp30"}


def get_wer(result_lid_path):
    if not os.path.isfile(result_lid_path):
        return None
    text = open(result_lid_path).read()
    m = re.search(r"Overall WER:\s*([0-9.]+)%", text)
    return float(m.group(1)) if m else None


def respin_row(code):
    base = os.path.join(ROOT, "respin", f"{code}_s5" if not code.endswith("_s5") else code)
    row = {"name": code}
    for key, sysname in SYS.items():
        suffix = RESPIN_WORD_OVERRIDE.get(code, sysname) if key == "word" else sysname
        model_dir = f"asr_otcrecip_respin_{code}_{suffix}"
        p = os.path.join(base, model_dir, "decode_ctc_asr_model_valid.loss.ave",
                          f"IISc_RESPIN_test_{code}", "result_lid.txt")
        row[key] = get_wer(p)
        row[f"{key}_path"] = p
    return row


def fleurs_row(code):
    base = os.path.join(ROOT, "fleurs", code)
    row = {"name": code}
    for key, sysname in SYS.items():
        suffix = FLEURS_WORD_OVERRIDE.get(code, sysname) if key == "word" else sysname
        model_dir = f"asr_otcrecip_fleurs_{code}_{suffix}"
        p = os.path.join(base, model_dir, "decode_ctc_asr_model_valid.loss.ave",
                          f"test_{code}", "result_lid.txt")
        row[key] = get_wer(p)
        row[f"{key}_path"] = p
    return row


def libri_row(testset, encoder_suffix=""):
    base = os.path.join(ROOT, "libri", "train_clean_100")
    row = {}
    for key, sysname in SYS.items():
        model_dir = f"asr_otcrecip_libri_en_{sysname}{encoder_suffix}"
        p = os.path.join(base, model_dir, "decode_ctc_asr_model_valid.loss.ave",
                          testset, "result_lid.txt")
        row[key] = get_wer(p)
        row[f"{key}_path"] = p
    return row


LABELSMOOTH_EPS = ["0p005", "0p01", "0p02", "0p05"]


def libri_labelsmooth_row(testset):
    """Sec. 4.4's regularisation control: CTC + label smoothing at eps in
    {0.005, 0.01, 0.02, 0.05}, no wildcard arcs. Not one of the seven Table 1
    systems, so kept in its own top-level key rather than in SYS."""
    base = os.path.join(ROOT, "libri", "train_clean_100")
    row = {}
    for eps in LABELSMOOTH_EPS:
        model_dir = f"asr_otcrecip_libri_en_labelsmooth_ls{eps}"
        p = os.path.join(base, model_dir, "decode_ctc_asr_model_valid.loss.ave",
                          testset, "result_lid.txt")
        row[eps] = get_wer(p)
        row[f"{eps}_path"] = p
    return row


def main():
    results = {
        "_generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "_note": "Regenerate: python3 extract_wer.py > wer_data.json. Cells read "
                 "None/missing at generation time (e.g. OTC-TW_H for RESPIN "
                 "ch/hi/kn) may since have finished -- rerun rather than trust "
                 "a stale copy of this file.",
        "respin": {code: respin_row(code) for code in RESPIN_LANGS.values()},
        "fleurs": {code: fleurs_row(code) for code in FLEURS_LANGS},
        "libri": {ts: libri_row(ts) for ts in ["test_clean", "test_other"]},
        "libri_conformer": {ts: libri_row(ts, "_conformer") for ts in ["test_clean", "test_other"]},
        "libri_labelsmooth": {ts: libri_labelsmooth_row(ts) for ts in ["test_clean", "test_other"]},
    }

    missing = []
    for corpus in ("respin", "fleurs"):
        for code, row in results[corpus].items():
            for key in ("ctc", "word", "token", "recip"):  # the four "core" systems
                if row.get(key) is None:
                    missing.append((corpus, code, key))
    if missing:
        print(f"WARNING: {len(missing)} core (ctc/word/token/recip) cells missing: "
              f"{missing}", file=sys.stderr)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
