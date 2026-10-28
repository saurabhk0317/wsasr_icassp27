import argparse
import re
from collections import defaultdict
from jiwer import cer, wer
from typing import Dict, Tuple, List, Optional

def get_parser():
    parser = argparse.ArgumentParser(description="Score ASR (+ optional LID)")
    parser.add_argument("--ref_text", type=str, required=True, help="Path to reference text")
    parser.add_argument("--hyp_text", type=str, required=True, help="Path to decoded text")
    parser.add_argument("--utt2dial", type=str, required=True, help="Path to utt2dial file")
    parser.add_argument("--out", type=str, default="result.txt", help="Output result file")
    parser.add_argument("--monolingual", action="store_true",
                        help="If set, ref/hyp lines are 'uttid text' with no LID; "
                             "report only dialect-wise and overall CER/WER.")
    parser.add_argument("--utt_tsv", type=str, default=None,
                        help="Optional path to save per-utterance TSV with columns: "
                             "uttid, cer, wer, ref, hyp")
    return parser

def load_text(path: str, monolingual: bool = False, is_ref: bool = False) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
        utt2lid: dict[uttid] = lid (empty if monolingual=True or lid absent)
        utt2text: dict[uttid] = text
    Notes:
        - If is_ref=True: discard lines with empty text
        - If is_ref=False (hyp): keep lines even if text is empty (treated as deletions)
    """
    utt2lid: Dict[str, str] = {}
    utt2text: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if monolingual:
                # Expect: uttid <text>
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    # ref: discard empty; hyp: keep as empty
                    if not is_ref and len(parts) == 1:
                        utt2text[parts[0]] = ""
                    continue
                uttid, text = parts[0], parts[1].strip()
                if is_ref and text == "":
                    continue  # discard empty text lines in ref
                utt2text[uttid] = text
            else:
                # Expect: uttid LID text  (LID may be like [bn_D3] or bn_D3)
                parts = line.split(maxsplit=2)
                if len(parts) < 3:
                    # Fallbacks
                    if len(parts) == 2:
                        uttid, text = parts[0], parts[1].strip()
                        if is_ref and text == "":
                            continue
                        utt2text[uttid] = text
                    elif len(parts) == 1:
                        # hyp-only: keep empty
                        if not is_ref:
                            utt2text[parts[0]] = ""
                    continue
                uttid, lid, text = parts[0], parts[1], parts[2].strip()
                if is_ref and text == "":
                    continue
                utt2lid[uttid] = lid
                utt2text[uttid] = text
    return utt2lid, utt2text

def load_utt2dial(path):
    utt2dial = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            utt, dial = parts
            utt2dial[utt] = dial
    return utt2dial

def strip_lid(text):
    # Remove a leading bracketed token like [bn_D3] and following spaces
    return re.sub(r"^\[[^\]]+\]\s*", "", text.strip())

def remove_punctuation(text: str) -> str:
    """
    Remove limited punctuation before scoring.
    Removes:
        ,   (comma)
        ?   (question mark)
        ।   (Devanagari Purna Viram)
    """
    return re.sub(r"[,\?।.]", "", text)

def extract_lang(dialect_id):
    # Assumes dialect_id like "bn_D3" -> "bn"
    return dialect_id.split("_")[0] if "_" in dialect_id else dialect_id

def write_and_print(msg, f):
    print(msg)
    f.write(msg + "\n")

def main():
    args = get_parser().parse_args()

    # Load inputs with new rules
    ref_lids, ref_texts = load_text(args.ref_text, monolingual=args.monolingual, is_ref=True)
    hyp_lids, hyp_texts = load_text(args.hyp_text, monolingual=args.monolingual, is_ref=False)
    utt2dial = load_utt2dial(args.utt2dial)

    # Evaluation set: only utts present in ref_texts (after filtering empties) AND utt2dial
    eval_utts = [u for u in ref_texts.keys() if u in utt2dial]

    # Global aggregates
    global_refs: List[str] = []
    global_hyps: List[str] = []

    # LID accuracy counters (only when LIDs are present for both)
    lid_total, lid_correct = 0, 0

    # Aggregation for overall bucket scoring (true overall CER/WER)
    lang_refs: Dict[str, List[str]] = defaultdict(list)
    lang_hyps: Dict[str, List[str]] = defaultdict(list)
    dial_refs: Dict[str, List[str]] = defaultdict(list)
    dial_hyps: Dict[str, List[str]] = defaultdict(list)

    # Optional per-utterance TSV rows
    per_utt_rows: List[Tuple[str, float, float, str, str]] = []

    with open(args.out, "w", encoding="utf-8") as out:
        for uttid in eval_utts:
            # Fetch texts; if hyp missing, treat as all deletions
            ref_raw = ref_texts[uttid].strip()
            hyp_raw = hyp_texts.get(uttid, "")

            # Strip LID prefix from the text only in bilingual mode
            if args.monolingual:
                ref = ref_raw
                hyp = hyp_raw
            else:
                ref = strip_lid(ref_raw)
                hyp = strip_lid(hyp_raw)
                
            ref = remove_punctuation(ref)
            hyp = remove_punctuation(hyp)

            dial = utt2dial[uttid]
            lang = extract_lang(dial)

            # Append to global
            global_refs.append(ref)
            global_hyps.append(hyp)

            # LID accuracy when both LIDs are provided in files
            if not args.monolingual and (uttid in ref_lids) and (uttid in hyp_lids):
                lid_total += 1
                if ref_lids[uttid] == hyp_lids[uttid]:
                    lid_correct += 1

            # Bucket aggregation (for true overall CER/WER per bucket)
            dial_refs[dial].append(ref)
            dial_hyps[dial].append(hyp)
            if not args.monolingual:
                lang_refs[lang].append(ref)
                lang_hyps[lang].append(hyp)

            # Per-utt metrics (for optional TSV)
            if args.utt_tsv:
                c = cer(ref, hyp)
                w = wer(ref, hyp)
                per_utt_rows.append((uttid, c, w, ref, hyp))

        # Reports: Language-wise (only multilingual)
        if not args.monolingual:
            write_and_print("Language-wise CER/WER (%):", out)
            for lang in sorted(lang_refs.keys()):
                avg_cer = 100 * cer(lang_refs[lang], lang_hyps[lang])
                avg_wer = 100 * wer(lang_refs[lang], lang_hyps[lang])
                write_and_print(f"{lang}\tCER: {avg_cer:.2f}%\tWER: {avg_wer:.2f}%", out)

        # Reports: Dialect-wise
        write_and_print(("\n" if not args.monolingual else "") + "Dialect-wise CER/WER (%):", out)
        for dial in sorted(dial_refs.keys()):
            dcer = 100 * cer(dial_refs[dial], dial_hyps[dial])
            dwer = 100 * wer(dial_refs[dial], dial_hyps[dial])
            write_and_print(f"{dial}\tCER: {dcer:.2f}%\tWER: {dwer:.2f}%", out)

        # Overall
        overall_cer = 100 * cer(global_refs, global_hyps)
        overall_wer = 100 * wer(global_refs, global_hyps)
        write_and_print(f"\nOverall CER: {overall_cer:.2f}%", out)
        write_and_print(f"Overall WER: {overall_wer:.2f}%", out)

        # LID accuracy
        if not args.monolingual:
            lid_acc = lid_correct / lid_total if lid_total > 0 else 0.0
            write_and_print(f"\nLID Accuracy: {lid_acc:.4f} ({lid_correct}/{lid_total})", out)

    # Optional TSV dump
    if args.utt_tsv:
        with open(args.utt_tsv, "w", encoding="utf-8") as tf:
            tf.write("uttid\tcer\twer\tref\thyp\n")
            for uttid, c, w, ref, hyp in per_utt_rows:
                tf.write(f"{uttid}\t{c:.6f}\t{w:.6f}\t{ref}\t{hyp}\n")

if __name__ == "__main__":
    main()
