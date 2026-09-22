"""DAY 4 - Translation quality checks.

Compares original English with the back-translation (round-trip quality),
saves per-note chrF (reused on Day 6), and exports a sheet for manual review.

Run:  python day4_mt_quality.py
      python day4_mt_quality.py --bertscore      (optional, slower; pip install bert-score)
"""
import argparse
import json

import pandas as pd
import sacrebleu

from config import DATA, RANDOM_STATE, RESULTS
from utils import load_condition

p = argparse.ArgumentParser()
p.add_argument("--bertscore", action="store_true")
p.add_argument("--n_manual", type=int, default=30)
args = p.parse_args()

en, bt = load_condition("en"), load_condition("en_bt")
ur, rom = load_condition("ur"), load_condition("ur_roman")
assert (en["id"] == bt["id"]).all(), "ids do not line up - re-run day4_translate.py"

refs, hyps = en["text"].tolist(), bt["text"].tolist()
summary = {
    "chrF_roundtrip": sacrebleu.corpus_chrf(hyps, [refs]).score,
    "BLEU_roundtrip": sacrebleu.corpus_bleu(hyps, [refs]).score,
    "n_notes": len(en),
}

per_note = pd.DataFrame({
    "id": en["id"],
    "chrf": [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(hyps, refs)],
    "len_ratio": bt["text"].str.split().str.len() / en["text"].str.split().str.len(),
})

if args.bertscore:
    from bert_score import score
    # distilbert keeps this laptop-friendly; say which model you used in the paper
    _, _, f1 = score(hyps, refs, model_type="distilbert-base-uncased", batch_size=16)
    per_note["bertscore_f1"] = f1.numpy()
    summary["BERTScore_F1_roundtrip"] = float(f1.mean())

per_note.to_csv(RESULTS / "mt_quality_per_note.csv", index=False)
json.dump(summary, open(RESULTS / "mt_quality_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print("\nLowest round-trip chrF (inspect these):")
print(per_note.nsmallest(5, "chrf").to_string(index=False))

# Manual review sheet (utf-8-sig so Excel shows Urdu correctly)
sample = en.sample(n=min(args.n_manual, len(en)), random_state=RANDOM_STATE)[["id", "label", "text"]]
sample = sample.rename(columns={"text": "english"})
sample["urdu"] = sample["id"].map(dict(zip(ur["id"], ur["text"])))
if rom is not None:
    sample["roman_urdu"] = sample["id"].map(dict(zip(rom["id"], rom["text"])))
sample["back_translation"] = sample["id"].map(dict(zip(bt["id"], bt["text"])))
for col in ["adequacy_1to5", "fluency_1to5", "medical_terms_ok_Y_N", "notes"]:
    sample[col] = ""
sample.to_csv(RESULTS / "manual_review_sheet.csv", index=False, encoding="utf-8-sig")
print(f"\nManual review sheet: {RESULTS / 'manual_review_sheet.csv'}")
