"""DAY 4 - Machine-translate the test set with NLLB-200 (distilled 600M).

Step 1, English -> Urdu (creates data/processed/test_ur.csv):
  python day4_translate.py --direction en2ur
Step 2, Urdu -> English back-translation (creates test_en_bt.csv):
  python day4_translate.py --direction ur2en

Translation is sentence by sentence and cached, so if it stops you can
re-run the same command and it continues where it left off.
Try --limit 5 first to check the output looks sensible.
"""
import argparse
import json
import re

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import DATA, NLLB

SETTINGS = {
    "en2ur": dict(src="eng_Latn", tgt="urd_Arab", infile="test_en.csv", outfile="test_ur.csv",
                  split=re.compile(r"(?<=[.!?])\s+")),
    # Urdu full stop is "۔" and question mark is "؟"
    "ur2en": dict(src="urd_Arab", tgt="eng_Latn", infile="test_ur.csv", outfile="test_en_bt.csv",
                  split=re.compile(r"(?<=[.!?۔؟])\s+")),
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--direction", choices=SETTINGS, required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--beams", type=int, default=2, help="4 is slightly better but ~2x slower")
    p.add_argument("--limit", type=int, default=None, help="only translate the first N notes")
    return p.parse_args()


def main():
    args = parse()
    s = SETTINGS[args.direction]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    df = pd.read_csv(DATA / s["infile"])
    if args.limit:
        df = df.head(args.limit)
    doc_sents = [[x for x in s["split"].split(str(t)) if x.strip()] for t in df["text"]]

    cache_file = DATA / f"mt_cache_{args.direction}.jsonl"
    cache = {}
    if cache_file.exists():
        for line in open(cache_file, encoding="utf-8"):
            r = json.loads(line)
            cache[r["src"]] = r["tgt"]

    todo = sorted({x for sents in doc_sents for x in sents if x not in cache}, key=len)
    print(f"Device: {device} | sentences left to translate: {len(todo)} (cached: {len(cache)})")

    if todo:
        tok = AutoTokenizer.from_pretrained(NLLB, src_lang=s["src"])
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB).to(device).eval()
        tgt_id = tok.convert_tokens_to_ids(s["tgt"])
        with open(cache_file, "a", encoding="utf-8") as fh:
            for i in range(0, len(todo), args.batch):
                batch = todo[i:i + args.batch]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                          max_length=256).to(device)
                with torch.inference_mode():
                    gen = model.generate(**enc, forced_bos_token_id=tgt_id,
                                         num_beams=args.beams, max_new_tokens=256)
                for src, tgt in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
                    cache[src] = tgt
                    fh.write(json.dumps({"src": src, "tgt": tgt}, ensure_ascii=False) + "\n")
                fh.flush()
                done = min(i + args.batch, len(todo))
                if done % (args.batch * 25) == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)}")

    out = df.copy()
    out["text"] = [" ".join(cache[x] for x in sents) for sents in doc_sents]
    out.to_csv(DATA / s["outfile"], index=False, encoding="utf-8")
    print(f"Saved {DATA / s['outfile']}")
    print("Example:\n  IN :", df["text"].iloc[0][:200], "\n  OUT:", out["text"].iloc[0][:200])


if __name__ == "__main__":
    main()
