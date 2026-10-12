import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import CONDITIONS, MAX_LEN, MODELS, MODELS_DIR, RESULTS, SEEDS
from utils import load_condition, load_preds, macro_f1, pred_path

p = argparse.ArgumentParser()
p.add_argument("--overwrite", action="store_true")
p.add_argument("--batch", type=int, default=32)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
tests = {c: load_condition(c) for c in CONDITIONS}
tests = {c: df for c, df in tests.items() if df is not None}
print("Conditions found:", list(tests))


def predict(model, tok, texts):
    out = []
    for i in range(0, len(texts), args.batch):
        enc = tok(texts[i:i + args.batch], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**enc).logits
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


for key in MODELS:
    for seed in SEEDS:
        mdir = MODELS_DIR / f"{key}_seed{seed}"
        if not mdir.exists():
            continue
        tok = model = None
        for cond, df in tests.items():
            if pred_path(key, seed, cond).exists() and not args.overwrite:
                continue
            if model is None:
                tok = AutoTokenizer.from_pretrained(mdir)
                model = AutoModelForSequenceClassification.from_pretrained(mdir).to(device).eval()
            probs = predict(model, tok, df["text"].astype(str).tolist())
            np.savez(pred_path(key, seed, cond), ids=df["id"].values.astype(str),
                     labels=df["label_id"].values, probs=probs)
            print(f"  saved {key} seed{seed} {cond}")

# Summary table
rows = []
for key in MODELS:
    for seed in SEEDS:
        for cond in tests:
            r = load_preds(key, seed, cond)
            if r is None:
                continue
            _, y, probs = r
            pred = probs.argmax(1)
            rows.append({"model": key, "seed": seed, "condition": cond,
                         "macro_f1": macro_f1(y, pred), "accuracy": accuracy_score(y, pred),
                         "n": len(y)})
summary = pd.DataFrame(rows)
summary.to_csv(RESULTS / "metrics_all_runs.csv", index=False)
print(summary.pivot_table(index=["model", "seed"], columns="condition", values="macro_f1").round(3))
