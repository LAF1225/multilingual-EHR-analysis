import pandas as pd, numpy as np
from sklearn.metrics import f1_score
from config import MODELS, SEEDS
from utils import load_preds

def max_trigram(text):
    w = str(text).lower().split()
    tri = [" ".join(w[i:i+3]) for i in range(len(w) - 2)]
    return pd.Series(tri).value_counts().max() if tri else 0

bt = pd.read_csv("data/processed/test_en_bt.csv")
ur = pd.read_csv("data/processed/test_ur.csv")
bad_bt = set(bt.loc[bt["text"].apply(max_trigram) >= 5, "id"])
bad_ur = set(ur.loc[ur["text"].apply(max_trigram) >= 5, "id"])
print(f"Degenerate back-translations: {len(bad_bt)} | degenerate Urdu: {len(bad_ur)} | in both: {len(bad_bt & bad_ur)}")

rows = []
for m in MODELS:
    for cond in ["en", "en_bt", "ur", "ur_roman"]:
        r = load_preds(m, SEEDS[0], cond)
        if r is None: continue
        ids, y, probs = r
        keep = ~np.isin(ids, list(bad_bt | bad_ur))
        rows.append({
            "model": m, "condition": cond,
            "all_notes": f1_score(y, probs.argmax(1), average="macro", zero_division=0),
            "clean_only": f1_score(y[keep], probs[keep].argmax(1), average="macro", zero_division=0),
            "n_clean": keep.sum(),
        })
out = pd.DataFrame(rows).round(3)
out.to_csv("results/sensitivity_degenerate.csv", index=False)
print(out.to_string(index=False))
