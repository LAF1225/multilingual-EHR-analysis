import pandas as pd
from config import MODELS, SEEDS
from utils import load_preds, ece

rows = []
for m in MODELS:
    for s in SEEDS:
        for c in ["en", "en_bt", "ur", "ur_roman"]:
            r = load_preds(m, s, c)
            if r is None:
                continue
            _, y, probs = r
            conf = probs.max(1)
            corr = probs.argmax(1) == y
            rows.append({
                "model": m, "seed": s, "condition": c,
                "ece": ece(probs, y), "mean_conf": conf.mean(), "acc": corr.mean(),
                "conf_right": conf[corr].mean() if corr.any() else float("nan"),
                "conf_wrong": conf[~corr].mean() if (~corr).any() else float("nan"),
                "gap": (conf[corr].mean() - conf[~corr].mean()) if corr.any() and (~corr).any() else float("nan"),
            })

df = pd.DataFrame(rows).round(3)
df.to_csv("results/calibration_all_seeds.csv", index=False)
print(df.pivot_table(index=["model", "condition"], columns="seed",
                     values=["ece", "gap"]).round(3).to_string())
