import numpy as np, pandas as pd
from itertools import combinations
from config import MODELS, SEEDS, CONDITIONS
from utils import load_preds, macro_f1

rng = np.random.default_rng(0)
rows = []
for seed in SEEDS:
    preds = {(m, c): load_preds(m, seed, c) for m in MODELS for c in CONDITIONS}
    if any(v is None for v in preds.values()): continue
    y = preds[(list(MODELS)[0], "en")][1]
    idx = rng.integers(0, len(y), size=(1000, len(y)))
    drops = {}
    for m in MODELS:
        p_en = preds[(m, "en")][2].argmax(1)
        for c in [x for x in CONDITIONS if x != "en"]:
            p_c = preds[(m, c)][2].argmax(1)
            drops[(m, c)] = np.array([(macro_f1(y[i], p_en[i]) - macro_f1(y[i], p_c[i])) / macro_f1(y[i], p_en[i]) for i in idx])
    for c in [x for x in CONDITIONS if x != "en"]:
        for a, b in combinations(MODELS, 2):
            d = drops[(a, c)] - drops[(b, c)]
            rows.append({"seed": seed, "condition": c, "a": a, "b": b,
                         "diff": d.mean(), "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)})
out = pd.DataFrame(rows).round(3)
out.to_csv("results/bootstrap_model_diffs_all_seeds.csv", index=False)
print(out.to_string(index=False))
