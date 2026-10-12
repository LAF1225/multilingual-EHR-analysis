import numpy as np, pandas as pd, statsmodels.formula.api as smf
from config import MODELS, SEEDS, RESULTS
from utils import load_preds

base = pd.read_csv(RESULTS / "example_features.csv")
keep = ["model", "condition", "id", "fertility", "unk_rate", "truncated", "term_pres_bt", "chrf"]
base = base[[c for c in keep if c in base.columns]]

for seed in SEEDS:
    rec = []
    for m in MODELS:
        for c in ["en", "en_bt", "ur", "ur_roman"]:
            r = load_preds(m, seed, c)
            if r is None:
                continue
            ids, y, probs = r
            rec.append(pd.DataFrame({"model": m, "condition": c, "id": ids,
                                     "correct": (probs.argmax(1) == y).astype(int)}))
    d_all = base.merge(pd.concat(rec), on=["model", "condition", "id"])

    wide = d_all.pivot_table(index=["model", "id"], columns="condition", values="correct").reset_index()
    flips = wide[(wide["en"] == 1) & (wide["ur"] == 0)].groupby("model").size().to_dict()
    print(f"\n========== seed {seed} | flips (right in EN, wrong in UR): {flips}")

    for cond in ["ur", "ur_roman"]:
        d = d_all[d_all.condition == cond].copy()
        preds = [p for p in ["fertility", "unk_rate", "truncated", "term_pres_bt", "chrf"]
                 if p in d and d[p].notna().any()]
        d = d.dropna(subset=preds)
        for v in preds:
            if v != "truncated" and d[v].std() > 0:
                d[v] = (d[v] - d[v].mean()) / d[v].std()
        preds = [p for p in preds if d[p].nunique() > 1]
        formula = "correct ~ " + " + ".join(preds + ["C(model)"])
        try:
            res = smf.logit(formula, data=d).fit(disp=0, cov_type="cluster",
                                                 cov_kwds={"groups": pd.factorize(d["id"])[0]})
            out = pd.DataFrame({"odds_ratio": np.exp(res.params), "p": res.pvalues}).round(3)
            print(f"\n-- {cond}\n{out.to_string()}")
        except Exception as e:
            print(f"[warn] {cond}: {e}")
