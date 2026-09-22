"""DAY 5 (and DAY 9) - Degradation, confidence intervals and significance tests.

Outputs (in results/):
  table_main.csv          mean +- std macro-F1 and relative degradation across seeds
  bootstrap_degradation.csv  95% CI for each model's degradation (primary seed)
  bootstrap_model_diffs.csv  is one model's degradation smaller than another's?
  mcnemar_tests.csv       paired model comparisons, Holm-corrected

Run:  python day5_stats.py
"""
from itertools import combinations

import numpy as np
import pandas as pd

from config import CONDITIONS, MODELS, RESULTS, SEEDS
from utils import holm, load_preds, macro_f1, mcnemar_p

N_BOOT = 1000
rng = np.random.default_rng(0)
SHIFTED = [c for c in CONDITIONS if c != "en"]

# ---------- 1. Mean +- std across seeds ----------
runs = pd.read_csv(RESULTS / "metrics_all_runs.csv")
en_f1 = runs[runs.condition == "en"].set_index(["model", "seed"])["macro_f1"]
runs["rel_drop"] = runs.apply(
    lambda r: (en_f1[(r.model, r.seed)] - r.macro_f1) / en_f1[(r.model, r.seed)], axis=1)
table = runs.groupby(["model", "condition"]).agg(
    f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
    rel_drop_mean=("rel_drop", "mean"), rel_drop_std=("rel_drop", "std"),
    n_seeds=("seed", "nunique")).round(4)
table.to_csv(RESULTS / "table_main.csv")
print(table, "\n")

# ---------- 2. Paired bootstrap on the primary seed ----------
seed = next(s for s in SEEDS if load_preds(list(MODELS)[0], s, "en") is not None)
print(f"Bootstrap and McNemar use seed {seed} (report this in the paper)\n")

preds = {}
for m in MODELS:
    for c in CONDITIONS:
        r = load_preds(m, seed, c)
        if r is not None:
            preds[(m, c)] = r

y = preds[(list(MODELS)[0], "en")][1]
ids = preds[(list(MODELS)[0], "en")][0]
for k, (i, yy, _) in preds.items():
    assert (i == ids).all(), f"id order mismatch in {k}"
n = len(y)
boot_idx = rng.integers(0, n, size=(N_BOOT, n))   # same resamples for every model -> paired

boot_drop = {}
rows = []
for m in MODELS:
    if (m, "en") not in preds:
        continue
    p_en = preds[(m, "en")][2].argmax(1)
    for c in SHIFTED:
        if (m, c) not in preds:
            continue
        p_c = preds[(m, c)][2].argmax(1)
        drops = []
        for idx in boot_idx:
            f_en = macro_f1(y[idx], p_en[idx])
            drops.append((f_en - macro_f1(y[idx], p_c[idx])) / f_en)
        drops = np.array(drops)
        boot_drop[(m, c)] = drops
        point = (macro_f1(y, p_en) - macro_f1(y, p_c)) / macro_f1(y, p_en)
        rows.append({"model": m, "condition": c, "rel_drop": point,
                     "ci_low": np.percentile(drops, 2.5), "ci_high": np.percentile(drops, 97.5)})
pd.DataFrame(rows).round(4).to_csv(RESULTS / "bootstrap_degradation.csv", index=False)
print(pd.DataFrame(rows).round(3), "\n")

rows = []
for c in SHIFTED:
    for a, b in combinations(MODELS, 2):
        if (a, c) in boot_drop and (b, c) in boot_drop:
            d = boot_drop[(a, c)] - boot_drop[(b, c)]
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            rows.append({"condition": c, "model_a": a, "model_b": b,
                         "drop_diff_a_minus_b": d.mean(),
                         "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5),
                         "p_boot": p})
diffs = pd.DataFrame(rows)
if len(diffs):
    diffs["p_holm"] = holm(diffs["p_boot"])
diffs.round(4).to_csv(RESULTS / "bootstrap_model_diffs.csv", index=False)
print(diffs.round(3), "\n")

# ---------- 3. McNemar tests ----------
rows = []
correct = {k: (v[2].argmax(1) == y) for k, v in preds.items()}
for c in CONDITIONS:                               # model vs model, same test set
    for a, b in combinations(MODELS, 2):
        if (a, c) in correct and (b, c) in correct:
            p, nb, nc = mcnemar_p(correct[(a, c)], correct[(b, c)])
            rows.append({"test": "model_vs_model", "condition": c, "a": a, "b": b,
                         "a_right_b_wrong": nb, "a_wrong_b_right": nc, "p": p})
for m in MODELS:                                   # English vs shifted, same model
    for c in SHIFTED:
        if (m, "en") in correct and (m, c) in correct:
            p, nb, nc = mcnemar_p(correct[(m, "en")], correct[(m, c)])
            rows.append({"test": "en_vs_shift", "condition": c, "a": f"{m}:en", "b": f"{m}:{c}",
                         "a_right_b_wrong": nb, "a_wrong_b_right": nc, "p": p})
mc = pd.DataFrame(rows)
mc["p_holm"] = holm(mc["p"])
mc.round(5).to_csv(RESULTS / "mcnemar_tests.csv", index=False)
print(mc.round(4).to_string(index=False))
