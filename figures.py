"""DAY 11 - Paper figures.

Run:  python day11_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CONDITIONS, FIGURES, MODELS, RESULTS

NAMES = {"bert": "BERT-base", "bioclinical": "Bio_ClinicalBERT", "xlmr": "XLM-R base"}
COND_NAMES = {"en": "English", "en_bt": "Back-translated EN", "ur_roman": "Roman Urdu", "ur": "Urdu"}
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

table = pd.read_csv(RESULTS / "table_main.csv")
conds = [c for c in CONDITIONS if c in set(table.condition)]
models = [m for m in MODELS if m in set(table.model)]
w = 0.8 / len(conds)
x = np.arange(len(models))

# Figure 1: macro-F1 per model and condition
fig, ax = plt.subplots(figsize=(6.5, 3.5))
for k, c in enumerate(conds):
    d = table[table.condition == c].set_index("model").reindex(models)
    ax.bar(x + (k - (len(conds) - 1) / 2) * w, d["f1_mean"], w,
           yerr=d["f1_std"].fillna(0), capsize=2, label=COND_NAMES[c])
ax.set_xticks(x, [NAMES[m] for m in models])
ax.set_ylabel("Macro-F1")
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES / "fig1_f1_by_condition.png", dpi=300)

# Figure 2: relative degradation (shifted conditions only)
shifted = [c for c in conds if c != "en"]
w2 = 0.8 / max(1, len(shifted))
fig, ax = plt.subplots(figsize=(6.5, 3.5))
for k, c in enumerate(shifted):
    d = table[table.condition == c].set_index("model").reindex(models)
    ax.bar(x + (k - (len(shifted) - 1) / 2) * w2, 100 * d["rel_drop_mean"], w2,
           yerr=100 * d["rel_drop_std"].fillna(0), capsize=2, label=COND_NAMES[c])
ax.set_xticks(x, [NAMES[m] for m in models])
ax.set_ylabel("Relative drop in macro-F1 (%)")
ax.axhline(0, color="black", lw=0.8)
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES / "fig2_relative_degradation.png", dpi=300)

# Figure 3: tokenizer fertility vs macro-F1
tok = pd.read_csv(RESULTS / "table_tokenization.csv")
merged = tok.merge(table, on=["model", "condition"])
fig, ax = plt.subplots(figsize=(5, 3.5))
markers = dict(zip(conds, "osD^"))
for m in models:
    d = merged[merged.model == m]
    ax.plot(d["fertility"], d["f1_mean"], "-", lw=0.8, alpha=0.5)
    for _, r in d.iterrows():
        ax.scatter(r.fertility, r.f1_mean, marker=markers[r.condition], s=45)
        ax.annotate(f"{NAMES[m]}\n{COND_NAMES[r.condition]}", (r.fertility, r.f1_mean),
                    fontsize=6, xytext=(4, 2), textcoords="offset points")
ax.set_xlabel("Fertility (subword tokens per word)")
ax.set_ylabel("Macro-F1")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES / "fig3_fertility_vs_f1.png", dpi=300)
print("Saved figures to", FIGURES)
