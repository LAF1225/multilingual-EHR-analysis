"""DAY 6 - Why do the models degrade? (RQ2 and RQ3)

Builds one row per (model, condition, test note) with:
  fertility, unk_rate, truncated   - how the tokenizer treats the note
  term_pres_bt                     - share of English medical terms that survive the round trip
  term_pres_surface                - share of English medical terms kept verbatim in the Urdu text
  chrf                             - round-trip translation quality (from Day 4)
  confidence, correct              - from the saved predictions
Then: calibration (ECE + reliability diagrams), a logistic regression,
confusion matrices, and an error-coding sheet for manual analysis.

Run:  python day6_analysis.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer

from config import CONDITIONS, DATA, FIGURES, MAX_LEN, MODELS, RANDOM_STATE, RESULTS, SEEDS
from utils import ece, load_condition, load_preds

SEED = SEEDS[0]
tests = {c: load_condition(c) for c in CONDITIONS}
tests = {c: d for c, d in tests.items() if d is not None}
en, bt, ur = tests["en"], tests.get("en_bt"), tests.get("ur")
labels = json.load(open(DATA / "labels.json"))


# ---------- 1. Medical-term preservation (model-independent) ----------
def term_preservation():
    try:
        import spacy
        nlp = spacy.load("en_core_sci_sm")
    except Exception as e:
        print(f"[warn] scispaCy not available ({e}); skipping term preservation.")
        return pd.DataFrame({"id": en["id"]})
    rows = []
    bt_map = dict(zip(bt["id"], bt["text"].str.lower())) if bt is not None else {}
    ur_map = dict(zip(ur["id"], ur["text"].str.lower())) if ur is not None else {}
    for doc_id, doc in zip(en["id"], nlp.pipe(en["text"].tolist(), batch_size=32)):
        terms = {e.text.lower() for e in doc.ents if len(e.text) > 2}
        row = {"id": doc_id, "n_terms": len(terms)}
        if terms and bt_map:
            row["term_pres_bt"] = np.mean([t in bt_map[doc_id] for t in terms])
        if terms and ur_map:
            row["term_pres_surface"] = np.mean([t in ur_map[doc_id] for t in terms])
        rows.append(row)
    return pd.DataFrame(rows)


terms = term_preservation()
mtq_path = RESULTS / "mt_quality_per_note.csv"
mtq = pd.read_csv(mtq_path)[["id", "chrf"]] if mtq_path.exists() else pd.DataFrame({"id": en["id"]})

# ---------- 2. Per-example features ----------
rows = []
for m, hf_id in MODELS.items():
    tok = AutoTokenizer.from_pretrained(hf_id)
    for c, df in tests.items():
        r = load_preds(m, SEED, c)
        if r is None:
            continue
        ids, y, probs = r
        text = dict(zip(df["id"], df["text"].astype(str)))
        for i, doc_id in enumerate(ids):
            t = text[doc_id]
            pieces = tok.tokenize(t)
            n_words = max(1, len(t.split()))
            rows.append({
                "model": m, "condition": c, "id": doc_id, "label": y[i],
                "pred": probs[i].argmax(), "confidence": probs[i].max(),
                "correct": int(probs[i].argmax() == y[i]),
                "fertility": len(pieces) / n_words,
                "unk_rate": sum(p == tok.unk_token for p in pieces) / max(1, len(pieces)),
                "truncated": int(len(pieces) > MAX_LEN - 2),
            })
feat = pd.DataFrame(rows).merge(terms, on="id", how="left").merge(mtq, on="id", how="left")
feat.to_csv(RESULTS / "example_features.csv", index=False)

tok_table = feat.groupby(["model", "condition"])[["fertility", "unk_rate", "truncated"]].mean().round(3)
tok_table.to_csv(RESULTS / "table_tokenization.csv")
print("Tokenization by model and condition:\n", tok_table, "\n")
if "term_pres_bt" in feat:
    print("Mean term preservation (round trip):", round(terms["term_pres_bt"].mean(), 3))
if "term_pres_surface" in feat:
    print("Mean terms kept verbatim in Urdu:", round(terms["term_pres_surface"].mean(), 3), "\n")

# ---------- 3. Calibration ----------
cal_rows = []
fig, axes = plt.subplots(len(MODELS), len(tests), figsize=(3 * len(tests), 3 * len(MODELS)),
                         squeeze=False)
bins = np.linspace(0, 1, 11)
for i, m in enumerate(MODELS):
    for j, c in enumerate(tests):
        ax = axes[i, j]
        r = load_preds(m, SEED, c)
        if r is None:
            ax.axis("off")
            continue
        _, y, probs = r
        conf, corr = probs.max(1), probs.argmax(1) == y
        cal_rows.append({"model": m, "condition": c, "ece": ece(probs, y),
                         "mean_conf": conf.mean(), "accuracy": corr.mean(),
                         "conf_when_right": conf[corr].mean() if corr.any() else np.nan,
                         "conf_when_wrong": conf[~corr].mean() if (~corr).any() else np.nan})
        centers, accs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf > lo) & (conf <= hi)
            if mask.sum() >= 5:
                centers.append((lo + hi) / 2)
                accs.append(corr[mask].mean())
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.bar(centers, accs, width=0.09, alpha=0.7)
        ax.set_title(f"{m} | {c}", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if i == len(MODELS) - 1: ax.set_xlabel("confidence")
        if j == 0: ax.set_ylabel("accuracy")
plt.tight_layout()
plt.savefig(FIGURES / "reliability_diagrams.png", dpi=200)
cal = pd.DataFrame(cal_rows).round(4)
cal.to_csv(RESULTS / "table_calibration.csv", index=False)
print("Calibration:\n", cal.to_string(index=False), "\n")

# ---------- 4. Logistic regression: what predicts a correct prediction on shifted text? ----------
# Confidence is left out on purpose: it is an output of the model, not a cause.
for c in [x for x in ("ur", "ur_roman") if x in tests]:
    d = feat[feat.condition == c].copy()
    preds = ["fertility", "unk_rate", "truncated"]
    preds += [v for v in ("term_pres_bt", "chrf") if v in d and d[v].notna().any()]
    d = d.dropna(subset=preds)
    for v in ("fertility", "unk_rate", "term_pres_bt", "chrf"):
        if v in preds and d[v].std() > 0:
            d[v] = (d[v] - d[v].mean()) / d[v].std()      # standardize -> comparable odds ratios
    preds = [v for v in preds if d[v].nunique() > 1]
    formula = "correct ~ " + " + ".join(preds + ["C(model)"])
    try:
        res = smf.logit(formula, data=d).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d["id"])[0]})
        out = pd.DataFrame({"coef": res.params, "odds_ratio": np.exp(res.params),
                            "p": res.pvalues}).round(4)
        out.to_csv(RESULTS / f"logit_{c}.csv")
        print(f"Logistic regression ({c}): {formula}\n{out}\n")
    except Exception as e:
        print(f"[warn] regression for {c} failed: {e}. Check for predictors that are "
              "almost perfectly correlated (e.g. unk_rate and model).")

# ---------- 5. Confusion matrices ----------
for m in MODELS:
    for c in tests:
        d = feat[(feat.model == m) & (feat.condition == c)]
        if len(d):
            cm = confusion_matrix(d["label"], d["pred"], labels=range(len(labels)))
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(
                RESULTS / "predictions" / f"confusion_{m}_{c}.csv")

# ---------- 6. Error-coding sheet ----------
# Notes a model got right in English but wrong in Urdu: the most informative errors.
if "ur" in tests:
    wide = feat.pivot_table(index=["model", "id"], columns="condition", values="correct").reset_index()
    flips = wide[(wide["en"] == 1) & (wide["ur"] == 0)]
    sample = pd.concat([g.sample(n=min(17, len(g)), random_state=RANDOM_STATE)
                        for _, g in flips.groupby("model")]).reset_index(drop=True)
    txt = {c: dict(zip(tests[c]["id"], tests[c]["text"])) for c in tests}
    ur_feat = feat[feat.condition == "ur"].set_index(["model", "id"])
    sheet = pd.DataFrame({
        "model": sample["model"], "id": sample["id"],
        "true_label": [labels[ur_feat.loc[(m, i), "label"]] for m, i in zip(sample["model"], sample["id"])],
        "pred_label_ur": [labels[ur_feat.loc[(m, i), "pred"]] for m, i in zip(sample["model"], sample["id"])],
        "english": sample["id"].map(txt["en"]),
        "urdu": sample["id"].map(txt["ur"]),
        "back_translation": sample["id"].map(txt["en_bt"]) if "en_bt" in txt else "",
    })
    # Suggested codes: TERM_LOST, ABBREV_MANGLED, NEGATION, MT_ERROR, TOKENIZER, AMBIGUOUS_LABEL, OTHER
    sheet["error_code"] = ""
    sheet["notes"] = ""
    sheet.to_csv(RESULTS / "error_coding_sheet.csv", index=False, encoding="utf-8-sig")
    print(f"Flipped (right in EN, wrong in UR) per model:\n{flips['model'].value_counts()}")
    print(f"Error coding sheet: {RESULTS / 'error_coding_sheet.csv'}")
