import json
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA, DATA_RAW, MAX_WORDS, N_CLASSES, RANDOM_STATE, RESULTS

# These are note TYPES, not specialties, and they duplicate notes from other
# categories. Keeping them would make the labels ambiguous.
EXCLUDE = {
    "Surgery", "Consult - History and Phy.", "SOAP / Chart / Progress Notes",
    "Discharge Summary", "Office Notes", "Letters", "Emergency Room Reports",
    "General Medicine", "IME-QME-Work Comp etc.", "Autopsy",
}

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def clean(text):
    text = str(text)
    text = re.sub(r":\s*,", ": ", text)       # MTSamples quirk: "HISTORY:,  The patient"
    text = re.sub(r",\s*,", ", ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate(text, max_words=MAX_WORDS):
    """Keep whole sentences until ~max_words, so translation sees complete sentences."""
    out, n = [], 0
    for sent in SENT_SPLIT.split(text):
        w = len(sent.split())
        if n + w > max_words:
            if not out:  # first sentence alone is too long: hard cut
                out.append(" ".join(sent.split()[:max_words]))
            break
        out.append(sent)
        n += w
    return " ".join(out)


def main():
    df = pd.read_csv(DATA_RAW / "mtsamples.csv")
    df = df.rename(columns={"transcription": "text", "medical_specialty": "label"})
    df["label"] = df["label"].astype(str).str.strip()
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].map(clean)
    df = df[df["text"].str.split().str.len() >= 30]          # drop near-empty notes
    df = df[~df["label"].isin(EXCLUDE)]

    # Drop notes that appear under more than one label, then exact duplicates
    n_labels = df.groupby("text")["label"].nunique()
    df = df[df["text"].map(n_labels) == 1].drop_duplicates("text")

    top = df["label"].value_counts().head(N_CLASSES).index.tolist()
    df = df[df["label"].isin(top)].copy()
    df["text"] = df["text"].map(truncate)

    labels = sorted(top)
    df["label_id"] = df["label"].map({l: i for i, l in enumerate(labels)})
    df = df.reset_index(drop=True)
    df["id"] = df.index.map(lambda i: f"doc{i:05d}")
    df = df[["id", "text", "label", "label_id"]]

    train_val, test = train_test_split(df, test_size=0.20, stratify=df["label_id"],
                                       random_state=RANDOM_STATE)
    train, val = train_test_split(train_val, test_size=0.125, stratify=train_val["label_id"],
                                  random_state=RANDOM_STATE)   # 0.125 * 0.8 = 10%

    train.to_csv(DATA / "train.csv", index=False)
    val.to_csv(DATA / "val.csv", index=False)
    test.to_csv(DATA / "test_en.csv", index=False)
    json.dump(labels, open(DATA / "labels.json", "w"), indent=2)

    # Dataset table for the paper
    stats = pd.DataFrame({
        "train": train["label"].value_counts(),
        "val": val["label"].value_counts(),
        "test": test["label"].value_counts(),
    }).fillna(0).astype(int)
    stats.loc["TOTAL"] = stats.sum()
    stats.to_csv(RESULTS / "dataset_stats.csv")
    print(stats)
    print(f"\nMean words per note: {df['text'].str.split().str.len().mean():.1f}")
    majority = test["label_id"].value_counts(normalize=True).max()
    print(f"Majority-class accuracy on test (sanity baseline): {majority:.3f}")
    print(f"Test sentences to translate: "
          f"{sum(len(SENT_SPLIT.split(t)) for t in test['text'])}")


if __name__ == "__main__":
    main()
