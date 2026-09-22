import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

from config import DATA, RESULTS
from utils import macro_f1

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test_en.csv")
labels = json.load(open(DATA / "labels.json"))

clf = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000, class_weight="balanced"),
)
clf.fit(train["text"], train["label_id"])
pred = clf.predict(test["text"])

res = {"macro_f1": macro_f1(test["label_id"], pred),
       "accuracy": accuracy_score(test["label_id"], pred)}
print(classification_report(test["label_id"], pred, target_names=labels, zero_division=0))
print(res)
json.dump(res, open(RESULTS / "baseline_tfidf_en.json", "w"), indent=2)
