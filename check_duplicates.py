import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test_en.csv")

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train = vec.fit_transform(train["text"])
X_test = vec.transform(test["text"])

sim = cosine_similarity(X_test, X_train)
best = sim.max(axis=1)
best_idx = sim.argmax(axis=1)

print("Similarity of each test note to its closest training note:")
print(f"  mean {best.mean():.2f} | median {pd.Series(best).median():.2f}")
for t in (0.9, 0.8, 0.7, 0.5):
    print(f"  share above {t}: {(best > t).mean():.1%}")

i = best.argmax()
print(f"\nClosest pair (similarity {best[i]:.2f}):")
print("TEST :", test['text'].iloc[i][:300])
print("TRAIN:", train['text'].iloc[best_idx[i]][:300])
