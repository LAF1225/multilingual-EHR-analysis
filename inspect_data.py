import pandas as pd
import re
from collections import Counter

df = pd.read_csv("data/processed/test_en.csv")

# 1. Read three notes from different specialties
for _, row in df.groupby("label").head(1).head(3).iterrows():
    print(f"\n=== {row['label']} ===\n{row['text']}\n")

# 2. How often does a note mention its own specialty by name?
keywords = {
    "Cardiovascular / Pulmonary": ["cardio", "pulmonar"],
    "Orthopedic": ["orthop"],
    "Gastroenterology": ["gastroenterolog"],
    "Urology": ["urolog"],
    "Obstetrics / Gynecology": ["obstetric", "gynecolog"],
    "ENT - Otolaryngology": ["otolaryngolog", " ent "],
}
print("Share of test notes that name their own specialty:")
for label, words in keywords.items():
    notes = df[df["label"] == label]["text"].str.lower()
    share = notes.apply(lambda t: any(w in t for w in words)).mean()
    print(f"  {label:30s} {share:.0%}")


# 3. Do OTHER specialties mention these words just as often?
df["mentions_cardio"] = df["text"].str.lower().str.contains(r"cardio|pulmonar")
print("\nShare of notes mentioning 'cardio' or 'pulmonar', by specialty:")
print(df.groupby("label")["mentions_cardio"].mean().map("{:.0%}".format))

# 4. Which exact words are matching, and in what context?
pattern = re.compile(r"\w*cardio\w*|\w*pulmonar\w*", re.IGNORECASE)
cv = df[(df["label"] == "Cardiovascular / Pulmonary") & df["mentions_cardio"]]
words = Counter(m.lower() for t in cv["text"] for m in pattern.findall(t))
print("\nMost common matching words in Cardiovascular notes:", words.most_common(8))
print("\nExample contexts:")
for t in cv["text"].head(6):
    m = pattern.search(t)
    print("  ...", t[max(0, m.start() - 50): m.end() + 50], "...")
