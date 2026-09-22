import pandas as pd, re
bt = pd.read_csv("data/processed/test_en_bt.csv")

def max_repeat(text):
    words = str(text).lower().split()
    best = run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    # also catch repeated 3-word phrases
    tri = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
    top = pd.Series(tri).value_counts().max() if tri else 0
    return best, top

stats = bt["text"].apply(max_repeat)
bt["max_word_run"] = [s[0] for s in stats]
bt["max_trigram_count"] = [s[1] for s in stats]
degenerate = bt[bt["max_trigram_count"] >= 5]
print(f"Notes with a 3-word phrase repeated 5+ times: {len(degenerate)} of {len(bt)} ({len(degenerate)/len(bt):.1%})")
print(degenerate[["id", "max_trigram_count"]].sort_values("max_trigram_count", ascending=False).head(10).to_string(index=False))
