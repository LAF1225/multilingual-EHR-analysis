import pandas as pd
q = pd.read_csv("results/mt_quality_per_note.csv")
en = pd.read_csv("data/processed/test_en.csv").set_index("id")
bt = pd.read_csv("data/processed/test_en_bt.csv").set_index("id")
print("Notes with ratio > 1.5:", (q.len_ratio > 1.5).sum(), "of", len(q))
i = "doc00693"
print("\nORIGINAL:\n", en.loc[i, "text"][:400])
print("\nBACK-TRANSLATED:\n", bt.loc[i, "text"][:600])
