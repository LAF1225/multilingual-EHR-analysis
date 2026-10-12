import pandas as pd

from config import DATA

try:
    import uroman as ur
except ImportError:
    raise SystemExit("Install it first:  pip install uroman")

df = pd.read_csv(DATA / "test_ur.csv")
u = ur.Uroman()
df["text"] = [u.romanize_string(str(t), lcode="urd") for t in df["text"]]
df.to_csv(DATA / "test_ur_roman.csv", index=False, encoding="utf-8")
print(df["text"].head(3).to_string())
print(f"Saved {DATA / 'test_ur_roman.csv'}")
