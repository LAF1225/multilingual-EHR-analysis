import pandas as pd
from transformers import AutoTokenizer
from config import MODELS, RESULTS


PAIRS = [
    ("The patient presented with chest pain and shortness of breath.",
     "مریض کو سینے میں درد اور سانس لینے میں دشواری کی شکایت تھی۔"),
    ("Blood pressure was 150/90 and the heart rate was 98.",
     "بلڈ پریشر 150/90 اور دل کی دھڑکن 98 تھی۔"),
    ("He is a known patient of diabetes and hypertension.",
     "وہ ذیابیطس اور بلند فشارِ خون کا پرانا مریض ہے۔"),
    ("An MRI of the knee showed a torn meniscus.",
     "گھٹنے کے ایم آر آئی میں مینیسکس پھٹا ہوا دکھائی دیا۔"),
    ("The patient was started on aspirin and metoprolol.",
     "مریض کو اسپرین اور میٹوپرولول شروع کروائی گئی۔"),
    ("She denies fever, nausea, or vomiting.",
     "وہ بخار، متلی یا الٹی کی شکایت سے انکار کرتی ہے۔"),
    ("Colonoscopy revealed a small polyp in the sigmoid colon.",
     "کولونوسکوپی میں سگمائیڈ کولون میں ایک چھوٹا پولپ پایا گیا۔"),
    ("The patient complains of severe headache and blurred vision.",
     "مریض شدید سر درد اور دھندلا نظر آنے کی شکایت کرتا ہے۔"),
]


def romanize(texts):
    try:
        import uroman as ur
        u = ur.Uroman()
        return [u.romanize_string(t, lcode="urd") for t in texts]
    except Exception as e:  # uroman missing or API changed
        print(f"[warn] uroman unavailable ({e}); skipping Roman Urdu.")
        return None


def token_stats(tok, texts):
    words = tokens = unks = multi = 0
    for t in texts:
        for w in t.split():
            pieces = tok.tokenize(w)
            words += 1
            tokens += len(pieces)
            unks += sum(p == tok.unk_token for p in pieces)
            multi += len(pieces) > 1
    return {
        "fertility": tokens / words,            # subword tokens per word
        "pct_words_split": 100 * multi / words, # words broken into >1 piece
        "pct_unk_tokens": 100 * unks / tokens,  # share of [UNK] tokens
    }


def main():
    en = [p[0] for p in PAIRS]
    ur = [p[1] for p in PAIRS]
    sets = {"en": en, "ur": ur}
    roman = romanize(ur)
    if roman:
        sets["ur_roman"] = roman
        print("Example Roman Urdu:", roman[0])

    rows = []
    for key, name in MODELS.items():
        tok = AutoTokenizer.from_pretrained(name)
        print(f"\n=== {key} ({name}) | vocab size {tok.vocab_size}")
        print("  EN :", tok.tokenize(en[0]))
        print("  UR :", tok.tokenize(ur[0]))
        for cond, texts in sets.items():
            rows.append({"model": key, "condition": cond, **token_stats(tok, texts)})

    df = pd.DataFrame(rows).round(2)
    print("\n", df.to_string(index=False))
    df.to_csv(RESULTS / "day1_tokenizer_check.csv", index=False)

    # Confirm Bio_ClinicalBERT and bert-base-cased really share a vocabulary
    a = AutoTokenizer.from_pretrained(MODELS["bert"]).get_vocab()
    b = AutoTokenizer.from_pretrained(MODELS["bioclinical"]).get_vocab()
    print(f"\nShared vocabulary bert vs bioclinical: {a == b} "
          f"(overlap {len(set(a) & set(b))}/{len(a)})")


if __name__ == "__main__":
    main()
