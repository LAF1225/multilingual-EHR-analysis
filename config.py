"""Shared settings for every script. Change things here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"            # put mtsamples.csv here
DATA = ROOT / "data" / "processed"          # cleaned splits + translations
MODELS_DIR = ROOT / "models"                # fine-tuned models
RESULTS = ROOT / "results"                  # tables, predictions, stats
FIGURES = ROOT / "figures"                  # plots for the paper

# The three models (short name -> Hugging Face ID)
MODELS = {
    "bert": "bert-base-cased",
    "bioclinical": "emilyalsentzer/Bio_ClinicalBERT",
    "xlmr": "xlm-roberta-base",
}

# Test conditions. "en_bt" = English back-translated from Urdu (a control for MT noise).
CONDITIONS = ["en", "en_bt", "ur_roman", "ur"]
CONDITION_FILES = {
    "en": "test_en.csv",
    "en_bt": "test_en_bt.csv",
    "ur_roman": "test_ur_roman.csv",
    "ur": "test_ur.csv",
}

SEEDS = [42, 43, 44]       # Week 1: use 42 only. Week 2: add 43 and 44.
MAX_LEN = 256              # max subword tokens fed to each classifier
MAX_WORDS = 150            # notes are cut to ~150 words (at a sentence boundary)
N_CLASSES = 6              # number of specialties to keep
RANDOM_STATE = 42          # for data splits and sampling

NLLB = "facebook/nllb-200-distilled-600M"

for d in (DATA_RAW, DATA, MODELS_DIR, RESULTS, RESULTS / "predictions", FIGURES):
    d.mkdir(parents=True, exist_ok=True)
