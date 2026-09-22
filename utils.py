"""Helper functions used by several scripts."""
import numpy as np
import pandas as pd
from scipy.stats import binom, chi2
from sklearn.metrics import f1_score

from config import DATA, CONDITION_FILES, RESULTS


def load_condition(cond):
    """Load one test condition, sorted by id so every condition lines up row by row."""
    path = DATA / CONDITION_FILES[cond]
    if not path.exists():
        return None
    return pd.read_csv(path).sort_values("id").reset_index(drop=True)


def pred_path(model_key, seed, cond):
    return RESULTS / "predictions" / f"{model_key}_seed{seed}_{cond}.npz"


def load_preds(model_key, seed, cond):
    """Returns (ids, labels, probs) or None if that prediction file does not exist."""
    p = pred_path(model_key, seed, cond)
    if not p.exists():
        return None
    z = np.load(p)
    return z["ids"], z["labels"], z["probs"]


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def ece(probs, labels, n_bins=15):
    """Expected Calibration Error (Guo et al., 2017), equal-width bins."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return total


def mcnemar_p(correct_a, correct_b):
    """McNemar test on paired correctness vectors.
    Exact binomial version when discordant pairs < 25, else chi-square with continuity correction."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return 1.0, b, c
    if n < 25:
        p = min(1.0, 2 * binom.cdf(min(b, c), n, 0.5))
    else:
        stat = (abs(b - c) - 1) ** 2 / n
        p = chi2.sf(stat, 1)
    return float(p), b, c


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj
