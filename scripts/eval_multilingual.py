#!/usr/bin/env python
"""Compare English-only reading with MindScope's multilingual + context pipeline on tests/data/multilingual_eval.csv.

    python scripts/eval_multilingual.py

English-only = the emotion model on the raw text, with the Hinglish lists switched off.
Pipeline     = Hindi translation + Hinglish phrase/word lists + past-vs-now and someone-else rules (src/multilingual.py).
The labels were written by hand for this project and should be checked by a second person.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.preprocess as pre  # noqa: E402
from src import multilingual as M  # noqa: E402
from src.inference import EmotionPredictor  # noqa: E402
from src.utils import LABELS, NEGATIVE_LABELS, load_config  # noqa: E402


def valence(p) -> str:
    neg = sum(p[LABELS.index(e)] for e in NEGATIVE_LABELS)
    pos = p[LABELS.index("Joy")]
    if neg >= 0.5:
        return "negative"
    return "positive" if pos >= 0.4 and pos > neg else "neutral"


def main():
    cfg = load_config()
    data = pd.read_csv(ROOT / "tests/data/multilingual_eval.csv")
    predictor = EmotionPredictor.from_config(cfg)
    translator = M.Translator(cfg["multilingual"]["translation_model"])

    pre.HINGLISH_ENABLED = False
    base = predictor.predict_proba(data["text"].tolist())
    pre.HINGLISH_ENABLED = True
    piped = np.array([M.read(t, predictor.predict_proba, translator, cfg).probs for t in data["text"]])

    rows = []
    for name, P in [("English-only", base), ("MindScope pipeline", piped)]:
        emo = [LABELS[i] for i in P.argmax(1)]
        val = [valence(p) for p in P]
        for lang, g in data.groupby("language"):
            idx = g.index
            rows.append({"system": name, "language": lang, "n": len(g),
                         "emotion_acc": float(np.mean([emo[i] == data.at[i, "emotion"] for i in idx])),
                         "valence_acc": float(np.mean([val[i] == data.at[i, "valence"] for i in idx]))})
        rows.append({"system": name, "language": "all", "n": len(data),
                     "emotion_acc": float(np.mean(np.array(emo) == data["emotion"].to_numpy())),
                     "valence_acc": float(np.mean(np.array(val) == data["valence"].to_numpy()))})
    res = pd.DataFrame(rows)
    out = ROOT / "results/multilingual_eval.csv"
    res.to_csv(out, index=False)
    print(res.round(3).to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
