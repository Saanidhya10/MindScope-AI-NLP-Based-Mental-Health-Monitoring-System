#!/usr/bin/env python
"""Evaluate trigger detection on the hand-labelled set tests/data/trigger_eval.csv.

Compares keywords only, zero-shot model only, and the hybrid used by the app. Writes results/trigger_eval.csv.
    python scripts/eval_triggers.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import triggers as T  # noqa: E402
from src.utils import load_config  # noqa: E402


def main():
    cfg = load_config()
    tc = cfg["triggers"]
    data = pd.read_csv(ROOT / "tests/data/trigger_eval.csv").fillna("")
    gold = [set(filter(None, s.split("|"))) for s in data["labels"]]
    texts = data["text"].tolist()
    model = T.TriggerDetector(cfg)._model_scores(texts)
    systems = {
        "keywords only": [T.keyword_hits(t) for t in texts],
        "model only": [{k for k, s in m.items() if s >= tc["threshold"]} for m in model],
        "hybrid (app)": [{k for k, _ in T.combine(m, t, tc["threshold"], tc["keyword_score"])}
                         for m, t in zip(model, texts)],
    }
    mlb = MultiLabelBinarizer(classes=[c.key for c in T.CATEGORIES]).fit([])
    Y = mlb.transform(gold)
    rows = []
    for name, pred in systems.items():
        P = mlb.transform(pred)
        for avg in ("micro", "macro"):
            p, r, f, _ = precision_recall_fscore_support(Y, P, average=avg, zero_division=0)
            rows.append({"system": name, "average": avg, "precision": p, "recall": r, "f1": f})
        exact = sum(a == b for a, b in zip(gold, pred)) / len(gold)
        rows.append({"system": name, "average": "exact match", "precision": None, "recall": None, "f1": exact})
    res = pd.DataFrame(rows)
    out = ROOT / "results/trigger_eval.csv"
    res.to_csv(out, index=False)
    print(res.round(3).to_string(index=False))
    print(f"\n{len(texts)} hand-labelled sentences. Saved {out}")
    for t, g, p in zip(texts, gold, systems["hybrid (app)"]):
        if g != p:
            print(f"  miss: {t[:60]!r} gold={sorted(g)} got={sorted(p)}")


if __name__ == "__main__":
    main()
