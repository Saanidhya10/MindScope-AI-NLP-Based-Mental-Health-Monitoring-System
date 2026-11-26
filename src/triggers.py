"""Emotional trigger analysis (improvement 4): what an entry seems to be ABOUT, alongside how it feels.

Hybrid of two signals, each covering the other's blind spots:
  * zero-shot classification (NLI model, multi-label) - understands paraphrases ("I might lose my job" -> Work)
  * a keyword list - catches explicit mentions the model under-weights ("my parents expect good marks" -> Family)
A category is reported when either signal is confident. These are topics the person wrote about, not causes.
Settings: config.yaml -> triggers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.utils import load_config


@dataclass(frozen=True)
class Category:
    key: str
    label: str          # shown to the user
    hypothesis: str     # completes "This text is about {}."
    keywords: str       # regex alternatives, matched as whole words, case-insensitive


CATEGORIES = [
    Category("academic", "Studies", "studies, exams or grades",
             r"exams?|tests?|assignments?|homework|studies|studying|study|grades?|marks|semester|college|university|"
             r"school|class(?:es)?|lectures?|revision|revise|thesis|project deadline|professors?|teachers?|graduation|degree"),
    Category("work", "Work", "a job or work",
             r"job|work(?:ing|load)?|office|boss|manager|colleagues?|interviews?|internship|career|promotion|fired|"
             r"layoffs?|shift|meetings?|clients?|deadlines?"),
    Category("money", "Money", "money or finances",
             r"money|rent|bills?|debt|loans?|emi|salary|paid off|afford|expenses?|broke|savings|fees|finances?"),
    Category("social", "Friends and loneliness", "loneliness or friendships",
             r"lonely|loneliness|alone|isolated|friends?|friendships?|nobody|no one|left out|ignored|"
             r"haven't talked|social"),
    Category("relationship", "Relationships", "a romantic relationship, a partner or a breakup",
             r"boyfriend|girlfriend|partner|husband|wife|relationship|break ?up|broke up|dating|crush|ex|divorce"),
    Category("family", "Family", "family",
             r"family|parents?|mom|mum|mother|dad|father|brother|sister|siblings?|grandparents?|relatives|home"),
    Category("health", "Sleep and health", "sleep, tiredness or physical health",
             r"sleep|slept|insomnia|tired|exhausted|fatigue|sick|ill|illness|pain|headaches?|migraines?|fever|health|"
             r"doctor|hospital|can't sleep|awake"),
    Category("self", "Self and future", "self-worth or the future",
             r"failure|worthless|not good enough|future|purpose|confidence|self[- ]esteem|hate myself|useless|"
             r"what i'm doing with my life|how i look"),
]
BY_KEY = {c.key: c for c in CATEGORIES}
_KEYWORD_RE = {c.key: re.compile(rf"\b(?:{c.keywords})\b", re.I) for c in CATEGORIES}


def keyword_hits(text: str) -> set[str]:
    return {k for k, rx in _KEYWORD_RE.items() if rx.search(text or "")}


def combine(model_scores: dict[str, float], text: str, threshold: float, keyword_score: float) -> list[tuple[str, float]]:
    """Merge model scores with keyword hits; return [(category, score)] above threshold, strongest first."""
    scores = dict(model_scores)
    for k in keyword_hits(text):
        scores[k] = max(scores.get(k, 0.0), keyword_score)
    return sorted(((k, s) for k, s in scores.items() if s >= threshold), key=lambda x: -x[1])


class TriggerDetector:
    def __init__(self, cfg: dict | None = None, use_model: bool = True):
        self.cfg = (cfg or load_config())["triggers"]
        self.use_model = use_model
        self._zs = None

    def _model_scores(self, texts: list[str]) -> list[dict[str, float]]:
        if not self.use_model:
            return [{} for _ in texts]
        if self._zs is None:
            from transformers import pipeline

            self._zs = pipeline("zero-shot-classification", model=self.cfg["model"], device="cpu")
        hyp = {c.hypothesis: c.key for c in CATEGORIES}
        out = self._zs(texts, candidate_labels=list(hyp), hypothesis_template="This text is about {}.",
                       multi_label=True)
        out = [out] if isinstance(out, dict) else out
        return [{hyp[l]: float(s) for l, s in zip(o["labels"], o["scores"])} for o in out]

    def detect(self, texts: list[str]) -> list[list[tuple[str, float]]]:
        return [combine(m, t, self.cfg["threshold"], self.cfg["keyword_score"])
                for m, t in zip(self._model_scores(texts), texts)]


DIFFICULT = 0.5   # an entry counts as "difficult" when at least half its emotion score is difficult feelings


def summary(entries: pd.DataFrame) -> pd.DataFrame:
    """Per category: entries mentioning it, how many were difficult, and the difficult share vs the overall share.
    `entries` has columns negative_score and triggers (list of keys), as from MoodDatabase.get_triggers."""
    cols = ["label", "entries", "difficult", "difficult_share", "overall_share"]
    if entries is None or entries.empty:
        return pd.DataFrame(columns=cols)
    overall = float((entries["negative_score"] >= DIFFICULT).mean())
    rows = []
    for c in CATEGORIES:
        hit = entries[entries["triggers"].map(lambda t: c.key in t)]
        if len(hit):
            diff = int((hit["negative_score"] >= DIFFICULT).sum())
            rows.append({"key": c.key, "label": c.label, "entries": len(hit), "difficult": diff,
                         "difficult_share": diff / len(hit), "overall_share": overall})
    return pd.DataFrame(rows, columns=["key", *cols]).set_index("key").sort_values("entries", ascending=False)
