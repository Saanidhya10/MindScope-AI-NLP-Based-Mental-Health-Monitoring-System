"""Tests for trigger analysis logic (src/triggers.py) that doesn't need the zero-shot model."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import triggers as T  # noqa: E402


def test_keywords_match_whole_words_only():
    assert T.keyword_hits("My exams start Monday") == {"academic"}
    assert "family" not in T.keyword_hits("I love momentum trading")  # 'mom' inside a word
    assert "money" not in T.keyword_hits("Please pay attention")


def test_combine_merges_model_and_keywords_above_threshold():
    model = {"work": 0.95, "money": 0.4, "self": 0.1}
    got = dict(T.combine(model, "My boss yelled and my parents are upset", threshold=0.6, keyword_score=0.65))
    assert got.keys() == {"work", "family"}
    assert got["work"] == 0.95 and got["family"] == 0.65


def test_detector_without_model_uses_keywords():
    det = T.TriggerDetector({"triggers": {"model": "x", "threshold": 0.6, "keyword_score": 0.65}}, use_model=False)
    assert {k for k, _ in det.detect(["I barely slept before the exam"])[0]} == {"academic", "health"}


def test_summary_counts_difficult_share():
    entries = pd.DataFrame({"negative_score": [0.8, 0.7, 0.2, 0.1],
                            "triggers": [["academic"], ["academic", "family"], ["family"], []]})
    s = T.summary(entries)
    assert s.loc["academic", "entries"] == 2 and s.loc["academic", "difficult"] == 2
    assert s.loc["family", "difficult_share"] == 0.5
    assert s.loc["academic", "overall_share"] == 0.5
