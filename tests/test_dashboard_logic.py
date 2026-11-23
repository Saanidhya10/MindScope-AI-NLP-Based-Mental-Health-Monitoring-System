"""Tests for the dashboard logic in src/ (trends.daily_frame / high_runs / personal_baseline,
recommend.assess / check_crisis). Uses synthetic probability vectors, so no trained model is needed."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import recommend, trends  # noqa: E402
from src.utils import load_config  # noqa: E402

CFG = load_config()
CALM = [0.6, 0.02, 0.02, 0.02, 0.02, 0.02, 0.1, 0.2]
HEAVY = [0.02, 0.4, 0.05, 0.05, 0.2, 0.25, 0.01, 0.02]


def frame(days, start="2025-01-01"):
    """One entry per (day offset, probability vector)."""
    ts = [pd.Timestamp(start) + pd.Timedelta(days=d, hours=10) for d, _ in days]
    return trends.build_prediction_frame(ts, [p for _, p in days])


def test_high_run_counts_consecutive_days():
    daily = trends.daily_frame(frame([(d, CALM) for d in range(5)] + [(d, HEAVY) for d in range(5, 12)]), CFG)
    assert daily["high_run"].iloc[0] == 0
    assert daily["high_run"].iloc[-1] >= CFG["trends"]["k"]


def test_long_gap_restarts_run():
    days = [(0, HEAVY), (1, HEAVY), (2, HEAVY), (3, HEAVY), (20, HEAVY)]
    daily = trends.daily_frame(frame(days), CFG)
    assert daily["high_run"].iloc[-1] == 1


def test_personal_baseline_defaults_until_two_entries():
    assert not trends.personal_baseline(frame([(0, CALM)]))["calibrated"]
    assert trends.personal_baseline(frame([(0, CALM), (1, HEAVY)]))["calibrated"]


def test_assess_sustained_distress_is_support_with_helplines():
    df = frame([(d, CALM) for d in range(6)] + [(d, HEAVY) for d in range(6, 14)])
    status = recommend.assess(trends.daily_frame(df, CFG), trends.personal_baseline(df), CFG)
    assert status["level"] == "support"
    assert status["helplines"]
    assert not recommend.check_wording(status["headline"])


def test_assess_calm_history_is_steady():
    df = frame([(d, CALM) for d in range(10)])
    status = recommend.assess(trends.daily_frame(df, CFG), trends.personal_baseline(df), CFG)
    assert status["level"] == "steady"
    assert status["helplines"] == []


def test_check_crisis_catches_phrases_and_ignores_ordinary_text():
    assert recommend.check_crisis("Some days I just want to end my life", CFG)
    assert recommend.check_crisis("marne ka man karta hai", CFG)
    assert not recommend.check_crisis("I killed it at the presentation today", CFG)


def test_every_configured_suggestion_is_non_diagnostic():
    for emotion in ["Joy", "Sadness", "Anger", "Fear", "Anxiety", "Stress", "Surprise", "Neutral"]:
        s = recommend.suggestion_for(emotion, CFG)
        assert s["title"] and s["text"]
        assert not recommend.check_wording(s["title"] + " " + s["text"])
    assert np.isfinite(CFG["trends"]["tau"])
