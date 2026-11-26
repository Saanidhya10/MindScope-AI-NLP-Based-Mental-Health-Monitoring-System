"""Tests for the personal emotional profile (src/fingerprint.py)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import fingerprint as F  # noqa: E402
from src.trends import build_prediction_frame  # noqa: E402

CALM = [0.6, 0.02, 0.02, 0.02, 0.02, 0.02, 0.1, 0.2]
HEAVY = [0.02, 0.4, 0.05, 0.05, 0.2, 0.25, 0.01, 0.02]


def hist(pattern, start="2026-01-01"):
    ts = [pd.Timestamp(start) + pd.Timedelta(days=i, hours=10) for i in range(len(pattern))]
    return build_prediction_frame(ts, pattern)


def dims(p):
    return {d.name: d for d in p["dimensions"]}


def test_recent_rise_in_difficult_feelings_is_reported_as_higher_than_usual():
    p = F.profile(hist([CALM] * 14 + [HEAVY] * 7))
    assert p["calibrated"]
    d = dims(p)["Difficult feelings"]
    assert d.change == "higher" and d.better == "lower"


def test_short_history_is_not_calibrated():
    assert not F.profile(hist([CALM] * 5))["calibrated"]


def test_recovery_days_measures_time_back_to_usual():
    idx = pd.date_range("2026-01-01", periods=8)
    s = pd.Series([0.2, 0.8, 0.7, 0.2, 0.2, 0.9, 0.2, 0.2], index=idx)
    assert F.recovery_days(s, usual_level=0.3) == 1.5  # 2 days, then 1 day


def test_unusual_days_needs_enough_history_and_explains_itself():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=30)
    feats = pd.DataFrame({"difficult": rng.normal(0.25, 0.03, 30).clip(0, 1),
                          "joy": rng.normal(0.4, 0.03, 30).clip(0, 1),
                          "sleep_hours": rng.normal(7.5, 0.3, 30)}, index=idx)
    feats.iloc[20, feats.columns.get_loc("difficult")] = 0.95
    feats.iloc[20, feats.columns.get_loc("sleep_hours")] = 3.5
    found = F.unusual_days(feats)
    assert any(u["day"] == idx[20] for u in found)
    reasons = next(u["reasons"] for u in found if u["day"] == idx[20])
    assert any("difficult feelings" in r or "sleep" in r for r in reasons)
    assert F.unusual_days(feats.head(10)) == []


def test_variability_labels():
    assert F.variability_label(0.05) == "low" and F.variability_label(0.3) == "high"
