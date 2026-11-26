"""Tests for the daily check-in (src/lifestyle.py and the check-in storage)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import lifestyle  # noqa: E402
from src.database import MoodDatabase  # noqa: E402


def days(n):
    return pd.date_range("2026-01-01", periods=n, freq="D")


def test_no_insights_before_minimum_days():
    idx = days(lifestyle.MIN_DAYS - 1)
    df = pd.DataFrame({"sleep_hours": np.arange(len(idx)), "mood": np.arange(len(idx))}, index=idx)
    assert lifestyle.associations(df) == []


def test_strong_same_day_association_is_found_and_worded_as_association():
    idx = days(20)
    sleep = np.linspace(4, 9, 20)
    df = pd.DataFrame({"sleep_hours": sleep, "mood": np.round(sleep)}, index=idx)
    found = lifestyle.associations(df)
    same_day = [a for a in found if a["habit"] == "sleep_hours" and a["outcome"] == "mood" and a["lag"] == 0]
    assert same_day and same_day[0]["rho"] > 0.9
    assert "tended to" in same_day[0]["text"] and "cause" not in same_day[0]["text"]


def test_next_day_association_uses_previous_calendar_day():
    rng = np.random.default_rng(0)
    idx = days(30)
    sleep = rng.uniform(4, 9, 30)
    mood = np.r_[5, sleep[:-1]]  # today's mood follows yesterday's sleep
    found = lifestyle.associations(pd.DataFrame({"sleep_hours": sleep, "mood": mood}, index=idx))
    assert any(a["lag"] == 1 and a["habit"] == "sleep_hours" and a["rho"] > 0.9 for a in found)


def test_unrelated_noise_is_not_reported():
    rng = np.random.default_rng(1)
    idx = days(25)
    df = pd.DataFrame({"screen_hours": rng.uniform(0, 8, 25), "stress": rng.integers(1, 10, 25)}, index=idx)
    assert not [a for a in lifestyle.associations(df) if abs(a["rho"]) > 0.6]


def test_checkins_need_consent_and_are_deleted_with_user_data():
    db = MoodDatabase(":memory:")
    with pytest.raises(PermissionError):
        db.save_checkin("u", "2026-01-01", {"mood": 5})
    db.set_consent("u", True)
    db.save_checkin("u", "2026-01-01", {"mood": 5, "sleep_hours": 7})
    db.save_checkin("u", "2026-01-01", {"mood": 8, "sleep_hours": 6})  # same day replaces
    got = db.get_checkins("u")
    assert len(got) == 1 and got.iloc[0]["mood"] == 8
    db.delete_user_data("u", keep_user=True)
    assert db.get_checkins("u").empty
