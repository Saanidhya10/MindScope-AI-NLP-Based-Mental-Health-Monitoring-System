"""Tests for adaptive activity ranking (src/activities.py) and the activity log storage."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import activities as A  # noqa: E402
from src.database import MoodDatabase  # noqa: E402


def log(rows):
    return pd.DataFrame(rows, columns=["timestamp", "activity", "mood_before", "mood_after", "helpful"])


def mean_rank(key, recent, history, runs=300):
    rng = np.random.default_rng(0)
    pos = [next(i for i, r in enumerate(A.rank(recent, history, rng)) if r["activity"].key == key) for _ in range(runs)]
    return np.mean(pos)


def test_every_activity_is_ranked_once():
    ranked = A.rank(["Stress"], None, np.random.default_rng(1))
    assert sorted(r["activity"].key for r in ranked) == sorted(a.key for a in A.CATALOG)


def test_suitable_activities_rank_higher_with_no_history():
    assert mean_rank("breathing", ["Anxiety"], None) < mean_rank("gratitude", ["Anxiety"], None)


def test_activities_that_helped_move_up_and_unhelpful_ones_move_down():
    helped = log([("t", "gratitude", 4, 7, 1)] * 6)
    unhelpful = log([("t", "breathing", 5, 5, 0)] * 6)
    assert mean_rank("gratitude", ["Anxiety"], helped) < mean_rank("gratitude", ["Anxiety"], None)
    assert mean_rank("breathing", ["Anxiety"], unhelpful) > mean_rank("breathing", ["Anxiety"], None)


def test_feedback_stats_and_reason_text():
    history = log([("t", "breathing", 4, 6, 1), ("t", "breathing", 5, 6, 1), ("t", "breathing", 5, 5, 0)])
    stats = A.feedback_stats(history)
    assert stats.loc["breathing", "done"] == 3 and stats.loc["breathing", "helped"] == 2
    assert stats.loc["breathing", "mood_change"] == pytest.approx(1.0)
    reason = next(r["reason"] for r in A.rank(["Stress"], history, np.random.default_rng(0))
                  if r["activity"].key == "breathing")
    assert reason == "Helped you 2 of 3 times"


def test_activity_log_goals_and_notes_need_consent_and_are_deleted():
    db = MoodDatabase(":memory:")
    with pytest.raises(PermissionError):
        db.log_activity("u", "breathing", 4, 6, True)
    db.set_consent("u", True)
    db.log_activity("u", "breathing", 4, 6, True)
    db.add_goal("u", "Sleep by 11:30")
    db.add_note("u", "gratitude", "Good chai with a friend")
    assert len(db.get_activity_log("u")) == 1 and len(db.get_goals("u")) == 1 and len(db.get_notes("u", "gratitude")) == 1
    db.delete_user_data("u", keep_user=True)
    assert db.get_activity_log("u").empty and db.get_goals("u").empty and db.get_notes("u", "gratitude").empty
