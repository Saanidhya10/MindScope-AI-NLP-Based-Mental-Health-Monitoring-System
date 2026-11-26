"""Tests for the emotional outlook (src/forecast.py). Uses small simulations so it stays fast."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import forecast as Fc  # noqa: E402


def frame(values, start="2026-01-01"):
    return pd.DataFrame({"difficult": values}, index=pd.date_range(start, periods=len(values)))


def test_features_need_three_observations_and_handle_gaps():
    df = frame([0.2, np.nan, 0.3, np.nan, 0.5])
    f = Fc.features_at(df, df.index[-1])
    assert f["last"] == 0.5 and f["gap"] == 0 and f["slope7"] > 0
    assert Fc.features_at(frame([0.2, 0.3]), pd.Timestamp("2026-01-02")) is None


def test_simple_methods():
    f = {"last": 0.6, "ema": 0.5, "slope7": 0.05}
    assert Fc.predict_simple("persistence", f, 3) == 0.6
    assert Fc.predict_simple("ema", f, 3) == 0.5
    assert 0.5 < Fc.predict_simple("trend", f, 3) <= 1


def test_backtest_runs_and_learned_model_is_competitive():
    scores, resid_q, _ = Fc.backtest(Fc.simulate(40, seed=0), Fc.simulate(15, seed=1))
    mae = scores.groupby("method")["mae"].mean()
    assert set(mae.index) == {"persistence", "ema", "trend", "gbm"}
    assert mae["gbm"] <= mae["persistence"] * 1.1
    lo, hi = resid_q["gbm"][7]
    assert lo < 0 < hi


def test_outlook_gives_seven_days_with_ranges_around_prediction():
    scores, resid_q, gbm = Fc.backtest(Fc.simulate(30, seed=2), Fc.simulate(10, seed=3))
    fc = Fc.Forecaster("gbm", gbm, resid_q, scores)
    daily = frame(np.linspace(0.2, 0.7, 14))
    out = fc.outlook(daily, daily.index[-1])
    assert len(out) == 7 and [o.horizon for o in out] == list(range(1, 8))
    assert all(o.low <= o.predicted <= o.high for o in out)
    assert out[0].day == daily.index[-1] + pd.Timedelta(days=1)


def test_enough_data_and_describe():
    daily = frame([0.3] * 10, start=str((pd.Timestamp.today() - pd.Timedelta(days=9)).date()))
    assert Fc.has_enough_data(daily, min_days=7)
    o = Fc.Outlook(pd.Timestamp.today(), 3, 0.6, 0.45, 0.75)
    assert Fc.describe(0.3, o) == ("may get heavier", "moderate")
    assert Fc.describe(0.6, o)[0] == "likely to stay about the same"
