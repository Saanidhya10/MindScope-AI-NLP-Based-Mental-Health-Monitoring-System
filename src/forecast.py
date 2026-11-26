"""Emotional outlook: forecasting the daily difficult-feelings share 1-7 days ahead (improvement 1). EXPERIMENTAL.

No public dataset has real per-person emotional timelines, so the learned model is trained and every method is
evaluated on SIMULATED timelines (regime changes, noise, skipped days, sleep/stress that move with mood). The backtest
shows how the methods compare under those assumptions; it cannot show accuracy on real people. The app labels the
outlook as a rough guide, and an outlook never triggers the support message by itself.

Methods: persistence (last value), EMA carried forward, damped 7-day trend, and gradient boosting on lag features.
Prediction intervals come from the backtest's residual quantiles per horizon.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZONS = range(1, 8)
FEATURES = ["last", "ema", "mean3", "slope7", "std7", "gap", "sleep", "stress", "h"]


# ------------------------------------------------------------------ simulated timelines
def simulate(n_series: int = 300, days: int = 60, seed: int = 0) -> list[pd.DataFrame]:
    """INVENTED daily timelines: difficult share drifting between regimes, with noise and ~30% skipped days.
    Sleep and stress loosely follow the difficult share. Columns: difficult, sleep_hours, stress."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_series):
        level, vals, d = rng.uniform(0.05, 0.6), [], 0
        while d < days:
            seg, target = int(rng.integers(5, 16)), rng.uniform(0.05, 0.85)
            for k in range(seg):
                level += (target - level) / max(seg - k, 1) + rng.normal(0, 0.03)
                vals.append(level)
            d += seg
        x = np.clip(np.array(vals[:days]) + rng.normal(0, 0.08, days), 0, 1)
        idx = pd.date_range("2025-01-01", periods=days)
        df = pd.DataFrame({"difficult": x,
                           "sleep_hours": np.clip(7.6 - 3.2 * (x - 0.3) + rng.normal(0, 0.6, days), 3, 10),
                           "stress": np.clip(np.round(2 + 7 * x + rng.normal(0, 1.0, days)), 1, 10)}, index=idx)
        df.loc[rng.random(days) < 0.3] = np.nan  # skipped days
        out.append(df)
    return out


# ------------------------------------------------------------------ features
def features_at(series: pd.DataFrame, t: pd.Timestamp, alpha: float = 0.3) -> dict | None:
    """Features from everything observed up to and including day t (calendar-indexed frame with NaN gaps)."""
    past = series.loc[:t]
    obs = past["difficult"].dropna()
    if len(obs) < 3:
        return None
    last7 = obs[obs.index > t - pd.Timedelta(days=7)]
    slope = np.polyfit((last7.index - t).days, last7.values, 1)[0] if len(last7) >= 3 else 0.0
    return {"last": obs.iloc[-1], "ema": obs.ewm(alpha=alpha, adjust=False).mean().iloc[-1],
            "mean3": obs.iloc[-3:].mean(), "slope7": slope, "std7": last7.std() if len(last7) >= 2 else 0.0,
            "gap": (t - obs.index[-1]).days,
            "sleep": past["sleep_hours"].dropna().iloc[-1] if "sleep_hours" in past and past["sleep_hours"].notna().any() else np.nan,
            "stress": past["stress"].dropna().iloc[-1] if "stress" in past and past["stress"].notna().any() else np.nan}


def training_rows(timelines: list[pd.DataFrame], min_day: int = 7) -> pd.DataFrame:
    rows = []
    for s in timelines:
        for t in s.index[min_day:]:
            f = features_at(s, t)
            if f is None:
                continue
            for h in HORIZONS:
                tgt_day = t + pd.Timedelta(days=h)
                if tgt_day in s.index and pd.notna(s.at[tgt_day, "difficult"]):
                    rows.append({**f, "h": h, "y": s.at[tgt_day, "difficult"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ methods
def predict_simple(method: str, f: dict, h: int) -> float:
    if method == "persistence":
        return f["last"]
    if method == "ema":
        return f["ema"]
    if method == "trend":  # damped: the trend fades over the horizon
        damp = sum(0.8 ** k for k in range(1, h + 1))
        return float(np.clip(f["ema"] + f["slope7"] * damp, 0, 1))
    raise ValueError(method)


def fit_gbm(rows: pd.DataFrame, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=24, random_state=seed)
    return model.fit(rows[FEATURES], rows["y"])


def backtest(train_tl: list[pd.DataFrame], test_tl: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict, object]:
    """MAE per method and horizon on held-out timelines, plus residual quantiles per horizon for each method."""
    gbm = fit_gbm(training_rows(train_tl))
    test = training_rows(test_tl)
    preds = {m: test.apply(lambda r: predict_simple(m, r, int(r["h"])), axis=1) for m in ("persistence", "ema", "trend")}
    preds["gbm"] = pd.Series(np.clip(gbm.predict(test[FEATURES]), 0, 1), index=test.index)
    rows, resid_q = [], {}
    for m, p in preds.items():
        err = test["y"] - p
        for h in HORIZONS:
            e = err[test["h"] == h]
            rows.append({"method": m, "horizon": h, "mae": float(e.abs().mean()), "n": int(len(e))})
            resid_q.setdefault(m, {})[h] = (float(e.quantile(0.1)), float(e.quantile(0.9)))
    return pd.DataFrame(rows), resid_q, gbm


# ------------------------------------------------------------------ the app's forecaster
@dataclass
class Outlook:
    day: pd.Timestamp
    horizon: int
    predicted: float
    low: float
    high: float


class Forecaster:
    """Trained once on simulated timelines (cached under models/forecast/), then applied to a person's own days."""

    def __init__(self, method: str, gbm, resid_q: dict, scores: pd.DataFrame):
        self.method, self.gbm, self.resid_q, self.scores = method, gbm, resid_q, scores

    @classmethod
    def build(cls, method: str | None = None, seed: int = 0) -> "Forecaster":
        train, test = simulate(300, seed=seed), simulate(80, seed=seed + 1)
        scores, resid_q, gbm = backtest(train, test)
        best = scores.groupby("method")["mae"].mean().idxmin()
        return cls(method or best, gbm, resid_q, scores)

    @classmethod
    def load_or_build(cls, path) -> "Forecaster":
        import joblib
        from pathlib import Path

        path = Path(path)
        if path.exists():
            return joblib.load(path)
        f = cls.build()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(f, path)
        return f

    def outlook(self, daily: pd.DataFrame, today: pd.Timestamp | None = None) -> list[Outlook]:
        """`daily` is calendar-indexed with a `difficult` column (and optional sleep_hours, stress)."""
        today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
        series = daily.reindex(pd.date_range(daily.index.min(), today)) if len(daily) else daily
        f = features_at(series, today)
        if f is None:
            return []
        out = []
        for h in HORIZONS:
            if self.method == "gbm":
                p = float(np.clip(self.gbm.predict(pd.DataFrame([{**f, "h": h}])[FEATURES])[0], 0, 1))
            else:
                p = predict_simple(self.method, f, h)
            lo_q, hi_q = self.resid_q[self.method][h]
            out.append(Outlook(today + pd.Timedelta(days=h), h, p, float(np.clip(p + lo_q, 0, 1)),
                               float(np.clip(p + hi_q, 0, 1))))
        return out


def has_enough_data(daily: pd.DataFrame, today: pd.Timestamp | None = None, min_days: int = 7) -> bool:
    """At least `min_days` days with entries in the last 14 days."""
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    recent = daily[daily.index > today - pd.Timedelta(days=14)]
    return int(recent["difficult"].notna().sum()) >= min_days


def describe(current: float, o: Outlook, tolerance: float = 0.08) -> tuple[str, str]:
    """(direction words, confidence words) for one outlook."""
    if o.predicted > current + tolerance:
        direction = "may get heavier"
    elif o.predicted < current - tolerance:
        direction = "may ease"
    else:
        direction = "likely to stay about the same"
    width = o.high - o.low
    confidence = "moderate" if width < 0.35 else "low"
    return direction, confidence
