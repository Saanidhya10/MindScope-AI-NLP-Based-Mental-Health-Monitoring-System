"""Personal emotional profile, the "emotional fingerprint" (improvement 2).

Compares what is USUAL for this person (all days before the recent window) with the RECENT window (last 7 days), across
several dimensions, so the same score can mean different things for different people. Unusual days are found with an
Isolation Forest over each day's features (journal + check-in), and explained by the features furthest from the
person's own median. Everything is relative to the person, not to population norms, and nothing here is a diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.trends import PROB_COLS, aggregate

RECENT_DAYS = 7
MIN_DAYS_FOR_ANOMALIES = 14
DIFFICULT = 0.5


@dataclass
class Dimension:
    name: str
    usual: str
    recent: str
    change: str          # "higher" | "lower" | "similar" | ""
    better: str          # which direction is better: "higher" | "lower" | ""
    note: str = ""


def daily_features(hist: pd.DataFrame, checkins: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per day with journal features (difficult share, joy share, entries) and check-in values."""
    if hist.empty:
        return pd.DataFrame()
    d = aggregate(hist, "daily")
    feats = pd.DataFrame({"difficult": d["neg_score"], "joy": d["p_Joy"], "entries": d["n_messages"]})
    feats.index = pd.to_datetime(feats.index).normalize()
    if checkins is not None and len(checkins):
        c = checkins.copy()
        c.index = pd.to_datetime(c.index).normalize()
        feats = feats.join(c[["sleep_hours", "mood", "stress", "energy"]], how="outer")
    return feats.sort_index()


def split_windows(feats: pd.DataFrame, recent_days: int = RECENT_DAYS) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """(usual, recent, calibrated). Usual falls back to all days when there isn't a week of earlier history."""
    if feats.empty:
        return feats, feats, False
    cut = feats.index.max() - pd.Timedelta(days=recent_days - 1)
    recent, earlier = feats[feats.index >= cut], feats[feats.index < cut]
    return (earlier, recent, True) if len(earlier) >= recent_days else (feats, recent, False)


def variability_label(std: float) -> str:
    return "low" if std < 0.12 else "medium" if std < 0.22 else "high"


def recovery_days(difficult: pd.Series, usual_level: float) -> float | None:
    """Average number of days from a difficult day until the difficult share is back at or below the usual level."""
    s = difficult.dropna()
    waits, i = [], 0
    while i < len(s):
        if s.iloc[i] >= DIFFICULT and s.iloc[i] > usual_level:
            j = i + 1
            while j < len(s) and s.iloc[j] > usual_level:
                j += 1
            if j < len(s):
                waits.append((s.index[j] - s.index[i]).days)
            i = j
        i += 1
    return float(np.mean(waits)) if waits else None


def _compare(name, usual, recent, fmt, better, tol, note=""):
    if usual is None or recent is None or pd.isna(usual) or pd.isna(recent):
        return Dimension(name, "Not enough data" if usual is None or pd.isna(usual) else fmt(usual),
                         "Not enough data" if recent is None or pd.isna(recent) else fmt(recent), "", better, note)
    change = "similar" if abs(recent - usual) <= tol else ("higher" if recent > usual else "lower")
    return Dimension(name, fmt(usual), fmt(recent), change, better, note)


def profile(hist: pd.DataFrame, checkins: pd.DataFrame | None = None,
            trigger_entries: pd.DataFrame | None = None) -> dict:
    """Everything the Emotional profile page shows."""
    feats = daily_features(hist, checkins)
    usual, recent, calibrated = split_windows(feats)
    pct = lambda v: f"{v:.0%}"  # noqa: E731
    dims = []
    if "difficult" in feats:
        dims.append(_compare("Difficult feelings", usual["difficult"].mean(), recent["difficult"].mean(), pct,
                             "lower", 0.07, "Share of sadness, anger, fear, anxiety and stress in your entries"))
        dims.append(_compare("Positive feelings", usual["joy"].mean(), recent["joy"].mean(), pct, "higher", 0.07,
                             "Share of joy in your entries"))
        u_std, r_std = usual["difficult"].std(), recent["difficult"].std()
        dims.append(Dimension("Ups and downs",
                              variability_label(u_std) if pd.notna(u_std) else "Not enough data",
                              variability_label(r_std) if pd.notna(r_std) else "Not enough data",
                              "" if pd.isna(u_std) or pd.isna(r_std) or variability_label(u_std) == variability_label(r_std)
                              else ("higher" if r_std > u_std else "lower"), "lower",
                              "How much difficult feelings change from day to day"))
        rec = recovery_days(feats["difficult"], float(usual["difficult"].mean()))
        dims.append(Dimension("Bouncing back", "Not enough data" if rec is None else
                              f"{rec:.1f} day{'s' if rec != 1 else ''}", "", "", "lower",
                              "Average time after a difficult day until you're back to your usual level"))
    for col, name, better, fmt, tol in [("sleep_hours", "Sleep", "higher", lambda v: f"{v:.1f} h", 0.5),
                                        ("mood", "Self-rated mood", "higher", lambda v: f"{v:.1f} / 10", 0.8),
                                        ("stress", "Self-rated stress", "lower", lambda v: f"{v:.1f} / 10", 0.8)]:
        if col in feats and feats[col].notna().sum() >= 3:
            dims.append(_compare(name, usual[col].mean(), recent[col].mean(), fmt, better, tol, "From your check-ins"))

    mix_usual = _emotion_mix(hist, usual.index if calibrated else None)
    mix_recent = _emotion_mix(hist, recent.index)
    topics = _difficult_topics(trigger_entries)
    return {"dimensions": dims, "calibrated": calibrated, "days": len(feats), "mix_usual": mix_usual,
            "mix_recent": mix_recent, "topics": topics, "unusual": unusual_days(feats)}


def _emotion_mix(hist: pd.DataFrame, days) -> pd.Series:
    h = hist if days is None else hist[hist["timestamp"].dt.normalize().isin(days)]
    return h[PROB_COLS].mean().rename(lambda c: c[2:]) if len(h) else pd.Series(dtype=float)


def _difficult_topics(entries: pd.DataFrame | None) -> list[tuple[str, int]]:
    from src.triggers import BY_KEY

    if entries is None or entries.empty:
        return []
    hard = entries[entries["negative_score"] >= DIFFICULT]
    counts = pd.Series([k for t in hard["triggers"] for k in t]).value_counts()
    return [(BY_KEY[k].label, int(n)) for k, n in counts.head(3).items() if k in BY_KEY]


FEATURE_WORDS = {"difficult": ("difficult feelings", "{:.0%}"), "joy": ("positive feelings", "{:.0%}"),
                 "entries": ("entries", "{:.0f}"), "sleep_hours": ("sleep", "{:.1f} h"),
                 "mood": ("self-rated mood", "{:.0f}/10"), "stress": ("self-rated stress", "{:.0f}/10"),
                 "energy": ("energy", "{:.0f}/10")}


def unusual_days(feats: pd.DataFrame, contamination: float = 0.1, seed: int = 0) -> list[dict]:
    """Days that look unlike the rest of this person's days (Isolation Forest), with the two features that differ most
    from their median, in plain words. Needs at least MIN_DAYS_FOR_ANOMALIES days."""
    from sklearn.ensemble import IsolationForest

    cols = [c for c in feats.columns if c != "entries" and feats[c].notna().mean() >= 0.6]
    X = feats[cols].dropna(how="all")
    if len(X) < MIN_DAYS_FOR_ANOMALIES or not cols:
        return []
    med, spread = X.median(), (X.quantile(0.75) - X.quantile(0.25)).replace(0, np.nan).fillna(X.std()).replace(0, 1)
    Xf = X.fillna(med)
    model = IsolationForest(contamination=contamination, random_state=seed).fit((Xf - med) / spread)
    flagged = Xf[model.predict((Xf - med) / spread) == -1]
    out = []
    for day, row in flagged.iterrows():
        z = ((row - med) / spread).abs().sort_values(ascending=False)
        reasons = []
        for col in z.index[:2]:
            if z[col] < 1:
                continue
            word, f = FEATURE_WORDS.get(col, (col, "{:.2f}"))
            direction = "higher" if row[col] > med[col] else "lower"
            reasons.append(f"{word} {f.format(row[col])}, {direction} than your usual {f.format(med[col])}")
        if reasons:
            out.append({"day": day, "reasons": reasons})
    return sorted(out, key=lambda r: r["day"], reverse=True)
