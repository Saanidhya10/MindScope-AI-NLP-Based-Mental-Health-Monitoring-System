"""Daily wellness check-in: field definitions, lifestyle-mood associations and demo data (improvement 8).

Insights are ASSOCIATIONS between the user's own self-reported habits and how they felt, measured with Spearman's
rank correlation on the same day and with a one-day lag (yesterday's sleep vs today's mood). They are only shown with
enough paired days and a clear-enough relationship, and are always worded as "tended to", never as causes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Field:
    key: str
    label: str        # sentence-case label for the form
    noun: str         # phrase insights use after "days with" ("more sleep", "a heavier workload")
    lo: float
    hi: float
    step: float
    default: float
    unit: str = ""


HABITS = [
    Field("sleep_hours", "Sleep last night (hours)", "more sleep", 0, 14, 0.5, 7, "h"),
    Field("activity_min", "Physical activity (minutes)", "more physical activity", 0, 240, 5, 20, "min"),
    Field("screen_hours", "Screen time outside work or study (hours)", "more screen time", 0, 16, 0.5, 3, "h"),
    Field("social", "Time with other people", "more time with other people", 1, 3, 1, 2),  # 1 low, 2 medium, 3 high
    Field("workload", "Study or work load (1 = light, 10 = heavy)", "a heavier workload", 1, 10, 1, 5),
]
FEELINGS = [
    Field("mood", "Mood (1 = very low, 10 = very good)", "mood", 1, 10, 1, 6),
    Field("energy", "Energy (1 = drained, 10 = full of energy)", "energy", 1, 10, 1, 6),
    Field("stress", "Stress (1 = calm, 10 = very stressed)", "stress", 1, 10, 1, 4),
]
SOCIAL_LEVELS = {1: "Low", 2: "Medium", 3: "High"}
OUTCOME_NOUNS = {"mood": "mood", "energy": "energy", "stress": "stress", "neg_affect": "difficult feelings in your entries"}

MIN_DAYS = 10        # paired days needed before an association is reported
MIN_RHO = 0.35       # minimum |Spearman rho|
MAX_P = 0.10


def merge_with_journal(checkins: pd.DataFrame, daily_journal: pd.DataFrame | None) -> pd.DataFrame:
    """Check-ins indexed by day, plus the journal's daily difficult-feelings share as `neg_affect` when available."""
    df = checkins.copy()
    df.index = pd.to_datetime(df.index).normalize()
    if daily_journal is not None and len(daily_journal):
        neg = daily_journal["neg_score"].copy()
        neg.index = pd.to_datetime(neg.index).normalize()
        df = df.join(neg.rename("neg_affect"), how="left")
    return df.apply(pd.to_numeric, errors="coerce")


def associations(df: pd.DataFrame) -> list[dict]:
    """Every habit x outcome pair (same day and next day) that passes the reporting thresholds, strongest first."""
    from scipy.stats import spearmanr

    found = []
    outcomes = [o for o in OUTCOME_NOUNS if o in df.columns]
    full = df.asfreq("D") if len(df) else df  # keep calendar gaps so the lag means "the previous day"
    for h in HABITS:
        if h.key not in df.columns:
            continue
        for o in outcomes:
            for lag in (0, 1):
                x = full[h.key].shift(lag) if lag else full[h.key]
                pair = pd.concat([x, full[o]], axis=1).dropna()
                if len(pair) < MIN_DAYS or pair.iloc[:, 0].nunique() < 3 or pair.iloc[:, 1].nunique() < 3:
                    continue
                rho, p = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                if np.isfinite(rho) and abs(rho) >= MIN_RHO and p <= MAX_P:
                    found.append({"habit": h.key, "outcome": o, "lag": lag, "rho": float(rho), "p": float(p),
                                  "n": len(pair), "text": describe(h, o, lag, rho, len(pair))})
    return sorted(found, key=lambda a: -abs(a["rho"]))


def describe(habit: Field, outcome: str, lag: int, rho: float, n: int) -> str:
    direction = "higher" if rho > 0 else "lower"
    when = "the next day" if lag else "on the same day"
    strength = "clearly" if abs(rho) >= 0.6 else "somewhat"
    return (f"After days with {habit.noun}, your {OUTCOME_NOUNS[outcome]} tended to be {strength} {direction} "
            f"{when}." if lag else
            f"On days with {habit.noun}, your {OUTCOME_NOUNS[outcome]} tended to be {strength} {direction}."
            ) + f" (Spearman ρ = {rho:+.2f}, {n} days)"


def sample_checkins(days: list, seed: int = 7) -> pd.DataFrame:
    """INVENTED check-ins for the demo profile: sleep and activity drop and workload rises as the sample journal
    drifts towards stress, with mood, energy and stress loosely following. Not real data."""
    rng = np.random.default_rng(seed)
    n = len(days)
    drift = np.linspace(0, 1, n)
    sleep = np.clip(7.8 - 2.4 * drift + rng.normal(0, 0.5, n), 3.5, 10).round(1)
    activity = np.clip(45 - 35 * drift + rng.normal(0, 10, n), 0, 120).round(-1)
    screen = np.clip(2.5 + 3 * drift + rng.normal(0, 0.7, n), 0, 12).round(1)
    workload = np.clip(np.round(4 + 5 * drift + rng.normal(0, 0.8, n)), 1, 10)
    social = np.clip(np.round(2.6 - 1.4 * drift + rng.normal(0, 0.4, n)), 1, 3)
    mood = np.clip(np.round(3 + 0.55 * sleep + 0.02 * activity - 0.25 * workload + rng.normal(0, 0.8, n)), 1, 10)
    energy = np.clip(np.round(1 + 0.75 * sleep + 0.015 * activity + rng.normal(0, 0.8, n)), 1, 10)
    stress = np.clip(np.round(1 + 0.75 * workload - 0.2 * sleep + rng.normal(0, 0.8, n)), 1, 10)
    return pd.DataFrame({"sleep_hours": sleep, "activity_min": activity, "screen_hours": screen, "social": social,
                         "workload": workload, "energy": energy, "mood": mood, "stress": stress},
                        index=pd.to_datetime(days))
