"""Wellness activities and adaptive ranking (improvement 9).

Each activity lists the feelings it is generally suited to. After doing one, the user can say whether it helped and
rate their mood before and after. Ranking combines (a) suitability for the user's recent feelings with (b) what has
actually helped *this* user, learned with Thompson sampling: each activity keeps a Beta(1 + helped, 1 + not helped)
belief, and we sample from it, so activities that help get suggested more while untried ones still get a chance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Activity:
    key: str
    title: str
    minutes: int
    summary: str
    suits: tuple[str, ...] = field(default_factory=tuple)   # emotions it is generally suited to


CATALOG = [
    Activity("breathing", "Guided breathing", 3, "Slow, paced breathing with an animated guide.",
             ("Stress", "Anxiety", "Anger", "Fear")),
    Activity("grounding", "5-4-3-2-1 grounding", 4, "Bring attention back to the room through your senses.",
             ("Anxiety", "Fear", "Stress")),
    Activity("gratitude", "Gratitude journal", 5, "Note three things that went well, however small.",
             ("Sadness", "Neutral", "Joy")),
    Activity("thought", "Thought journal", 10, "Write down a difficult thought and look at it from another angle.",
             ("Anxiety", "Sadness", "Anger", "Fear")),
    Activity("study", "Study planner", 10, "Break a big task into small steps you can start today.",
             ("Stress", "Anxiety")),
    Activity("sleep", "Sleep routine", 5, "Check tonight's wind-down against habits that support sleep.",
             ("Stress", "Anxiety", "Sadness")),
    Activity("mindfulness", "Three-minute body scan", 3, "Notice each part of your body without judging it.",
             ("Stress", "Anxiety", "Anger")),
    Activity("movement", "Movement break", 10, "A short walk or stretch to shift your energy.",
             ("Sadness", "Stress", "Anger", "Neutral")),
    Activity("connect", "Reach out to someone", 5, "Send a short message to someone you trust.",
             ("Sadness", "Fear", "Neutral")),
    Activity("goals", "Personal goals", 5, "Set small wellness goals and tick them off.",
             ("Neutral", "Joy", "Sadness")),
]
BY_KEY = {a.key: a for a in CATALOG}


def feedback_stats(log: pd.DataFrame) -> pd.DataFrame:
    """Per activity: times done, times rated helpful / not helpful, average mood change (after - before)."""
    cols = ["done", "helped", "not_helped", "mood_change"]
    if log is None or log.empty:
        return pd.DataFrame(columns=cols)
    g = log.groupby("activity")
    out = pd.DataFrame({
        "done": g.size(),
        "helped": g["helpful"].apply(lambda s: int((s == 1).sum())),
        "not_helped": g["helpful"].apply(lambda s: int((s == 0).sum())),
        "mood_change": g.apply(lambda d: (d["mood_after"] - d["mood_before"]).mean(), include_groups=False),
    })
    return out


def rank(recent: list[str], log: pd.DataFrame | None, rng: np.random.Generator | None = None,
         suit_weight: float = 0.35) -> list[dict]:
    """Activities ordered for this user now. `recent` = the user's recent emotions, most important first.
    Score = Thompson sample of P(helpful) + suit_weight * suitability. Returns dicts with a plain-language reason."""
    rng = rng or np.random.default_rng()
    stats = feedback_stats(log)
    ranked = []
    for a in CATALOG:
        helped = int(stats.loc[a.key, "helped"]) if a.key in stats.index else 0
        not_helped = int(stats.loc[a.key, "not_helped"]) if a.key in stats.index else 0
        p_helpful = rng.beta(1 + helped, 1 + not_helped)
        suit = next((1.0 - 0.25 * i for i, e in enumerate(recent[:3]) if e in a.suits), 0.0)
        if helped + not_helped:
            reason = f"Helped you {helped} of {helped + not_helped} time{'s' if helped + not_helped > 1 else ''}"
        elif suit:
            reason = f"Often suits {next(e for e in recent[:3] if e in a.suits).lower()}"
        else:
            reason = "You haven't tried this yet"
        ranked.append({"activity": a, "score": p_helpful + suit_weight * suit, "reason": reason})
    return sorted(ranked, key=lambda r: -r["score"])
