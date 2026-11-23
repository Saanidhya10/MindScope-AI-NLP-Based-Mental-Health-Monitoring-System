"""Rule-based, NON-DIAGNOSTIC recommendations and safety escalation (paper, Sec. 5.7).

Rules are deliberately transparent rather than learned. Escalation depends on persistence
(see trends.escalation_flags), never on a single message, with one exception: ``check_crisis``
surfaces helplines immediately when a message contains a crisis phrase, whatever the model predicts.
Wording never names a condition; ``check_wording`` enforces that in tests/test_recommend.py.

All user-facing copy (suggestions, support message, helplines, crisis phrases) lives in
config.yaml -> recommend.
"""
from __future__ import annotations

import re

import pandas as pd

from src.utils import LABELS, NEGATIVE_LABELS, load_config

# Terms that must never appear in user-facing text (no diagnosis, no clinical labels, no alarmism).
BANNED_PATTERNS = [
    r"\bdepress(ed|ion|ive)\b", r"\bdisorder\b", r"(?<!cannot )\bdiagnos(e|ed|is)\b", r"\byou (have|are suffering)\b",
    r"\bsuffer(ing)? from\b", r"\bmentally ill\b", r"\bsuicid", r"\bcrisis\b", r"\bdanger(ous)?\b",
]
HELPLINE_NOTE = "If things feel like too much, a qualified professional or a helpline such as Tele-MANAS (14416) can help."


def check_wording(text: str) -> list[str]:
    """Return the banned patterns found in ``text`` (empty list == OK)."""
    return [p for p in BANNED_PATTERNS if re.search(p, text, flags=re.I)]


def get_helplines(cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_config()
    return list(cfg.get("recommend", {}).get("helplines", [])) or [{"name": "Tele-MANAS", "number": "14416"}]


def suggestion_for(emotion: str, cfg: dict | None = None) -> dict:
    """{'title', 'text'} for one of the 8 emotions."""
    if emotion not in LABELS:
        raise ValueError(f"Unknown emotion: {emotion}")
    cfg = cfg or load_config()
    return dict(cfg["recommend"]["suggestions"][emotion.lower()])


def get_suggestion(emotion: str, cfg: dict | None = None) -> str:
    s = suggestion_for(emotion, cfg)
    return f"{s['title']}. {s['text']}"


def check_crisis(text: str, cfg: dict | None = None) -> bool:
    """True when the text contains a phrase that should surface helplines straight away."""
    if not text:
        return False
    cfg = cfg or load_config()
    lower = " ".join(text.lower().split())
    return any(p in lower for p in cfg["recommend"].get("crisis_phrases", []))


def should_escalate(scores, tau: float = 0.55, k: int = 3, alpha: float = 0.3) -> bool:
    """Plain-list version of the escalation rule: EMA of ``scores`` above tau for >= k consecutive steps."""
    if not len(scores):
        return False
    run, s = 0, scores[0]
    for i, x in enumerate(scores):
        s = x if i == 0 else alpha * x + (1.0 - alpha) * s
        run = run + 1 if s > tau else 0
        if run >= k:
            return True
    return False


def assess(daily: pd.DataFrame, baseline: dict, cfg: dict | None = None) -> dict:
    """One status for the dashboard, so the page never shows two conflicting verdicts.

    Order of precedence: sustained distress (escalation) > a heavy day versus the personal baseline >
    a negative emotion that keeps coming up > a lighter day than usual > steady.
    ``daily`` comes from trends.daily_frame (needs dominant_emotion, neg_score, smoothed, high_run).
    Returns level ('support' | 'watch' | 'steady'), headline, detail, suggestion {title, text},
    helplines (only for 'support') and the recent dominant emotion.
    """
    cfg = cfg or load_config()
    tau, k = cfg["trends"]["tau"], cfg["trends"]["k"]
    last = daily.iloc[-1]
    dom = str(last["dominant_emotion"])
    latest = float(last["neg_score"])
    std = baseline["std"] if baseline["std"] > 0.02 else 0.10
    z = (latest - baseline["mean"]) / std
    out = {"emotion": dom, "suggestion": suggestion_for(dom, cfg), "helplines": [], "z": z}

    run = int(last["high_run"])
    if run >= k:
        return {**out, "level": "support",
                "headline": f"Difficult feelings have stayed high for {run} check-in days in a row.",
                "detail": " ".join(cfg["recommend"]["support_message"].split()),
                "suggestion": suggestion_for(dom if dom in NEGATIVE_LABELS else "Stress", cfg),
                "helplines": get_helplines(cfg)}
    if z >= 1.75 and latest > 0.48:
        return {**out, "level": "watch",
                "headline": "Today's entries are heavier than usual for you.",
                "detail": f"Difficult feelings made up {latest:.0%} of today, against your usual {baseline['mean']:.0%}.",
                "suggestion": suggestion_for(dom if dom in NEGATIVE_LABELS else "Stress", cfg)}
    if dom in NEGATIVE_LABELS and float(last["smoothed"]) > 0.65 * tau:
        return {**out, "level": "watch",
                "headline": f"{dom} has been coming up a lot lately.",
                "detail": "It hasn't been sustained long enough to be a pattern yet. Keep writing and see how it moves."}
    if z <= -1.3 and latest < 0.30:
        return {**out, "level": "steady",
                "headline": "Today feels lighter than usual for you.",
                "detail": f"Difficult feelings made up {latest:.0%} of today, below your usual {baseline['mean']:.0%}."}
    return {**out, "level": "steady",
            "headline": "Your recent entries are close to your usual.",
            "detail": f"Difficult feelings are around {latest:.0%} of today, near your usual {baseline['mean']:.0%}."}
