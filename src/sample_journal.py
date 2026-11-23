"""An invented three-week journal (a student drifting from joy towards stress and sadness).

Used by scripts/demo.py and the dashboard's "Add sample entries" button. The entries are run through the
real model, so the sample shows what the trained classifier actually does. NOT real user data.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

# (days before the last day, hour, text) - an invented journal, days 0..20
JOURNAL = [
    (20, 9, "Had a wonderful weekend with friends, I feel so happy and relaxed."),
    (19, 20, "Got the internship offer! I'm thrilled and so grateful."),
    (17, 13, "Lunch with my sister was lovely, laughed all afternoon."),
    (15, 8, "Another ordinary day. Class at ten, then the library."),
    (14, 21, "Feeling pretty good about the semester so far, proud of my progress."),
    (12, 10, "The meeting is moved to Thursday, I'll check the schedule later."),
    (11, 22, "Slightly nervous about the presentation next week, but it should be fine."),
    (10, 9, "Three deadlines landed on the same day and I don't know where to start."),
    (9, 23, "I'm so behind. There's too much to do and I keep falling further back."),
    (8, 15, "Skipped lunch again to finish the report, I feel completely overwhelmed."),
    (7, 7, "Barely slept. My mind keeps racing about everything that could go wrong."),
    (6, 20, "I'm worried I'll fail the course and let everyone down."),
    (5, 11, "Everything feels heavy lately and I can't shake this sadness."),
    (5, 22, "I feel so alone, I haven't talked to anyone in days."),
    (4, 9, "The pressure at work is exhausting, I don't know how much more I can take."),
    (3, 14, "Cried in the bathroom between classes. I just feel empty."),
    (3, 23, "Can't focus on anything. My chest is tight and my thoughts won't stop."),
    (2, 10, "Another day of being swamped and anxious, nothing is getting better."),
    (1, 19, "I feel hopeless about all of it, so tired of feeling like this."),
    (0, 8, "Woke up dreading the day again. I'm exhausted and overwhelmed."),
]


def sample_entries(last_day: date | None = None) -> list[tuple[datetime, str]]:
    """(timestamp, text) pairs ending on ``last_day`` (default: today)."""
    last_day = last_day or date.today()
    return [(datetime.combine(last_day - timedelta(days=d), time(hour=h)), t) for d, h, t in JOURNAL]
