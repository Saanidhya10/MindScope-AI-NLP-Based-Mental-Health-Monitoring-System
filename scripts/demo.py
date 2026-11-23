#!/usr/bin/env python
"""End-to-end demo: raw text -> emotion -> stored predictions -> mood trend -> supportive suggestion.

Runs the invented three-week journal in src/sample_journal.py through the real pipeline
(src.inference, src.database, src.trends, src.recommend) and prints what the dashboard would show.

NOT A DIAGNOSTIC TOOL. Nothing is kept unless you pass --keep-db.

Usage
-----
  python scripts/demo.py                    # scripted journal, model from config.yaml -> inference
  python scripts/demo.py --plot             # also save results/figures/demo_mood_trend.png
  python scripts/demo.py --text "I can't sleep, my mind keeps racing about tomorrow"
  python scripts/demo.py --app              # launch the Streamlit dashboard instead
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import recommend, trends  # noqa: E402
from src.database import MoodDatabase  # noqa: E402
from src.inference import EmotionPredictor  # noqa: E402
from src.sample_journal import sample_entries  # noqa: E402
from src.utils import LABELS, load_config  # noqa: E402

DISCLAIMER = ("This is a wellness-support and early-awareness demo. It is NOT a diagnostic tool and cannot "
              "replace a qualified mental-health professional.")


def bar(v: float, width: int = 20) -> str:
    n = int(round(max(0.0, min(1.0, v)) * width))
    return "#" * n + "." * (width - n)


def run_single(predictor: EmotionPredictor, text: str, cfg: dict) -> None:
    p = predictor.predict_proba([text])[0]
    print(f'\nText: "{text}"')
    for i in np.argsort(p)[::-1][:3]:
        print(f"  {LABELS[i]:<9} {p[i]:.2f}  {bar(p[i])}")
    if recommend.check_crisis(text, cfg):
        print("\n  This message mentions wanting to hurt yourself. Please talk to someone now:")
        for h in recommend.get_helplines(cfg):
            print(f"  - {h['name']}: {h['number']}")
        return
    s = recommend.suggestion_for(LABELS[int(p.argmax())], cfg)
    print(f"\n{s['title']}: {s['text']}")


def run_journal(predictor: EmotionPredictor, cfg: dict, args) -> int:
    entries = sample_entries()
    proba = predictor.predict_proba([t for _, t in entries])

    db_path = Path(args.db) if args.db else ROOT / "results" / "demo_mood.db"
    db_path.unlink(missing_ok=True)
    db = MoodDatabase(db_path, cfg)
    db.set_consent("demo-user", True)
    for (ts, _), p in zip(entries, proba):
        db.add_prediction("demo-user", p, ts)  # raw text is NOT stored
    stored = db.get_predictions("demo-user")

    print("\nMessages -> emotion")
    for (ts, text), p in zip(entries, proba):
        print(f"  {ts:%d %b %H:%M}  {LABELS[int(p.argmax())]:<9} {p.max():.2f}  {text[:62]}")

    tc = cfg["trends"]
    daily = trends.daily_frame(stored, cfg)
    print(f"\nDaily trend (difficult-feelings share, EMA alpha={tc['alpha']}; support suggested after "
          f"{tc['k']}+ days above {tc['tau']})")
    for idx, r in daily.iterrows():
        flag = "  <-- sustained" if r["high_run"] >= tc["k"] else ""
        print(f"  {idx:%Y-%m-%d}  {r['dominant_emotion']:<9} n={int(r['n_messages'])}  "
              f"{bar(r['smoothed'])} {r['smoothed']:.2f}{flag}")

    status = recommend.assess(daily, trends.personal_baseline(stored, tc["tau"]), cfg)
    print(f"\n--- {status['headline']}\n{status['detail']}")
    print(f"\n{status['suggestion']['title']}: {status['suggestion']['text']}")
    for h in status["helplines"]:
        print(f"  - {h['name']}: {h['number']}  ({h.get('note', '')})")
    print(f"\n{DISCLAIMER}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(stored["timestamp"], stored["neg_score"], "o", color="#9aa5b1", ms=5, label="entry")
        ax.plot(daily.index, daily["smoothed"], color="#2D5F73", lw=2, label="smoothed trend")
        ax.axhline(tc["tau"], color="#B23A2E", ls="--", lw=1, label=f"support line ({tc['tau']})")
        ax.set_ylim(0, 1)
        ax.set_ylabel("difficult-feelings share")
        ax.set_title("Mood trend, sample journal (invented entries)", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.autofmt_xdate()
        fig.tight_layout()
        out = ROOT / "results" / "figures" / "demo_mood_trend.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"\nSaved {out}")

    if not args.keep_db:
        db_path.unlink(missing_ok=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="classify one message instead of the scripted journal")
    ap.add_argument("--plot", action="store_true", help="save results/figures/demo_mood_trend.png")
    ap.add_argument("--db", help="SQLite path for the demo (default results/demo_mood.db)")
    ap.add_argument("--keep-db", action="store_true", help="keep the SQLite file after the demo")
    ap.add_argument("--app", action="store_true", help="launch the Streamlit dashboard (app/app.py)")
    args = ap.parse_args(argv)

    print("=" * 72 + f"\n{DISCLAIMER}\n" + "=" * 72)
    if args.app:
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "app.py")], cwd=ROOT)
    cfg = load_config()
    predictor = EmotionPredictor.from_config(cfg)
    if args.text:
        run_single(predictor, args.text, cfg)
        return 0
    return run_journal(predictor, cfg, args)


if __name__ == "__main__":
    sys.exit(main())
