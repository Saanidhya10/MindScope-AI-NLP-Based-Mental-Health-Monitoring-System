#!/usr/bin/env python
"""Backtest the outlook methods on held-out SIMULATED timelines and save results/forecast_eval.csv.

    python scripts/eval_forecast.py

Simulated data shows how the methods compare under the simulation's assumptions only, not accuracy on real people.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import forecast as Fc  # noqa: E402


def main():
    f = Fc.Forecaster.build()
    table = f.scores.pivot(index="method", columns="horizon", values="mae")
    table["mean"] = table.mean(axis=1)
    out = ROOT / "results/forecast_eval.csv"
    f.scores.to_csv(out, index=False)
    print("Mean absolute error of the daily difficult-feelings share (0-1), by days ahead:\n")
    print(table.round(3).sort_values("mean").to_string())
    print(f"\nBest: {f.method}. Saved {out}")


if __name__ == "__main__":
    main()
