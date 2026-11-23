"""Rendering for the Streamlit dashboard: styles, entry result, mood ribbon, status, charts.

All logic lives in src/ (inference, database, trends, recommend); this module only draws.
"""
from __future__ import annotations

from datetime import date, timedelta
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.trends import aggregate
from src.utils import LABELS

INK, MUTED, RULE, LAKE = "#18262B", "#56656A", "#CBD5D3", "#2D5F73"
SANS = "Lexend, system-ui, sans-serif"

# One hue family per kind of feeling: warm for agitation, cool for low mood, light for positive.
EMOTION_COLORS = {
    "Joy": "#E2AE2A",
    "Surprise": "#4FAE9F",
    "Neutral": "#AEB8B6",
    "Sadness": "#4766A8",
    "Fear": "#76599F",
    "Anxiety": "#DB7A34",
    "Stress": "#BC3F67",
    "Anger": "#A8372B",
}
LEVEL_COLORS = {"steady": "#3E8E7E", "watch": "#C77A2E", "support": "#A8372B"}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600&family=Literata:opsz,wght@7..72,400;7..72,600&display=swap');

:root {
  --paper: #EEF2F1; --surface: #FFFFFF; --ink: #18262B; --muted: #56656A; --rule: #CBD5D3; --lake: #2D5F73;
  --serif: 'Literata', Georgia, serif;
  --sans: 'Lexend', system-ui, sans-serif;
}

.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp [data-baseweb],
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {
  font-family: var(--sans) !important;
}
.stApp [data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
.stApp textarea { font-family: var(--serif) !important; font-size: 1.1rem !important; line-height: 1.6 !important; }

[data-testid="stMainBlockContainer"] { max-width: 1120px; padding-top: 3rem; padding-bottom: 4rem; }

/* ---- type ---- */
.stApp .mj-title { font-family: var(--serif) !important; font-weight: 600; font-size: clamp(2.25rem, 4.2vw, 3rem);
  line-height: 1.1; letter-spacing: -0.012em; color: var(--ink); margin: 0 0 0.6rem; }
.stApp .mj-lede { font-size: 1.125rem; line-height: 1.55; color: var(--muted); max-width: 60ch; margin: 0 0 1.25rem; }
.stApp .mj-help { border-left: 3px solid var(--lake); padding: 0.2rem 0 0.2rem 0.9rem; font-size: 0.95rem;
  line-height: 1.5; color: var(--ink); max-width: 72ch; margin: 0; }
.stApp .mj-h2 { font-family: var(--serif) !important; font-weight: 600; font-size: 1.5rem; line-height: 1.25; color: var(--ink);
  margin: 2.75rem 0 0.35rem; }
.stApp .mj-note { color: var(--muted); font-size: 0.95rem; line-height: 1.5; margin: 0 0 1rem; max-width: 70ch; }
.stApp .mj-num { font-variant-numeric: tabular-nums; }

/* ---- writing surface ---- */
.st-key-mj-entry { box-sizing: border-box; background: var(--surface); border: 1px solid var(--rule); border-radius: 12px; padding: 1.25rem 1.25rem 1rem; }
.st-key-mj-entry [data-testid="stForm"] { border: 0; padding: 0; }

/* ---- result ---- */
.stApp .mj-reading { font-family: var(--serif) !important; font-size: 1.85rem; line-height: 1.2; color: var(--ink); margin: 0.2rem 0 0.35rem; }
.stApp .mj-certainty { color: var(--muted); font-size: 0.95rem; margin: 0 0 1.1rem; }
.stApp .mj-bars { display: grid; grid-template-columns: 5.5rem 1fr 2.75rem; gap: 0.55rem 0.75rem; align-items: center; font-size: 0.95rem; }
.stApp .mj-bars .val { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
.stApp .mj-track { height: 0.55rem; background: #E2E8E7; border-radius: 999px; overflow: hidden; }
.stApp .mj-fill { height: 100%; border-radius: 999px; transform-origin: left; animation: mj-grow 0.7s cubic-bezier(.2,.7,.2,1) both; }
@keyframes mj-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
.stApp .mj-empty { color: var(--muted); font-size: 1rem; line-height: 1.55; padding: 1.5rem 0; max-width: 40ch; }
.stApp .mj-model { color: var(--muted); font-size: 0.85rem; margin-top: 1rem; }

/* ---- crisis + status ---- */
.stApp .mj-crisis { background: #F7ECEA; border-left: 4px solid #A8372B; border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin-bottom: 1.25rem; }
.stApp .mj-crisis p { margin: 0 0 0.6rem; line-height: 1.5; }
.stApp .mj-status { background: var(--surface); border-left: 4px solid var(--level); border-radius: 0 10px 10px 0; padding: 1.35rem 1.6rem; }
.stApp .mj-status-head { font-family: var(--serif) !important; font-size: 1.4rem; line-height: 1.3; color: var(--ink); margin: 0 0 0.4rem; max-width: 34ch; }
.stApp .mj-status p { line-height: 1.55; margin: 0; max-width: 62ch; }
.stApp .mj-try { margin-top: 1.1rem; padding-top: 1rem; border-top: 1px solid #E2E8E7; }
.stApp .mj-try strong { display: block; margin-bottom: 0.2rem; }
.stApp .mj-lines { margin: 1rem 0 0; padding: 0; list-style: none; }
.stApp .mj-lines li { padding: 0.45rem 0; border-top: 1px solid #E2E8E7; line-height: 1.45; }
.stApp .mj-lines li span { color: var(--muted); font-size: 0.9rem; display: block; }

/* ---- mood ribbon ---- */
.stApp .mj-ribbon { display: grid; grid-template-columns: repeat(28, minmax(0, 1fr)); gap: 5px; }
.stApp .mj-day { display: flex; flex-direction: column; align-items: center; gap: 0.3rem; }
.stApp .mj-cell { width: 100%; aspect-ratio: 1 / 1.7; border-radius: 4px; }
.stApp .mj-cell.none { border: 1px dashed var(--rule); }
.stApp .mj-date { font-size: 0.72rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.stApp .mj-day.today .mj-date { color: var(--ink); font-weight: 700; }
.stApp .mj-day.monday .mj-date { color: var(--ink); }
.stApp .mj-legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1.1rem; margin-top: 0.9rem; font-size: 0.9rem; color: var(--muted); }
.stApp .mj-legend i { display: inline-block; width: 0.75rem; height: 0.75rem; border-radius: 2px; margin-right: 0.4rem; vertical-align: -1px; }

.stApp .mj-footer { color: var(--muted); font-size: 0.85rem; line-height: 1.5; margin-top: 3.5rem; padding-top: 1rem;
  border-top: 1px solid var(--rule); max-width: 80ch; }

@media (max-width: 760px) {
  [data-testid="stMainBlockContainer"] { padding-top: 4rem; }
  .stApp .mj-ribbon { grid-template-columns: repeat(14, minmax(0, 1fr)); row-gap: 0.75rem; }
  .stApp .mj-status { padding: 1.1rem 1.1rem; }
}
@media (prefers-reduced-motion: reduce) { .mj-fill { animation: none; } }
</style>
"""


def _md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def inject_css() -> None:
    _md(_CSS)


def header(helplines: list[dict]) -> None:
    first = helplines[0] if helplines else {"name": "Tele-MANAS", "number": "14416"}
    _md(
        '<p class="mj-title" role="heading" aria-level="1">Mood Journal</p>'
        '<p class="mj-lede">Write a few lines about your day. The app picks out the feelings in what you write '
        "and shows how they change over the weeks.</p>"
        f'<p class="mj-help">This is not a medical tool and it can be wrong. If you need to talk to someone now, '
        f'call {escape(first["name"])} on <strong class="mj-num">{escape(first["number"])}</strong>, '
        "free and open day and night.</p>"
    )


def section(title: str, note: str | None = None) -> None:
    _md(f'<p class="mj-h2" role="heading" aria-level="2">{escape(title)}</p>'
        + (f'<p class="mj-note">{note}</p>' if note else ""))


# ------------------------------------------------------------------ single entry
def _reading(probs: np.ndarray) -> tuple[str, str]:
    """Plain-language summary of one prediction: ('Mostly stress, with some sadness', certainty note)."""
    order = np.argsort(probs)[::-1]
    top, second = LABELS[order[0]], LABELS[order[1]]
    p1, p2 = float(probs[order[0]]), float(probs[order[1]])
    headline = f"Mostly {top.lower()}"
    if p2 >= 0.16 and p2 >= 0.35 * p1:
        headline += f", with some {second.lower()}"
    q = np.clip(probs, 1e-12, 1.0)
    entropy = float(-(q * np.log2(q)).sum() / np.log2(len(q)))
    if entropy < 0.5 or p1 - p2 > 0.35:
        note = "A clear signal."
    elif entropy <= 0.72:
        note = "More than one feeling comes through."
    else:
        note = "Hard to read. The model isn't sure about this one, so take it lightly."
    return headline, note


def entry_result(probs: np.ndarray, crisis: bool, helplines: list[dict], model_label: str) -> None:
    if crisis:
        lines = "".join(f'<li><strong>{escape(h["name"])}</strong> <span class="mj-num">{escape(h["number"])}</span>'
                        f'<span>{escape(h.get("note", ""))}</span></li>' for h in helplines)
        _md('<div class="mj-crisis"><p><strong>What you wrote sounds like you might be thinking about hurting '
            "yourself.</strong> You deserve support right now. Please call one of these, or tell someone near you.</p>"
            f'<ul class="mj-lines">{lines}</ul></div>')
    headline, note = _reading(probs)
    if crisis:
        note = "The model only picks out emotions, so it can miss what matters most in an entry like this."
    rows = "".join(
        f"<span>{e}</span>"
        f'<div class="mj-track"><div class="mj-fill" style="width:{p * 100:.1f}%;background:{EMOTION_COLORS[e]}"></div></div>'
        f'<span class="val">{p * 100:.0f}%</span>'
        for e, p in sorted(zip(LABELS, map(float, probs)), key=lambda x: -x[1])
    )
    _md(f'<p class="mj-reading">{headline}</p><p class="mj-certainty">{note}</p>'
        f'<div class="mj-bars" role="list" aria-label="Emotion scores">{rows}</div>'
        f'<p class="mj-model">Read by the {escape(model_label)} model.</p>')


def entry_placeholder() -> None:
    _md('<p class="mj-empty">After you analyse an entry, the feelings the model finds in it will show here, '
        "strongest first.</p>")


# ------------------------------------------------------------------ history
def mood_ribbon(hist: pd.DataFrame, days: int = 28) -> None:
    """One block per day for the last ``days`` days, coloured by that day's main emotion."""
    daily = aggregate(hist, "daily") if len(hist) else pd.DataFrame()
    by_day = {ts.date(): r for ts, r in daily.iterrows()}
    today = date.today()
    cells, seen = [], set()
    for i in range(days - 1, -1, -1):
        d = today - timedelta(days=i)
        r = by_day.get(d)
        cls = "mj-day" + (" today" if d == today else "") + (" monday" if d.weekday() == 0 else "")
        label = "Today" if d == today else f"{d:%a %d %b}"
        if r is None:
            cells.append(f'<div class="{cls}" title="{label}: no entries"><div class="mj-cell none"></div>'
                         f'<span class="mj-date">{d.day}</span></div>')
            continue
        e, n = r["dominant_emotion"], int(r["n_messages"])
        seen.add(e)
        cells.append(f'<div class="{cls}" title="{label}: mostly {e.lower()}, {n} entr{"y" if n == 1 else "ies"}">'
                     f'<div class="mj-cell" style="background:{EMOTION_COLORS[e]}"></div>'
                     f'<span class="mj-date">{d.day}</span></div>')
    legend = "".join(f'<span><i style="background:{EMOTION_COLORS[e]}"></i>{e}</span>' for e in LABELS if e in seen)
    _md(f'<div class="mj-ribbon">{"".join(cells)}</div>'
        f'<div class="mj-legend">{legend or "<span>No entries in the last four weeks yet.</span>"}</div>')


def status_card(status: dict) -> None:
    color = LEVEL_COLORS[status["level"]]
    s = status["suggestion"]
    lines = "".join(f'<li><strong>{escape(h["name"])}</strong> <span class="mj-num">{escape(h["number"])}</span>'
                    f'<span>{escape(h.get("note", ""))}</span></li>' for h in status["helplines"])
    _md(f'<div class="mj-status" style="--level:{color}">'
        f'<p class="mj-status-head">{escape(status["headline"])}</p><p>{escape(status["detail"])}</p>'
        + (f'<ul class="mj-lines">{lines}</ul>' if lines else "")
        + f'<div class="mj-try"><strong>Something to try: {escape(s["title"].lower())}</strong>'
        f'<p>{escape(s["text"])}</p></div></div>')


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family=SANS, color=INK, size=14),
                      hoverlabel=dict(font=dict(family=SANS)), showlegend=False)
    return fig


def emotion_mix(hist: pd.DataFrame, days: int = 30) -> None:
    recent = hist[hist["timestamp"] >= hist["timestamp"].max() - pd.Timedelta(days=days)]
    counts = recent["emotion"].value_counts().reindex(LABELS).dropna()
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.62, sort=True, direction="clockwise",
        marker=dict(colors=[EMOTION_COLORS[e] for e in counts.index], line=dict(color="#FFFFFF", width=2)),
        textinfo="none", hovertemplate="%{label}: %{value} entries (%{percent})<extra></extra>"))
    fig.add_annotation(text=f"<b>{len(recent)}</b><br>entries", showarrow=False,
                       font=dict(family=SANS, size=15, color=INK))
    st.plotly_chart(_layout(fig, 250), use_container_width=True, config={"displayModeBar": False})
    top = counts.sort_values(ascending=False)
    _md('<div class="mj-legend">' + "".join(
        f'<span><i style="background:{EMOTION_COLORS[e]}"></i>{e} <span class="mj-num">{int(n)}</span></span>'
        for e, n in top.items()) + "</div>")


def trend_chart(hist: pd.DataFrame, daily: pd.DataFrame, view: str, baseline: dict, tau: float) -> None:
    if view == "Day":
        x, y, smooth = daily.index, daily["neg_score"], daily["smoothed"]
    else:
        agg = aggregate(hist, "weekly" if view == "Week" else "monthly")
        x, y, smooth = agg.index, agg["neg_score"], None
    fig = go.Figure()
    # label "Your usual" only when it is far enough from the support line not to collide with it
    usual = dict(text="Your usual", font=dict(color=MUTED, size=12)) if abs(baseline["mean"] - tau) > 0.06 else None
    fig.add_hline(y=baseline["mean"] * 100, line=dict(color=MUTED, width=1, dash="dash"),
                  annotation=usual, annotation_position="bottom right")
    fig.add_hline(y=tau * 100, line=dict(color=LEVEL_COLORS["support"], width=1, dash="dot"),
                  annotation=dict(text="Support line", font=dict(color=LEVEL_COLORS["support"], size=12)),
                  annotation_position="top left")
    fig.add_trace(go.Scatter(x=x, y=y * 100, mode="markers" if smooth is not None else "lines+markers",
                             marker=dict(size=7, color="#9FB0B4" if smooth is not None else LAKE),
                             line=dict(color=LAKE, width=2.5),
                             hovertemplate="%{x|%d %b}: %{y:.0f}%<extra></extra>"))
    if smooth is not None:
        fig.add_trace(go.Scatter(x=x, y=smooth * 100, mode="lines", line=dict(color=LAKE, width=2.5),
                                 hovertemplate="%{x|%d %b}: %{y:.0f}% smoothed<extra></extra>"))
    fig.update_yaxes(range=[0, 100], ticksuffix="%", gridcolor="#DDE4E2", zeroline=False, tickfont=dict(color=MUTED))
    fig.update_xaxes(showgrid=False, tickformat="%d %b", tickfont=dict(color=MUTED), linecolor=RULE)
    st.plotly_chart(_layout(fig, 290), use_container_width=True, config={"displayModeBar": False})


def footer(model_label: str) -> None:
    _md(f'<p class="mj-footer">Emotions are estimated from your words by the {escape(model_label)} model, trained on '
        "English Reddit comments. It often misreads short, sarcastic or mixed entries, and it cannot diagnose "
        "anything. Only the scores and the time of each entry are saved, on this computer.</p>")
