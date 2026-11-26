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
.stApp .mj-tagline { font-family: var(--serif) !important; font-size: 1.2rem; font-style: italic; color: var(--ink);
  margin: -0.2rem 0 0.9rem; max-width: 60ch; }
.stApp .mj-page-title { font-family: var(--serif) !important; font-weight: 600; font-size: 2.1rem; line-height: 1.15;
  color: var(--ink); margin: 0 0 0.5rem; }
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
.stApp .mj-heard-label { font-size: 0.9rem; color: var(--muted); margin: 0 0 0.3rem; }
.stApp .mj-heard { font-family: var(--serif) !important; font-size: 1.05rem; line-height: 1.55; color: var(--ink);
  border-left: 3px solid var(--rule); margin: 0 0 0.8rem; padding: 0.1rem 0 0.1rem 0.9rem; max-width: 60ch; }
.stApp .mj-tone { font-size: 0.95rem; line-height: 1.5; color: var(--ink); margin: 0 0 1.1rem; max-width: 60ch; }
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

.stApp .mj-insights { margin: 0 0 1rem; padding: 0; list-style: none; max-width: 62ch; }
.stApp .mj-insights li { padding: 0.6rem 0 0.6rem 0.9rem; border-left: 3px solid var(--lake); margin-bottom: 0.5rem;
  background: var(--surface); line-height: 1.5; }
[class*="st-key-mj-act-"] { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
  padding: 1rem 1.1rem; height: 100%; }
.stApp .mj-act-title { font-family: var(--serif) !important; font-size: 1.2rem; font-weight: 600; margin: 0 0 0.2rem; }
.stApp .mj-act-meta { color: var(--lake); font-size: 0.88rem; margin: 0 0 0.5rem; }
.stApp .mj-act-summary { color: var(--muted); font-size: 0.95rem; line-height: 1.45; margin: 0 0 0.4rem; }
.stApp .mj-table { border-collapse: collapse; width: 100%; max-width: 760px; background: var(--surface); font-size: 0.95rem; }
.stApp .mj-table th { text-align: left; font-weight: 600; padding: 0.55rem 0.8rem; border-bottom: 2px solid var(--rule); }
.stApp .mj-table td { padding: 0.5rem 0.8rem; border-bottom: 1px solid #E2E8E7; }
.stApp .mj-table td.mj-num, .stApp .mj-table th { white-space: nowrap; }
.stApp .mj-chip { display: inline-block; background: #E2ECEE; color: var(--ink); border-radius: 999px;
  padding: 0.1rem 0.65rem; margin: 0 0.3rem 0.3rem 0; font-size: 0.88rem; }
.stApp .mj-sub { color: var(--muted); font-size: 0.85rem; }
.stApp .mj-outlook { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 1.1rem 1.25rem; }
.stApp .mj-outlook-label { color: var(--muted); font-size: 0.9rem; margin: 0 0 0.2rem; }
.stApp .mj-outlook-head { font-family: var(--serif) !important; font-size: 1.35rem; line-height: 1.3; margin: 0 0 0.4rem; }
.stApp .mj-outlook-body { color: var(--ink); font-size: 0.95rem; line-height: 1.5; margin: 0; }
.stApp .mj-rec { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
  padding: 0.95rem 1.15rem; margin-bottom: 0.7rem; max-width: 70ch; }
.stApp .mj-rec-title { font-family: var(--serif) !important; font-size: 1.12rem; font-weight: 600; margin: 0 0 0.3rem; }
.stApp .mj-rec-body { line-height: 1.55; margin: 0 0 0.4rem; }
.stApp .mj-rec-why { color: var(--lake); font-size: 0.9rem; margin: 0 0 0.25rem; }
.stApp .mj-rec-src { color: var(--muted); font-size: 0.85rem; margin: 0; }
.stApp .mj-rec-src a { color: var(--lake); }
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


def help_line(helplines: list[dict]) -> None:
    first = helplines[0] if helplines else {"name": "Tele-MANAS", "number": "14416"}
    _md(f'<p class="mj-help">This is not a medical tool and it can be wrong. If you need to talk to someone now, '
        f'call {escape(first["name"])} on <strong class="mj-num">{escape(first["number"])}</strong>, '
        "free and open day and night.</p>")


def brand_header(helplines: list[dict]) -> None:
    """Home page masthead: product name, what it is, and the helpline."""
    from common import APP_NAME, APP_TAGLINE

    _md(f'<p class="mj-title" role="heading" aria-level="1">{APP_NAME}</p>'
        f'<p class="mj-tagline">{APP_TAGLINE}</p>'
        '<p class="mj-lede">Write or speak a few lines each day. MindScope picks out the feelings in what you share, '
        "learns what is usual for you, and shows how things change over the weeks.</p>")
    help_line(helplines)


def page_header(title: str, lede: str | None = None) -> None:
    _md(f'<p class="mj-page-title" role="heading" aria-level="1">{escape(title)}</p>'
        + (f'<p class="mj-lede">{lede}</p>' if lede else ""))


def empty_state(text: str) -> None:
    _md(f'<p class="mj-empty">{text}</p>')


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


TONE_WORDS = {"Neutral": "calm", "Joy": "upbeat", "Anger": "tense", "Sadness": "low"}


def _voice_block(voice: dict) -> str:
    """What Whisper heard, and how the voice sounded."""
    html = (f'<p class="mj-heard-label">What the app heard ({voice["seconds"]:.0f} seconds)</p>'
            f'<blockquote class="mj-heard">{escape(voice["transcript"])}</blockquote>')
    tone = voice.get("tone")
    if tone is None:
        return html + '<p class="mj-tone">Tone of voice wasn\'t read for this entry.</p>'
    order = [i for i in np.argsort(tone)[::-1] if tone[i] > 0]
    top = LABELS[order[0]]
    parts = ", ".join(f"{TONE_WORDS[LABELS[i]]} {tone[i] * 100:.0f}%" for i in order)
    effect = ("Your tone pointed to more difficult feelings than your words, so it was added to the reading."
              if voice.get("tone_used") else
              "The reading is based on your words. Tone is only added when it points to more difficult feelings, "
              "never to fewer.")
    return html + (f'<p class="mj-tone">Your voice sounded mostly <strong>{TONE_WORDS[top]}</strong> '
                   f'<span class="mj-num">({parts})</span>. {effect}</p>')


def _trigger_line(triggers: list[str] | None) -> str:
    if triggers is None:
        return ""
    from src.triggers import BY_KEY

    if not triggers:
        return '<p class="mj-tone">No particular topic stood out in this entry.</p>'
    chips = "".join(f'<span class="mj-chip">{escape(BY_KEY[k].label)}</span>' for k in triggers)
    return f'<p class="mj-tone">It seems to be about {chips}</p>'


def entry_result(probs: np.ndarray, crisis: bool, helplines: list[dict], model_label: str,
                 voice: dict | None = None, triggers: list[str] | None = None,
                 notes: list[str] | None = None) -> None:
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
    model_note = f"Read by the {escape(model_label)} model"
    if voice is not None:
        model_note += ", from a transcript made with Whisper" + (" and a speech-emotion model for tone" if voice.get(
            "tone") is not None else "")
    _md(f'<p class="mj-reading">{headline}</p><p class="mj-certainty">{note}</p>'
        + (_voice_block(voice) if voice is not None else "")
        + _trigger_line(triggers)
        + "".join(f'<p class="mj-tone mj-sub">{escape(n)}</p>' for n in (notes or []))
        + f'<div class="mj-bars" role="list" aria-label="Emotion scores">{rows}</div>'
        f'<p class="mj-model">{model_note}.</p>')


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


def insight_list(items: list[str]) -> None:
    _md('<ul class="mj-insights">' + "".join(f"<li>{escape(t)}</li>" for t in items) + "</ul>")


def checkin_chart(df: pd.DataFrame) -> None:
    """Mood, energy and stress (1-10) with sleep hours, for recent check-ins."""
    fig = go.Figure()
    for col, name, color, dash in [("mood", "Mood", LAKE, "solid"), ("energy", "Energy", "#4FAE9F", "solid"),
                                   ("stress", "Stress", LEVEL_COLORS["support"], "dot"),
                                   ("sleep_hours", "Sleep (h)", "#9FB0B4", "dash")]:
        if col in df:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name, mode="lines+markers",
                                     line=dict(color=color, width=2, dash=dash), marker=dict(size=5),
                                     hovertemplate=f"%{{x|%d %b}}: {name} %{{y}}<extra></extra>"))
    fig.update_yaxes(range=[0, 11], gridcolor="#DDE4E2", zeroline=False, tickfont=dict(color=MUTED))
    fig.update_xaxes(showgrid=False, tickformat="%d %b", tickfont=dict(color=MUTED), linecolor=RULE)
    fig = _layout(fig, 280)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.2, x=0, font=dict(color=MUTED)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def activity_card(act, reason: str) -> None:
    _md(f'<p class="mj-act-title">{escape(act.title)}</p>'
        f'<p class="mj-act-meta">{escape(reason)}. About {act.minutes} minutes.</p>'
        f'<p class="mj-act-summary">{escape(act.summary)}</p>')


def breathing_guide(phases: list[tuple[str, int]]) -> None:
    """CSS-only breathing circle: grows while breathing in, holds, shrinks while breathing out.
    With reduced motion the circle stays still and the phases are listed instead."""
    total = sum(sec for _, sec in phases)
    scale_frames, label_rules, t = [], [], 0.0
    size = 0.55
    for i, (name, sec) in enumerate(phases):
        start, end = 100 * t / total, 100 * (t + sec) / total
        if name == "Breathe in":
            size_end = 1.0
        elif name == "Breathe out":
            size_end = 0.55
        else:
            size_end = size
        scale_frames.append(f"{start:.2f}% {{ transform: scale({size}); }} {end:.2f}% {{ transform: scale({size_end}); }}")
        size = size_end
        label_rules.append(
            f"@keyframes mj-lab{i} {{ 0%, {max(start - 0.01, 0):.2f}% {{ opacity: 0; }} {start:.2f}%, {end - 0.5:.2f}% "
            f"{{ opacity: 1; }} {end:.2f}%, 100% {{ opacity: 0; }} }} "
            f".mj-breath-lab{i} {{ animation: mj-lab{i} {total}s linear infinite; }}")
        t += sec
    labels = "".join(f'<span class="mj-breath-lab mj-breath-lab{i}">{escape(n)} ({sec}s)</span>'
                     for i, (n, sec) in enumerate(phases))
    steps = " → ".join(f"{n} for {sec}" for n, sec in phases)
    st.html(f"""<style>
@keyframes mj-breath {{ {' '.join(scale_frames)} }}
{' '.join(label_rules)}
.mj-breath {{ position: relative; height: 260px; display: flex; align-items: center; justify-content: center; }}
.mj-breath-circle {{ width: 220px; height: 220px; border-radius: 50%; background: radial-gradient(circle, #CFE0E2, #8FB6BD);
  animation: mj-breath {total}s ease-in-out infinite; }}
.mj-breath-lab {{ position: absolute; font-family: 'Literata', Georgia, serif; font-size: 1.35rem; color: #18262B; opacity: 0; }}
.mj-breath-static {{ display: none; font-family: 'Lexend', sans-serif; color: #18262B; }}
@media (prefers-reduced-motion: reduce) {{
  .mj-breath-circle, .mj-breath-lab {{ animation: none; }} .mj-breath-lab {{ display: none; }}
  .mj-breath-static {{ display: block; position: absolute; }} }}
</style>
<div class="mj-breath" role="img" aria-label="Breathing guide: {escape(steps)}">
  <div class="mj-breath-circle"></div>{labels}<span class="mj-breath-static">{escape(steps)}</span>
</div>""")


def timer_bar(seconds: int) -> None:
    """A bar that fills over `seconds` (CSS only), for timed exercises."""
    st.html(f"""<style>
@keyframes mj-fillbar {{ from {{ width: 0; }} to {{ width: 100%; }} }}
.mj-timer {{ height: 8px; background: #E2E8E7; border-radius: 999px; overflow: hidden; margin: 4px 0 4px; }}
.mj-timer div {{ height: 100%; background: #2D5F73; animation: mj-fillbar {seconds}s linear forwards; }}
@media (prefers-reduced-motion: reduce) {{ .mj-timer div {{ animation: none; width: 0; }} }}
</style><div class="mj-timer" role="progressbar" aria-label="{seconds // 60} minute timer"><div></div></div>
<p style="font-family: Lexend, sans-serif; color: #56656A; font-size: 0.9rem; margin: 0;">
About {seconds // 60} minutes. The bar fills as you go.</p>""")


def activity_stats(stats: pd.DataFrame) -> None:
    from src.activities import BY_KEY

    rows = "".join(
        f"<tr><td>{escape(BY_KEY[k].title) if k in BY_KEY else escape(k)}</td><td class='mj-num'>{int(r['done'])}</td>"
        f"<td class='mj-num'>{int(r['helped'])} of {int(r['helped'] + r['not_helped'])}</td>"
        f"<td class='mj-num'>{'' if pd.isna(r['mood_change']) else f'{r.mood_change:+.1f}'}</td></tr>"
        for k, r in stats.sort_values("done", ascending=False).iterrows())
    _md('<table class="mj-table"><thead><tr><th>Activity</th><th>Times done</th><th>Helped</th>'
        f'<th>Average mood change</th></tr></thead><tbody>{rows}</tbody></table>')


def trigger_chart(summ: pd.DataFrame) -> None:
    """Horizontal bars: entries per topic, split into difficult and other entries."""
    summ = summ.iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=summ["label"], x=summ["difficult"], orientation="h", name="Difficult entries",
                         marker_color=LEVEL_COLORS["support"], hovertemplate="%{y}: %{x} difficult<extra></extra>"))
    fig.add_trace(go.Bar(y=summ["label"], x=summ["entries"] - summ["difficult"], orientation="h", name="Other entries",
                         marker_color="#B9CBCE", hovertemplate="%{y}: %{x} other<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(gridcolor="#DDE4E2", tickfont=dict(color=MUTED), title=None, dtick=1 if summ["entries"].max() <= 12 else None)
    fig.update_yaxes(tickfont=dict(color=INK))
    fig = _layout(fig, 70 + 38 * len(summ))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.15, x=0, font=dict(color=MUTED)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def trigger_table(entries: pd.DataFrame) -> None:
    from src.triggers import BY_KEY

    rows = "".join(
        f"<tr><td class='mj-num'>{r.timestamp:%d %b}</td><td>{escape(r.emotion)}</td><td>"
        + ("".join(f'<span class="mj-chip">{escape(BY_KEY[k].label)}</span>' for k in r.triggers if k in BY_KEY)
           or '<span style="color: var(--muted)">None</span>') + "</td></tr>"
        for r in entries.itertuples())
    _md('<table class="mj-table"><thead><tr><th>Day</th><th>Main feeling</th><th>Topics</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>")


def profile_table(dims) -> None:
    """Usual vs recent per dimension, with a plain-words change column coloured by whether it's a better direction."""
    rows = []
    for d in dims:
        if d.change in ("higher", "lower"):
            good = d.change == d.better
            color = LEVEL_COLORS["steady"] if good else LEVEL_COLORS["watch"]
            change = f'<span style="color:{color}; font-weight:600">{d.change.capitalize()} than usual</span>'
        elif d.change == "similar":
            change = '<span style="color: var(--muted)">About the same</span>'
        else:
            change = ""
        rows.append(f"<tr><td><strong>{escape(d.name)}</strong><br><span class='mj-sub'>{escape(d.note)}</span></td>"
                    f"<td class='mj-num'>{escape(d.usual)}</td><td class='mj-num'>{escape(d.recent)}</td>"
                    f"<td>{change}</td></tr>")
    _md('<table class="mj-table"><thead><tr><th></th><th>Usual</th><th>Last 7 days</th><th></th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>")


def mix_compare(usual: pd.Series, recent: pd.Series) -> None:
    order = [e for e in LABELS if usual.get(e, 0) + recent.get(e, 0) > 0.02]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=order, x=[usual.get(e, 0) * 100 for e in order], orientation="h", name="Usual",
                         marker_color="#B9CBCE", hovertemplate="%{y}: %{x:.0f}% usually<extra></extra>"))
    fig.add_trace(go.Bar(y=order, x=[recent.get(e, 0) * 100 for e in order], orientation="h", name="Last 7 days",
                         marker_color=LAKE, hovertemplate="%{y}: %{x:.0f}% recently<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(ticksuffix="%", gridcolor="#DDE4E2", tickfont=dict(color=MUTED))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=INK))
    fig = _layout(fig, 60 + 46 * len(order))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.12, x=0, font=dict(color=MUTED)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def outlook_cards(cards: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(cards), gap="medium")
    for col, (label, head, body) in zip(cols, cards):
        with col:
            _md(f'<div class="mj-outlook"><p class="mj-outlook-label">{escape(label)}</p>'
                f'<p class="mj-outlook-head">{escape(head)}</p><p class="mj-outlook-body">{escape(body)}</p></div>')


def outlook_chart(obs: pd.Series, outlooks, tau: float) -> None:
    """Observed daily difficult share, then the outlook with its 80% range."""
    xs = [o.day for o in outlooks]
    start = [obs.index[-1]] if len(obs) else []
    first = [float(obs.iloc[-1]) * 100] if len(obs) else []
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=start + xs + xs[::-1] + start, fill="toself", fillcolor="rgba(45,95,115,0.14)",
                             y=first + [o.high * 100 for o in outlooks] + [o.low * 100 for o in outlooks][::-1] + first,
                             mode="lines", line=dict(width=0), hoverinfo="skip", name="Expected range"))
    fig.add_trace(go.Scatter(x=obs.index, y=obs * 100, mode="lines+markers", name="Your days",
                             line=dict(color=LAKE, width=2.5), marker=dict(size=6),
                             hovertemplate="%{x|%d %b}: %{y:.0f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=start + xs, y=first + [o.predicted * 100 for o in outlooks], mode="lines+markers",
                             name="Outlook", line=dict(color=LAKE, width=2, dash="dot"), marker=dict(size=5),
                             hovertemplate="%{x|%d %b}: about %{y:.0f}%<extra></extra>"))
    fig.add_hline(y=tau * 100, line=dict(color=LEVEL_COLORS["support"], width=1, dash="dot"),
                  annotation=dict(text="Support line", font=dict(color=LEVEL_COLORS["support"], size=12)),
                  annotation_position="top left")
    fig.update_yaxes(range=[0, 100], ticksuffix="%", gridcolor="#DDE4E2", zeroline=False, tickfont=dict(color=MUTED))
    fig.update_xaxes(showgrid=False, tickformat="%d %b", tickfont=dict(color=MUTED), linecolor=RULE)
    fig = _layout(fig, 320)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.15, x=0, font=dict(color=MUTED)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def rec_list(recs: list[dict]) -> None:
    """Library suggestions: title, summary, why it was suggested, and the source it's based on."""
    items = []
    for r in recs:
        e = r["entry"]
        why = f'<p class="mj-rec-why">{escape(r["reason"])}.</p>' if r.get("reason") else ""
        items.append(f'<div class="mj-rec"><p class="mj-rec-title">{escape(e["title"])}</p>'
                     f'<p class="mj-rec-body">{escape(e["summary"])}</p>{why}'
                     f'<p class="mj-rec-src">Based on <a href="{escape(e["url"])}" target="_blank" rel="noopener">'
                     f'{escape(e["source"])}</a></p></div>')
    _md("".join(items))


def footer(model_label: str) -> None:
    _md(f'<p class="mj-footer">MindScope AI estimates emotions from your words with the {escape(model_label)} model, '
        "trained on English Reddit comments. It often misreads short, sarcastic or mixed entries, and it cannot diagnose "
        "anything. Only the scores and the time of each entry are saved, on this computer.</p>")
