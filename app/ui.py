"""Streamlit rendering helpers (layout, charts, cards). Logic lives in components.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from components import (
        EMOTIONS, HELPLINES, NEGATIVE, PCOLS,
        check_crisis_safety, compute_uncertainty_metrics, get_multilabel_emotions,
    )
except ImportError:
    from app.components import (
        EMOTIONS, HELPLINES, NEGATIVE, PCOLS,
        check_crisis_safety, compute_uncertainty_metrics, get_multilabel_emotions,
    )

# ── Dark-violet palette ──────────────────────────────────────────────────────
VIOLET = {
    "bg":        "#0b0714",   # near-black deep violet — page bg
    "bg2":       "#100d20",   # sidebar bg
    "card":      "#160f2b",   # card surface
    "card2":     "#1e1540",   # card hover / deeper
    "border":    "#342660",   # border — visible on dark bg
    "primary":   "#a67df0",   # bright lavender-violet — buttons, links
    "primary2":  "#7a52cc",   # deeper violet — gradient start
    "accent":    "#d4c4ff",   # very light lavender — headings, badges
    "text":      "#f2eeff",   # near-white lavender — ALL body text
    "muted":     "#b0a8d0",   # readable mid-tone — secondary text
    "dim":       "#6a6090",   # dim — captions only
    "success":   "#5ec4a0",   # teal-green
    "warn":      "#e0a060",   # amber
    "danger":    "#e06080",   # rose-pink
    "shadow":    "rgba(5,2,18,0.70)",
    "glow":      "rgba(166,125,240,0.22)",
}

EMOTION_COLORS = {
    "Joy":      "#c4b0f0",
    "Sadness":  "#7c5cbf",
    "Anger":    "#e06080",
    "Fear":     "#9575cd",
    "Anxiety":  "#e0a060",
    "Stress":   "#b06090",
    "Surprise": "#60b8e8",
    "Neutral":  "#7a7490",
}

# ── Global CSS ───────────────────────────────────────────────────────────────
_CSS = f"""
<style>
/* ═══════════════════════════════════════════════════════════════
   ROOT VARS
═══════════════════════════════════════════════════════════════ */
:root {{
    --vi-bg:      {VIOLET["bg"]};
    --vi-bg2:     {VIOLET["bg2"]};
    --vi-card:    {VIOLET["card"]};
    --vi-card2:   {VIOLET["card2"]};
    --vi-border:  {VIOLET["border"]};
    --vi-primary: {VIOLET["primary"]};
    --vi-p2:      {VIOLET["primary2"]};
    --vi-accent:  {VIOLET["accent"]};
    --vi-text:    {VIOLET["text"]};
    --vi-muted:   {VIOLET["muted"]};
    --vi-dim:     {VIOLET["dim"]};
    --vi-shadow:  {VIOLET["shadow"]};
    --vi-glow:    {VIOLET["glow"]};
}}

/* ═══════════════════════════════════════════════════════════════
   FORCE DARK BASE ON EVERY STREAMLIT WRAPPER
═══════════════════════════════════════════════════════════════ */
html, body,
.stApp,
.main,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="block-container"],
section.main > div,
.element-container,
.stMarkdown,
.stForm,
[data-testid="stForm"],
[data-testid="stHorizontalBlock"],
[data-testid="column"] {{
    background-color: transparent !important;
    color: {VIOLET["text"]} !important;
}}

/* The ONE true background — the app root */
.stApp {{
    background: {VIOLET["bg"]} !important;
    background-image: radial-gradient(
        ellipse at 20% 10%, rgba(90,52,160,0.18) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 90%, rgba(60,30,110,0.14) 0%, transparent 55%) !important;
}}

/* ── Top toolbar ── */
[data-testid="stHeader"],
[data-testid="stToolbar"] {{
    background: {VIOLET["bg2"]} !important;
    border-bottom: 1.5px solid {VIOLET["border"]} !important;
}}
[data-testid="stToolbar"] svg,
[data-testid="stHeader"] svg {{
    fill: {VIOLET["muted"]} !important;
}}

/* ── Sidebar Styling — comfortably spaced & visually pleasing ── */
[data-testid="stSidebar"] {{
    background: {VIOLET["bg2"]} !important;
    border-right: 1.5px solid {VIOLET["border"]} !important;
}}
[data-testid="stSidebar"] .block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 1.2rem !important;
    padding-left: 1.1rem !important;
    padding-right: 1.1rem !important;
}}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0.65rem !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    font-size: 0.95rem !important;
    margin-bottom: 0.25rem !important;
}}
[data-testid="stSidebar"] .stButton button {{
    padding: 0.42rem 1rem !important;
    font-size: 0.84rem !important;
    height: 36px !important;
}}
[data-testid="stSidebar"] .stDownloadButton button {{
    padding: 0.42rem 1rem !important;
    font-size: 0.84rem !important;
    height: 36px !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    margin-bottom: 0.45rem !important;
    border-radius: 12px !important;
    border: 1px solid var(--vi-border) !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
    padding: 0.42rem 0.75rem !important;
    font-size: 0.82rem !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stVerticalBlock"] {{
    padding: 0.35rem 0.5rem !important;
    gap: 0.35rem !important;
}}
[data-testid="stSidebar"] .stTextInput {{
    margin-bottom: 0.3rem !important;
}}
[data-testid="stSidebar"] .stTextInput input {{
    padding: 0.4rem 0.9rem !important;
    font-size: 0.86rem !important;
    height: 36px !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] {{
    margin-bottom: 0.3rem !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    padding: 0.22rem 0.75rem !important;
    font-size: 0.8rem !important;
    margin-right: 0.3rem !important;
}}
[data-testid="stSidebar"] hr {{
    margin: 0.55rem 0 !important;
    border-color: var(--vi-border) !important;
}}
[data-testid="stSidebar"] .stCaption {{
    font-size: 0.74rem !important;
    line-height: 1.35 !important;
}}

/* ═══════════════════════════════════════════════════════════════
   TYPOGRAPHY — HIGH CONTRAST
═══════════════════════════════════════════════════════════════ */
*,
*::before,
*::after {{
    color: inherit;
}}

body, .stApp, p, span, div, li, td, th,
label, strong, em, small, blockquote,
[class*="st"] {{
    color: {VIOLET["text"]};
}}

h1, h2, h3, h4, h5, h6,
[data-testid="stHeading"],
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3 {{
    color: {VIOLET["accent"]} !important;
    font-weight: 700 !important;
}}
h1 {{ font-size: 2.1rem !important; letter-spacing: -0.02em; }}
h2 {{ font-size: 1.45rem !important; }}
h3 {{ font-size: 1.15rem !important; }}

.stMarkdown p,
.stMarkdown li {{
    color: {VIOLET["text"]} !important;
    line-height: 1.65 !important;
}}

.stMarkdown strong {{
    color: {VIOLET["accent"]} !important;
}}

small,
.stCaption,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {{
    color: {VIOLET["dim"]} !important;
    font-size: 0.82rem !important;
}}

a {{ color: {VIOLET["primary"]} !important; }}
a:hover {{ color: {VIOLET["accent"]} !important; text-decoration: underline; }}

/* ═══════════════════════════════════════════════════════════════
   BUBBLE / CARD
═══════════════════════════════════════════════════════════════ */
.vi-card {{
    background: linear-gradient(145deg, {VIOLET["card"]} 0%, {VIOLET["card2"]} 100%) !important;
    border: 1.5px solid {VIOLET["border"]};
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 30px {VIOLET["shadow"]}, 0 0 0 1px rgba(166,125,240,0.07);
    transition: box-shadow 0.22s ease, transform 0.22s ease;
}}
.vi-card:hover {{
    box-shadow: 0 12px 38px {VIOLET["shadow"]}, 0 0 20px {VIOLET["glow"]};
}}

/* ── Metric bubble ── */
.vi-metric {{
    background: linear-gradient(135deg, {VIOLET["card2"]} 0%, #26184e 100%) !important;
    border: 1.5px solid {VIOLET["border"]};
    border-radius: 16px;
    padding: 0.85rem 1rem;
    text-align: center;
    box-shadow: 0 4px 16px {VIOLET["shadow"]}, 0 0 10px {VIOLET["glow"]};
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    cursor: default;
}}
.vi-metric:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px {VIOLET["shadow"]}, 0 0 18px {VIOLET["glow"]};
}}
.vi-metric-value {{
    font-size: 1.7rem;
    font-weight: 800;
    color: {VIOLET["accent"]} !important;
    line-height: 1.1;
}}
.vi-metric-label {{
    font-size: 0.72rem;
    color: {VIOLET["muted"]} !important;
    margin-top: 0.3rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}

/* ── Emotion badge ── */
.vi-badge {{
    display: inline-block;
    background: linear-gradient(135deg, {VIOLET["primary2"]}, {VIOLET["primary"]});
    color: #ffffff !important;
    border-radius: 999px;
    padding: 0.4rem 1.25rem;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    box-shadow: 0 4px 18px rgba(166,125,240,0.50), 0 0 12px rgba(166,125,240,0.25);
}}

/* ── Recommendation cards ── */
.vi-rec-escalation {{
    background: linear-gradient(135deg, #200c18, #2e1020) !important;
    border-left: 5px solid {VIOLET["danger"]};
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 4px 20px rgba(224,96,128,0.22);
}}
.vi-rec-suggestion {{
    background: linear-gradient(135deg, #22160c, #301d0a) !important;
    border-left: 5px solid {VIOLET["warn"]};
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 4px 20px rgba(224,160,96,0.22);
}}
.vi-rec-none {{
    background: linear-gradient(135deg, #0a1e16, #0d2820) !important;
    border-left: 5px solid {VIOLET["success"]};
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 4px 20px rgba(94,196,160,0.22);
}}
.vi-rec-title {{
    font-size: 1rem;
    font-weight: 700;
    color: {VIOLET["accent"]} !important;
    margin-bottom: 0.4rem;
}}
.vi-rec-text {{
    font-size: 0.9rem;
    color: {VIOLET["text"]} !important;
    line-height: 1.6;
}}

/* ── Disclaimer with Emergency Helplines ── */
.vi-disclaimer {{
    background: linear-gradient(135deg, {VIOLET["card"]}, {VIOLET["card2"]}) !important;
    border: 1.5px solid {VIOLET["border"]};
    border-left: 5px solid {VIOLET["primary"]};
    border-radius: 16px;
    padding: 1rem 1.4rem;
    font-size: 0.88rem;
    color: {VIOLET["muted"]} !important;
    line-height: 1.6;
    margin-bottom: 1.1rem;
    box-shadow: 0 4px 20px {VIOLET["shadow"]};
}}
.vi-disclaimer strong {{ color: {VIOLET["accent"]} !important; }}

.vi-helpline-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.65rem;
}}
.vi-phone-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: linear-gradient(135deg, #24143a, #2f1848);
    border: 1px solid var(--vi-border);
    border-radius: 999px;
    padding: 0.28rem 0.85rem;
    font-size: 0.8rem;
    color: var(--vi-accent) !important;
    text-decoration: none !important;
    box-shadow: 0 2px 8px var(--vi-shadow);
    transition: transform 0.15s ease, border-color 0.15s ease;
}}
.vi-phone-chip:hover {{
    transform: translateY(-1px);
    border-color: var(--vi-primary);
    color: #ffffff !important;
    box-shadow: 0 4px 12px var(--vi-glow);
}}

/* ── Consent box ── */
.vi-consent {{
    background: linear-gradient(135deg, {VIOLET["card"]}, {VIOLET["card2"]}) !important;
    border: 1.5px solid {VIOLET["border"]};
    border-radius: 16px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 16px {VIOLET["shadow"]};
}}

/* ── Pattern Insight Box ── */
.vi-pattern-box {{
    background: linear-gradient(135deg, #1d1338, #251848);
    border: 1px solid var(--vi-border);
    border-radius: 14px;
    padding: 0.85rem 1.1rem;
    box-shadow: 0 4px 16px var(--vi-shadow);
}}

/* ═══════════════════════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════════════════════ */
.stButton > button,
.stButton > button:focus {{
    background: linear-gradient(135deg, {VIOLET["primary2"]} 0%, {VIOLET["primary"]} 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.55rem 1.8rem !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 20px rgba(166,125,240,0.42),
                0 1px 0 rgba(255,255,255,0.10) inset !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, #5e38a8 0%, {VIOLET["primary2"]} 100%) !important;
    box-shadow: 0 6px 28px rgba(166,125,240,0.60),
                0 0 16px rgba(166,125,240,0.28) !important;
    transform: translateY(-2px) !important;
}}
.stButton > button:active {{
    transform: translateY(0) !important;
    box-shadow: 0 2px 12px rgba(166,125,240,0.38) !important;
}}

/* Download button */
[data-testid="stDownloadButton"] > button {{
    background: linear-gradient(135deg, #1e5c3a 0%, {VIOLET["success"]} 100%) !important;
    color: #fff !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 18px rgba(94,196,160,0.38) !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    background: linear-gradient(135deg, #17502f 0%, #4ab890 100%) !important;
    box-shadow: 0 6px 24px rgba(94,196,160,0.55) !important;
    transform: translateY(-2px) !important;
}}

/* Form submit button */
[data-testid="stFormSubmitButton"] > button {{
    background: linear-gradient(135deg, {VIOLET["primary2"]} 0%, {VIOLET["primary"]} 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 4px 22px rgba(166,125,240,0.48) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}}
[data-testid="stFormSubmitButton"] > button:hover {{
    background: linear-gradient(135deg, #4e2888 0%, {VIOLET["primary2"]} 100%) !important;
    box-shadow: 0 6px 30px rgba(166,125,240,0.65),
                0 0 18px rgba(166,125,240,0.32) !important;
    transform: translateY(-2px) !important;
}}
[data-testid="stFormSubmitButton"] > button:disabled {{
    background: {VIOLET["card2"]} !important;
    color: {VIOLET["dim"]} !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
}}

/* ═══════════════════════════════════════════════════════════════
   INPUTS
═══════════════════════════════════════════════════════════════ */
textarea,
.stTextArea textarea,
input[type="text"],
.stTextInput input {{
    background: {VIOLET["bg"]} !important;
    border: 1.5px solid {VIOLET["border"]} !important;
    color: {VIOLET["text"]} !important;
    caret-color: {VIOLET["primary"]} !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}}
textarea,
.stTextArea textarea {{
    border-radius: 14px !important;
    padding: 0.9rem 1.1rem !important;
}}
input[type="text"],
.stTextInput input {{
    border-radius: 999px !important;
    padding: 0.45rem 1rem !important;
}}
textarea:focus, .stTextArea textarea:focus,
input[type="text"]:focus, .stTextInput input:focus {{
    border-color: {VIOLET["primary"]} !important;
    box-shadow: 0 0 0 3px rgba(166,125,240,0.24) !important;
    outline: none !important;
}}
textarea::placeholder,
input::placeholder {{
    color: {VIOLET["dim"]} !important;
}}

/* ── Radio pills ── */
[data-testid="stRadio"] label {{
    background: {VIOLET["card"]} !important;
    border: 1.5px solid {VIOLET["border"]} !important;
    border-radius: 999px !important;
    padding: 0.25rem 0.85rem !important;
    margin-right: 0.3rem !important;
    font-size: 0.85rem !important;
    color: {VIOLET["muted"]} !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
}}
[data-testid="stRadio"] label:hover {{
    background: {VIOLET["card2"]} !important;
    border-color: {VIOLET["primary"]} !important;
    color: {VIOLET["accent"]} !important;
}}

/* ── Checkbox ── */
[data-testid="stCheckbox"] span,
[data-testid="stCheckbox"] p {{
    color: {VIOLET["text"]} !important;
    font-size: 0.92rem !important;
}}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    background: {VIOLET["card"]} !important;
    border: 1.5px solid {VIOLET["border"]} !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 12px {VIOLET["shadow"]} !important;
    overflow: hidden !important;
}}
[data-testid="stExpander"] > div {{
    background: {VIOLET["card"]} !important;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {{
    font-weight: 600 !important;
    color: {VIOLET["accent"]} !important;
}}
[data-testid="stExpander"] summary:hover {{
    background: rgba(166,125,240,0.08) !important;
}}

/* ── Dataframe & Table ── */
[data-testid="stDataFrame"],
.stTable,
[data-testid="stTable"] {{
    border: 1.5px solid {VIOLET["border"]} !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 14px {VIOLET["shadow"]} !important;
}}
[data-testid="stDataFrame"] th,
.stTable th,
[data-testid="stTable"] th {{
    background: {VIOLET["card2"]} !important;
    color: {VIOLET["accent"]} !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    border-bottom: 1.5px solid {VIOLET["border"]} !important;
}}
[data-testid="stDataFrame"] td,
.stTable td,
[data-testid="stTable"] td {{
    background: {VIOLET["card"]} !important;
    color: {VIOLET["text"]} !important;
    font-size: 0.8rem !important;
    border-bottom: 1px solid rgba(52,38,96,0.35) !important;
}}

/* ═══════════════════════════════════════════════════════════════
   PLOTLY GRAPHS — FULL VISIBILITY, FULL ASPECT RATIO, ZERO SCROLLBARS
═══════════════════════════════════════════════════════════════ */

/* Allow parent containers to adapt naturally to graph height without scrollbars */
[data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]),
.element-container:has([data-testid="stPlotlyChart"]),
[data-testid="column"],
[data-testid="stHorizontalBlock"],
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"] {{
    overflow: visible !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}}

/* stPlotlyChart container: fully visible, auto-height, crisp border, zero scrollbar */
[data-testid="stPlotlyChart"],
.stPlotlyChart {{
    border: 1.5px solid {VIOLET["border"]} !important;
    border-radius: 16px !important;
    padding: 0.2rem !important;
    margin: 0 !important;
    background: {VIOLET["bg"]} !important;
    box-shadow: 0 4px 18px {VIOLET["shadow"]} !important;
    overflow: visible !important;
    box-sizing: border-box !important;
    height: auto !important;
    max-height: none !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}}

/* Zoom/pan is already disabled via fixedrange=True on axes and scrollZoom:false in config */

/* Force hide all scrollbars on any graph or graph wrapper */
[data-testid="stPlotlyChart"],
[data-testid="stPlotlyChart"] *,
.stPlotlyChart,
.stPlotlyChart *,
.js-plotly-plot,
.js-plotly-plot *,
.plot-container,
.plot-container *,
.svg-container,
.svg-container *,
[data-testid="stElementContainer"],
.element-container,
[data-testid="column"],
[data-testid="column"] *,
[data-testid="stHorizontalBlock"],
[data-testid="stVerticalBlock"] {{
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}}

[data-testid="stPlotlyChart"]::-webkit-scrollbar,
[data-testid="stPlotlyChart"] *::-webkit-scrollbar,
.stPlotlyChart::-webkit-scrollbar,
.stPlotlyChart *::-webkit-scrollbar,
.js-plotly-plot::-webkit-scrollbar,
.js-plotly-plot *::-webkit-scrollbar,
.plot-container::-webkit-scrollbar,
.plot-container *::-webkit-scrollbar,
.svg-container::-webkit-scrollbar,
[data-testid="stElementContainer"]::-webkit-scrollbar,
[data-testid="stElementContainer"] *::-webkit-scrollbar,
.element-container::-webkit-scrollbar,
[data-testid="column"]::-webkit-scrollbar,
[data-testid="column"] *::-webkit-scrollbar,
[data-testid="stHorizontalBlock"]::-webkit-scrollbar,
[data-testid="stVerticalBlock"]::-webkit-scrollbar {{
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}}

/* ── Alert boxes ── */
[data-testid="stAlert"],
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {{
    background: {VIOLET["card"]} !important;
    border: 1.5px solid {VIOLET["border"]} !important;
    border-radius: 14px !important;
    color: {VIOLET["text"]} !important;
}}

/* ── Toast ── */
[data-testid="stToast"],
[data-testid="stToast"] p {{
    background: {VIOLET["card2"]} !important;
    border: 1.5px solid {VIOLET["border"]} !important;
    border-radius: 14px !important;
    color: {VIOLET["text"]} !important;
    box-shadow: 0 4px 22px {VIOLET["shadow"]} !important;
}}

/* ── Sidebar profile badge ── */
.vi-profile-badge {{
    background: linear-gradient(135deg, {VIOLET["card2"]}, #28185a) !important;
    border: 1.5px solid {VIOLET["border"]};
    border-radius: 10px;
    padding: 0.5rem 0.85rem;
    font-size: 0.83rem;
    color: {VIOLET["accent"]} !important;
    text-align: center;
    margin-bottom: 0.6rem;
    box-shadow: 0 2px 10px {VIOLET["shadow"]};
}}

/* ── Scrollbars: EXCLUSIVELY FOR MAIN APP VIEWPORT ── */
html::-webkit-scrollbar,
body::-webkit-scrollbar,
[data-testid="stAppViewContainer"]::-webkit-scrollbar,
section.main::-webkit-scrollbar {{
    width: 6px !important;
    height: 6px !important;
}}
html::-webkit-scrollbar-track,
body::-webkit-scrollbar-track,
[data-testid="stAppViewContainer"]::-webkit-scrollbar-track,
section.main::-webkit-scrollbar-track {{
    background: {VIOLET["bg"]} !important;
}}
html::-webkit-scrollbar-thumb,
body::-webkit-scrollbar-thumb,
[data-testid="stAppViewContainer"]::-webkit-scrollbar-thumb,
section.main::-webkit-scrollbar-thumb {{
    background: {VIOLET["primary2"]} !important;
    border-radius: 999px !important;
}}
html::-webkit-scrollbar-thumb:hover,
body::-webkit-scrollbar-thumb:hover,
[data-testid="stAppViewContainer"]::-webkit-scrollbar-thumb:hover,
section.main::-webkit-scrollbar-thumb:hover {{
    background: {VIOLET["primary"]} !important;
}}

/* ── GLOBALLY SUPPRESS SCROLLBARS ON EVERY OTHER ELEMENT, CONTAINER, CARD & GRAPH ── */
*:not(html):not(body):not([data-testid="stAppViewContainer"]):not(section.main)::-webkit-scrollbar {{
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}}
*:not(html):not(body):not([data-testid="stAppViewContainer"]):not(section.main) {{
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Universal Plotly Config — completely disable all graph scrolling/zooming ──
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": False,
    "doubleClick": False,
    "showAxisDragHandles": False,
    "showAxisRangeEntryBoxes": False,
    "staticPlot": False,
}


# ── Plotly dark base template ─────────────────────────────────────────────────
def _vi_layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,10,28,0.0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=VIOLET["text"], size=11),
        margin=dict(l=0, r=0, t=28, b=0),
        autosize=True,
        legend=dict(
            orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)",
            font=dict(color=VIOLET["muted"], size=10),
        ),
    )
    # Default axes locked with fixedrange=True to completely eliminate any graph scrolling
    default_xaxis = dict(
        fixedrange=True,
        gridcolor="rgba(61,45,107,0.45)",
        linecolor=VIOLET["border"],
        tickfont=dict(color=VIOLET["muted"], size=10),
    )
    default_yaxis = dict(
        fixedrange=True,
        gridcolor="rgba(61,45,107,0.45)",
        linecolor=VIOLET["border"],
        tickfont=dict(color=VIOLET["muted"], size=10),
    )

    user_x = kwargs.pop("xaxis", None)
    if user_x:
        default_xaxis.update(user_x)
    default_xaxis["fixedrange"] = True
    base["xaxis"] = default_xaxis

    user_y = kwargs.pop("yaxis", None)
    if user_y:
        default_yaxis.update(user_y)
    default_yaxis["fixedrange"] = True
    base["yaxis"] = default_yaxis

    base.update(kwargs)
    return base


# ── Render functions ──────────────────────────────────────────────────────────

def render_header():
    inject_css()
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.4rem;">
            <span style="font-size:2.8rem;filter:drop-shadow(0 0 12px rgba(157,120,232,0.6));">🧠</span>
            <div>
                <h1 style="margin:0;color:{VIOLET['accent']};
                           text-shadow:0 0 20px rgba(157,120,232,0.4);">Mood Journal</h1>
                <p style="margin:0;color:{VIOLET['muted']};font-size:0.95rem;">
                    Emotion-aware journaling &nbsp;·&nbsp; track how your mood changes over time
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer():
    st.markdown(
        f"""
        <div class="vi-disclaimer">
            <div style="display:flex; align-items:flex-start; gap:0.6rem; margin-bottom:0.4rem;">
                <span style="font-size:1.25rem;">ℹ️</span>
                <div>
                    <strong style="color:{VIOLET['accent']}; font-size:0.95rem;">Wellness & Early-Awareness Tool — Not a Diagnostic Instrument</strong><br>
                    <span style="font-size:0.86rem; color:{VIOLET['text']}; line-height:1.55;">
                        This application estimates emotional patterns from written text. It does not provide medical diagnoses or replace a certified therapist or doctor.
                        <strong>If you or someone you know is in distress or needs immediate support, please contact these verified 24x7 free helplines:</strong>
                    </span>
                </div>
            </div>
            <div class="vi-helpline-row">
                <a href="tel:14416" class="vi-phone-chip" title="Call Tele-MANAS toll-free">
                    <span>📞</span>
                    <span><strong>Tele-MANAS (24x7 Free):</strong> 14416 / 1800-891-4416</span>
                </a>
                <a href="tel:112" class="vi-phone-chip" title="Call Emergency Services">
                    <span>🚨</span>
                    <span><strong>National Emergency:</strong> 112</span>
                </a>
                <a href="tel:18005990019" class="vi-phone-chip" title="Call KIRAN Helpline">
                    <span>📞</span>
                    <span><strong>KIRAN Helpline:</strong> 1800-599-0019</span>
                </a>
                <a href="tel:9999666555" class="vi-phone-chip" title="Call Vandrevala Support">
                    <span>💬</span>
                    <span><strong>Vandrevala Crisis Support:</strong> +91 9999 666 555</span>
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(icon: str, title: str, subtitle: str | None = None):
    """Renders section header situated prominently ABOVE separator containers."""
    sub_html = (
        f'<span style="font-size:0.78rem; color:{VIOLET["muted"]}; font-weight:400; margin-left:0.4rem;">'
        f'{subtitle}</span>'
        if subtitle else ""
    )
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:0.5rem; margin-top:0.4rem; margin-bottom:0.45rem; padding-left:0.15rem;">
            <span style="font-size:1.25rem;">{icon}</span>
            <span style="font-size:1.1rem; font-weight:700; color:{VIOLET['accent']}; letter-spacing:0.01em;">{title}</span>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_consent() -> bool:
    render_section_title("🔒", "Privacy & Consent", "Local Processing · Full Anonymity")
    st.markdown('<div class="vi-consent">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="font-size:0.86rem; color:{VIOLET['text']}; line-height:1.5; margin-bottom:0.35rem;">
            🔒 Your entries are analyzed securely in real time. <strong>Zero raw text is saved</strong> — only private emotion scores and timestamps are stored so you can view your personal trends.
        </div>
        """,
        unsafe_allow_html=True,
    )
    result = st.checkbox("I agree to analyze my check-in entries (Pre-consented; uncheck to pause)", value=True, key="consent")
    st.markdown("</div>", unsafe_allow_html=True)
    return result


def render_entry_result(probs: np.ndarray, model_label: str, raw_text: str = ""):
    # ── Safety & Crisis Guardrail ──
    safety = check_crisis_safety(raw_text)
    if safety["triggered"]:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg, #320d1e, #451028); border:1.5px solid {VIOLET['danger']}; border-radius:14px; padding:0.85rem 1.1rem; margin-bottom:0.75rem; box-shadow:0 4px 20px rgba(224,96,128,0.35);">
                <div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.25rem;">
                    <span style="font-size:1.25rem;">🚨</span>
                    <strong style="color:#ffffff; font-size:0.95rem;">Immediate Support & Safety Notice</strong>
                </div>
                <p style="font-size:0.84rem; color:{VIOLET['text']}; margin:0 0 0.45rem 0; line-height:1.45;">
                    {safety['message']}
                </p>
                <div class="vi-helpline-row" style="margin-top:0.3rem;">
                    <a href="tel:14416" class="vi-phone-chip" style="background:#501430; border-color:{VIOLET['danger']};">
                        <span>📞</span> <span><strong>Tele-MANAS:</strong> 14416 (24x7 Free)</span>
                    </a>
                    <a href="tel:112" class="vi-phone-chip" style="background:#501430; border-color:{VIOLET['danger']};">
                        <span>🚨</span> <span><strong>Emergency:</strong> 112</span>
                    </a>
                    <a href="tel:18005990019" class="vi-phone-chip" style="background:#501430; border-color:{VIOLET['danger']};">
                        <span>📞</span> <span><strong>KIRAN:</strong> 1800-599-0019</span>
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Multi-Label Emotion Detection ──
    ml_emotions = get_multilabel_emotions(probs)
    unc = compute_uncertainty_metrics(probs)

    badges_html = []
    if len(ml_emotions) >= 1:
        primary = ml_emotions[0]
        badges_html.append(
            f'<span class="vi-badge">Primary: {primary["emotion"]} &nbsp;·&nbsp; {primary["prob"]:.0%}</span>'
        )
    if len(ml_emotions) >= 2:
        secondary = ml_emotions[1]
        badges_html.append(
            f'<span class="vi-badge" style="background:linear-gradient(135deg, #5a3c8c, #7a52cc); font-size:0.92rem; margin-left:0.4rem;">Co-occurring: {secondary["emotion"]} &nbsp;·&nbsp; {secondary["prob"]:.0%}</span>'
        )

    st.markdown(
        f"""
        <div style="margin:0.4rem 0 0.45rem;">
            <div style="color:{VIOLET['dim']}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">
                Detected Emotional Signature (Multi-Label)
            </div>
            {''.join(badges_html)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Emotion Confidence & Uncertainty Metrics ──
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.55rem; font-size:0.78rem; background:rgba(30,21,64,0.65); padding:0.3rem 0.75rem; border-radius:999px; border:1px solid {VIOLET['border']}; width:fit-content;">
            <span style="color:{unc['certainty_color']}; font-weight:700;">● {unc['certainty_label']}</span>
            <span style="color:{VIOLET['muted']};">|</span>
            <span style="color:{VIOLET['text']};">Confidence: <strong>{unc['confidence']}%</strong></span>
            <span style="color:{VIOLET['muted']};">|</span>
            <span style="color:{VIOLET['dim']};">{unc['explanation']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Graph 1 of 2: Real-Time Emotion Probability Distribution ──
    order = np.argsort(probs)[::-1]
    fig = go.Figure(go.Bar(
        x=[probs[i] for i in order][::-1],
        y=[EMOTIONS[i] for i in order][::-1],
        orientation="h",
        marker=dict(
            color=[EMOTION_COLORS[EMOTIONS[i]] for i in order][::-1],
            opacity=0.88,
            line=dict(width=0),
        ),
        text=[f"{probs[i]:.0%}" for i in order][::-1],
        textposition="outside",
        textfont=dict(color=VIOLET["muted"], size=10),
    ))
    fig.update_layout(
        **_vi_layout(
            height=280,
            margin=dict(l=75, r=45, t=10, b=10),
            xaxis=dict(range=[0, 1.18], visible=False, gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=VIOLET["text"], size=11)),
        )
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.caption(f"Model: {model_label}  ·  Fine-grained emotion mapping  ·  Multilingual Hinglish active.")


def render_summary_metrics(daily: pd.DataFrame, n_entries: int):
    c1, c2, c3 = st.columns(3)
    items = [
        (c1, str(n_entries),     "Entries Analysed",       "📝"),
        (c2, str(len(daily)),    "Days with Entries",      "📅"),
        (c3, f"{daily['S_ema'].iloc[-1]:.2f}" if len(daily) else "—",
              "Latest Neg. Affect", "📊"),
    ]
    for col, val, label, icon in items:
        col.markdown(
            f"""
            <div class="vi-metric">
                <div class="vi-metric-value">{icon} {val}</div>
                <div class="vi-metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_recommendation(rec: dict):
    level = rec.get("level", "none")
    title = rec.get("title", "")
    text  = rec.get("text", "")

    if level == "escalation":
        st.markdown(
            f'<div class="vi-rec-escalation">'
            f'<div class="vi-rec-title">💙 {title}</div>'
            f'<div class="vi-rec-text">{text}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<span style="color:{VIOLET["muted"]};font-size:0.84rem;font-weight:600;">'
            "Recommended Support Contacts</span>",
            unsafe_allow_html=True,
        )
        chips = "".join(
            f'<span class="vi-phone-chip">📞 {name}: <strong>{number}</strong></span>'
            for name, number in HELPLINES.items()
        )
        st.markdown(f'<div class="vi-helpline-row">{chips}</div>', unsafe_allow_html=True)

    elif level == "suggestion":
        st.markdown(
            f'<div class="vi-rec-suggestion">'
            f'<div class="vi-rec-title">🌿 {title}</div>'
            f'<div class="vi-rec-text">{text}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="vi-rec-none">'
            f'<div class="vi-rec-title">✅ {title}</div>'
            f'<div class="vi-rec-text">{text}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_recent_mood_analysis(
    hist: pd.DataFrame,
    daily: pd.DataFrame | None,
    baseline: dict | None = None,
    anomaly: dict | None = None,
):
    """Longitudinal explanation, personal baseline comparison, and anomaly monitoring — ZERO graphs for clarity."""
    if hist.empty or daily is None or daily.empty:
        return

    baseline = baseline or {"mean": 0.35, "std": 0.12, "tau_personal": 0.55, "status": "Baseline"}
    anomaly = anomaly or {"status": "Stable", "badge": "🟢 Stable Baseline", "color": "#7c5cbf", "desc": "Scores are tracking within typical ranges."}

    recent_entries = hist.tail(10)
    counts = recent_entries["emotion"].value_counts()
    dominant_recent = counts.index[0]
    dom_pct = (counts.iloc[0] / len(recent_entries)) * 100
    avg_s = float(recent_entries["s"].mean())
    mean_base = baseline.get("mean", 0.35)
    tau_p = baseline.get("tau_personal", 0.55)

    st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.95rem; font-weight:700; color:{VIOLET["accent"]}; margin-bottom:0.4rem;">'
        '📊 Longitudinal Trajectory & Personalized Baseline</div>',
        unsafe_allow_html=True,
    )

    # Narrative explanation
    if avg_s > 0.55:
        tone_class = "Elevated Strain"
        tone_color = VIOLET["danger"]
        pattern_desc = (
            f"Recent check-ins show emotional strain with **{dominant_recent}** leading at **{dom_pct:.0f}%**. "
            f"Negative affect averages **{avg_s:.2f}**, which exceeds your baseline ({mean_base:.2f}). "
            "Pacing daily demands and grounding breaks is strongly advised."
        )
    elif avg_s > 0.35:
        tone_class = "Moderate Mix"
        tone_color = VIOLET["warn"]
        pattern_desc = (
            f"Recent check-ins show a balanced mix, with **{dominant_recent}** as the primary tone ({dom_pct:.0f}%). "
            f"Negative affect averages **{avg_s:.2f}**, remaining close to your personal norm ({mean_base:.2f})."
        )
    else:
        tone_class = "Positive & Resilient"
        tone_color = VIOLET["success"]
        pattern_desc = (
            f"Recent check-ins indicate calm stability, with **{dominant_recent}** leading ({dom_pct:.0f}%). "
            f"Negative affect is healthy and low at **{avg_s:.2f}** (baseline: {mean_base:.2f})."
        )

    st.markdown(
        f"""
        <div class="vi-pattern-box" style="margin-top:0.2rem; margin-bottom:0.65rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                <span style="font-weight:700; color:{VIOLET['accent']}; font-size:0.86rem;">📈 Trajectory Interpretation</span>
                <span style="font-size:0.74rem; font-weight:700; color:{tone_color}; background:rgba(0,0,0,0.35); padding:0.14rem 0.6rem; border-radius:999px;">{tone_class}</span>
            </div>
            <p style="font-size:0.84rem; color:{VIOLET['text']}; line-height:1.55; margin:0;">
                {pattern_desc}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Personalized Baseline & Anomaly Monitor Card (Pure Card — No extra graphs)
    anomaly_badge = anomaly.get("badge", "🟢 Stable Baseline")
    anomaly_color = anomaly.get("color", "#7c5cbf")
    anomaly_desc = anomaly.get("desc", f"Current affect aligns with personal baseline ({mean_base:.2f}).")

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, {VIOLET['card2']}, #251648); border:1.5px solid {VIOLET['border']}; border-radius:14px; padding:0.85rem 1.1rem; margin-bottom:0.6rem; box-shadow:0 4px 16px {VIOLET['shadow']};">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                <span style="font-size:0.86rem; font-weight:700; color:{VIOLET['accent']};">🎯 Personalized Baseline & Anomaly Monitor</span>
                <span style="font-size:0.74rem; font-weight:700; color:#ffffff; background:{anomaly_color}; padding:0.16rem 0.65rem; border-radius:999px;">
                    {anomaly_badge}
                </span>
            </div>
            <p style="font-size:0.82rem; color:{VIOLET['muted']}; margin:0 0 0.45rem 0; line-height:1.45;">
                {anomaly_desc}
            </p>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.6rem; font-size:0.78rem; border-top:1px solid rgba(52,38,96,0.45); padding-top:0.45rem;">
                <div>
                    <span style="color:{VIOLET['dim']};">Personal Baseline (μ):</span>
                    <strong style="color:#5ec4a0; margin-left:4px;">{mean_base:.2f}</strong>
                </div>
                <div>
                    <span style="color:{VIOLET['dim']};">Dynamic Alert Threshold:</span>
                    <strong style="color:{VIOLET['danger']}; margin-left:4px;">{tau_p:.2f}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trends(
    daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame,
    tau: float, flags: np.ndarray, view: str,
    baseline: dict | None = None,
):
    """Renders Graph 2: The single unified longitudinal trajectory with Personal Baseline & Support Threshold + History Table."""
    data = {"Daily": daily, "Weekly": weekly, "Monthly": monthly}[view]
    if data.empty:
        st.info("No recorded mood entries available for this view yet.", icon="🌱")
        return

    baseline = baseline or {"mean": 0.35, "std": 0.12, "tau_personal": tau}
    mean_base = baseline.get("mean", 0.35)

    # Automated Pattern Explanation Narrative
    avg_s = data["S"].mean()
    most_common_dom = data["dominant"].mode()[0] if not data["dominant"].empty else "Neutral"
    dom_count = (data["dominant"] == most_common_dom).sum()
    dom_share = (dom_count / len(data)) * 100

    if len(data) >= 4:
        half = len(data) // 2
        s_early = data["S"].iloc[:half].mean()
        s_late = data["S"].iloc[half:].mean()
        diff = s_late - s_early
        if diff > 0.10:
            trend_dir = f"trending higher in negative affect (+{diff:.2f})"
            trend_tone = "Alert"
            trend_color = VIOLET["danger"]
        elif diff < -0.10:
            trend_dir = f"trending calmer and recovering (-{abs(diff):.2f})"
            trend_tone = "Improving"
            trend_color = VIOLET["success"]
        else:
            trend_dir = "maintaining a steady, stable trajectory"
            trend_tone = "Stable"
            trend_color = VIOLET["muted"]
    else:
        trend_dir = "reflecting your baseline entries"
        trend_tone = "Baseline"
        trend_color = VIOLET["primary"]

    st.markdown(
        f"""
        <div class="vi-pattern-box" style="margin-bottom:0.8rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                <span style="font-weight:700; color:{VIOLET['accent']}; font-size:0.9rem;">
                    🧠 {view} Trajectory & Longitudinal Trends
                </span>
                <span style="font-size:0.74rem; font-weight:700; color:{trend_color}; background:rgba(0,0,0,0.4); padding:0.15rem 0.65rem; border-radius:999px;">
                    {trend_tone}
                </span>
            </div>
            <p style="font-size:0.85rem; color:{VIOLET['text']}; line-height:1.55; margin:0;">
                Across your <strong>{view.lower()}</strong> history ({len(data)} recording points), primary emotion has been <strong>{most_common_dom}</strong> ({dom_share:.0f}% of time).
                Average negative affect is <strong>{avg_s:.2f}</strong> (Personal baseline: <strong>{mean_base:.2f}</strong>), and trajectory is <strong>{trend_dir}</strong>.
                {f"On {int(flags.sum())} day(s), negative affect exceeded the support threshold ({tau:.2f})." if flags.sum() > 0 else "Scores have remained comfortably within typical ranges without persistent distress."}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    wellness = np.clip((1.0 - data["S"]) * 100, 0, 100)
    wellness_ema = np.clip((1.0 - data["S_ema"]) * 100, 0, 100)
    mean_base_pct = np.clip((1.0 - mean_base) * 100, 0, 100)
    tau_alert_pct = np.clip((1.0 - tau) * 100, 0, 100)

    # Clean date strings for x-axis so gaps don't draw artificial diagonal plunges
    x_labels = [d.strftime("%b %d") for d in data.index]

    # ── Graph 2 of 2: Unified Stable Mood Trajectory (0–100%) ──
    fig = go.Figure()

    # 1. Shaded Healthy Equilibrium Zone (60% to 85%)
    fig.add_hrect(
        y0=60, y1=85,
        fillcolor="rgba(94,196,160,0.08)", line_width=0,
        annotation_text="Healthy Equilibrium Zone (60%–85%)",
        annotation_position="top left",
        annotation_font_color="#5ec4a0", annotation_font_size=9,
    )

    # 2. Check-in Stability points
    fig.add_trace(go.Scatter(
        x=x_labels, y=wellness,
        mode="lines+markers", name=f"{view} Check-in",
        line=dict(color="rgba(176,168,208,0.45)", width=1.5, dash="dot"),
        marker=dict(size=7, color=VIOLET["accent"]),
        text=[f"{v:.0f}%" for v in wellness],
        hovertemplate="<b>%{x}</b><br>Mood Stability: <b>%{y:.1f}%</b><extra></extra>",
    ))

    if view == "Daily":
        fig.add_trace(go.Scatter(
            x=x_labels, y=wellness_ema,
            mode="lines", name="Smoothed Stability Trend (EMA)",
            line=dict(color=VIOLET["primary"], width=3),
            hovertemplate="<b>%{x}</b><br>Smoothed Trend: <b>%{y:.1f}%</b><extra></extra>",
        ))

        # Personal Baseline Line
        fig.add_hline(
            y=mean_base_pct, line_dash="dash", line_color="#5ec4a0",
            annotation_text=f"Personal Baseline ({mean_base_pct:.0f}%)",
            annotation_position="right",
            annotation_font_color="#5ec4a0",
            annotation_font_size=10,
            annotation_xanchor="left",
        )

        # Support Alert Threshold
        fig.add_hline(
            y=tau_alert_pct, line_dash="dot", line_color=VIOLET["danger"],
            annotation_text=f"Support Alert ({tau_alert_pct:.0f}%)",
            annotation_position="right",
            annotation_font_color=VIOLET["danger"],
            annotation_font_size=10,
            annotation_xanchor="left",
        )

    fig.update_layout(
        **_vi_layout(
            height=460,
            autosize=True,
            title=dict(
                text=f"Mood Stability Index (0–100%) vs. Personal Baseline ({view})",
                font=dict(color=VIOLET["accent"], size=12),
                y=0.97,
            ),
            margin=dict(l=55, r=120, t=45, b=55),
            yaxis=dict(
                range=[0, 110],
                title="Stability Index (%)",
                ticksuffix="%",
                gridcolor="rgba(61,45,107,0.45)",
                tickfont=dict(color=VIOLET["muted"], size=10),
                title_font=dict(color=VIOLET["muted"], size=10),
                fixedrange=True,
            ),
            xaxis=dict(
                type="category",
                gridcolor="rgba(61,45,107,0.35)",
                tickfont=dict(color=VIOLET["muted"], size=10),
                fixedrange=True,
            ),
        )
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ── History Records Table ──
    st.markdown(
        f'<div style="font-size:0.86rem; font-weight:700; color:{VIOLET["accent"]}; margin:0.8rem 0 0.35rem;">'
        f'📋 Recent {view} History Records</div>',
        unsafe_allow_html=True,
    )
    t = data[["dominant"]].copy()
    t.index = t.index.strftime("%Y-%m-%d")
    t["Dominant Emotion"] = data["dominant"]
    t["Mood Stability"] = [f"{v:.0f}%" for v in wellness]
    t["Negative Affect"] = data["S"].round(2)
    t_display = t[["Dominant Emotion", "Mood Stability", "Negative Affect"]].tail(6)
    st.table(t_display)
