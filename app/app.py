"""Mood Journal - Streamlit app.   Run from the repo root:  streamlit run app/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st  # noqa: E402

import ui  # noqa: E402
from src import recommend, trends  # noqa: E402
from src.database import MoodDatabase  # noqa: E402
from src.inference import EmotionPredictor  # noqa: E402
from src.sample_journal import sample_entries  # noqa: E402
from src.utils import load_config  # noqa: E402

st.set_page_config(page_title="Mood Journal", page_icon="📓", layout="wide")


@st.cache_resource(show_spinner="Loading the emotion model…")
def get_predictor() -> EmotionPredictor:
    return EmotionPredictor.from_config(cfg)


cfg = load_config()
db = MoodDatabase(cfg=cfg)
helplines = recommend.get_helplines(cfg)
ui.inject_css()

# ───────────────────────── sidebar: profile, grouping, data controls ─────────────────────────
with st.sidebar:
    user_id = st.text_input("Profile name", value="me",
                            help="Entries are kept separately for each name, on this computer.").strip() or "me"
    view = st.radio("Group the trend by", ["Day", "Week", "Month"], horizontal=True)
    hist = db.get_predictions(user_id)

    st.divider()
    st.markdown(f"**Your data**  \n{len(hist)} entr{'y' if len(hist) == 1 else 'ies'} saved for *{user_id}*.")
    st.download_button("Download my scores (CSV)",
                       hist.drop(columns=["user_id"], errors="ignore").to_csv(index=False).encode(),
                       file_name=f"{user_id}_mood_scores.csv", mime="text/csv", disabled=hist.empty)
    with st.expander("Delete all my entries"):
        sure = st.checkbox("I understand this can't be undone", key=f"del_{user_id}")
        if st.button("Delete entries", disabled=not sure):
            n = db.delete_user_data(user_id, keep_user=True)
            st.toast(f"Deleted {n} entries")
            st.rerun()

# ───────────────────────── header ─────────────────────────
ui.header(helplines)

# ───────────────────────── today's entry + result ─────────────────────────
left, right = st.columns([1.1, 1], gap="large")
with left:
    ui.section("Today's entry")
    with st.container(key="mj-entry"):
        consent = st.checkbox("Analyse my entries and keep the scores on this computer. The text itself is never saved.",
                              value=db.has_consent(user_id), key=f"consent_{user_id}")
        if consent != db.has_consent(user_id):
            db.set_consent(user_id, consent)
        with st.form("entry", clear_on_submit=True, border=False):
            text = st.text_area("What happened today, and how did it feel?", height=170,
                                max_chars=cfg["app"]["max_input_chars"],
                                placeholder="Work felt overwhelming and I'm nervous about tomorrow's presentation, "
                                            "but finishing the slides was a relief.")
            submitted = st.form_submit_button("Analyse entry", type="primary", disabled=not consent)
        if not consent:
            st.caption("Tick the box above to start. You can untick it any time to pause.")
        if hist.empty and consent:
            if st.button("Add sample entries",
                         help="Runs an invented three-week journal through the model so you can see how trends look."):
                entries = sample_entries()
                for (ts, _), p in zip(entries, get_predictor().predict_proba([t for _, t in entries])):
                    db.add_prediction(user_id, p, ts)
                st.toast(f"Added {len(entries)} sample entries")
                st.rerun()

    if submitted:
        if len(text.split()) < 3:
            st.warning("Write at least a few words. Very short entries are hard to read reliably.")
        else:
            probs = get_predictor().predict_proba(text)[0]
            db.add_prediction(user_id, probs)
            st.session_state["last"] = (probs, recommend.check_crisis(text, cfg))
            st.toast("Entry analysed and saved")
            st.rerun()

with right:
    ui.section("What came through")
    if "last" in st.session_state:
        probs, crisis = st.session_state["last"]
        ui.entry_result(probs, crisis, helplines, get_predictor().label)
    else:
        ui.entry_placeholder()

# ───────────────────────── history ─────────────────────────
ui.section("The last four weeks", "Each block is one day, coloured by the feeling that came through most.")
ui.mood_ribbon(hist)

if not hist.empty:
    daily = trends.daily_frame(hist, cfg)
    baseline = trends.personal_baseline(hist, cfg["trends"]["tau"])
    status = recommend.assess(daily, baseline, cfg)

    ui.section("How you've been")
    ui.status_card(status)

    mix, trend = st.columns([1, 1.6], gap="large")
    with mix:
        ui.section("Emotion mix", "Main feeling of each entry in the last 30 days.")
        ui.emotion_mix(hist)
    with trend:
        ui.section("Difficult feelings over time",
                   f"Share of sadness, anger, fear, anxiety and stress in your entries, by {view.lower()}. "
                   f"Support is suggested after {cfg['trends']['k']} days above the support line.")
        ui.trend_chart(hist, daily, view, baseline, cfg["trends"]["tau"])

ui.footer(get_predictor().label)
