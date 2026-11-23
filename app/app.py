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
from src.voice import VoiceAnalyzer, fuse, load_audio  # noqa: E402

st.set_page_config(page_title="Mood Journal", page_icon="📓", layout="wide")


@st.cache_resource(show_spinner="Loading the emotion model…")
def get_predictor() -> EmotionPredictor:
    return EmotionPredictor.from_config(cfg)


@st.cache_resource(show_spinner="Loading the speech models (first time only, this takes a minute)…")
def get_voice() -> VoiceAnalyzer:
    return VoiceAnalyzer(cfg)


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
def save_entry(text: str, voice_info: dict | None = None, tone=None) -> None:
    """Score the text (and tone, if given), store only the scores, remember the result for the right column."""
    probs = get_predictor().predict_proba(text)[0]
    used = False
    if tone is not None:
        probs, used = fuse(probs, tone, cfg["voice"]["tone_weight"])
    db.add_prediction(user_id, probs)
    if voice_info is not None:
        voice_info.update(tone=tone, tone_used=used)
    st.session_state["last"] = {"probs": probs, "crisis": recommend.check_crisis(text, cfg), "voice": voice_info}
    st.toast("Entry analysed and saved")
    st.rerun()


left, right = st.columns([1.1, 1], gap="large")
with left:
    ui.section("Today's entry")
    with st.container(key="mj-entry"):
        consent = st.checkbox("Analyse my entries and keep the scores on this computer. The text itself is never saved.",
                              value=db.has_consent(user_id), key=f"consent_{user_id}")
        if consent != db.has_consent(user_id):
            db.set_consent(user_id, consent)
        mode = st.radio("How do you want to add it?", ["Write", "Record"], horizontal=True, key="mode",
                        label_visibility="collapsed")

        if mode == "Write":
            with st.form("entry", clear_on_submit=True, border=False):
                text = st.text_area("What happened today, and how did it feel?", height=170,
                                    max_chars=cfg["app"]["max_input_chars"],
                                    placeholder="Work felt overwhelming and I'm nervous about tomorrow's presentation, "
                                                "but finishing the slides was a relief.")
                submitted = st.form_submit_button("Analyse entry", type="primary", disabled=not consent)
            if submitted:
                if len(text.split()) < 3:
                    st.warning("Write at least a few words. Very short entries are hard to read reliably.")
                else:
                    save_entry(text)
        else:
            audio = st.audio_input("Talk about your day, in English. A minute or two is plenty.", key="recording",
                                   disabled=not consent)
            use_tone = st.checkbox("Also read my tone of voice. The recording is processed on this computer and "
                                   "never saved.", value=True, key="use_tone", disabled=not consent)
            if st.button("Analyse recording", type="primary", disabled=not (consent and audio)):
                try:
                    wav, seconds = load_audio(audio.getvalue())
                except Exception:
                    st.error("Couldn't read that recording. Try recording again.")
                else:
                    if seconds > cfg["voice"]["max_seconds"]:
                        st.warning(f"That recording is {seconds / 60:.0f} minutes long. "
                                   f"Keep it under {cfg['voice']['max_seconds'] // 60} minutes.")
                    else:
                        with st.spinner("Listening to your recording…"):
                            heard = get_voice().transcribe(wav)
                            tone = get_voice().tone_proba(wav) if use_tone else None
                        if len(heard.split()) < cfg["voice"]["min_words"]:
                            st.warning("Couldn't make out enough words. Speak a little longer or closer to the microphone.")
                        else:
                            save_entry(heard, {"transcript": heard, "seconds": seconds}, tone)

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

with right:
    ui.section("What came through")
    if "last" in st.session_state:
        last = st.session_state["last"]
        ui.entry_result(last["probs"], last["crisis"], helplines, get_predictor().label, last["voice"])
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
