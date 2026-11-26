"""Journal: write or record an entry and see what came through."""
import streamlit as st

import common
import ui
from src import recommend
from src.voice import fuse, load_audio

cfg, db, uid = common.get_cfg(), common.get_db(), common.user_id()
ui.page_header("Journal", "Write or say a few lines about your day. The app picks out the feelings in it.")
ui.help_line(common.helplines())


def save_entry(text: str, voice_info: dict | None = None, tone=None) -> None:
    """Score the text (and tone, if given), store only the scores, remember the result for the right column."""
    probs = common.get_predictor().predict_proba(text)[0]
    used = False
    if tone is not None:
        probs, used = fuse(probs, tone, cfg["voice"]["tone_weight"])
    found = [k for k, _ in common.get_triggers().detect([text])[0]]
    db.add_prediction(uid, probs, triggers=found)
    if voice_info is not None:
        voice_info.update(tone=tone, tone_used=used)
    st.session_state["last"] = {"probs": probs, "crisis": recommend.check_crisis(text, cfg), "voice": voice_info,
                                "triggers": found}
    st.toast("Entry analysed and saved")
    st.rerun()


left, right = st.columns([1.1, 1], gap="large")
with left:
    ui.section("Today's entry")
    with st.container(key="mj-entry"):
        consent = common.consent_box("journal")
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
                            heard = common.get_voice().transcribe(wav)
                            tone = common.get_voice().tone_proba(wav) if use_tone else None
                        if len(heard.split()) < cfg["voice"]["min_words"]:
                            st.warning("Couldn't make out enough words. Speak a little longer or closer to the "
                                       "microphone.")
                        else:
                            save_entry(heard, {"transcript": heard, "seconds": seconds}, tone)
        if not consent:
            st.caption("Tick the box above to start. You can untick it any time to pause.")

with right:
    ui.section("What came through")
    if "last" in st.session_state:
        last = st.session_state["last"]
        ui.entry_result(last["probs"], last["crisis"], common.helplines(), common.get_predictor().label, last["voice"],
                        last.get("triggers"))
    else:
        ui.entry_placeholder()

ui.footer(common.get_predictor().label)
