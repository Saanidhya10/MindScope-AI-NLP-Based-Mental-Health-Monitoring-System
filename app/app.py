"""MindScope AI - Streamlit app.   Run from the repo root:  streamlit run app/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st  # noqa: E402

from common import APP_NAME, APP_TAGLINE  # noqa: E402
import ui  # noqa: E402

st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="wide")
ui.inject_css()

with st.sidebar:
    st.session_state["user_id"] = st.text_input(
        "Profile name", value=st.session_state.get("user_id", "me"),
        help="Everything is kept separately for each name, on this computer.").strip() or "me"
    st.caption(f"{APP_NAME}: {APP_TAGLINE.lower()}. Not a medical tool.")

VIEWS = Path(__file__).parent / "views"
pages = {
    "": [st.Page(VIEWS / "home.py", title="Home", icon=":material/home:", default=True),
         st.Page(VIEWS / "journal.py", title="Journal", icon=":material/edit_note:")],
    "Track": [st.Page(VIEWS / "checkin.py", title="Daily check-in", icon=":material/fact_check:")],
    "Account": [st.Page(VIEWS / "settings.py", title="Settings & privacy", icon=":material/shield_person:")],
}
st.navigation(pages).run()
