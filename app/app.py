"""Mood Journal – Streamlit app.   Run from the repo root:  streamlit run app/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

import components as C
import ui

st.set_page_config(
    page_title="Mood Journal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading model…")
def get_predictor():
    return C.Predictor()


def main():
    ui.render_header()
    ui.render_disclaimer()

    params = C.load_params()

    # ───────────────────────── Sidebar (Comfortably Spaced) ─────────────
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.6rem; padding:0.4rem 0 0.8rem;">
                <span style="font-size:1.6rem; filter:drop-shadow(0 0 10px rgba(166,125,240,0.6));">🧠</span>
                <div>
                    <strong style="color:#d4c4ff; font-size:1.05rem; line-height:1.2;">Mood Journal</strong><br>
                    <span style="font-size:0.75rem; color:#b0a8d0;">Private Wellness Space</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<span style="font-size:0.78rem; font-weight:700; color:#d4c4ff; text-transform:uppercase; letter-spacing:0.05em;">👤 Profile</span>', unsafe_allow_html=True)
        user_id = (
            st.text_input(
                "Profile",
                value="me",
                help="Entries are saved per profile on this machine.",
                placeholder="Enter profile name",
                label_visibility="collapsed",
            ).strip() or "me"
        )

        st.markdown('<span style="font-size:0.78rem; font-weight:700; color:#d4c4ff; text-transform:uppercase; letter-spacing:0.05em;">📊 Timeline View</span>', unsafe_allow_html=True)
        view = st.radio(
            "Timeline View",
            ["Daily", "Weekly", "Monthly"],
            horizontal=True,
            label_visibility="collapsed",
        )

        hist = C.load_history(user_id)
        st.markdown(
            f'<div class="vi-profile-badge">'
            f'📝 <strong>{len(hist)}</strong> entries recorded for <em>{user_id}</em></div>',
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇️ Export Data (CSV)",
            C.export_history_csv(user_id),
            file_name=f"{user_id}_mood_scores.csv",
            mime="text/csv",
        )

        st.markdown("<div style='height:0.2rem;'></div>", unsafe_allow_html=True)

        with st.expander("🗑️ Clear My History"):
            sure = st.checkbox("Confirm permanent deletion", key="confirm_del")
            if st.button("Delete My Entries", disabled=not sure):
                deleted_count = C.delete_history(user_id)
                st.toast(f"Deleted {deleted_count} records")
                st.rerun()

        st.markdown(
            """
            <div style="font-size:0.72rem; color:#6a6090; text-align:center; margin-top:0.6rem;">
                Mood Journal · Wellness Support · Not Medical
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ───────────────────────── Consent ─────────────────────────────────────
    consent = ui.render_consent()

    # ───────────────────────── Main Two-Column Layout ──────────────────────
    left, right = st.columns([1, 1], gap="large")

    # ── Left Column: Journal Entry ──────────────────────────────────────────
    with left:
        ui.render_section_title("✍️", "How are you feeling?", "Daily Emotional Check-in")
        st.markdown('<div class="vi-card">', unsafe_allow_html=True)
        st.caption("Write a few honest sentences. The model identifies fine-grained emotions in real time.")

        with st.form("entry", clear_on_submit=True):
            text = st.text_area(
                "Your entry",
                height=140,
                placeholder="e.g. Work felt overwhelming today and I'm anxious about the presentation tomorrow, but relieved to finish the slides…",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button(
                "✨ Analyse my mood",
                disabled=not consent,
            )

        if not consent:
            st.caption("☝️ Tick the consent box above to enable analysis.")

        if submitted:
            if len(text.split()) < 3:
                st.warning("Please write at least a few words — very short text is hard to classify reliably.")
            else:
                try:
                    pred = get_predictor()
                except FileNotFoundError as e:
                    st.error(str(e))
                else:
                    probs = pred.predict_proba(text)
                    C.add_prediction(user_id, probs)
                    st.session_state["last"] = (probs, pred.label, text)
                    # Refresh to ensure right column and trend graphs update immediately
                    st.rerun()

        if "last" in st.session_state:
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            ui.render_entry_result(*st.session_state["last"])

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Right Column: Suggestions & Real-Time Insights ──────────────────────
    hist = C.load_history(user_id)
    daily = C.daily_summary(hist, params["alpha"]) if len(hist) else None
    baseline = C.compute_personalized_baseline(hist, params["tau"])
    anomaly = (
        C.detect_anomalies_and_trend(hist, daily, baseline, baseline["tau_personal"])
        if daily is not None and not daily.empty
        else {"status": "Awaiting", "badge": "🟢 Stable", "color": "#7c5cbf", "desc": "Record entries to begin personalized trend detection."}
    )

    with right:
        ui.render_section_title("💡", "Suggestions & Real-Time Insights", "Personalized Support & Patterns")
        st.markdown('<div class="vi-card">', unsafe_allow_html=True)

        if daily is None or daily.empty:
            st.info(
                "Add your first journal entry on the left to see instant emotional insights, suggestions, "
                "and personalized baseline patterns here.",
                icon="🌱",
            )
        else:
            flags = C.escalation_flags(daily, params["tau"], params["k"], params["max_gap_days"])
            ui.render_summary_metrics(daily, len(hist))
            st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
            ui.render_recommendation(C.recommend(daily, flags, params["tau"], anomaly))

            # Narrative explanation & personalized baseline monitoring (Zero graphs)
            ui.render_recent_mood_analysis(hist, daily, baseline, anomaly)

        st.markdown("</div>", unsafe_allow_html=True)

    # ───────────────────────── Bottom Section: Long-Term Trends ────────────
    ui.render_section_title("📈", f"Mood Stability & Trajectory ({view} View)", "Unified Longitudinal History")
    st.markdown('<div class="vi-card">', unsafe_allow_html=True)

    if daily is None or daily.empty:
        st.info(
            "No historical mood entries recorded yet for this profile. Submit a check-in above to track your stability trajectory.",
            icon="📊",
        )
    else:
        flags = C.escalation_flags(daily, params["tau"], params["k"], params["max_gap_days"])
        ui.render_trends(
            daily,
            C.window_summary(hist, "7D"),
            C.window_summary(hist, "MS"),
            params["tau"],
            flags,
            view,
            baseline,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ───────────────────────── Footer ──────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding:1.2rem 0 0.5rem;
                    color:#6a6090; font-size:0.8rem; opacity:0.85;">
            🧠 Mood Journal &nbsp;·&nbsp; A wellness-support tool, not a medical device &nbsp;·&nbsp;
            Trends reflect self-reported words, not your clinical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


main()
