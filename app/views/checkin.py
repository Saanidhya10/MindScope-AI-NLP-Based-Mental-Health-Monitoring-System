"""Daily wellness check-in (improvement 8): habits and how you felt, and what tends to go together."""
import datetime as dt

import pandas as pd
import streamlit as st

import common
import ui
from src import lifestyle, trends

cfg, db, uid = common.get_cfg(), common.get_db(), common.user_id()
ui.page_header("Daily check-in", "A minute a day: how you slept, moved and felt. Over a few weeks MindScope shows "
                                 "which habits tend to go with better or harder days for you.")

checkins = db.get_checkins(uid)
left, right = st.columns([1, 1.2], gap="large")

with left:
    ui.section("Today's check-in")
    with st.container(key="mj-entry"):
        consent = common.consent_box("checkin")
        day = st.date_input("Day", value=dt.date.today(), max_value=dt.date.today(), format="DD/MM/YYYY")
        saved = checkins.loc[pd.Timestamp(day)] if pd.Timestamp(day) in checkins.index else None
        prev = (lambda k, d: d if saved is None or pd.isna(saved[k]) else saved[k])
        with st.form("checkin", border=False):
            values = {}
            for f in lifestyle.HABITS:
                if f.key == "social":
                    level = st.select_slider(f.label, options=[1, 2, 3], value=int(prev(f.key, f.default)),
                                             format_func=lifestyle.SOCIAL_LEVELS.get)
                    values[f.key] = level
                elif f.step >= 1 and f.hi <= 10:
                    values[f.key] = st.slider(f.label, int(f.lo), int(f.hi), int(prev(f.key, f.default)))
                else:
                    values[f.key] = st.number_input(f.label, float(f.lo), float(f.hi), float(prev(f.key, f.default)), float(f.step),
                                                    format="%.0f" if f.step >= 1 else "%.1f")
            st.markdown("**How you felt**")
            for f in lifestyle.FEELINGS:
                values[f.key] = st.slider(f.label, int(f.lo), int(f.hi), int(prev(f.key, f.default)))
            label = "Update check-in" if saved is not None else "Save check-in"
            if st.form_submit_button(label, type="primary", disabled=not consent):
                db.save_checkin(uid, day, values)
                st.toast(f"Check-in for {day:%d %b} saved")
                st.rerun()
        if saved is not None:
            st.caption(f"You already checked in for {day:%d %b}. Saving again replaces it.")

with right:
    ui.section("What tends to go together", "Patterns in your own check-ins and journal. These are associations, "
                                            "not proof that one thing causes another.")
    hist = common.history()
    daily = trends.daily_frame(hist, cfg) if len(hist) else None
    found = lifestyle.associations(lifestyle.merge_with_journal(checkins, daily)) if len(checkins) else []
    if len(checkins) < lifestyle.MIN_DAYS:
        ui.empty_state(f"Insights appear after {lifestyle.MIN_DAYS} check-ins. You have {len(checkins)} so far.")
    elif not found:
        ui.empty_state("No clear patterns yet. That's normal; they can take a few weeks to show up.")
    else:
        ui.insight_list([a["text"] for a in found[:6]])
        if len(found) > 6:
            st.caption(f"{len(found) - 6} weaker patterns hidden.")

    if len(checkins):
        ui.section("Your last two weeks")
        ui.checkin_chart(checkins.tail(14))

ui.footer(common.get_predictor().label)
