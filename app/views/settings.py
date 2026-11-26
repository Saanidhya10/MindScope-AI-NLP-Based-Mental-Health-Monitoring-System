"""Settings & privacy: consent, sample data, export and delete."""
import streamlit as st

import common
import ui
import pandas as pd

from src import lifestyle
from src.sample_journal import sample_entries

db, uid = common.get_db(), common.user_id()
hist = common.history()
ui.page_header("Settings & privacy", f"Everything here applies to the profile “{uid}” on this computer.")

ui.section("Consent")
consent = common.consent_box("settings")
st.caption("Untick to pause analysis. Your saved scores stay until you delete them below.")

ui.section("Your data", f"{len(hist)} entr{'y' if len(hist) == 1 else 'ies'} saved for this profile.")
st.download_button("Download my scores (CSV)", hist.drop(columns=["user_id"], errors="ignore").to_csv(index=False).encode(),
                   file_name=f"{uid}_mood_scores.csv", mime="text/csv", disabled=hist.empty)
with st.expander("Delete all my data"):
    sure = st.checkbox("I understand this can't be undone", key=f"del_{uid}")
    if st.button("Delete everything for this profile", disabled=not sure):
        n = db.delete_user_data(uid, keep_user=True)
        st.toast(f"Deleted {n} entries")
        st.rerun()

ui.section("Sample data", "Runs an invented three-week journal through the real model and adds invented daily "
                          "check-ins, so you can see how MindScope looks with some history. Only available on an empty profile.")
if st.button("Add sample entries", disabled=not (consent and hist.empty)):
    entries = sample_entries()
    for (ts, _), p in zip(entries, common.get_predictor().predict_proba([t for _, t in entries])):
        db.add_prediction(uid, p, ts)
    days = sorted({ts.date() for ts, _ in entries})
    span = pd.date_range(days[0], days[-1])  # one invented check-in per day across the same three weeks
    for day, row in lifestyle.sample_checkins(list(span)).iterrows():
        db.save_checkin(uid, day, row.to_dict())
    st.toast(f"Added {len(entries)} sample journal entries and {len(span)} check-ins")
    st.rerun()
