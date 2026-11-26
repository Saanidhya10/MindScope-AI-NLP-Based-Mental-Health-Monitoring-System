"""Emotional profile, the "emotional fingerprint" (improvement 2): what is usual for you, and how lately compares."""
import streamlit as st

import common
import ui
from src import fingerprint as F
from src import trends

cfg, db, uid = common.get_cfg(), common.get_db(), common.user_id()
ui.page_header("Your emotional profile", "What is usual for you, learned from your own journal and check-ins, and "
               "how the last seven days compare. Everyone's normal is different, so MindScope compares you with you.")

hist = common.history()
if hist.empty:
    ui.empty_state("Your profile builds up as you write. Add a few journal entries or check-ins first.")
else:
    p = F.profile(hist, db.get_checkins(uid), db.get_triggers(uid))
    if not p["calibrated"]:
        st.caption(f"Still learning: {p['days']} days of history so far. “Usual” uses everything until there's more "
                   f"than a week before the last {F.RECENT_DAYS} days.")
    ui.section("Usual and recent", f"“Usual” is every day before the last {F.RECENT_DAYS}; “recent” is the last "
                                   f"{F.RECENT_DAYS} days.")
    ui.profile_table(p["dimensions"])

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        ui.section("Your feelings, usual and recent", "Average share of each feeling in your entries.")
        ui.mix_compare(p["mix_usual"], p["mix_recent"])
    with right:
        ui.section("Topics on difficult days")
        if p["topics"]:
            ui.insight_list([f"{label}: {n} difficult entr{'y' if n == 1 else 'ies'}" for label, n in p["topics"]])
        else:
            ui.empty_state("None yet. Topics come from journal entries.")
        baseline = trends.personal_baseline(hist, cfg["trends"]["tau"])
        ui.section("Your personal alert line")
        ui.insight_list([f"MindScope treats a day as unusually heavy for you when difficult feelings pass about "
                         f"{baseline['tau_personal']:.0%}, based on your own history, rather than one fixed "
                         "line for everyone."])

    ui.section("Unusual days", "Days that looked unlike your others, found with an Isolation Forest over your daily "
                               "journal and check-in values. Unusual doesn't mean bad.")
    if p["unusual"]:
        ui.insight_list([f"{u['day']:%a %d %b}: " + "; ".join(u["reasons"]) for u in p["unusual"][:6]])
    else:
        ui.empty_state(f"Unusual days are picked out once there are at least {F.MIN_DAYS_FOR_ANOMALIES} days of "
                       "history, and none stand out yet.")

ui.footer(common.get_predictor().label)
