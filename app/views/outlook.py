"""Emotional outlook (improvement 1, experimental): where the next few days might be heading, with uncertainty."""
import pandas as pd
import streamlit as st

import common
import ui
from src import forecast as Fc
from src.fingerprint import daily_features

cfg, db, uid = common.get_cfg(), common.get_db(), common.user_id()
ui.page_header("Outlook", "A rough guide to where the next few days might be heading, based on your recent entries "
               "and check-ins. It is experimental and often wrong, so treat it as a nudge to check in, not a "
               "prediction about you.")

hist = common.history()
daily = daily_features(hist, db.get_checkins(uid)) if len(hist) else pd.DataFrame()
today = pd.Timestamp.today().normalize()

if daily.empty or "difficult" not in daily or not Fc.has_enough_data(daily, today, cfg["forecast"]["min_days"]):
    n = 0 if daily.empty else int(daily[daily.index > today - pd.Timedelta(days=14)]["difficult"].notna().sum())
    ui.empty_state(f"An outlook appears once you have journal entries on at least {cfg['forecast']['min_days']} of "
                   f"the last 14 days. You have {n}.")
else:
    fc = common.get_forecaster()
    outlooks = fc.outlook(daily, today)
    if db.has_consent(uid):
        db.save_forecasts(uid, today, outlooks)
    obs = daily["difficult"].dropna()
    current = float(obs.iloc[-3:].mean())
    week_change = float(Fc.features_at(daily.reindex(pd.date_range(daily.index.min(), today)), today)["slope7"] * 7)

    ui.section("The next few days", f"Right now difficult feelings are around {current:.0%} of your entries "
                                    "(average of your last three days with entries).")
    cards = []
    for h in (3, 7):
        o = outlooks[h - 1]
        direction, confidence = Fc.describe(current, o)
        cards.append((f"{h}-day outlook", f"Difficult feelings {direction}",
                      f"Most likely between {o.low:.0%} and {o.high:.0%} by {o.day:%a %d %b}. Confidence: {confidence}."))
    ui.outlook_cards(cards)
    if abs(week_change) >= 0.2:
        ui.insight_list([f"Over the past week difficult feelings have {'risen' if week_change > 0 else 'fallen'} by "
                         f"about {abs(week_change):.0%}, which is a noticeable change for you."])
    heavier = outlooks[2].predicted > current + 0.08
    staying_high = outlooks[2].predicted >= cfg["trends"]["tau"]
    if heavier or staying_high:
        st.markdown("Things may get heavier. " if heavier else
                    "Even if it eases a little, difficult feelings may stay fairly high for a few days. ")
        st.markdown("A quick check-in or a short activity now can help you stay ahead of it.")
        c1, c2 = st.columns(2)
        c1.page_link("views/checkin.py", label="Do a daily check-in", icon=":material/fact_check:")
        c2.page_link("views/activities.py", label="Try a wellness activity", icon=":material/self_improvement:")

    ui.section("Recent days and the outlook", "Shaded area: the range the outlook expects 8 times out of 10.")
    ui.outlook_chart(obs[obs.index > today - pd.Timedelta(days=21)], outlooks, cfg["trends"]["tau"])

    past = db.get_forecasts(uid)
    past = past[past["target_day"] <= today].merge(obs.rename("actual"), left_on="target_day", right_index=True)
    ui.section("Earlier outlooks and what happened")
    if past.empty:
        ui.empty_state("Outlooks are saved each day you visit this page. Once those days arrive and you've written "
                       "entries, they're compared with what actually happened here.")
    else:
        inside = ((past["actual"] >= past["low"]) & (past["actual"] <= past["high"])).mean()
        mae = (past["actual"] - past["predicted"]).abs().mean()
        ui.insight_list([f"{len(past)} past outlooks checked. What happened fell inside the expected range "
                         f"{inside:.0%} of the time, and the outlook was off by {mae:.0%} on average."])

    with st.expander("How the outlook works"):
        st.markdown(
            "MindScope looks at your recent daily difficult-feelings share (and sleep and stress from check-ins, if "
            "you log them) and compares four methods: repeating the last value, a smoothed average, a fading trend, "
            "and a gradient-boosting model. Because no real long-term data is available, the model learned from "
            "**simulated** timelines, and the methods were compared on held-out simulated timelines too. The table "
            f"shows their average error there, by days ahead. The app uses **{fc.method}**, the best of the four. "
            "That says how the methods compare under the simulation's assumptions, not how accurate the outlook is "
            "for you, which is what the section above tracks.")
        table = fc.scores.pivot(index="method", columns="horizon", values="mae")
        table.columns = [f"{h} day{'s' if h > 1 else ''}" for h in table.columns]
        st.dataframe(table.round(3), use_container_width=True)

ui.footer(common.get_predictor().label)
