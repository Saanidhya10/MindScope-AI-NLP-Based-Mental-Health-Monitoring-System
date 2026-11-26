"""Home: how you've been, the last four weeks, emotion mix and trend."""
import streamlit as st

import common
import ui
from src import recommend, trends

cfg = common.get_cfg()
hist = common.history()
ui.brand_header(common.helplines())

if hist.empty:
    ui.empty_state("Nothing here yet. Write or record your first entry, or add sample entries from Settings & privacy "
                   "to see how the dashboard looks.")
    st.page_link("views/journal.py", label="Write today's entry", icon=":material/edit_note:")
else:
    daily = trends.daily_frame(hist, cfg)
    baseline = trends.personal_baseline(hist, cfg["trends"]["tau"])
    ui.section("How you've been")
    ui.status_card(recommend.assess(daily, baseline, cfg))

    ui.section("The last four weeks", "Each block is one day, coloured by the feeling that came through most.")
    ui.mood_ribbon(hist)

    mix, trend = st.columns([1, 1.6], gap="large")
    with mix:
        ui.section("Emotion mix", "Main feeling of each entry in the last 30 days.")
        ui.emotion_mix(hist)
    with trend:
        view = st.radio("Group by", ["Day", "Week", "Month"], horizontal=True, key="trend_view")
        ui.section("Difficult feelings over time",
                   f"Share of sadness, anger, fear, anxiety and stress in your entries, by {view.lower()}. "
                   f"Support is suggested after {cfg['trends']['k']} days above the support line.")
        ui.trend_chart(hist, daily, view, baseline, cfg["trends"]["tau"])

ui.footer(common.get_predictor().label)
