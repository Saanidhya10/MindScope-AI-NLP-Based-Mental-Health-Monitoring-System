"""Trigger analysis (improvement 4): what your entries are about, and which topics come up on difficult days."""
import streamlit as st

import common
import ui
from src import triggers as T

db, uid = common.get_db(), common.user_id()
ui.page_header("What your entries are about", "MindScope tags each journal entry with the topics it mentions. "
               "Over time this shows which topics tend to come up on harder days. Topics are what you wrote about, "
               "not proof of what caused how you felt.")

entries = db.get_triggers(uid)
if entries.empty:
    ui.empty_state("No topics yet. They're picked out from journal entries written from now on.")
else:
    summ = T.summary(entries)
    ui.section("Recurring topics", f"Across {len(entries)} entries. An entry counts as difficult when at least half of "
                                   "what came through was sadness, anger, fear, anxiety or stress.")
    if summ.empty:
        ui.empty_state("No particular topics have come up yet.")
    else:
        ui.trigger_chart(summ)
        notes = []
        for _, r in summ[summ["entries"] >= 3].iterrows():
            notes.append(f"{r['label']} came up in {r['entries']} entries, and {r['difficult']} of them "
                         f"({r['difficult_share']:.0%}) were difficult, compared with {r['overall_share']:.0%} of all "
                         "your entries.")
        if notes:
            ui.insight_list(notes)
        else:
            st.caption("Sentences appear for topics that come up in at least three entries.")

    ui.section("Recent entries")
    ui.trigger_table(entries.tail(10).iloc[::-1])

ui.footer(common.get_predictor().label)
