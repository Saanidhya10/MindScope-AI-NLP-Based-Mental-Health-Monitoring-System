"""Knowledge centre (improvement 3): search the curated, sourced wellness library and find professional support."""
import streamlit as st

import common
import ui
from src.triggers import CATEGORIES

kb = common.get_knowledge()
ui.page_header("Knowledge centre", f"{len(kb.entries)} short, practical guides written for MindScope from public "
               "sources such as the NHS, WHO and NIMH. Each links to the page it's based on. General wellness "
               "information, not medical advice.")

ui.section("Talk to someone", "Free and confidential. If you or someone else is in immediate danger, call 112.")
lines = [f"{h['name']}: {h['number']}" + (f" ({h['note']})" if h.get("note") else "") for h in common.helplines()]
ui.insight_list(lines)

ui.section("Find guidance")
query = st.text_input("Describe what's going on, in your own words",
                      placeholder="I can't switch off at night before exams", key="kb_query")
c1, c2 = st.columns(2)
feelings = sorted({e for x in kb.entries for e in x.get("emotions", [])})
feeling = c1.selectbox("Feeling", ["Any"] + feelings)
topic_labels = {c.label: c.key for c in CATEGORIES}
topic = c2.selectbox("Topic", ["Any"] + list(topic_labels))

if query.strip():
    results = [r for r in kb.search(query, k=len(kb.entries))]
else:
    results = [{"entry": e} for e in kb.entries]
results = [r for r in results
           if (feeling == "Any" or feeling in r["entry"].get("emotions", []))
           and (topic == "Any" or topic_labels[topic] in r["entry"].get("topics", []))]
shown = results[:8] if query.strip() else results
if not shown:
    ui.empty_state("Nothing matches those filters. Try “Any”.")
else:
    st.caption(f"Showing {len(shown)} of {len(kb.entries)}" + (", closest match first." if query.strip() else "."))
    ui.rec_list(shown)

ui.footer(common.get_predictor().label)
