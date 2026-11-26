"""Wellness activities (improvement 9): guided exercises, feedback, and suggestions that adapt to what helps you."""
import datetime as dt
import hashlib

import numpy as np
import streamlit as st

import common
import ui
from src import activities as A
from src.trends import PROB_COLS

db, uid = common.get_db(), common.user_id()
ui.page_header("Wellness activities", "Short, practical exercises. Tell MindScope whether each one helped, and its "
                                      "suggestions will lean towards what works for you.")
consent = common.consent_box("activities")

# what the user has been feeling lately: strongest emotions over the last 7 days
hist = common.history()
recent = ["Neutral"]
if len(hist):
    last = hist[hist["timestamp"] >= hist["timestamp"].max() - np.timedelta64(7, "D")]
    means = last[PROB_COLS].mean().sort_values(ascending=False)
    recent = [c[2:] for c in means.index]
log = db.get_activity_log(uid)
seed = int(hashlib.sha256(f"{uid}-{dt.date.today()}-{len(log)}".encode()).hexdigest(), 16) % 2**32
ranked = A.rank(recent, log, np.random.default_rng(seed))

ui.section("Suggested for you now")
cols = st.columns(3, gap="medium")
for col, r in zip(cols, ranked[:3]):
    with col, st.container(key=f"mj-act-{r['activity'].key}"):
        ui.activity_card(r["activity"], r["reason"])
        if st.button("Start", key=f"start_{r['activity'].key}"):
            st.session_state["activity"] = r["activity"].key

keys = [a.key for a in A.CATALOG]
current = st.session_state.get("activity", ranked[0]["activity"].key)
choice = st.selectbox("Or choose any activity", keys, index=keys.index(current),
                      format_func=lambda k: f"{A.BY_KEY[k].title} ({A.BY_KEY[k].minutes} min)")
st.session_state["activity"] = choice
act = A.BY_KEY[choice]

ui.section(act.title, act.summary)
with st.container(key="mj-entry"):
    if choice == "breathing":
        pattern = st.radio("Pattern", ["Box breathing (4-4-4-4)", "Relaxing breath (4-7-8)"], horizontal=True)
        phases = ([("Breathe in", 4), ("Hold", 4), ("Breathe out", 4), ("Hold", 4)] if pattern.startswith("Box")
                  else [("Breathe in", 4), ("Hold", 7), ("Breathe out", 8)])
        ui.breathing_guide(phases)
        st.caption("Follow the circle for four or five rounds. Stop if you feel light-headed.")
    elif choice == "grounding":
        for n, sense in [(5, "things you can see"), (4, "things you can touch"), (3, "things you can hear"),
                         (2, "things you can smell"), (1, "thing you can taste")]:
            st.text_input(f"Name {n} {sense}", key=f"g_{n}", placeholder="Say them out loud or type them here")
        st.caption("Nothing you type here is saved.")
    elif choice in ("gratitude", "thought"):
        prompts = (["Something that went well today", "Someone you're glad was around", "A small thing you enjoyed"]
                   if choice == "gratitude" else
                   ["What happened?", "What went through your mind?", "What supports that thought?",
                    "What doesn't fit with it?", "A kinder or more balanced way to see it"])
        answers = [st.text_area(pr, key=f"{choice}_{i}", height=70) for i, pr in enumerate(prompts)]
        keep = st.checkbox("Save this to my journal on this computer", value=False, key=f"keep_{choice}",
                           help="Off by default. Saved notes can be deleted from Settings & privacy.")
        if keep and st.button("Save note", disabled=not (consent and any(a.strip() for a in answers))):
            db.add_note(uid, choice, "\n".join(f"{p}: {a.strip()}" for p, a in zip(prompts, answers) if a.strip()))
            st.toast("Note saved")
        notes = db.get_notes(uid, choice)
        if len(notes):
            with st.expander(f"Your saved notes ({len(notes)})"):
                for _, n in notes.head(10).iterrows():
                    st.markdown(f"**{n['timestamp'][:10]}**  \n" + n["body"].replace("\n", "  \n"))
    elif choice == "study":
        task = st.text_input("What's the big task?", placeholder="Revise chapter 4 for Friday's exam")
        steps = st.text_area("Break it into steps you could each finish in under 30 minutes (one per line)",
                             placeholder="Re-read the summary\nMake flashcards for key terms\nDo 5 practice questions")
        lines = [s.strip() for s in steps.splitlines() if s.strip()]
        if lines:
            st.markdown("**Start with:** " + lines[0])
        if st.button("Add these steps to my goals", disabled=not (consent and lines)):
            for line in lines:
                db.add_goal(uid, f"{task.strip()}: {line}" if task.strip() else line)
            st.toast(f"Added {len(lines)} goals")
    elif choice == "sleep":
        habits = ["Wake up at about the same time every day", "Wind down for 30 to 60 minutes before bed",
                  "No caffeine after mid-afternoon", "Put screens away before bed",
                  "Keep the bedroom dark, quiet and cool", "Avoid heavy meals and alcohol late in the evening",
                  "Get some daylight in the morning"]
        done = sum(st.checkbox(h, key=f"sleep_{i}") for i, h in enumerate(habits))
        st.markdown(f"**{done} of {len(habits)}** tonight. Pick one more to try tomorrow rather than all at once.")
    elif choice == "mindfulness":
        ui.timer_bar(180)
        for step in ["Sit or lie comfortably and let your eyes close or soften.",
                     "Notice your feet: warmth, pressure, tingling. Just notice.",
                     "Move slowly up through your legs, stomach, chest and shoulders.",
                     "Notice your arms, hands, neck and face. Let your jaw loosen.",
                     "If your mind wanders, that's normal. Gently come back to the body.",
                     "Finish with three slow breaths and open your eyes."]:
            st.markdown(f"- {step}")
    elif choice == "movement":
        for idea in ["A 10-minute walk outside, without your phone if you can",
                     "Stretch your neck, shoulders and back for five minutes",
                     "Put on two songs you like and move to them", "Walk up and down the stairs a few times"]:
            st.markdown(f"- {idea}")
    elif choice == "connect":
        st.markdown("You don't need a reason or a long message. A few ideas:")
        for idea in ["“Hey, I was thinking of you. How's your week going?”",
                     "“Do you have ten minutes to talk later? It's been a long day.”",
                     "“Want to grab chai/coffee this week?”"]:
            st.markdown(f"- {idea}")
    elif choice == "goals":
        new = st.text_input("Add a small, specific goal", placeholder="Go to bed by 11:30 on weeknights")
        if st.button("Add goal", disabled=not (consent and new.strip())):
            db.add_goal(uid, new)
            st.rerun()
        goals = db.get_goals(uid)
        for _, g in goals.iterrows():
            c1, c2 = st.columns([6, 1])
            done = c1.checkbox(g["title"], value=bool(g["done"]), key=f"goal_{g['id']}")
            if done != bool(g["done"]):
                db.set_goal_done(uid, int(g["id"]), done)
                st.rerun()
            if c2.button("Remove", key=f"del_goal_{g['id']}"):
                db.delete_goal(uid, int(g["id"]))
                st.rerun()
        if goals.empty:
            ui.empty_state("No goals yet. Small and specific works best.")

ui.section("How did it go?", "Optional, but it's how MindScope learns which activities help you.")
with st.form("feedback", border=False):
    c1, c2 = st.columns(2)
    before = c1.slider("Mood before (1 to 10)", 1, 10, 5)
    after = c2.slider("Mood after (1 to 10)", 1, 10, 5)
    helped = st.radio("Did it help?", ["Yes", "No", "Not sure"], horizontal=True, index=2)
    if st.form_submit_button("Save feedback", type="primary", disabled=not consent):
        db.log_activity(uid, choice, before, after, {"Yes": True, "No": False}.get(helped))
        st.toast(f"Thanks. Logged {act.title.lower()}.")
        st.rerun()

stats = A.feedback_stats(log)
if len(stats):
    ui.section("What has helped you so far")
    ui.activity_stats(stats)

ui.footer(common.get_predictor().label)
