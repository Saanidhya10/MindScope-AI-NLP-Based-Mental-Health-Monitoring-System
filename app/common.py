"""Shared state for every page: config, database, models and the current profile."""
from __future__ import annotations

import streamlit as st

from src import recommend
from src.database import MoodDatabase
from src.inference import EmotionPredictor
from src.forecast import Forecaster
from src.knowledge import KnowledgeBase
from src.triggers import TriggerDetector
from src.utils import resolve_path
from src.utils import load_config
from src.voice import VoiceAnalyzer

APP_NAME = "MindScope AI"
APP_TAGLINE = "An Intelligent Personal Emotional Wellness and Monitoring Platform"


@st.cache_resource
def get_cfg() -> dict:
    return load_config()


@st.cache_resource
def get_db() -> MoodDatabase:
    return MoodDatabase(cfg=get_cfg())


@st.cache_resource(show_spinner="Loading the emotion model…")
def get_predictor() -> EmotionPredictor:
    return EmotionPredictor.from_config(get_cfg())


@st.cache_resource(show_spinner="Loading the speech models (first time only, this takes a minute)…")
def get_voice() -> VoiceAnalyzer:
    return VoiceAnalyzer(get_cfg())


@st.cache_resource(show_spinner="Loading the topic model (first time only, this takes a minute)…")
def get_triggers() -> TriggerDetector:
    return TriggerDetector(get_cfg())


@st.cache_resource(show_spinner="Preparing the outlook model (first time only, about 15 seconds)…")
def get_forecaster() -> Forecaster:
    return Forecaster.load_or_build(resolve_path(get_cfg()["forecast"]["model_path"]))


@st.cache_resource(show_spinner="Loading the wellness library…")
def get_knowledge() -> KnowledgeBase:
    return KnowledgeBase(get_cfg())


def user_id() -> str:
    return st.session_state.get("user_id", "me")


def helplines() -> list[dict]:
    return recommend.get_helplines(get_cfg())


def history():
    return get_db().get_predictions(user_id())


def consent_box(key_suffix: str) -> bool:
    """The analysis-consent checkbox, kept in sync with the database (shown on several pages)."""
    db, uid = get_db(), user_id()
    consent = st.checkbox("Analyse my entries and keep the scores on this computer. The text itself is never saved.",
                          value=db.has_consent(uid), key=f"consent_{key_suffix}_{uid}")
    if consent != db.has_consent(uid):
        db.set_consent(uid, consent)
        st.rerun()
    return consent
