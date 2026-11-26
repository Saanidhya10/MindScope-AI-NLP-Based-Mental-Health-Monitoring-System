"""Multilingual and context-aware reading of journal text (improvement 10). Voice stays English-only.

Language
  * Hindi in Devanagari is translated to English locally (MarianMT, Helsinki-NLP/opus-mt-hi-en), then read by the
    English emotion model.
  * Romanised Hinglish goes through the phrase and word lists in src.preprocess (applied inside the predictor).
  * There is no labelled Hindi/Hinglish emotion dataset to train MuRIL/XLM-R on, so this translate-and-normalise
    approach is what's realistic; scripts/eval_multilingual.py compares it with English-only reading.

Context (applies to every language, after translation)
  * Past vs now: "I used to be anxious, but now I feel fine" -> the part after "now" counts most.
  * Someone else's feelings: "My friend is really sad" with no feeling of the writer's own -> counts for less.
Settings: config.yaml -> multilingual.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from src.utils import LABELS, load_config

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
HINGLISH_MARKERS = {"hai", "hain", "nahi", "nahin", "bahut", "yaar", "mujhe", "mera", "meri", "kya", "aaj", "raha",
                    "rahi", "hoon", "hu", "kuch", "kyun", "abhi", "bhi", "aur", "lekin", "matlab", "wajah", "ki", "ko",
                    "se", "ho", "gaya", "gayi", "tha", "thi", "kal", "sab", "kaise", "accha", "achha"}
EMOTION_WORDS = r"(?:sad|upset|angry|mad|furious|anxious|worried|nervous|scared|afraid|stressed|depressed|down|" \
                r"happy|excited|lonely|hurt|annoyed|frustrated|heartbroken|tense)"
NOW_SPLIT = re.compile(r"\b(?:but|and)?\s*(?:now|these days|nowadays|at the moment)\b[, ]*", re.I)
PAST_CUE = re.compile(r"\b(?:used to|was|were|had been|last (?:week|month|year)|earlier|before)\b", re.I)
THIRD_PERSON_FEELING = re.compile(rf"\b(?:my \w+|he|she|they|his \w+|her \w+|their \w+)\s+(?:is|was|are|were|seems|"
                                  rf"looked|looks|feels|felt|got|gets)\s+(?:\w+\s+)?{EMOTION_WORDS}\b", re.I)
FIRST_PERSON_FEELING = re.compile(rf"\b(?:i|i'm|im|i've|me|my heart)\b[^.!?]{{0,40}}\b(?:{EMOTION_WORDS[3:-1]}|feel|"
                                  rf"felt|feeling|worry|cry|cried|hate|love)\b", re.I)


def detect_language(text: str) -> str:
    """'hi' (Devanagari), 'hinglish' (romanised Hindi mixed with English) or 'en'."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(bool(DEVANAGARI.match(c)) for c in letters) / len(letters) > 0.3:
        return "hi"
    words = re.findall(r"[a-z']+", text.lower())
    return "hinglish" if words and sum(w in HINGLISH_MARKERS for w in words) >= 2 else "en"


class Translator:
    """Hindi -> English with MarianMT, loaded on first use (~300 MB)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tok = self._model = None

    def __call__(self, texts: list[str]) -> list[str]:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self._model is None:
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).eval()
        with torch.no_grad():
            enc = self._tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            out = self._model.generate(**enc, max_new_tokens=128)
        return self._tok.batch_decode(out, skip_special_tokens=True)


@dataclass
class Reading:
    model_text: str                     # what the English models read
    language: str                       # en | hi | hinglish
    translated: bool
    probs: np.ndarray
    notes: list[str] = field(default_factory=list)


def present_part(text: str) -> str | None:
    """The part after "now" when the entry contrasts a past feeling with the present, else None."""
    parts = NOW_SPLIT.split(text)
    if len(parts) >= 2 and PAST_CUE.search(parts[0]) and parts[-1].strip():
        return parts[-1].strip()
    return None


def about_someone_else(text: str) -> bool:
    return bool(THIRD_PERSON_FEELING.search(text)) and not FIRST_PERSON_FEELING.search(text)


def read(text: str, predict_proba, translator=None, cfg: dict | None = None) -> Reading:
    """Translate if needed, score with `predict_proba(list[str]) -> (n, 8)`, then apply the context rules."""
    mc = (cfg or load_config())["multilingual"]
    lang = detect_language(text)
    model_text, translated = text, False
    if lang == "hi" and translator is not None:
        model_text, translated = translator([text])[0], True
    probs = np.asarray(predict_proba([model_text])[0], dtype=float)
    notes = []
    if translated:
        notes.append(f"Translated from Hindi before reading: “{model_text}”")
    now = present_part(model_text)
    if now:
        w = mc["present_weight"]
        probs = w * np.asarray(predict_proba([now])[0]) + (1 - w) * probs
        notes.append("You described a change, so how you feel now counted more than how you felt before.")
    if about_someone_else(model_text):
        neutral = np.eye(len(LABELS))[LABELS.index("Neutral")]
        probs = mc["other_person_weight"] * probs + (1 - mc["other_person_weight"]) * neutral
        notes.append("This seems to be mostly about someone else's feelings, so it counted for less in your trends.")
    return Reading(model_text, lang, translated, probs / probs.sum(), notes)
