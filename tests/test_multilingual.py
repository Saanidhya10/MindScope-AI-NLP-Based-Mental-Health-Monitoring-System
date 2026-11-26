"""Tests for multilingual and context-aware reading (src/multilingual.py, Hinglish lists in src/preprocess.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import multilingual as M  # noqa: E402
from src.preprocess import normalize_hinglish  # noqa: E402
from src.recommend import check_crisis  # noqa: E402
from src.utils import LABELS, load_config  # noqa: E402

CFG = load_config()


def onehot(label):
    v = np.full(len(LABELS), 0.01)
    v[LABELS.index(label)] = 1.0
    return v / v.sum()


def fake_predict(texts):
    """Joy if 'fine'/'better' appears, Sadness if 'sad', else Anxiety."""
    out = []
    for t in texts:
        t = t.lower()
        out.append(onehot("Joy" if ("fine" in t or "better" in t) else "Sadness" if "sad" in t else "Anxiety"))
    return np.array(out)


def test_language_detection():
    assert M.detect_language("मुझे अपनी परीक्षा की चिंता हो रही है।") == "hi"
    assert M.detect_language("Yaar exams ki wajah se bahut tension hai") == "hinglish"
    assert M.detect_language("I went to the main office today.") == "en"


def test_hinglish_glosses_only_apply_to_hinglish():
    assert normalize_hinglish("I went to the main office") == "I went to the main office"
    assert "happy" in normalize_hinglish("Main bahut khush hoon aaj")
    assert normalize_hinglish("aaj khush nahi hoon").startswith("today not happy")


def test_hindi_is_translated_before_reading():
    r = M.read("मैं बहुत उदास हूँ", fake_predict, translator=lambda xs: ["I am very sad"], cfg=CFG)
    assert r.translated and r.model_text == "I am very sad"
    assert LABELS[int(r.probs.argmax())] == "Sadness"
    assert any("Translated from Hindi" in n for n in r.notes)


def test_present_counts_more_than_past():
    r = M.read("I used to be anxious about exams but now I feel fine.", fake_predict, cfg=CFG)
    assert LABELS[int(r.probs.argmax())] == "Joy"
    assert M.present_part("I feel fine now") is None  # no past cue -> no split


def test_someone_elses_feelings_count_for_less():
    r = M.read("My friend is really sad about her breakup.", fake_predict, cfg=CFG)
    assert r.probs[LABELS.index("Sadness")] < 0.6 and r.probs[LABELS.index("Neutral")] > 0.4
    mine = M.read("My friend is sad and I feel sad too.", fake_predict, cfg=CFG)
    assert LABELS[int(mine.probs.argmax())] == "Sadness" and not mine.notes


def test_hindi_and_hinglish_crisis_phrases():
    assert check_crisis("main jeena nahi chahta", CFG)
    assert check_crisis("मैं जीना नहीं चाहता", CFG)
