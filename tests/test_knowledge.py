"""Tests for the wellness library and retrieval ranking (src/knowledge.py), without loading the embedding model."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.knowledge import KnowledgeBase  # noqa: E402
from src.recommend import check_wording  # noqa: E402
from src.triggers import BY_KEY  # noqa: E402
from src.utils import LABELS  # noqa: E402


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


def test_library_entries_are_complete_sourced_and_non_diagnostic(kb):
    ids = [e["id"] for e in kb.entries]
    assert len(ids) == len(set(ids)) and len(ids) >= 25
    for e in kb.entries:
        assert e["title"] and e["summary"] and e["source"]
        assert e["url"].startswith("https://")
        assert set(e.get("emotions", [])) <= set(LABELS)
        assert set(e.get("topics", [])) <= set(BY_KEY)
        assert not check_wording(e["title"] + " " + e["summary"]), e["id"]


def fake_kb(kb, entries):
    """A KnowledgeBase over `entries` with one-hot embeddings, so ranking can be tested without the model."""
    k = KnowledgeBase.__new__(KnowledgeBase)
    k.cfg, k.entries = kb.cfg, entries
    k._emb = np.eye(len(entries))
    return k


def test_topic_and_emotion_boosts_and_reasons(kb):
    entries = [{"id": "a", "title": "A", "summary": "", "emotions": ["Stress"], "topics": ["academic"]},
               {"id": "b", "title": "B", "summary": "", "emotions": ["Joy"], "topics": []},
               {"id": "c", "title": "C", "summary": "", "emotions": [], "topics": []}]
    k = fake_kb(kb, entries)
    query = np.array([0.3, 0.35, 0.33])  # raw similarity slightly favours b
    ranked = k.rank(query, ["Stress"], ["academic"])
    assert ranked[0]["entry"]["id"] == "a"
    assert ranked[0]["reason"].startswith("You wrote about studies")
    assert next(r for r in ranked if r["entry"]["id"] == "c")["reason"] == "Closest match to what you wrote"


def test_mmr_avoids_near_duplicates(kb):
    entries = [{"id": i, "title": i, "summary": "", "emotions": [], "topics": []} for i in "abc"]
    k = fake_kb(kb, entries)
    k._emb = np.array([[1, 0], [0.995, 0.1], [0.6, 0.8]])
    k._emb = k._emb / np.linalg.norm(k._emb, axis=1, keepdims=True)
    q = np.array([0.9, 0.436])
    k.embed = lambda texts: np.array([q / np.linalg.norm(q)])
    picked = [r["entry"]["id"] for r in k.recommend("x", [], [], k=2, diversity=0.5)]
    assert picked == ["b", "c"]  # a is almost identical to b, so the different-but-relevant c comes second
