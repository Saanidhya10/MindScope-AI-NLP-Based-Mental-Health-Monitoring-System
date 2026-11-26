"""Evidence-based recommendations by retrieval (improvement 3, without a language model).

A curated library (data/knowledge/wellness_kb.yaml) of short wellness guidance, each entry linked to the public source
it is based on (NHS, WHO, NIMH, Tele-MANAS). For a journal entry we embed the text with a small local sentence-embedding
model, rank library entries by cosine similarity, and add a boost when an entry is tagged for the entry's detected
feelings or topics. Each suggestion comes with a plain-language reason and its source link. Nothing is generated, so
the advice is exactly what was curated, and crisis messages never reach this module (helplines are shown instead).
Settings: config.yaml -> knowledge.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from src.utils import load_config, resolve_path


def load_library(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class KnowledgeBase:
    def __init__(self, cfg: dict | None = None):
        self.cfg = (cfg or load_config())["knowledge"]
        self.path = resolve_path(self.cfg["library"])
        self.entries = load_library(self.path)
        self._model = None
        self._emb = None

    # ---- embeddings ------------------------------------------------------------------
    def _encoder(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg["embedding_model"], device="cpu")
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._encoder().encode(texts, normalize_embeddings=True, show_progress_bar=False))

    @staticmethod
    def entry_text(e: dict) -> str:
        return f"{e['title']}. {e['summary']} Helps with: {', '.join(e.get('emotions', []))}."

    def library_embeddings(self) -> np.ndarray:
        """Embeddings of every library entry, cached on disk and rebuilt when the library file changes."""
        if self._emb is None:
            digest = hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]
            cache = resolve_path(self.cfg["cache_dir"]) / f"kb_{digest}.npy"
            if cache.exists():
                self._emb = np.load(cache)
            else:
                self._emb = self.embed([self.entry_text(e) for e in self.entries])
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self._emb)
        return self._emb

    # ---- ranking ---------------------------------------------------------------------
    def rank(self, query_vec: np.ndarray, emotions: list[str], topics: list[str]) -> list[dict]:
        """Score = cosine similarity + boosts for tagged feelings (first feeling counts most) and matching topics."""
        from src.triggers import BY_KEY

        sims = self.library_embeddings() @ query_vec
        out = []
        for e, sim in zip(self.entries, sims):
            emo_hits = [m for m in emotions[:2] if m in e.get("emotions", [])]
            emo_boost = (self.cfg["emotion_boost"] if emotions and emotions[0] in e.get("emotions", []) else
                         self.cfg["emotion_boost"] / 2 if emo_hits else 0.0)
            topic_hits = [t for t in topics if t in e.get("topics", [])]
            score = float(sim) + emo_boost + min(len(topic_hits), 2) * self.cfg["topic_boost"]
            if topic_hits:
                reason = "You wrote about " + " and ".join(BY_KEY[t].label.lower() for t in topic_hits[:2] if t in BY_KEY)
                if emo_hits:
                    reason += f", with {emo_hits[0].lower()} coming through"
            elif emo_hits:
                reason = f"Often helps with {emo_hits[0].lower()}"
            else:
                reason = "Closest match to what you wrote"
            out.append({"entry": e, "score": score, "similarity": float(sim), "reason": reason})
        return sorted(out, key=lambda r: -r["score"])

    def recommend(self, text: str, emotions: list[str], topics: list[str], k: int = 3,
                  diversity: float = 0.35) -> list[dict]:
        """Top-k with maximal marginal relevance: each next pick is penalised by its similarity to picks so far,
        so three near-identical tips (e.g. three sleep tips) don't crowd out a different, relevant one."""
        ranked = self.rank(self.embed([text])[0], emotions, topics)
        emb = self.library_embeddings()
        index = {e["id"]: i for i, e in enumerate(self.entries)}
        chosen: list[dict] = []
        pool = ranked[: max(12, k)]
        while pool and len(chosen) < k:
            def mmr(r):
                if not chosen:
                    return r["score"]
                v = emb[index[r["entry"]["id"]]]
                return r["score"] - diversity * max(float(v @ emb[index[c["entry"]["id"]]]) for c in chosen)
            best = max(pool, key=mmr)
            chosen.append(best)
            pool.remove(best)
        return chosen

    def search(self, query: str, k: int = 8) -> list[dict]:
        sims = self.library_embeddings() @ self.embed([query])[0]
        order = np.argsort(sims)[::-1][:k]
        return [{"entry": self.entries[i], "similarity": float(sims[i])} for i in order]
