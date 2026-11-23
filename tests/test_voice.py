"""Tests for src/voice.py logic that doesn't need the speech models: audio loading, tone mapping, fusion."""
import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

sf = pytest.importorskip("soundfile")
from src import voice  # noqa: E402
from src.trends import negative_scores  # noqa: E402
from src.utils import LABELS, load_config  # noqa: E402

TONE_MAP = load_config()["voice"]["tone_labels"]
CALM_TEXT = np.array([0.6, 0.02, 0.02, 0.02, 0.02, 0.02, 0.1, 0.2])
HEAVY_TEXT = np.array([0.02, 0.4, 0.05, 0.05, 0.2, 0.25, 0.01, 0.02])


def tone(label):
    v = np.zeros(len(LABELS))
    v[LABELS.index(label)] = 1.0
    return v


def test_load_audio_resamples_to_16k_mono():
    sr = 44_100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    stereo = np.stack([np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    wav, seconds = voice.load_audio(buf.getvalue())
    assert wav.ndim == 1 and wav.dtype == np.float32
    assert abs(seconds - 1.0) < 0.01 and abs(len(wav) - 16_000) <= 1


def test_map_tone_covers_only_known_labels_and_sums_to_one():
    out = voice.map_tone([{"label": "sad", "score": 0.6}, {"label": "neu", "score": 0.3},
                          {"label": "unknown", "score": 0.1}], TONE_MAP)
    assert out.sum() == pytest.approx(1.0)
    assert out[LABELS.index("Sadness")] == pytest.approx(0.6 / 0.9)
    assert out[LABELS.index("Stress")] == 0


def test_sad_tone_adds_weight_to_calm_words():
    fused, used = voice.fuse(CALM_TEXT, tone("Sadness"), 0.25)
    assert used
    assert negative_scores(fused) > negative_scores(CALM_TEXT)
    assert fused.sum() == pytest.approx(1.0)


def test_cheerful_tone_never_cancels_distressed_words():
    fused, used = voice.fuse(HEAVY_TEXT, tone("Joy"), 0.25)
    assert not used
    assert np.allclose(fused, HEAVY_TEXT)


def test_empty_tone_leaves_text_unchanged():
    fused, used = voice.fuse(CALM_TEXT, np.zeros(len(LABELS)), 0.25)
    assert not used and np.allclose(fused, CALM_TEXT)
