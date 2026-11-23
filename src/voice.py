"""Voice entries (English only): speech -> text with Whisper, plus tone of voice from a speech-emotion model.

    analyzer = VoiceAnalyzer(cfg)
    wav, seconds = load_audio(wav_bytes)
    text = analyzer.transcribe(wav)                    # then run the normal text model on it
    tone = analyzer.tone_proba(wav)                    # 8-class vector, zeros for classes the model lacks
    probs, used = fuse(text_probs, tone, cfg["voice"]["tone_weight"])

Recordings are only held in memory. Tone is combined one-directionally: it can add evidence of distress but
never lowers the difficult-feelings share of what was said, because speech-emotion models trained on acted
speech are unreliable on everyday voices (a cheerful-sounding "I can't do this anymore" must not read as fine).
Settings: config.yaml -> voice.
"""
from __future__ import annotations

import io
from math import gcd

import numpy as np

from src.trends import negative_scores
from src.utils import LABELS, NUM_CLASSES, get_logger, load_config

log = get_logger("voice")
SAMPLE_RATE = 16_000


def load_audio(data: bytes) -> tuple[np.ndarray, float]:
    """WAV/FLAC/OGG bytes -> mono float32 at 16 kHz, plus duration in seconds."""
    import soundfile as sf
    from scipy.signal import resample_poly

    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = gcd(sr, SAMPLE_RATE)
        wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype("float32")
    return wav, len(wav) / SAMPLE_RATE


def map_tone(scores: list[dict], label_map: dict) -> np.ndarray:
    """Speech-emotion pipeline output [{'label': 'sad', 'score': .7}, ...] -> 8-class vector summing to 1."""
    out = np.zeros(NUM_CLASSES)
    for s in scores:
        if s["label"] in label_map:
            out[LABELS.index(label_map[s["label"]])] += s["score"]
    total = out.sum()
    return out / total if total > 0 else out


def fuse(text_probs: np.ndarray, tone_probs: np.ndarray, weight: float) -> tuple[np.ndarray, bool]:
    """Blend tone into the text distribution only when it points to more distress than the words.
    Returns (probabilities, whether tone was used)."""
    text_probs = np.asarray(text_probs, dtype=float)
    tone_probs = np.asarray(tone_probs, dtype=float)
    if tone_probs.sum() == 0 or negative_scores(tone_probs) <= negative_scores(text_probs):
        return text_probs, False
    blended = (1 - weight) * text_probs + weight * tone_probs
    return blended / blended.sum(), True


class VoiceAnalyzer:
    """Loads Whisper and the tone model lazily, so text-only users never pay for them."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = (cfg or load_config())["voice"]
        self._asr = None
        self._tone = None

    def transcribe(self, wav: np.ndarray) -> str:
        if self._asr is None:
            from faster_whisper import WhisperModel

            self._asr = WhisperModel(self.cfg["asr_model"], device="cpu", compute_type=self.cfg["asr_compute_type"])
        segments, _ = self._asr.transcribe(wav, language="en", vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()

    def tone_proba(self, wav: np.ndarray) -> np.ndarray:
        if self._tone is None:
            from transformers import pipeline

            self._tone = pipeline("audio-classification", model=self.cfg["tone_model"], device="cpu")
        scores = self._tone({"raw": wav, "sampling_rate": SAMPLE_RATE}, top_k=None)
        return map_tone(scores, self.cfg["tone_labels"])
