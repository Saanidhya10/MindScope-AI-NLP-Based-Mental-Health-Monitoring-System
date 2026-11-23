"""Unified inference wrapper: load any trained model and get an 8-class probability distribution.

Kinds: ``classical`` (TF-IDF pipeline), ``bilstm``, ``transformer`` (our fine-tuned DistilBERT/RoBERTa) and
``goemotions`` - a public model fine-tuned on GoEmotions' 28 labels (e.g. SamLowe/roberta-base-go_emotions),
collapsed to our 8 classes with config.yaml -> label_mapping.goemotions. GoEmotions has no Stress label, so
that model never predicts Stress; stressful text lands on Anxiety/Sadness/Fear instead.

    predictor = EmotionPredictor.from_config(cfg)          # reads cfg['inference']
    predictor.predict("I can't stop worrying about tomorrow")
    predictor.predict_proba(["text 1", "text 2"])          # (n, 8) array, columns ordered as utils.LABELS
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.preprocess import TextPreprocessor
from src.utils import LABELS, NUM_CLASSES, get_logger, load_config, load_json, resolve_path, softmax

log = get_logger("inference")


class EmotionPredictor:
    def __init__(self, kind: str, path: str | Path, cfg: dict | None = None, device: str | None = None,
                 batch_size: int = 64):
        self.cfg = cfg or load_config()
        self.kind = kind
        # a Hugging Face Hub id ("org/name") is kept as-is; everything else is a path in the repo
        self.path = path if kind == "goemotions" and not resolve_path(path).exists() else resolve_path(path)
        self.batch_size = batch_size
        self.pre = TextPreprocessor(self.cfg)
        self._device = device
        self._load()

    @classmethod
    def from_config(cls, cfg: dict | None = None, **kw) -> "EmotionPredictor":
        cfg = cfg or load_config()
        ic = cfg.get("inference", {})
        if "kind" in ic and "path" in ic:
            try:
                return cls(ic["kind"], ic["path"], cfg, **kw)
            except (ImportError, OSError) as e:  # torch/transformers missing, or the model can't be downloaded
                log.warning("Could not load %s model '%s' (%s); falling back to a local model.", ic["kind"], ic["path"], e)

        for kind in ["distilbert", "roberta"]:
            p = resolve_path(f"models/{kind}/final")
            if (p / "config.json").exists():
                return cls("transformer", p, cfg, **kw)

        for cand in [
            "models/classical/logreg_baseline/pipeline.joblib",
            "models/classical/logreg_baseline",
            "models/classical/svm_baseline/pipeline.joblib",
            "models/classical/rf_baseline/pipeline.joblib",
        ]:
            p = resolve_path(cand)
            if p.exists():
                return cls("classical", p, cfg, **kw)

        raise FileNotFoundError(
            "No trained model found. Expected trained transformer or classical pipeline in models/."
        )

    # ---------------------------------------------------------------- loading
    def _load(self) -> None:
        if self.kind == "classical":
            f = self.path / "pipeline.joblib" if self.path.is_dir() else self.path
            self.model = joblib.load(f)
        elif self.kind == "bilstm":
            import torch

            from src.datasets import Vocabulary
            from src.train_bilstm import BiLSTMClassifier

            self.device = torch.device(self._device or ("cuda" if torch.cuda.is_available() else "cpu"))
            hp = load_json(self.path / "hparams.json")
            self.vocab = Vocabulary.load(self.path / "vocab.json")
            self.max_len = hp["max_len"]
            self.model = BiLSTMClassifier(len(self.vocab), hp["emb_dim"], hp["hidden"], hp["dropout"],
                                          pad_idx=self.vocab.pad_idx)
            self.model.load_state_dict(torch.load(self.path / "model.pt", map_location=self.device))
            self.model.to(self.device).eval()
        elif self.kind == "goemotions":
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device = torch.device(self._device or "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.path)).to(self.device).eval()
            self.max_len = self.cfg.get("inference", {}).get("max_len", 128)
            group = {name: cls for cls, names in self.cfg["label_mapping"]["goemotions"].items() for name in names}
            id2label = self.model.config.id2label
            # column j of the 28-label output -> index of our class
            self._group_idx = np.array([LABELS.index(group[id2label[j]]) for j in range(len(id2label))])
        elif self.kind in ("distilbert", "roberta", "transformer"):
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device = torch.device(self._device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.path)).to(self.device).eval()
            self.max_len = self.cfg.get("transformer", {}).get("max_len", 64)
        else:
            raise ValueError(f"Unknown model kind '{self.kind}'")
        log.info("Loaded %s model from %s", self.kind, self.path)

    # ---------------------------------------------------------------- prediction
    def predict_proba(self, texts) -> np.ndarray:
        texts = [texts] if isinstance(texts, str) else list(texts)
        if self.kind == "classical":
            return self._proba_classical(texts)
        if self.kind == "bilstm":
            return self._proba_bilstm(texts)
        if self.kind == "goemotions":
            return self._proba_goemotions(texts)
        return self._proba_transformer(texts)

    def _proba_classical(self, texts) -> np.ndarray:
        X = self.pre.classical_many(texts)
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X)
        else:  # LinearSVC: no probabilities -> softmax over decision scores (uncalibrated)
            p = softmax(self.model.decision_function(X))
        out = np.zeros((len(texts), NUM_CLASSES))
        out[:, np.asarray(self.model.classes_, dtype=int)] = p   # classes_ are label ids (may miss a class)
        return out

    def _proba_bilstm(self, texts) -> np.ndarray:
        from src.train_bilstm import predict_proba_bilstm

        return predict_proba_bilstm(self.model, self.vocab, [self.pre.seq(t) for t in texts], self.max_len,
                                    self.device, self.batch_size)

    def _proba_transformer(self, texts) -> np.ndarray:
        import torch

        clean = [self.pre.transformer(t) or "." for t in texts]
        out = []
        with torch.no_grad():
            for i in range(0, len(clean), self.batch_size):
                enc = self.tokenizer(clean[i:i + self.batch_size], truncation=True, max_length=self.max_len,
                                     padding=True, return_tensors="pt").to(self.device)
                out.append(torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy())
        return np.concatenate(out)

    @property
    def label(self) -> str:
        """Short human-readable model name, e.g. 'logreg_baseline (classical)'."""
        if self.kind == "goemotions":
            return f"{str(self.path).split('/')[-1]} (pretrained)"
        name = self.path.parent.name if self.path.name == "pipeline.joblib" else self.path.name
        return f"{name} ({self.kind})"

    def _proba_goemotions(self, texts) -> np.ndarray:
        """Sigmoid per GoEmotions label, then each of our classes takes its strongest sub-label (max, not sum,
        so Joy's 11 sub-labels don't outweigh single-label classes like Anxiety), renormalised to sum to 1."""
        import torch

        clean = [self.pre.transformer(t) or "." for t in texts]
        out = np.zeros((len(clean), NUM_CLASSES))
        with torch.no_grad():
            for i in range(0, len(clean), self.batch_size):
                enc = self.tokenizer(clean[i:i + self.batch_size], truncation=True, max_length=self.max_len,
                                     padding=True, return_tensors="pt").to(self.device)
                sig = torch.sigmoid(self.model(**enc).logits).cpu().numpy()
                for j, c in enumerate(self._group_idx):
                    out[i:i + len(sig), c] = np.maximum(out[i:i + len(sig), c], sig[:, j])
        out = out + 1e-9
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, text: str) -> dict:
        p = self.predict_proba([text])[0]
        i = int(p.argmax())
        return {"emotion": LABELS[i], "confidence": float(p[i]),
                "probabilities": {l: float(v) for l, v in zip(LABELS, p)}}

    def predict_many(self, texts) -> list[dict]:
        P = self.predict_proba(texts)
        return [{"emotion": LABELS[int(p.argmax())], "confidence": float(p.max()),
                 "probabilities": {l: float(v) for l, v in zip(LABELS, p)}} for p in P]


_default_predictor = None


def get_default_predictor(cfg: dict | None = None) -> EmotionPredictor:
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = EmotionPredictor.from_config(cfg)
    return _default_predictor


def predict(text: str) -> dict:
    if text is None or not str(text).strip():
        raise ValueError("Input text cannot be empty")
    return get_default_predictor().predict(str(text))


def predict_proba(texts) -> np.ndarray:
    return get_default_predictor().predict_proba(texts)

