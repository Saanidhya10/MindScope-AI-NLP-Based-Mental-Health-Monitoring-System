"""Backend building blocks for the Streamlit app (no Streamlit imports here, so it is easy to test).

Self-contained on purpose: you can later move these pieces into src/inference.py, src/trends.py,
src/recommend.py and src/database.py and import them from there instead.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data/processed/mood_app.db"

EMOTIONS = ["Joy", "Sadness", "Anger", "Fear", "Anxiety", "Stress", "Surprise", "Neutral"]
NEGATIVE = ["Sadness", "Anger", "Fear", "Anxiety", "Stress"]
PCOLS = [f"p_{e}" for e in EMOTIONS]
NEG_PCOLS = [f"p_{e}" for e in NEGATIVE]

# Verify these before any real deployment - numbers change and differ by region.
HELPLINES = {
    "Tele-MANAS (24x7 National Toll-Free)": "14416 / 1800-891-4416",
    "National Emergency Response": "112",
    "KIRAN Mental Health Support (Govt)": "1800-599-0019",
    "Vandrevala Foundation (24x7 Helpline)": "+91 9999 666 555",
}

# ---------------------------------------------------------------- parameters
DEFAULT_PARAMS = {"alpha": 0.3, "tau": 0.55, "k": 3, "max_gap_days": 3}


def load_params() -> dict:
    """tau / k / alpha calibrated in notebook 06 (results/trend_params.json), else conservative defaults."""
    p = dict(DEFAULT_PARAMS)
    f = ROOT / "results/trend_params.json"
    if f.exists():
        try:
            p.update({k: v for k, v in json.loads(f.read_text()).items() if k in p})
        except Exception:
            pass
    return p


# ---------------------------------------------------------------- preprocessing & multilingual
_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<[^>]+>|&\w+;")
_HANDLE = re.compile(r"(?<!\w)@\w+|(?<!\w)/?u/\w+")
_REPEAT = re.compile(r"([!?.])\1{2,}")
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
_classical_cache: dict = {}

# Multilingual Hindi / Hinglish sentiment transliteration lexicon
HINGLISH_MAP = {
    # Sadness / Depression
    "udaas": "sad", "udasi": "sadness", "udaasi": "sadness", "dukh": "sadness", "dukhi": "sad",
    "gham": "grief", "dard": "pain", "rona": "crying", "aansu": "tears", "aansoo": "tears",
    "akela": "lonely", "akelapan": "loneliness", "tanha": "lonely", "tanhai": "loneliness",
    "mayus": "hopeless", "mayusi": "hopelessness",
    # Anxiety / Stress / Worry
    "tension": "anxiety stress", "chinta": "anxiety", "fikr": "worry", "fakar": "worry",
    "pareshaan": "worried distressed", "pareshan": "worried distressed",
    "pareshaani": "distress", "pareshani": "distress", "ghabrahat": "anxiety panic",
    "ghabra": "anxious", "dar": "fear", "darr": "fear", "khauf": "fear", "bhay": "fear",
    "thaka": "exhausted tired", "thakaan": "exhaustion fatigue",
    # Anger / Frustration
    "gussa": "anger angry", "krodh": "anger", "naraz": "angry upset",
    "chid": "irritated", "chirh": "irritated", "nafrat": "hate",
    # Joy / Relief
    "khush": "happy joy", "khushi": "joy happiness", "prasann": "happy",
    "sukoon": "calm relief", "shanti": "peace", "raahat": "relief", "rahat": "relief",
    "maza": "fun enjoyment", "anand": "joy delight",
    # Devanagari Hindi Script
    "उदास": "sad", "उदासी": "sadness", "दुःख": "sadness", "दुखी": "sad",
    "चिंता": "anxiety", "परेशान": "worried", "घबराहट": "anxiety panic",
    "डर": "fear", "गुस्सा": "anger", "क्रोध": "anger", "खुश": "happy",
    "खुशी": "joy", "सुकून": "calm relief", "अकेला": "lonely", "रोना": "crying",
}

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die", "self harm",
    "cutting myself", "hurt myself", "hang myself", "slit my wrist",
    "take my own life", "mar jaana", "marne ka man", "jaan dena",
]


def normalize_multilingual(text: str) -> str:
    """Replaces Hindi/Hinglish emotional terms with English sentiment proxies."""
    if not text:
        return ""
    words = text.split()
    mapped = []
    for w in words:
        w_lower = w.lower().strip(".,!?:;\"'()[]")
        mapped.append(HINGLISH_MAP.get(w_lower, w))
    return " ".join(mapped)


def check_crisis_safety(text: str) -> dict:
    """Checks for acute distress or self-harm keywords to surface immediate safety helplines."""
    if not text:
        return {"triggered": False, "helplines": HELPLINES}
    lower = text.lower()
    triggered = any(kw in lower for kw in CRISIS_KEYWORDS)
    return {
        "triggered": triggered,
        "message": (
            "If you are feeling overwhelmed, in deep pain, or having thoughts of self-harm, "
            "please know that immediate compassionate support is available 24/7."
        ),
        "helplines": HELPLINES,
    }


def preprocess_tf(text: str) -> str:
    """Light cleaning for transformers + multilingual token normalization."""
    t = normalize_multilingual(str(text))
    t = _HANDLE.sub(" ", _HTML.sub(" ", _URL.sub(" ", t)))
    try:
        import emoji
        t = emoji.demojize(t, delimiters=(" ", " ")).replace("_", " ")
    except ImportError:
        pass
    return _WS.sub(" ", _REPEAT.sub(r"\1\1", t)).strip()


def preprocess_classical(text: str) -> str:
    """TF-IDF preprocessing pipeline: multilingual normalization, lowercase, stop-words kept/removed, lemmatised."""
    if not _classical_cache:
        import nltk
        for pkg in ["stopwords", "wordnet", "omw-1.4"]:
            nltk.download(pkg, quiet=True)
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        neg_base = {"no", "nor", "not", "never", "none", "nothing", "neither", "nobody", "cannot", "don", "didn",
                    "doesn", "isn", "wasn", "aren", "couldn", "shouldn", "wouldn", "hasn", "haven", "hadn", "won",
                    "weren", "needn", "mustn", "shan", "mightn", "ain"}
        sw = set(stopwords.words("english"))
        negations = {w for w in sw if w in neg_base or "n't" in w} | neg_base
        _classical_cache["stop"] = sw - negations
        _classical_cache["lem"] = WordNetLemmatizer()
    t = preprocess_tf(text).lower().replace("\u2019", "'")
    return " ".join(_classical_cache["lem"].lemmatize(w) for w in _TOKEN.findall(t) if w not in _classical_cache["stop"])


# ---------------------------------------------------------------- model
class Predictor:
    """Text -> probability distribution over the 8 emotions. Transformer if trained, else classical LR."""

    def __init__(self):
        self.model = None
        self.pipe = None
        meta = ROOT / "models/final_model.json"
        keys = ([json.loads(meta.read_text())["model"]] if meta.exists() else []) + ["distilbert", "roberta"]
        for key in keys:
            path = ROOT / "models" / key / "final"
            if (path / "config.json").exists():
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self._torch = torch
                self.tok = AutoTokenizer.from_pretrained(str(path))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).eval()
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                self.name, self.kind = key, "transformer"
                return
        lr_candidates = [
            ROOT / "models/classical/logreg_baseline/pipeline.joblib",
            ROOT / "models/classical/logreg_baseline.joblib",
            ROOT / "models/classical/svm_baseline/pipeline.joblib",
            ROOT / "models/classical/rf_baseline/pipeline.joblib",
        ]
        for lr in lr_candidates:
            if lr.exists():
                import joblib
                self.pipe = joblib.load(lr)
                self.name = lr.parent.name if lr.name == "pipeline.joblib" else lr.stem
                self.kind, self.device = "classical", "cpu"
                return
        raise FileNotFoundError(
            "No trained model found. Run notebooks 02 (classical) or 04 (transformers) first - "
            "expected models/<model>/final/ or models/classical/logreg_baseline/pipeline.joblib.")

    def predict_proba(self, text: str) -> np.ndarray:
        if self.kind == "classical":
            return self.pipe.predict_proba([preprocess_classical(text)])[0]
        enc = self.tok(preprocess_tf(text), truncation=True, max_length=64, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            return self._torch.softmax(self.model(**enc).logits, -1)[0].cpu().numpy()

    @property
    def label(self) -> str:
        return f"{self.name} ({self.kind})"


# ---------------------------------------------------------------- multi-label & uncertainty
def get_multilabel_emotions(probs: np.ndarray, threshold: float = 0.16) -> list[dict]:
    """Identifies primary emotion and secondary co-occurring emotions."""
    order = np.argsort(probs)[::-1]
    results = []
    top_p = float(probs[order[0]])
    for idx in order:
        p = float(probs[idx])
        if len(results) == 0 or (p >= threshold and p >= 0.35 * top_p):
            results.append({"emotion": EMOTIONS[idx], "prob": p})
        if len(results) >= 2:
            break
    return results


def compute_uncertainty_metrics(probs: np.ndarray) -> dict:
    """Computes Normalized Shannon Entropy and Margin to quantify model certainty and emotional ambiguity."""
    order = np.argsort(probs)[::-1]
    top_p = float(probs[order[0]])
    sec_p = float(probs[order[1]]) if len(probs) > 1 else 0.0

    p_clean = np.clip(probs, 1e-12, 1.0)
    p_clean = p_clean / p_clean.sum()
    h_norm = float(-np.sum(p_clean * np.log2(p_clean)) / np.log2(len(probs)))
    margin = float(top_p - sec_p)

    if h_norm < 0.50 or margin > 0.35:
        label = "High Certainty"
        color = "#5ec4a0"
        expl = "Clear and focused emotional signal detected."
    elif h_norm <= 0.72:
        label = "Moderate Certainty"
        color = "#e0a060"
        expl = "Nuanced expression with co-occurring emotional tones."
    else:
        label = "High Ambiguity / Mixed"
        color = "#b06090"
        expl = "Complex emotional state where multiple feelings co-exist."

    return {
        "confidence": round(top_p * 100),
        "entropy_pct": round(h_norm * 100),
        "margin": round(margin, 2),
        "certainty_label": label,
        "certainty_color": color,
        "explanation": expl,
    }


# ---------------------------------------------------------------- storage (derived scores only, never raw text)
def _con() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions (
        user_id TEXT, timestamp TEXT, emotion TEXT, confidence REAL, sentiment_score REAL, probs TEXT)""")
    return con


def add_prediction(user_id: str, probs: np.ndarray, ts: datetime | None = None) -> None:
    ts = ts or datetime.now()
    probs = np.asarray(probs, float)
    s = float(probs[[EMOTIONS.index(e) for e in NEGATIVE]].sum())
    with _con() as con:
        con.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?)",
                    (user_id, ts.isoformat(timespec="seconds"), EMOTIONS[int(probs.argmax())],
                     float(probs.max()), s, json.dumps([round(float(x), 5) for x in probs])))


def load_history(user_id: str) -> pd.DataFrame:
    with _con() as con:
        df = pd.read_sql("SELECT * FROM predictions WHERE user_id=? ORDER BY timestamp", con, params=(user_id,))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df[PCOLS] = np.array([json.loads(p) for p in df["probs"]])
    df["s"] = df[NEG_PCOLS].sum(axis=1)
    return df


def delete_history(user_id: str) -> int:
    with _con() as con:
        return con.execute("DELETE FROM predictions WHERE user_id=?", (user_id,)).rowcount


def export_history_csv(user_id: str) -> bytes:
    df = load_history(user_id)
    cols = ["timestamp", "emotion", "confidence", "sentiment_score"] + PCOLS
    return (df[cols] if not df.empty else pd.DataFrame(columns=cols)).to_csv(index=False).encode()


def add_demo_history(user_id: str, days: int = 28, seed: int = 0) -> None:
    """SIMULATED data drifting from positive to negative mood - for demos only."""
    rng = np.random.default_rng(seed)
    pos = np.array([.45, .03, .02, .02, .02, .03, .13, .30])
    neg = np.array([.02, .30, .08, .05, .20, .27, .01, .07])
    start = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0) - pd.Timedelta(days=days - 1)
    for d in range(days):
        w = d / (days - 1)
        for _ in range(int(rng.integers(1, 4))):
            p = rng.dirichlet(60 * ((1 - w) * pos + w * neg) + 0.3)
            add_prediction(user_id, p, start + pd.Timedelta(days=d, hours=int(rng.integers(0, 12)),
                                                              minutes=int(rng.integers(0, 60))))


# ---------------------------------------------------------------- temporal aggregation & baseline
def daily_summary(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Per-day mean probabilities, dominant emotion, negative-affect S and its EMA S~ (days with entries only)."""
    d = df.set_index("timestamp")[PCOLS + ["s"]].resample("D").mean().dropna(subset=["s"])
    d["n_entries"] = df.set_index("timestamp")["s"].resample("D").size().reindex(d.index)
    d["dominant"] = d[PCOLS].idxmax(axis=1).str[2:]
    d["S"] = d["s"]
    d["S_ema"] = d["S"].ewm(alpha=alpha, adjust=False).mean()
    return d


def window_summary(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Weekly ('7D') or monthly ('MS') summaries from message-level probabilities."""
    w = df.set_index("timestamp")[PCOLS + ["s"]].resample(rule).mean().dropna(subset=["s"])
    w["dominant"] = w[PCOLS].idxmax(axis=1).str[2:]
    w["S"] = w["s"]
    return w


def escalation_flags(daily: pd.DataFrame, tau: float, k: int, max_gap_days: int = 3) -> np.ndarray:
    """True on days where S~ has exceeded tau for >= k consecutive entry-days (a gap > max_gap_days resets the run)."""
    run, prev, flags = 0, None, []
    for ts, v in zip(daily.index, daily["S_ema"].values):
        gap = (ts - prev).days if prev is not None else 0
        prev = ts
        run = (run + 1 if gap <= max_gap_days else 1) if v > tau else 0
        flags.append(run >= k)
    return np.array(flags, dtype=bool)


def compute_personalized_baseline(hist: pd.DataFrame, default_tau: float = 0.55) -> dict:
    """Computes the user's empirical personal mood baseline and dynamic alert threshold."""
    if hist.empty or len(hist) < 2:
        return {
            "mean": 0.35,
            "std": 0.12,
            "tau_personal": default_tau,
            "status": "Establishing Baseline",
            "entries": len(hist),
        }
    s_series = hist["s"]
    mean_val = float(s_series.mean())
    std_val = float(s_series.std()) if len(s_series) > 1 else 0.12
    tau_p = float(np.clip(mean_val + 1.25 * std_val, 0.40, 0.85))
    return {
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "tau_personal": round(tau_p, 2),
        "status": "Calibrated Baseline",
        "entries": len(hist),
    }


def detect_anomalies_and_trend(hist: pd.DataFrame, daily: pd.DataFrame, baseline: dict, tau: float) -> dict:
    """Detects acute stress spikes, sustained strain streaks, and emotional relief uplifts."""
    if daily.empty:
        return {
            "status": "Awaiting Data",
            "badge": "🌱 Starting",
            "color": "#a67df0",
            "desc": "Add check-in entries to unlock intelligent trend and anomaly detection.",
        }

    latest_s = float(daily["S"].iloc[-1])
    mean_s = baseline["mean"]
    std_s = baseline["std"] if baseline["std"] > 0.02 else 0.10
    z_score = (latest_s - mean_s) / std_s

    # Acute spike anomaly
    if z_score >= 1.75 and latest_s > 0.48:
        return {
            "status": "Acute Stress Spike",
            "badge": "⚠️ Stress Surge",
            "color": "#e0a060",
            "desc": f"Today's negative affect ({latest_s:.2f}) jumped +{z_score:.1f}σ above your personal norm.",
            "is_spike": True,
        }

    # Calming recovery dip
    if z_score <= -1.3 and latest_s < 0.30:
        return {
            "status": "Significant Uplift",
            "badge": "🌟 Calm Uplift",
            "color": "#5ec4a0",
            "desc": f"Today's check-in shows a restorative calm ({latest_s:.2f}), {-z_score:.1f}σ below your baseline.",
            "is_recovery": True,
        }

    # Sustained strain streak
    recent_tail = daily["S_ema"].tail(3)
    if len(recent_tail) >= 3 and (recent_tail > tau).all():
        return {
            "status": "Sustained Strain",
            "badge": "🚨 Persistent Strain",
            "color": "#e06080",
            "desc": f"Negative affect has remained elevated above threshold ({tau:.2f}) over multiple recent days.",
            "is_sustained": True,
        }

    return {
        "status": "Stable Equilibrium",
        "badge": "🟢 Stable Baseline",
        "color": "#7c5cbf",
        "desc": f"Current affect ({latest_s:.2f}) aligns closely with your personal baseline ({mean_s:.2f}).",
        "is_normal": True,
    }


# ---------------------------------------------------------------- rule-based recommendations
SUGGESTIONS = {
    "Stress": ("Try a slow breathing exercise",
               "Stress seems present in your recent entries. A few rounds of 4-7-8 breathing "
               "(inhale for 4s, hold for 7s, exhale slowly for 8s) can calm nervous system activation."),
    "Anxiety": ("Practice a 5-4-3-2-1 grounding exercise",
                "Things may feel uncertain or on edge. Ground yourself by naming 5 things you can see, "
                "4 you can physically feel, 3 you hear, 2 you smell, and 1 you can taste right now."),
    "Sadness": ("Practice self-compassion and gentle pacing",
                "Your recent entries feel heavy. Give yourself permission to slow down, reach out to someone "
                "comforting, or engage in a quiet, low-demand restorative activity."),
    "Anger": ("Step back for emotional defusion",
              "Frustration or tension is showing up. A brief walk, sipping cold water, or writing out raw thoughts "
              "without judgment helps create helpful mental distance."),
    "Fear": ("Break worries into actionable pieces",
             "Fear often magnifies uncertainties. Writing down the single immediate worry and identifying "
             "one tiny concrete step you control can bring back your sense of agency."),
    "Joy": ("Savor and anchor this positive moment",
            "You are experiencing positive well-being! Take 30 seconds to intentionally notice and anchor "
            "what contributed to this calm or happiness today."),
    "Neutral": ("Steady and balanced headspace",
                "Your mood is balanced and steady. A great time for focused work, light hobbies, or restorative routines."),
    "Surprise": ("Reflect and absorb unexpected events",
                 "Unplanned changes can be stimulating or disorienting. Take a mindful breath to process the new context."),
}

ESCALATION_TITLE = "Consider talking to someone who can support you"
ESCALATION_TEXT = ("Over the past days your entries have shown a persistent pattern of distress. This tool cannot "
                   "diagnose anything, but speaking with a professional or trusted counselor can make a genuine difference. "
                   "You do not have to carry everything alone.")


def recommend(daily: pd.DataFrame, flags: np.ndarray, tau: float, anomaly: dict | None = None) -> dict:
    """Generates context-aware, non-diagnostic coping recommendations."""
    if daily.empty:
        return {"level": "none", "title": "Welcome to Mood Journal", "text": "Record your first check-in to receive personalized support.", "dominant": "Neutral"}
    last, dom = daily.iloc[-1], daily.iloc[-1]["dominant"]

    # Check escalation streak
    if len(flags) and bool(flags[-1]):
        return {"level": "escalation", "title": ESCALATION_TITLE, "text": ESCALATION_TEXT, "dominant": dom}

    # Check acute anomaly spike
    if anomaly and anomaly.get("is_spike"):
        t, x = SUGGESTIONS.get("Stress", ("Take a mindful pause", "A sudden wave of tension was detected."))
        return {"level": "suggestion", "title": f"⚡ Acute Surge: {t}", "text": x, "dominant": dom}

    if dom in SUGGESTIONS and last["S_ema"] > 0.65 * tau:
        t, x = SUGGESTIONS[dom]
        return {"level": "suggestion", "title": t, "text": x, "dominant": dom}

    if dom in ["Joy", "Neutral", "Surprise"]:
        t, x = SUGGESTIONS[dom]
        return {"level": "none", "title": f"✨ {t}", "text": x, "dominant": dom}

    return {"level": "none", "title": "Steady Emotional State",
            "text": "Your entries do not show persistent distress. Continue nurturing daily well-being.", "dominant": dom}
