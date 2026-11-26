"""SQLite mood-history store (the synopsis lists MongoDB/MySQL/Firebase as options; SQLite needs no server).

Privacy by design (paper, Sec. 10.2):
  * nothing is stored without explicit, revocable consent
  * by default only derived scores + timestamps are kept, never raw text
  * users can view, export and delete their own data
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.trends import PROB_COLS, build_prediction_frame, negative_scores
from src.utils import ID2LABEL, LABELS, load_config, resolve_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS consent (user_id TEXT PRIMARY KEY, granted INTEGER NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    emotion TEXT NOT NULL,
    confidence REAL NOT NULL,
    negative_score REAL NOT NULL,
    probs TEXT NOT NULL,
    text TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_user_time ON predictions(user_id, timestamp);
CREATE TABLE IF NOT EXISTS checkins (          -- daily self-reported wellness check-in
    user_id TEXT NOT NULL, day TEXT NOT NULL,
    sleep_hours REAL, activity_min REAL, screen_hours REAL, social INTEGER,
    workload INTEGER, energy INTEGER, mood INTEGER, stress INTEGER,
    PRIMARY KEY (user_id, day)
);
CREATE TABLE IF NOT EXISTS activity_log (      -- completed wellness activities and how they felt
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, timestamp TEXT NOT NULL,
    activity TEXT NOT NULL, mood_before INTEGER, mood_after INTEGER, helpful INTEGER
);
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, created TEXT NOT NULL,
    title TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS notes (             -- gratitude / thought journals: only with explicit opt-in
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, timestamp TEXT NOT NULL,
    kind TEXT NOT NULL, body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecasts (         -- saved outlooks, compared later with what actually happened
    user_id TEXT NOT NULL, made_on TEXT NOT NULL, target_day TEXT NOT NULL,
    predicted REAL NOT NULL, low REAL NOT NULL, high REAL NOT NULL,
    PRIMARY KEY (user_id, made_on, target_day)
);
"""
USER_TABLES = ["predictions", "checkins", "activity_log", "goals", "notes", "forecasts"]
CHECKIN_FIELDS = ["sleep_hours", "activity_min", "screen_hours", "social", "workload", "energy", "mood", "stress"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MoodDatabase:
    def __init__(self, path: str | Path | None = None, cfg: dict | None = None):
        if path is None:
            cfg = cfg or load_config()
            path = resolve_path(cfg["database"]["path"])
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn = sqlite3.connect(":memory:") if str(path) == ":memory:" else None
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(predictions)")}
            if "triggers" not in cols:  # added after the first release: detected trigger categories (JSON list)
                c.execute("ALTER TABLE predictions ADD COLUMN triggers TEXT")

    @contextmanager
    def _conn(self):
        conn = self._memory_conn or sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            if not self._memory_conn:
                conn.close()

    # ---- consent -----------------------------------------------------------------
    def set_consent(self, user_id: str, granted: bool, delete_data_on_revoke: bool = False) -> None:
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO users VALUES (?, ?)", (user_id, _now()))
            c.execute("INSERT INTO consent VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET "
                      "granted=excluded.granted, updated_at=excluded.updated_at", (user_id, int(granted), _now()))
        if not granted and delete_data_on_revoke:
            self.delete_user_data(user_id, keep_user=True)

    def has_consent(self, user_id: str) -> bool:
        with self._conn() as c:
            row = c.execute("SELECT granted FROM consent WHERE user_id=?", (user_id,)).fetchone()
        return bool(row and row[0])

    # ---- write -------------------------------------------------------------------
    def add_prediction(self, user_id: str, probs, timestamp=None, text: str | None = None,
                       store_text: bool = False, triggers: list[str] | None = None) -> None:
        """Store one prediction. Raises PermissionError without consent. Raw text only if store_text=True.
        `triggers` are category keys from src.triggers (topics, not text)."""
        if not self.has_consent(user_id):
            raise PermissionError(f"User '{user_id}' has not granted consent; nothing was stored.")
        probs = np.asarray(probs, dtype=float).ravel()
        ts = pd.Timestamp(timestamp or datetime.now()).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute(
                "INSERT INTO predictions (user_id, timestamp, emotion, confidence, negative_score, probs, text, "
                "triggers) VALUES (?,?,?,?,?,?,?,?)",
                (user_id, ts, ID2LABEL[int(probs.argmax())], float(probs.max()),
                 float(negative_scores(probs[None, :])[0]), json.dumps(probs.tolist()),
                 text if (store_text and text) else None, None if triggers is None else json.dumps(triggers)))

    def get_triggers(self, user_id: str) -> pd.DataFrame:
        """One row per entry that has trigger information: timestamp, emotion, negative_score, triggers (list)."""
        with self._conn() as c:
            df = pd.read_sql("SELECT timestamp, emotion, negative_score, triggers FROM predictions "
                             "WHERE user_id=? AND triggers IS NOT NULL ORDER BY timestamp", c, params=(user_id,))
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        df["triggers"] = df["triggers"].map(json.loads)
        return df

    # ---- read --------------------------------------------------------------------
    def get_predictions(self, user_id: str, since=None) -> pd.DataFrame:
        """Message-level frame in the format expected by trends.aggregate / mood_summary."""
        q, args = "SELECT timestamp, probs FROM predictions WHERE user_id=?", [user_id]
        if since is not None:
            q += " AND timestamp >= ?"
            args.append(pd.Timestamp(since).isoformat())
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY timestamp", args).fetchall()
        if not rows:
            return pd.DataFrame(columns=["timestamp", *PROB_COLS, "neg_score", "emotion", "confidence"])
        return build_prediction_frame([r[0] for r in rows], [json.loads(r[1]) for r in rows], user_id=user_id)

    def export_user_data(self, user_id: str) -> dict:
        with self._conn() as c:
            cons = c.execute("SELECT granted, updated_at FROM consent WHERE user_id=?", (user_id,)).fetchone()
            preds = c.execute("SELECT timestamp, emotion, confidence, negative_score, probs, text "
                              "FROM predictions WHERE user_id=? ORDER BY timestamp", (user_id,)).fetchall()
        return {
            "user_id": user_id,
            "consent": {"granted": bool(cons[0]), "updated_at": cons[1]} if cons else None,
            "predictions": [
                {"timestamp": t, "emotion": e, "confidence": cf, "negative_score": ns,
                 "probabilities": dict(zip(LABELS, json.loads(p))), "text": tx}
                for t, e, cf, ns, p, tx in preds],
        }

    def delete_user_data(self, user_id: str, keep_user: bool = False) -> int:
        with self._conn() as c:
            n = c.execute("DELETE FROM predictions WHERE user_id=?", (user_id,)).rowcount
            for table in USER_TABLES[1:]:
                c.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
            if not keep_user:
                c.execute("DELETE FROM consent WHERE user_id=?", (user_id,))
                c.execute("DELETE FROM users WHERE user_id=?", (user_id,))
        return n

    # ---- daily check-ins ---------------------------------------------------------
    def save_checkin(self, user_id: str, day, values: dict) -> None:
        """Insert or replace one day's check-in. Raises PermissionError without consent."""
        if not self.has_consent(user_id):
            raise PermissionError(f"User '{user_id}' has not granted consent; nothing was stored.")
        row = [values.get(f) for f in CHECKIN_FIELDS]
        with self._conn() as c:
            c.execute(f"INSERT OR REPLACE INTO checkins (user_id, day, {', '.join(CHECKIN_FIELDS)}) "
                      f"VALUES (?, ?, {', '.join('?' * len(CHECKIN_FIELDS))})",
                      [user_id, pd.Timestamp(day).date().isoformat(), *row])

    def get_checkins(self, user_id: str) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql("SELECT * FROM checkins WHERE user_id=? ORDER BY day", c, params=(user_id,))
        df["day"] = pd.to_datetime(df["day"])
        return df.drop(columns=["user_id"]).set_index("day")

    # ---- wellness activities, goals and journal notes --------------------------------
    def _require_consent(self, user_id: str) -> None:
        if not self.has_consent(user_id):
            raise PermissionError(f"User '{user_id}' has not granted consent; nothing was stored.")

    def log_activity(self, user_id: str, activity: str, mood_before: int | None, mood_after: int | None,
                     helpful: bool | None, timestamp=None) -> None:
        self._require_consent(user_id)
        ts = pd.Timestamp(timestamp or datetime.now()).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("INSERT INTO activity_log (user_id, timestamp, activity, mood_before, mood_after, helpful) "
                      "VALUES (?,?,?,?,?,?)", (user_id, ts, activity, mood_before, mood_after,
                                               None if helpful is None else int(helpful)))

    def get_activity_log(self, user_id: str) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT timestamp, activity, mood_before, mood_after, helpful FROM activity_log "
                               "WHERE user_id=? ORDER BY timestamp", c, params=(user_id,))

    def add_goal(self, user_id: str, title: str) -> None:
        self._require_consent(user_id)
        with self._conn() as c:
            c.execute("INSERT INTO goals (user_id, created, title) VALUES (?,?,?)", (user_id, _now(), title.strip()))

    def get_goals(self, user_id: str) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT id, created, title, done FROM goals WHERE user_id=? ORDER BY done, id",
                               c, params=(user_id,))

    def set_goal_done(self, user_id: str, goal_id: int, done: bool) -> None:
        with self._conn() as c:
            c.execute("UPDATE goals SET done=? WHERE id=? AND user_id=?", (int(done), goal_id, user_id))

    def delete_goal(self, user_id: str, goal_id: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM goals WHERE id=? AND user_id=?", (goal_id, user_id))

    def add_note(self, user_id: str, kind: str, body: str) -> None:
        """Gratitude / thought journal text. Only called when the user has opted in to saving notes."""
        self._require_consent(user_id)
        with self._conn() as c:
            c.execute("INSERT INTO notes (user_id, timestamp, kind, body) VALUES (?,?,?,?)",
                      (user_id, _now(), kind, body))

    def get_notes(self, user_id: str, kind: str) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT id, timestamp, body FROM notes WHERE user_id=? AND kind=? ORDER BY id DESC",
                               c, params=(user_id, kind))

    def list_users(self) -> list[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT user_id FROM users ORDER BY user_id")]
