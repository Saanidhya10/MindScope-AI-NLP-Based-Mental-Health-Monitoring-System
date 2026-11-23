# Change log: dashboard overhaul and model upgrade

Branch: `dashboard-overhaul`. Everything below was verified by running the app and taking screenshots
(empty profile, sample data, crisis entry, sustained distress, 390px phone) and by `pytest` (58 passing).

## 1. Model: pretrained GoEmotions RoBERTa replaces the TF-IDF baseline in the app

- New `goemotions` model type in `src/inference.py`, set as the default in `config.yaml -> inference`
  (`SamLowe/roberta-base-go_emotions`, about 500 MB, downloaded on first run).
- The 28 GoEmotions labels are collapsed to our 8 classes with the existing `label_mapping.goemotions`.
  Each class takes its strongest sub-label (max, not sum, so Joy's 11 sub-labels don't drown out Anxiety).
- If torch/transformers are missing or the download fails, it falls back to the local
  `logreg_baseline` with a warning.

GoEmotions test split, single-class rows (4,997), 7 classes:

| Model | Accuracy | Macro-F1 | Anxiety F1 |
|---|---|---|---|
| logreg_baseline (old) | 0.725 | 0.504 | 0.00 |
| roberta-base-go_emotions (new) | 0.746 | 0.648 | 0.54 |

The baseline was trained on splits cut from all of GoEmotions, so it may have seen some of these rows; its score
is an upper bound.

**Known limit:** the model never predicts Stress (GoEmotions has no stress label). Stressful text lands on
Sadness, Anxiety or Fear, which still count towards the difficult-feelings trend. This is not the project's own
trained model; the report still needs the fine-tuned DistilBERT (see "Next steps").

## 2. Merge: the app now uses `src/` directly

- Deleted `app/components.py`, which duplicated the predictor, database, trend and recommendation logic
  (two SQLite files, two sets of rules, helplines that differed from the config).
- Moved what was useful into `src/`:
  - `preprocess.py`: Hindi/Hinglish feeling-word mapping (`normalize_hinglish`), applied in `clean_basic`.
  - `recommend.py`: rewritten. Suggestions, support message, helplines and crisis phrases come from
    `config.yaml -> recommend`. `check_crisis()` and one `assess()` that returns a single status
    (`steady` / `watch` / `support`), so the page can no longer show conflicting verdicts.
  - `trends.py`: `daily_frame()` (daily aggregate + EMA + current high run), `high_runs()` (a gap longer than
    `max_gap_days` restarts the run), `personal_baseline()`.
  - `sample_journal.py` (new): the invented three-week journal, shared by `scripts/demo.py` and the app's
    "Add sample entries" button.
- Consent is now stored per profile in the database (`MoodDatabase.set_consent`) instead of resetting each session.

## 3. Bugs fixed

- `config.yaml` was missing the sections that `src/train_bilstm.py`, `src/train_transformer.py` and
  `src/inference.py` read (`bilstm`, `transformer`, `seeds`), so training crashed with a KeyError.
- `config.yaml` was missing `trends.alpha/tau/k`, and `trends.simulation` didn't match what the code reads.
- `src/preprocess.py` read `cfg["preprocess"]`; the config key is `preprocessing`.
- `scripts/demo.py` imported a non-existent `MoodDB` and called functions that don't exist. Rewritten against
  the real API; `python scripts/demo.py` and `--text "..."` both run.
- Mixed timestamp formats (with/without microseconds) crashed history loading. The database now stores
  seconds precision, and `build_prediction_frame` parses any ISO-8601 string.
- After a crisis message the CLI demo still printed a cheerful suggestion. It now shows only the helplines.

## 4. Dead code removed

- About 20 alias functions (`classify`, `predict_emotion`, `suggest`, `ema`, `neg_score`, `clean`, ...) that existed only
  because the tests looked functions up under several names. The tests now call the real names.
- `trends.mood_summary`, `recommend.recommend_from_summary`, the old `Recommendation` dataclass, `add_demo_history`.
- About 500 lines of unused CSS in `app/ui.py`.
- Config keys nothing read: `trends.min_confidence/sentiment_scores/windows/dominant_emotion_method`,
  the old `recommend.escalation` block, the `privacy` flags (privacy is enforced in `src/database.py`), and most of
  `app`. KIRAN helpline removed (status uncertain; a dead number is harmful in a crisis).

## 5. Dashboard redesign (`app/app.py`, `app/ui.py`, `.streamlit/config.toml`)

- Light, calm palette (mist `#EEF2F1`, white writing surface, slate ink `#18262B`, lake blue `#2D5F73`)
  replacing neon violet on near-black. Literata for the journal voice (title, headings, the writing box,
  status sentences); Lexend for the interface.
- **Today's entry**: opt-in consent, a serif writing box, "Analyse entry". The result reads in plain words
  ("Mostly sadness, with some anxiety"), notes how sure the model is, and shows all 8 scores as bars.
- **Crisis messages** show helplines first, whatever the model predicts.
- **The last four weeks**: a ribbon with one coloured block per day by main feeling.
- **How you've been**: one status card with one suggestion. Sustained distress adds the helplines.
- **Emotion mix** donut (last 30 days) and **difficult feelings over time** (day / week / month), with
  "your usual" and "support line" markers.
- Fixed the empty boxes and raw `**markdown**` from the old split-`<div>` layout. Works at phone width, respects
  reduced motion, and the Deploy button is hidden.
- Sidebar: profile, trend grouping, CSV export, delete with confirmation, and "Add sample entries" for empty profiles.

## 6. Tests

- New `tests/test_dashboard_logic.py` (7 tests): run counting and gap reset, baseline calibration,
  `assess()` levels, crisis phrases, non-diagnostic wording of every configured suggestion.
- Existing tests point at canonical function names. 58 passing.

## How to run

```bash
pip install -r requirements.txt          # includes torch + transformers
streamlit run app/app.py                 # from the repo root
python scripts/demo.py                   # CLI walkthrough of the sample journal
pytest -q
```

## Next steps

1. Fine-tune DistilBERT on the project's own splits (`scripts/setup_data.py`, then `scripts/train_all.py`), add
   short stress examples so Stress isn't learned only from long Dreaddit posts, and set `inference.kind: transformer`.
2. Fix `scripts/run_experiments.py`, which calls `trends.trend_quality` (doesn't exist), then fill in the
   results tables in `results/` and the README.
3. Verify every helpline number in `config.yaml` (all marked `verified: false`).
