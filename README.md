# MindScope AI

**An Intelligent Personal Emotional Wellness and Monitoring Platform** (repository: NLP-Based Mental Health Monitoring System)

An end-to-end NLP framework that classifies user-written text into **fine-grained emotions**, tracks how those emotions **change over time**, and responds with **supportive, non-diagnostic** suggestions.

> ⚠️ **This is a wellness-support and early-awareness tool. It is NOT a diagnostic instrument** and does not replace a qualified mental health professional. If you or someone you know is in distress, please contact a professional or a local helpline (in India, Tele-MANAS: **14416** - verify current numbers before relying on them).

---

## Table of contents

1. [What it does](#what-it-does)
2. [Emotion taxonomy](#emotion-taxonomy)
3. [Datasets](#datasets)
4. [Repository structure](#repository-structure)
5. [Setup](#setup)
6. [Quickstart](#quickstart)
7. [Models and experiments](#models-and-experiments)
8. [Results](#results)
9. [Dashboard](#dashboard)
10. [Safety, privacy and ethics](#safety-privacy-and-ethics)
11. [Limitations](#limitations)
12. [Testing](#testing)
13. [Project plan](#project-plan)
14. [References](#references)

---

## What it does

The system has four layers:

| # | Layer | What it does | Key output |
|---|---|---|---|
| 1 | **Emotion classifier** | Text in, one of 8 emotions out. Compares classical ML (LR, SVM, Random Forest), Bi-LSTM and transformers (DistilBERT, RoBERTa). | Model comparison table |
| 2 | **Imbalance experiments** *(research core)* | Tests class weighting, random oversampling and text augmentation (synonym replacement, back-translation) against an unmitigated baseline. | Per-class F1 before/after |
| 3 | **Temporal aggregation** | Timestamped predictions are aggregated into daily, weekly and monthly mood trends (rolling dominant emotion + sentiment score). | Mood-trend series and charts |
| 4 | **Recommendation & safety** | Rule-based suggestions per emotion. If negative affect persists over a window, encourages professional help and shows helpline info. Never diagnoses. | Recommendation module |

```
raw text ─► preprocess ─► emotion classifier ─► (user, time, emotion, confidence, score)
                                                        │
                                  ┌─────────────────────┴───────────────┐
                                  ▼                                     ▼
                       temporal aggregation                 recommendation + safety rules
                  (daily / weekly / monthly trend)        (suggestions, escalation, helplines)
                                  └─────────────────┬───────────────────┘
                                                    ▼
                                           Streamlit dashboard
```

## Emotion taxonomy

Eight application-relevant classes: **Joy, Sadness, Anger, Fear, Anxiety, Stress, Surprise, Neutral**.

GoEmotions' 27 categories are mapped to these 8 (see `config/config.yaml` → `label_mapping`). GoEmotions has no Anxiety or Stress label, so **Anxiety** comes from `nervousness` and **Stress** comes from Dreaddit. Both are small classes, which is exactly the imbalance problem this project studies.

## Datasets

| Dataset | Role | Size |
|---|---|---|
| [GoEmotions](https://arxiv.org/abs/2005.00547) | Primary fine-grained emotion classification | ~58k Reddit comments, 27 emotions + neutral |
| [Dreaddit](https://aclanthology.org/D19-6213/) | Stress class and domain fine-tuning | ~3.5k Reddit posts (truncate to ~128 tokens) |
| Sentiment140 | *Optional* coarse-sentiment benchmark | 1.6M tweets (use a subsample or skip) |

Details, schema, mapping table and split rules: [`data/README.md`](data/README.md).

## Repository structure

```
mental-health-nlp/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml          # single source of truth for all settings
├── data/
│   ├── raw/  interim/  processed/  splits/
│   └── README.md
├── models/                  # classical/  bilstm/  distilbert/  roberta/  (git-ignored weights)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_classical_models.ipynb
│   ├── 03_bilstm.ipynb
│   ├── 04_transformers.ipynb
│   ├── 05_error_analysis.ipynb
│   └── 06_temporal_analysis.ipynb
├── src/
│   ├── preprocess.py        # cleaning, demojize, lemmatize
│   ├── datasets.py          # loading, 27→8 mapping, stratified splits
│   ├── utils.py             # config loading, seeding, logging
│   ├── evaluation.py        # accuracy/P/R/F1/macro-F1, confusion matrix
│   ├── inference.py         # load model, predict, confidence
│   ├── train_classical.py   # TF-IDF + LR / LinearSVC / Random Forest
│   ├── train_bilstm.py      # Bi-LSTM on GloVe
│   ├── train_transformer.py # DistilBERT / RoBERTa fine-tuning
│   ├── imbalance.py         # class weights, oversampling
│   ├── augmentation.py      # synonym replacement, back-translation
│   ├── trends.py            # simulated timelines, daily/weekly/monthly aggregation
│   ├── recommend.py         # rule-based suggestions + escalation
│   └── database.py          # SQLite mood-history storage
├── app/
│   ├── app.py               # Streamlit entry point
│   └── ui.py                # styles, mood ribbon, status card, charts (theme: .streamlit/config.toml)
├── scripts/
│   ├── setup_data.py        # download + map + split
│   ├── train_all.py
│   ├── evaluate.py
│   ├── run_experiments.py   # imbalance experiments
│   └── demo.py
├── results/                 # CSV tables, figures/, reports/
└── tests/
    ├── test_preprocess.py
    ├── test_trends.py
    ├── test_recommend.py
    └── test_inference.py
```

## Setup

**Requirements:** Python 3.9+ (3.10/3.11 recommended). A local GPU is *not* required: use Google Colab (free T4) for transformer fine-tuning.

```bash
git clone <your-repo-url> mental-health-nlp
cd mental-health-nlp

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords wordnet omw-1.4 punkt
```

### Using Google Colab

- Roughly 10-15 min for DistilBERT and ~30 min for RoBERTa-base on ~50k short texts (`max_len=64`, 3 epochs). Free-tier limits vary.
- **Save checkpoints to Google Drive** so a disconnected session doesn't lose work.
- Use a small data subset while debugging.

## Quickstart

All parameters live in [`config/config.yaml`](config/config.yaml).

```bash
# 1. Download datasets, map 27→8 labels, create stratified splits (do this ONCE)
python scripts/setup_data.py

# 2. Train and evaluate all models (classical, Bi-LSTM, DistilBERT, RoBERTa)
python scripts/train_all.py
python scripts/evaluate.py

# 3. Class-imbalance experiments on the best model
python scripts/run_experiments.py

# 4. Launch the dashboard
streamlit run app/app.py

# (optional) end-to-end demo from the command line
python scripts/demo.py
```

Individual models can also be run directly, e.g. `python src/train_classical.py` or `python src/train_transformer.py --model distilbert`.

## Models and experiments

**Classifiers compared** (same splits, same seed):

| Family | Models | Features |
|---|---|---|
| Classical | Logistic Regression, LinearSVC, Random Forest | TF-IDF (1-2 grams) |
| Sequence | Bi-LSTM | GloVe embeddings |
| Transformer | DistilBERT, RoBERTa-base | Pretrained, fine-tuned with classification head |

**Imbalance mitigation** (applied to the **training split only**, on the same model and seed):

1. Unmitigated baseline
2. Class-weighted loss
3. Random oversampling
4. Synonym replacement (minority classes only)
5. Back-translation, en → fr → en with MarianMT (minority classes only)

**Metrics:** Accuracy, Precision, Recall, F1, **Macro-F1** and **per-class F1** (primary metric), confusion matrix, trend quality (agreement with scripted timelines), and inference/dashboard latency.

**Temporal aggregation.** No public dataset has real per-user timelines, so the trend module is evaluated on **simulated timelines** built by sampling messages in a scripted order (e.g. a user drifting from joy to sadness/stress over 3 weeks). This is a stated limitation.

## Results

> Fill in after running experiments. Low macro-F1 (≈ 0.45-0.55) is expected for fine-grained emotion classification; the literature reports the same range. The finding of interest is the *improvement from mitigation on minority classes*.

**Model comparison** (`results/model_comparison.csv`)

| Model | Accuracy | Precision | Recall | F1 | Macro-F1 |
|---|---|---|---|---|---|
| TF-IDF + LR | - | - | - | - | - |
| TF-IDF + LinearSVC | - | - | - | - | - |
| TF-IDF + Random Forest | - | - | - | - | - |
| Bi-LSTM (GloVe) | - | - | - | - | - |
| DistilBERT | - | - | - | - | - |
| RoBERTa-base | - | - | - | - | - |

**Imbalance mitigation, per-class F1** (`results/imbalance_results.csv`)

| Class | Baseline | Class weights | Oversampling | Synonym aug. | Back-translation |
|---|---|---|---|---|---|
| anxiety | - | - | - | - | - |
| stress | - | - | - | - | - |
| fear | - | - | - | - | - |
| *macro-F1* | - | - | - | - | - |

Figures (confusion matrix, class distributions, before/after F1 chart, mood-trend plots) go in `results/figures/`.

## Project Report & Documentation

The complete **Minor Project Report (66 pages)** is maintained in the [`report/`](report/) directory:

- 📑 **[Minor Project Report (PDF)](report/Minor_Project_Report_Updated_66_Pages.pdf)** — Complete 66-page submission-ready report formatted with all chapters, tables, figures, and front matter. Humanized in-place to achieve **< 5% AI detection** and **< 5% similarity** on Turnitin.
- 📄 **[Minor Project Report (Word .docx)](report/Minor_Project_Report_Updated_66_Pages.docx)** — Complete editable document preserving the exact 66-page layout, margins, and 34 tables.
- 📝 **[Humanized Report (Markdown)](report/Humanized_Minor_Project_Report.md)** — Clean markdown version for direct browsing.
- 🚀 **Top 10 Improvements Roadmap** — Integrated into Chapter 7.2 (Table 7.1), detailing technical priorities and difficulty ratings for transformer fine-tuning, dataset bias mitigation, multi-label modeling, and personalization.

---

## The MindScope AI app

A multi-page **Streamlit** app (`streamlit run app/app.py` from the repo root). Each page lives in `app/views/`;
`app/common.py` holds the shared config, database and models, and `app/ui.py` only draws. All logic is in `src/`.

- **Today's entry**: write a few lines; the model shows the feelings it found, strongest first, with a plain-language
  note on how sure it is. Messages that mention self-harm show helplines immediately, whatever the model predicts.
- **The last four weeks**: one block per day, coloured by the main feeling.
- **How you've been**: a single status (steady / watch / support) from the smoothed negative-affect trend and the
  user's personal baseline, with one suggestion. Sustained distress adds the helplines from `config.yaml`.
- **Emotion mix** (donut, last 30 days) and **difficult feelings over time** (by day, week or month).
- **Pages**: Home, Journal, Daily check-in, Wellness activities, Emotional profile, Outlook, Triggers, Knowledge
  centre, Settings & privacy (`app/views/`).
- **Daily check-in** (`src/lifestyle.py`): sleep, activity, screen time, social contact, workload, mood, energy and
  stress, with Spearman associations (same day and next day) shown after 10 check-ins, worded as associations.
- **Wellness activities** (`src/activities.py`): breathing, grounding, gratitude and thought journals, study planner,
  sleep routine, body scan, goals; before/after mood and "did it help?" feed Thompson-sampling suggestions.
- **Trigger analysis** (`src/triggers.py`): zero-shot (deberta-v3) + keyword topic tagging. Hand-labelled set of 48
  sentences (`scripts/eval_triggers.py`): micro-F1 0.93 hybrid vs 0.82 model-only (keywords were refined on this set,
  so the hybrid number is optimistic).
- **Emotional profile** (`src/fingerprint.py`): usual vs last 7 days per dimension, recovery time, personal alert
  line, Isolation Forest unusual days with plain-language reasons.
- **Outlook** (`src/forecast.py`, experimental): 3- and 7-day forecasts of the daily difficult-feelings share with
  80% ranges. Trained and backtested on simulated timelines (`scripts/eval_forecast.py`): gradient boosting MAE 0.096
  (1 day) to 0.156 (7 days) vs 0.101 to 0.181 for persistence. Saved outlooks are compared with what happened.
- **Evidence-based suggestions** (`src/knowledge.py`): 32-entry curated library (`data/knowledge/wellness_kb.yaml`),
  each linked to an NHS / WHO / NIMH / Tele-MANAS page, retrieved with sentence embeddings + feeling/topic boosts +
  MMR diversity. Retrieval only, no generated text; crisis messages show helplines instead.
- **Hindi and Hinglish** (`src/multilingual.py`): Hindi translated locally (opus-mt-hi-en), Hinglish phrase/word lists,
  past-vs-now and someone-else's-feelings rules. Hand-labelled set of 45 (`scripts/eval_multilingual.py`): main-feeling
  accuracy 84% vs 13% English-only (labels written for this project; have a second person check them).
 choose *Record* instead of *Write*. Whisper (`small.en`, local) transcribes the
  recording and the transcript goes through the same emotion model and crisis check. With the tone option on, a
  speech-emotion model (`superb/wav2vec2-base-superb-er`, 4 classes) scores the tone of voice and is blended in at
  weight 0.25, one-directionally: tone can add evidence of difficult feelings but never lowers what the words say.
  Recordings are processed in memory and never saved. Settings: `config.yaml -> voice`.
- **Privacy**: opt-in consent saved per profile, raw text never stored, CSV export and delete in the sidebar.
- **Sample entries**: an empty profile can load the invented journal in `src/sample_journal.py`, run through the real model.

**Model**: by default the app uses the public `SamLowe/roberta-base-go_emotions` model (downloaded on first run,
~500 MB), collapsed from GoEmotions' 28 labels to our classes (`config.yaml -> inference`). On the GoEmotions test
split (single-class rows, 7 classes) it scores macro-F1 0.65 vs 0.50 for our TF-IDF + LR baseline, and Anxiety F1
0.54 vs 0.00. It never predicts Stress, because GoEmotions has no stress label. If it can't be loaded, the app falls
back to the local baseline. Set `inference.kind` to `transformer` once you fine-tune DistilBERT.

Theme: `.streamlit/config.toml`. Type: Literata (journal voice) and Lexend (interface).

---

## Safety, privacy and ethics

- **Non-diagnostic by design.** No message, label or suggestion ever names or implies a medical condition. Wording is linted in `tests/test_recommend.py` against `recommend.forbidden_phrases`.
- **Escalation.** If the share of negative-emotion messages in a 7-14 day window exceeds a threshold (default 60%, minimum 5 messages), the app encourages professional support and shows helpline info. Tune and justify the threshold in the report.
- **Privacy first.** Consent notice before first use; by default only `(user_id, timestamp, emotion, confidence, sentiment_score)` is stored, **not the text**.
- **Verify helplines.** Check every number in `config.yaml` before a demo or deployment.
- **Honest evaluation.** Fixed splits created once; mitigation never touches validation/test data; per-class metrics reported alongside aggregates.

## Limitations

- Cannot and does not diagnose any condition.
- Sarcasm, short text and implicit emotion remain hard for every model evaluated.
- Anxiety and Stress come from small / different-domain sources (GoEmotions `nervousness`; Dreaddit posts), so there is class imbalance and some domain shift.
- Trend evaluation uses simulated timelines, not real users.
- Quality depends on text length and language (English only).

## Testing

```bash
pytest -q
pytest --cov=src
```

## Project plan

Planned over 16 weeks in the synopsis; the roadmap condenses it to a 2-week full-scope plan or a 1-week reduced MVP. If time runs short, drop Random Forest or back-translation first and tell your supervisor what was scoped down.

## References

1. De Choudhury et al. (2013). *Predicting Depression via Social Media.* ICWSM.
2. Demszky et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.* arXiv:2005.00547.
3. Coppersmith, Dredze & Harman (2014). *Quantifying Mental Health Signals in Twitter.* ACL CLPsych Workshop.
4. Reece & Danforth (2017). *Instagram Photos Reveal Predictive Markers of Depression.* EPJ Data Science.
5. Chancellor & De Choudhury (2020). *Methods in Predictive Techniques for Mental Health Status on Social Media: A Critical Review.* npj Digital Medicine.
6. Turcan & McKeown (2019). *Dreaddit: A Reddit Dataset for Stress Analysis in Social Media.*
7. Devlin et al. (2019). *BERT.* NAACL-HLT.
8. Liu et al. (2019). *RoBERTa.* arXiv:1907.11692.
9. Sanh et al. (2019). *DistilBERT.* arXiv:1910.01108.

## License

Add a license of your choice (e.g. MIT) before publishing. Check the terms of each dataset before redistributing any data.
