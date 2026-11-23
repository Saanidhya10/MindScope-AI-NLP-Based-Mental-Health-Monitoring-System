# chapter4.py
"""
Chapter 4: Description of Various Modules
Detailed modular breakdown of the software engineering architecture.
High burstiness, authentic student engineering explanations.
"""

CHAPTER_4_TITLE = "CHAPTER 4: DESCRIPTION OF VARIOUS MODULES"

SECTIONS = [
    ("4.1 System Architecture Overview", """The system is architected into two cleanly separated pipelines: an Offline Model Training Pipeline and an Online Inference and Monitoring Pipeline.

The Offline Pipeline ingests raw public corpora, harmonizes disparate taxonomy labels into an eight-class affective schema, applies stratified data partitioning, isolates class-imbalance mitigations strictly to the training fold, and benchmarks candidate classification models. The best-performing model—selected based on validation macro-averaged F1—is serialized and deployed into the production runtime.

The Online Pipeline serves the user-facing web interface. When an individual submits a text reflection, the system tokenizes the input, extracts model-specific features, performs single-pass inference, computes calibrated emotion probabilities, passes the numerical scores to the temporal smoothing engine, records the derived metrics in a local SQLite store, and immediately purges the raw text from memory. If the smoothed negative affect metric breaches safety thresholds over multiple consecutive days, the recommendation engine transitions from personal wellness exercises to active crisis intervention, displaying national support contact information."""),

    ("4.2 Data Ingestion and Harmonization Module", """The data ingestion module bridges two distinct public datasets: GoEmotions and Dreaddit. Because these datasets were collected under different experimental settings, this module standardizes their formats into a uniform data schema.

Key engineering tasks executed by this module include:
- Label Mapping: GoEmotions provides 27 fine-grained emotion labels. We mapped these into seven core classes (Joy, Sadness, Anger, Fear, Anxiety, Surprise, Neutral) based on established psychological overlaps. Dreaddit contributed the eighth class: Stress.
- De-duplication and Text Sanitization: Raw comments were stripped of automated bot watermarks, deleted author tags, and broken URL strings. Duplicate entries within and across datasets were pruned to prevent data contamination.
- Stratified Partitioning: The combined corpus of 51,115 records was partitioned into an 80% training set (40,892 samples), a 10% validation set (5,111 samples), and a 10% test set (5,112 samples). We enforced strict stratification across all eight classes to ensure that rare minority classes maintained identical relative proportions in every fold. Partitions were frozen and persisted to disk in immutable Parquet format to guarantee full experimental reproducibility."""),

    ("4.3 Multi-View Pre-processing and Feature Extraction Module", """A common failure in text pipelines is applying a one-size-fits-all pre-processing routine across fundamentally different model families. For instance, aggressive lowercasing, stop-word removal, and stemming help high-dimensional bag-of-words models by collapsing vocabulary sparsity; however, they destroy the subtle grammatical signals that transformer models rely upon.

To resolve this, our module implements three distinct text views:
1. Classical Pipeline View: Text is lowercased; URL patterns and user handles are removed; common English stop-words are filtered out; but critical negation tokens ('not', 'no', 'never', 'cannot') are preserved via a custom negative whitelist. Contractions are expanded ('don\\'t' becomes 'do not'). Emojis are translated into descriptive English tokens using the emoji library (e.g., '😭' becomes 'loudly crying face') rather than stripped, preserving vital affective cues.
2. Sequence / Bi-LSTM View: Text undergoes sentence-boundary tokenization, is lowercased, truncated or padded to a fixed sequence length of 128 tokens, and mapped to 300-dimensional pre-trained GloVe embedding vectors (glove.6B.300d). Out-of-vocabulary tokens are mapped to a dedicated learned embedding token.
3. Transformer Tokenizer View: Raw text is passed directly into Hugging Face tokenizers (DistilBertTokenizerFast or RobertaTokenizerFast) using WordPiece or Byte-Pair Encoding (BPE). Case information, punctuation rhythms, and exclamation marks are retained to leverage subword contextual attention mechanisms."""),

    ("4.4 Model Development and Evaluation Module", """This module establishes a unified Object-Oriented interface for training, serializing, and querying multiple model architectures under identical evaluation conditions.

All models inherit from a common base class (`BaseEmotionClassifier`) implementing standard `.fit(X, y)` and `.predict_proba(X)` methods. This design pattern ensures that the downstream temporal engine and Streamlit UI remain completely decoupled from the specific underlying classification algorithm:
- Classical Models: Scikit-learn pipelines pairing sublinear TF-IDF vectorization (unigrams + bigrams, maximum vocabulary of 40,000 to 60,000 features) with Logistic Regression, Linear Support Vector Classifiers (LinearSVC with Platt scaling for probability estimates), and Random Forest ensembles.
- Recurrent Architecture: A PyTorch-based Bidirectional LSTM featuring an initial 300-d GloVe embedding layer, a 2-layer Bi-LSTM with 128 hidden units per direction, a dropout layer (p = 0.3) for regularization, and a dense linear classification head.
- Transformer Architectures: Pre-trained DistilBERT (distilbert-base-uncased) and RoBERTa (roberta-base) models fine-tuned with AdamW optimization, linear learning rate warm-up, and cross-entropy loss over three training epochs."""),

    ("4.5 Temporal Analysis and Anomaly Tracking Module", """The temporal module is the core algorithmic component that elevates our system from a simple text classifier into a longitudinal mental wellness monitoring tool. It tracks emotional dynamics over time, filtering transient emotional noise to isolate persistent negative trajectories.

The module operates according to a three-step mathematical formulation:
1. Daily Affect Aggregation: For any given day t containing n entries, the system computes the daily mean negative affect score A(t) by summing the predicted probabilities of Sadness, Anger, Fear, Anxiety, and Stress:
   A(t) = (1 / n) * sum_{i=1}^{n} P(negative_classes | text_i)
2. Exponential Moving Average (EMA) Smoothing: To prevent sudden one-day spikes from dominating the trajectory, the system updates a smoothed affective index S(t):
   S(t) = alpha * A(t) + (1 - alpha) * S(t-1)
   where alpha in (0, 1] represents the smoothing parameter (calibrated to 0.3 to balance responsiveness and stability).
3. Persistence Window and Escalation Logic: A warning state is declared only if the smoothed index S(t) exceeds a calibrated threshold tau (set to 0.55) for k consecutive days (set to 3 days):
   Condition = (S(t) > tau) AND (S(t-1) > tau) AND (S(t-2) > tau)
   If this condition holds, the system escalates the interface state to 'Chronic Distress', notifying the user of their sustained negative pattern and presenting clinical resources."""),

    ("4.6 Interactive Web Dashboard Module", """The user interface is engineered as a clean, responsive web application using Streamlit. To ensure an uncluttered, distraction-free environment for users in emotional distress, the interface adheres to strict visual design principles:
- Simplified Two-Graph Layout: Rather than overwhelming users with a dozen competing charts, the dashboard displays exactly two stable, high-value visualizations:
  1. Emotion Distribution (Polar / Donut chart showing relative balance of recent affective states).
  2. Mood Stability Trajectory (Line chart depicting smoothed daily affect against the baseline threshold across 7-day, 14-day, and 30-day windows).
- Header Hierarchy and Separation: All instructional and category headers are positioned prominently above content containers, ensuring that card borders and separator lines do not fragment visual reading flow.
- Non-Diagnostic Suggestions: When negative affect is elevated but below crisis thresholds, the dashboard surfaces gentle, evidence-based coping exercises (e.g., 4-7-8 breathing techniques, progressive muscle relaxation, journaling prompts).
- Zero Fluff: Experimental configuration sliders, developer debugging parameters, and synthetic profile selectors are completely isolated from the standard user interface."""),

    ("4.7 Privacy, Consent, and Safety Layer", """Data privacy and psychological safety are fundamental architectural requirements of this system:
- Ephemeral In-Memory Text Processing: User text submissions exist solely in transient memory during tokenization and model inference. Once the prediction vector is calculated, the raw string is garbage-collected. No text is written to disk, server logs, or local storage.
- Local SQLite Metric Store: The local database records only five fields: user_id, timestamp, predicted_primary_emotion, confidence_score, and composite_negative_affect_score. Even if the local database file were inspected, no personal reflections or journal entries can be recovered.
- Consent and Data Erasure: Users must explicitly toggle a consent switch before any affective scores are recorded. A dedicated 'Purge My Data' button allows instantaneous, irrevocable deletion of all historical records from the SQLite database.
- Crisis Escalation: If the persistence rule triggers an escalation, the interface displays prominent, clickable contact cards for verified national helplines:
  - Tele-MANAS (Government of India): Dial 14416 (24x7 toll-free)
  - KIRAN Mental Health Helpline: 1800-599-0019
  - Vandrevala Foundation Helpline: +91 9999 666 555""")
]
