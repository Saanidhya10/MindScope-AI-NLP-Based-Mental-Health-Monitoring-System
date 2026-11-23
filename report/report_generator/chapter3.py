# chapter3.py
"""
Chapter 3: Problem Statement
Clear, rigorous engineering formulation. High burstiness, student perspective.
"""

CHAPTER_3_TITLE = "CHAPTER 3: PROBLEM STATEMENT"

SECTIONS = [
    ("3.1 Problem Definition and Context", """Contemporary computational emotion systems suffer from four distinct structural weaknesses that prevent their practical use in personal mental wellness monitoring:

First, the vast majority of existing sentiment tools evaluate text using coarse polarity labels: positive, negative, or neutral. This broad grouping obscures critical emotional differences. A person experiencing explosive anger requires completely different support than someone sinking into quiet despair or experiencing paralyzing social anxiety. Treating all three as generic 'Negative' text prevents meaningful personal insights.

Second, almost all natural language processing benchmarks evaluate models on isolated, single-message inputs. However, an isolated angry or sad comment is rarely a clinical red flag. Everyday life is filled with transient frustrations: a missed bus, a difficult exam, or a quarrel with a friend. These temporary emotional dips are entirely normal. What actually signals deteriorating psychological health is persistence: an unbroken downward spiral extending across days or weeks. Single-message classifiers trigger constant false alarms, causing alarm fatigue and driving users away.

Third, real-world conversational datasets suffer from extreme class imbalance. Neutral statements and mild conversational exchanges dominate public corpora by orders of magnitude, while acute negative emotions like grief, remorse, and intense anxiety are rare. Standard machine learning models trained on such data learn to predict the majority classes reliably while completely ignoring rare distress signals—precisely the signals that matter most.

Finally, existing mental health apps frequently present serious ethical and privacy vulnerabilities. Many commercial tools upload personal reflections to external cloud servers, run opaque generative language models that hallucinate unsafe advice, or make inappropriate pseudo-diagnostic claims that mislead vulnerable individuals.

There is a clear need for an integrated, privacy-first NLP system that classifies fine-grained emotional states, tracks emotional trajectories across time with mathematical smoothing, ignores transient noise, and provides transparent, non-diagnostic support while escalating persistent crises to qualified human care."""),

    ("3.2 Concrete Project Objectives and Deliverables", """To resolve the challenges outlined above, this project defines four concrete technical objectives:
1. Data Engineering & Harmonization: Construct a standardized, reproducible multi-class emotion benchmark of over 50,000 samples by mapping GoEmotions and Dreaddit into an eight-class affective spectrum, using strictly frozen stratified splits to guarantee reproducibility.
2. Multi-Model Baseline Benchmarking: Implement and evaluate candidate classification pipelines across linear (TF-IDF + Logistic Regression, Linear SVM, Random Forest), recurrent (Bi-LSTM), and transformer families (DistilBERT, RoBERTa), measuring performance through macro-averaged F1 scores and class-level recall.
3. Longitudinal Aggregation & Anomaly Filtering: Build a temporal processing layer combining daily affect aggregation, Exponential Moving Average (EMA) smoothing, and a multi-day persistence requirement to filter short-term emotional noise and detect genuine chronic distress.
4. Privacy-First Interactive Dashboard: Engineer a clean, two-graph Streamlit interface backed by local SQLite storage that records only aggregated affect metrics, explicitly destroys raw user text immediately following inference, and surfaces verified emergency hotlines (Tele-MANAS) during sustained crises."""),

    ("3.3 Scope and Delimitations", """To maintain ethical integrity and technical rigor, this project establishes the following explicit boundaries:
- Strictly Non-Diagnostic: The software does not diagnose, treat, or prevent any medical or psychological disorder. It operates exclusively as an emotional self-reflection aid.
- In-Scope Data: The pipeline handles English text inputs, including informal conversational prose, internet slang, abbreviations, contractions, and Unicode emojis commonly found in personal journals and student chats.
- Out-of-Scope Capabilities: The current release does not perform multimodal analysis (voice, facial video, biometric sensor streams), nor does it process non-English vernacular languages (e.g., Hindi or code-mixed Hinglish), which are marked for future releases.
- Zero Cloud Text Retention: Text inference runs entirely on the host environment; no user text is transmitted to external proprietary APIs or persisted in backend databases."""),

    ("3.4 Key Engineering Challenges", """Executing this framework presents several difficult technical hurdles:
- Extreme Class Skew: Addressing a 190:1 frequency ratio between majority Neutral comments and minority Grief/Remorse samples without inducing catastrophic false positive rates across other classes.
- Dataset-Source Confounding: Ensuring that models do not rely on superficial document-length shortcuts when discriminating between short GoEmotions comments and long Dreaddit stress narratives.
- Inference Latency vs. Representation Power: Balancing the rich contextual representations of transformer architectures against the sub-second inference requirements of a responsive local desktop dashboard.
- Temporal Parameter Calibration: Tuning the smoothing coefficient (alpha) and persistence threshold (k days) to avoid both hypersensitive false alarms and delayed crisis detection.""")
]
