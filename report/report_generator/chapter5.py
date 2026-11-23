# chapter5.py
"""
Chapter 5: Methodology Adopted
Detailed algorithmic formulations, data mapping, training hyperparameters, and experimental protocols.
High burstiness, authentic student engineering explanations.
"""

CHAPTER_5_TITLE = "CHAPTER 5: METHODOLOGY ADOPTED"

SECTIONS = [
    ("5.1 Dataset Selection and Schema Harmonization", """The foundation of this project rests upon curating an affective dataset that captures both everyday conversational sentiment and acute psychological distress. To achieve this, we combined two distinct public data sources: GoEmotions (Demszky et al., 2020) and Dreaddit (Turcan & McKeown, 2019).

GoEmotions contains 58,008 curated Reddit comments originally annotated across 27 distinct emotion categories plus a neutral baseline. While linguistically rich, many categories exhibit high semantic overlap (e.g., 'annoyance' vs. 'anger', 'nervousness' vs. 'fear', 'grief' vs. 'sadness') and sparse representation. Dreaddit comprises 3,553 long-form posts collected from five acute stress support communities (r/anxiety, r/ptsd, r/stress, r/financial_hardship, r/almosthomeless).

We constructed a mapping function that consolidates these sources into eight distinct affective classes:
- Joy: Mapped from GoEmotions 'admiration', 'amusement', 'approval', 'caring', 'desire', 'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', and 'relief'.
- Sadness: Mapped from GoEmotions 'sadness', 'disappointment', 'grief', and 'remorse'.
- Anger: Mapped from GoEmotions 'anger', 'annoyance', and 'disapproval'.
- Fear: Mapped from GoEmotions 'fear'.
- Anxiety: Mapped from GoEmotions 'nervousness'.
- Surprise: Mapped from GoEmotions 'surprise' and 'realization'.
- Neutral: Mapped from GoEmotions 'neutral' and 'curiosity'.
- Stress: Sourced directly from Dreaddit posts annotated with acute stress markers.

The consolidated corpus totals 51,115 messages. We partitioned this corpus into strictly frozen splits: 80% training (40,892 records), 10% validation (5,111 records), and 10% test (5,112 records), stratified across all eight classes."""),

    ("5.2 Multi-Stage Pre-processing Protocol", """Text preprocessing was tailored to preserve affective nuance while standardizing lexical representations across candidate model architectures.

The pipeline applies a multi-step transformation:
1. Normalization and Cleaning: URLs, website links, and Reddit user handles ('u/username') were removed using regular expressions. HTML entities (e.g., '&amp;', '&lt;') were unescaped.
2. Contraction Expansion: Contractions carry vital sentiment indicators. For example, 'don\\'t like' carries completely opposite sentiment from 'like'. We expanded all standard English contractions (e.g., 'can\\'t' -> 'cannot', 'won\\'t' -> 'will not', 'isn\\'t' -> 'is not') using a deterministic lookup map.
3. Negation Whitelisting: Standard stop-word removal algorithms strip out negation terms ('no', 'not', 'never', 'none', 'cannot'). We constructed a custom stop-word filter that explicitly preserves all negation words, appending a negation marker to subsequent tokens within a three-word window (e.g., 'not happy' -> 'not_happy').
4. Emoji Preservation and Translation: In modern digital communication, emojis frequently carry the primary emotional payload. Rather than discarding Unicode symbols, our pipeline parses emojis into descriptive English phrases using the emoji library (e.g., '😊' -> 'smiling face with smiling eyes', '💔' -> 'broken heart').
5. Sublinear TF-IDF Vectorization: For classical classifiers, text was converted into n-gram features (unigrams and bigrams). We applied sublinear term-frequency scaling (replace tf with 1 + log(tf)) to reduce the distorting influence of excessively repeated words, setting maximum feature dimensionality between 40,000 and 60,000 features with minimum document frequency min_df = 2."""),

    ("5.3 Candidate Model Architectures and Mathematical Formulations", """We benchmarked six distinct model architectures spanning three model families:

1. TF-IDF + Logistic Regression (Multinomial):
   Models the conditional probability distribution over K classes using the softmax function:
   P(y = k | x) = exp(w_k^T x + b_k) / sum_{j=1}^{K} exp(w_j^T x + b_j)
   Trained via cross-entropy loss with L2 regularization (C = 1.0) using the L-BFGS optimization algorithm.

2. TF-IDF + Linear Support Vector Classifier (Linear SVM):
   Optimizes a one-vs-rest hinge loss function to find the maximum-margin hyperplane separating each class from all others:
   min_{w_k, b_k} (1/2) ||w_k||^2 + C * sum_{i=1}^{N} max(0, 1 - y_{ik} (w_k^T x_i + b_k))
   Calibrated posterior probabilities P(y = k | x) are estimated using Platt scaling (fitting a sigmoid over SVM decision margins).

3. TF-IDF + Random Forest Classifier:
   An ensemble of 100 decision trees constructed via bootstrap aggregating (bagging) with random feature subspace sampling. Tree splits are selected by minimizing Gini impurity:
   Gini(p) = 1 - sum_{k=1}^{K} p_k^2

4. Bidirectional LSTM (Bi-LSTM):
   A deep recurrent neural network that processes sequential text in both forward and backward directions. Word tokens are embedded into 300-dimensional dense vectors initialized with pre-trained GloVe embeddings (glove.6B.300d). The forward and backward hidden state vectors are concatenated at each time step:
   h_t = [h_t^{forward}; h_t^{backward}]
   A temporal max-pooling layer extracts the most salient hidden representations across the sequence, which are fed into a 128-unit dense layer with dropout (p = 0.3) followed by an 8-way softmax classification layer.

5. DistilBERT (Transformer Baseline):
   A 6-layer, 768-dimensional transformer architecture pre-trained through knowledge distillation on BERT base. DistilBERT reduces parameters by 40% while preserving 97% of BERT's language comprehension. Fine-tuning uses AdamW optimizer with learning rate 2e-5, linear warmup, and batch size 32 over 3 epochs.

6. RoBERTa (Optimized Transformer):
   A 12-layer transformer trained on larger mini-batches and diverse corpora without next-sentence prediction, utilizing dynamic masking. Fine-tuned with learning rate 1.5e-5 and weight decay 0.01."""),

    ("5.4 Class Imbalance Mitigation Protocols", """Severe class skew (ranging from 14,000 Neutral samples down to fewer than 100 minority samples) distorts decision boundaries. We implemented four comparative mitigation protocols, strictly confined to the training partition:
- Protocol 1: Inverse Class Frequency Weighting. Assigns sample weight w_k = N / (K * N_k) to class k, penalizing minority classification mistakes during gradient descent.
- Protocol 2: Balanced Random Undersampling. Randomly sub-samples majority classes to cap the maximum imbalance ratio at 5:1 relative to the median class.
- Protocol 3: Synthetic Minority Over-sampling (SMOTE). Generates synthetic examples along minority feature vectors using k = 5 nearest neighbors. Applied cautiously due to sparsity issues in TF-IDF spaces.
- Protocol 4: Focal Loss. Dynamically dampens the loss contributed by easy examples:
  FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
  with focusing parameter gamma = 2.0 to concentrate updates on ambiguous minority samples."""),

    ("5.5 Evaluation Strategy and Metrics Protocol", """In severely imbalanced multi-class problems, standard classification accuracy is a deceptive metric. A naive classifier that predicts 'Neutral' on every single sample in our dataset would achieve approximately 30% accuracy, while completely failing on every single distress, sadness, or anxiety signal.

To prevent deceptive benchmarking, our evaluation protocol enforces strict rules:
- Primary Metric: Macro-Averaged F1 Score:
  Macro-F1 = (1 / K) * sum_{k=1}^{K} (2 * P_k * R_k) / (P_k + R_k)
  where P_k and R_k denote precision and recall for class k. By weighting every class equally regardless of sample frequency, Macro-F1 penalizes models that ignore minority classes.
- Per-Class Precision and Recall: Evaluated to inspect exactly which affective categories suffer from misclassification.
- Confusion Matrix Analysis: Visualizes systematic inter-class confusion patterns (e.g., whether Anxiety is misclassified as Fear or absorbed into Neutral).
- Inference Latency Benchmarking: Measures mean wall-clock latency (milliseconds per sentence) across 1,000 test queries on CPU hardware to ensure dashboard responsiveness."""),

    ("5.6 Experimental Design and Hardware Specifications", """All experiments were executed within a dual-environment configuration:
- Local Development & Baseline Environment: Intel Core i7-12700H CPU @ 2.30 GHz, 16 GB DDR4 RAM, running Windows 11 and Python 3.11. This environment hosted data ingestion, baseline scikit-learn model training, SQLite persistence, Streamlit UI development, and unit test execution.
- GPU Training Environment: Google Colab environment equipped with an NVIDIA Tesla T4 GPU (16 GB VRAM) and Intel Xeon CPU @ 2.20 GHz for PyTorch Bi-LSTM training and Hugging Face transformer fine-tuning.
- Reproducibility Controls: Random seed was locked to 42 across NumPy, Scikit-learn, and PyTorch. All split indices were saved as immutable Parquet files.""")
]

TABLE_5_1_DATA = {
    "title": "Table 5.1: Emotion Mapping Schema from Source Datasets to 8-Class Spectrum",
    "headers": ["Target Emotion", "Primary Source Dataset", "Original Source Categories Included", "Clinical/Affective Rationale"],
    "rows": [
        ["Joy", "GoEmotions", "admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief", "Aggregates positive valence states signifying wellness and psychological resilience"],
        ["Sadness", "GoEmotions", "sadness, disappointment, grief, remorse", "Captures low-arousal negative affect indicative of depressive or mournful states"],
        ["Anger", "GoEmotions", "anger, annoyance, disapproval", "Identifies high-arousal negative affect reflecting interpersonal conflict or frustration"],
        ["Fear", "GoEmotions", "fear", "Measures acute alarm, dread, or existential threat responses"],
        ["Anxiety", "GoEmotions", "nervousness", "Captures anticipatory dread, apprehension, and rumination (distinct from acute fear)"],
        ["Surprise", "GoEmotions", "surprise, realization", "Identifies sudden cognitive orientation changes (affectively neutral to high arousal)"],
        ["Neutral", "GoEmotions", "neutral, curiosity", "Serves as baseline non-affective or informational conversational statements"],
        ["Stress", "Dreaddit", "Acute stress narratives across 5 support subreddits", "Captures long-form psychological strain, cognitive overload, and burnout"]
    ]
}

TABLE_5_2_DATA = {
    "title": "Table 5.2: Class Distribution Across Stratified Dataset Splits (Total: 51,115 samples)",
    "headers": ["Target Emotion Class", "Training Split (80%)", "Validation Split (10%)", "Test Split (10%)", "Total Samples", "Percentage (%)"],
    "rows": [
        ["Joy", "11,878", "1,485", "1,485", "14,848", "29.05%"],
        ["Neutral", "11,466", "1,433", "1,433", "14,332", "28.04%"],
        ["Anger", "5,448", "681", "681", "6,810", "13.32%"],
        ["Sadness", "4,264", "533", "533", "5,330", "10.43%"],
        ["Stress", "2,842", "355", "356", "3,553", "6.95%"],
        ["Surprise", "2,608", "326", "326", "3,260", "6.38%"],
        ["Fear", "1,350", "169", "169", "1,688", "3.30%"],
        ["Anxiety", "1,036", "129", "130", "1,295", "2.53%"],
        ["Total", "40,892", "5,111", "5,112", "51,115", "100.00%"]
    ]
}
