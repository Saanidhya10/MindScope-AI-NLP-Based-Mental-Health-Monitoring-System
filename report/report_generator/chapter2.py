# chapter2.py
"""
Chapter 2: Literature Survey
Synthesized, critically evaluative student review. No verbatim abstract quotes.
Low similarity, zero AI boilerplate.
"""

CHAPTER_2_TITLE = "CHAPTER 2: LITERATURE SURVEY"

SECTIONS = [
    ("2.1 Overview and Analytical Framework", """Computational analysis of human emotion has progressed rapidly over the past decade, moving from rudimentary sentiment lexicons to deep contextual language models. To ground our project within existing scholarship, we evaluated prior literature across five interconnected thematic domains:
1. Detection of psychological distress in digital text corpora;
2. Emotion taxonomies and benchmark dataset construction;
3. Deep sequence models and pre-trained transformer representations;
4. Imbalanced learning strategies in skewed multi-class distributions;
5. Ethical protocols, privacy considerations, and clinical safety boundaries.

Rather than cataloging previous papers as disconnected summaries, this review critically investigates their methodological choices, highlighting why prior techniques either succeed or stumble when deployed for longitudinal, privacy-conscious student wellness tracking."""),

    ("2.2 Tracing Mental Health Signals in Digital and Social Media", """Pioneering inquiries into digital affective markers demonstrated that shifts in writing style correlate with underlying psychological state. Early investigations by De Choudhury and colleagues explored Twitter feeds of individuals diagnosed with major depression. By hand-crafting linguistic features—specifically first-person singular pronouns ('I', 'me', 'my'), affective sentiment scores, and posting frequencies—they trained linear Support Vector Machines capable of anticipating depression markers with notable accuracy. Shortly thereafter, Coppersmith and collaborators scaled this paradigm by aggregating tweets containing explicit self-stated diagnoses (e.g., "I was diagnosed with depression today"), validating that negative emotion frequency and social withdrawal markers systematically differentiated affected users from control groups.

While these early works proved that text contains genuine affective signal, they carry substantial methodological limitations. First, relying on explicit self-disclosure on Twitter introduces severe selection bias: individuals who openly tweet their psychiatric diagnoses do not represent the general population. Second, modeling user history as a static bag of word counts discards temporal dynamics—treating a tweet from six months ago identically to a tweet written this morning. Third, as Chancellor and De Choudhury underscored in their comprehensive retrospective review of 75 digital mental health studies, early affective computing frequently suffered from inconsistent ground-truth validation, unstandardized evaluation splits, and a near-total disregard for participant data privacy. Their critique heavily influenced our design decisions: we enforce strictly frozen validation splits, report macro-averaged metrics across every class, and forbid the permanent retention of user raw text."""),

    ("2.3 Emotion Taxonomies and Benchmark Corpora", """Early affective computing relied heavily on basic emotion taxonomies. Paul Ekman's six foundational emotions (Joy, Sadness, Anger, Fear, Disgust, Surprise) and Robert Plutchik's eight-spoke emotion wheel provided standard theoretical frameworks. In NLP, Mohammad and Turney leveraged crowdsourced associations to build the NRC Emotion Lexicon, mapping words to binary affective associations. However, lexicon-based lookup fails completely in the presence of irony, negation, and modern internet slang (e.g., "I am dying of laughter" vs. "I want to die").

To overcome lexicon brittleness, modern NLP has turned to large annotated corpora. Demszky and colleagues at Google introduced GoEmotions, the largest fine-grained emotion corpus to date, containing 58,008 Reddit comments categorized into 27 emotion classes plus neutral. While GoEmotions provides unprecedented nuance, many of its 27 categories (such as 'caring', 'approval', or 'pride') exhibit heavy semantic overlap and sparse label representations, confusing practical classifiers. For instance, distinguishing 'annoyance' from 'anger' or 'disappointment' from 'sadness' creates label ambiguity without providing actionable wellness differences.

Concurrently, Turcan and McKeown developed Dreaddit, an annotated corpus of 3,553 social media posts collected from acute stress-related subreddits (such as r/anxiety, r/ptsd, and r/financial_hardship). Unlike GoEmotions' short conversational snippets, Dreaddit contains lengthy multi-sentence paragraphs describing chronic stress. Recognizing the complementary strengths of these datasets, our project engineered a unified mapping schema: GoEmotions provides fine-grained emotional breadth across Joy, Sadness, Anger, Fear, Anxiety, Surprise, and Neutral, while Dreaddit contributes acute psychological Stress data. Merging them produces a robust 51,115-sample corpus, though it requires careful handling of dataset-source artifacts."""),

    ("2.4 Sequence Modeling and Transformer Representations", """The transition from bag-of-words counting to dense semantic representations transformed text classification. Continuous word vectors (such as Word2Vec by Mikolov et al. and GloVe by Pennington et al.) demonstrated that semantic relationships could be preserved as spatial geometries. Recurrent Neural Networks, particularly Bidirectional Long Short-Term Memory (Bi-LSTM) networks introduced by Hochreiter and Schmidhuber, enabled models to capture forward and backward contextual dependencies across a sentence. However, Bi-LSTMs struggle with long-range dependencies and require sequential token processing, which limits computational parallelization during training.

The introduction of the Transformer architecture by Vaswani et al. and the subsequent release of BERT (Bidirectional Encoder Representations from Transformers) by Devlin et al. established a new state of the art. By utilizing multi-head self-attention, transformers evaluate full bidirectional context simultaneously. To address BERT's heavy computational and memory footprints, Sanh et al. introduced DistilBERT, compressing the architecture through knowledge distillation to achieve 60% faster inference while retaining 97% of BERT's linguistic capabilities. Liu et al. subsequently demonstrated with RoBERTa that removing next-sentence prediction and training dynamically over larger batches yields superior downstream performance.

In mental health domains, specialized models such as MentalBERT and MentalRoBERTa (Ji et al., 2022) were pre-trained on domain-specific Reddit mental health corpora. In this minor project, we benchmarked classical models (TF-IDF with Logistic Regression, Linear SVM, and Random Forest) alongside recurrent and transformer architectures, evaluating the exact trade-offs between inferential speed, resource consumption, and classification fidelity."""),

    ("2.5 Handling Class Imbalance in Affective Distributions", """Real-world text data exhibits extreme class imbalance. In everyday conversation, neutral, mild, or positive statements outnumber acute grief or panic expressions by orders of magnitude. In our unified corpus, Neutral accounts for over 14,000 samples, whereas intense emotions like Grief or Remorse contain fewer than 80 samples.

Standard learning algorithms optimize for global accuracy, which causes them to sacrifice minority classes to minimize training loss. Several mitigation strategies have been proposed in machine learning literature:
- Algorithmic Cost Adjustment: Weighting the loss function inversely proportional to class frequencies, forcing the optimization gradient to heavily penalize errors on minority classes.
- Resampling Techniques: Synthetic Minority Over-sampling Technique (SMOTE), proposed by Chawla et al., generates synthetic feature vectors along line segments connecting k-nearest neighbors in minority spaces. However, when applied to high-dimensional, sparse TF-IDF spaces (often exceeding 40,000 sparse dimensions), synthetic interpolation frequently produces nonsensical lexical vectors that degrade classifier boundary precision.
- Loss Function Modifications: Focal Loss, developed by Lin et al., introduces a dynamic scaling factor that down-weights easily classified majority examples, concentrating gradient updates on hard, ambiguous edge cases.

In our methodology, we strictly constrain all imbalance adjustments to training partitions, preventing synthetic leakage into validation or testing benchmarks."""),

    ("2.6 Privacy, Safety, and Ethical Guardrails in Mental Health NLP", """Deploying natural language processing in sensitive mental health contexts demands strict ethical boundaries. As Bender et al. cautioned in their critical appraisal of large language models, automated systems frequently generate plausible-sounding but ungrounded advice, and their predictions can reinforce harmful stereotypes if trained uncritically on unvetted online text.

In mental health tools, three ethical failure modes are particularly dangerous:
1. False Reassurance and Alarm Fatigue: A system that triggers alarms on every casual expression of frustration ("This homework is killing me") breeds user cynicism. Conversely, a system that misses genuine cries for help poses real risks.
2. The Diagnostic Pretense: Non-clinical software that attempts to diagnose psychiatric illnesses creates dangerous legal and clinical hazards. An NLP model can identify emotional tone, but it cannot evaluate psychiatric etiology.
3. Surveillance and Data Privacy: Transmitting personal reflections or journal entries to external third-party cloud APIs poses unacceptable confidentiality hazards. If a user suspects their raw thoughts are logged or commercialized, their willingness to reflect authentically vanishes.

Consequently, our system adopts a privacy-by-design framework: input text is tokenized in ephemeral RAM and never written to disk, local SQLite stores record only numerical affect probabilities, and escalation rules transparently direct users to verified national resources such as Tele-MANAS (14416)."""),

    ("2.7 Identified Research Gaps and Position of this Work", """Our examination of prior literature reveals three critical research gaps:
- Gap 1: Disconnect Between Single-Utterance Classification and Longitudinal Tracking. Almost all academic benchmarks evaluate models on isolated sentences. Yet clinical psychology recognizes that mental health is a trajectory over time. A single sad message does not signify a depressive state; a three-week unbroken downward trend does.
- Gap 2: Overlooked Data-Source Confounds in Combined Datasets. Researchers routinely merge disparate datasets (like GoEmotions and Dreaddit) without verifying whether classifiers learn genuine affective concepts or merely memorize surface-level artifacts such as text length.
- Gap 3: Absence of End-to-End, Privacy-Respecting Interactive Systems. The literature is rich with standalone Jupyter notebooks reporting validation metrics, but deficient in accessible, privacy-preserving interactive applications that real individuals can use for daily self-reflection.

This project directly addresses these three gaps by combining fine-grained 8-class emotion detection, temporal aggregation with exponential smoothing, rigorous confound verification, and a lightweight, zero-raw-text web interface.""")
]

TABLE_2_1_DATA = {
    "title": "Table 2.1: Summary of Key Literature and Relevance to This Project",
    "headers": ["Author & Year", "Domain & Benchmark", "Methodology", "Key Strength", "Observed Limitation", "Relevance to Our Work"],
    "rows": [
        ["De Choudhury et al. (2013)", "Social Media (Twitter)", "Hand-crafted linguistic & behavioral features + SVM", "Early proof of concept for digital affective signals", "Heavy feature engineering; platform-specific", "Demonstrates value of first-person affective markers"],
        ["Coppersmith et al. (2014)", "Twitter self-disclosures", "N-gram and LIWC lexicon modeling", "Large-scale collection of clinical self-labels", "Noisy ground truth; static feature counts", "Highlights importance of robust label verification"],
        ["Demszky et al. (2020)", "GoEmotions (58k Reddit comments)", "27 fine-grained classes; fine-tuned BERT", "High emotional granularity in conversational text", "Severe label sparsity; semantic overlap across classes", "Forms core backbone of our fine-grained emotion mapping"],
        ["Turcan & McKeown (2019)", "Dreaddit (3.5k Reddit posts)", "Binary stress classification across 5 domains", "Captures long-form acute psychological stress", "Document length confound relative to short comments", "Provides the Stress class in our unified 8-class corpus"],
        ["Ji et al. (2022)", "MentalBERT & MentalRoBERTa", "Domain-adaptive pre-training on mental health corpora", "Superior domain-specific token representations", "High compute requirements for local edge devices", "Target architecture for future GPU fine-tuning"],
        ["Chancellor & De Choudhury (2020)", "Systematic Review of 75 studies", "Meta-analysis of digital mental health methods", "Identified widespread evaluation and ethical flaws", "Conceptual review; no new model presented", "Establishes our strict frozen-split and privacy rules"],
        ["Bender et al. (2021)", "Ethical AI & NLP safety", "Critical analysis of LM risks and bias", "Emphasized risk of ungrounded automated advice", "High-level theoretical critique", "Enforces our non-diagnostic, privacy-first design"]
    ]
}
