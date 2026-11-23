# front_matter.py
"""
Front matter content for the Minor Project Report.
Contains title page details, certificate, declaration, abstract, acknowledgements,
and table of contents metadata.
"""

TITLE_DATA = {
    "title": "Emotional and Mental Health Monitoring: NLP-Based Sentiment Analysis",
    "degree": "Bachelor of Technology in Artificial Intelligence and Data Science (AI-DS)",
    "department": "Department of Artificial Intelligence and Data Science",
    "institution": "University School of Automation and Robotics (USAR)\nGuru Gobind Singh Indraprastha University (GGSIPU)\nEast Delhi Campus, Surajmal Vihar, Delhi – 110092",
    "academic_year": "Academic Year 2025–2026",
    "author": "Utkarsh & Project Team",
    "roll_number": "Student Roll Numbers",
    "supervisor": "Faculty Supervisor, Department of AI-DS, USAR",
}

CERTIFICATE_TEXT = """This is to certify that the project report entitled "Emotional and Mental Health Monitoring: NLP-Based Sentiment Analysis", submitted by the project team in partial fulfilment of the requirements for the award of the degree of Bachelor of Technology in Artificial Intelligence and Data Science at the University School of Automation and Robotics (USAR), Guru Gobind Singh Indraprastha University, Delhi, represents authentic work carried out under my direct supervision during the academic year 2025–2026.

To the best of my knowledge, the matter embodied in this report has not been submitted in whole or in part to any other university or institute for the award of any other degree or diploma. The experimental protocols, data engineering steps, and baseline models reported herein adhere to accepted research ethics and institutional academic norms.

Supervisor Signature: __________________________
Department of Artificial Intelligence and Data Science
University School of Automation and Robotics
Guru Gobind Singh Indraprastha University, Delhi
Date: November 2026
"""

DECLARATION_TEXT = """We hereby declare that this project report, titled "Emotional and Mental Health Monitoring: NLP-Based Sentiment Analysis", is our original work conducted at the University School of Automation and Robotics, GGSIPU, Delhi. All sources of literature, datasets, open-source libraries, and foundational research referenced in this report have been explicitly acknowledged and cited.

We confirm that this report does not infringe upon any intellectual property rights or incorporate uncredited third-party material. The code implementation, empirical benchmarks, and system prototypes discussed within were engineered by our team as part of the curriculum requirements.

Student Names: Utkarsh & Team Members
Programme: B.Tech (AI-DS), 7th Semester
USAR, GGSIPU, Delhi
Date: November 2026
"""

ABSTRACT_TEXT = """Mental health struggles frequently go unnoticed until individuals reach a breaking point. While millions express feelings of pressure, sadness, and fatigue in everyday text, traditional mental healthcare remains inaccessible to many due to cost, stigma, and provider shortages. This project designs and implements an end-to-end Natural Language Processing (NLP) framework that tracks emotional state over time rather than treating isolated messages as diagnostic judgements.

We unified the GoEmotions and Dreaddit corpora into an eight-category emotional spectrum: Joy, Sadness, Anger, Fear, Anxiety, Stress, Surprise, and Neutral. This created a standardized benchmark of 51,115 messages partitioned into fixed 80/10/10 stratified splits. Six classification models across three distinct architectural families were evaluated: classical linear pipelines (TF-IDF with Logistic Regression, Linear SVM, and Random Forest), recurrent networks (GloVe-initialized Bi-LSTM), and pre-trained transformers (DistilBERT and RoBERTa). On our test split, classical models attained overall accuracies between 0.685 and 0.693, but their macro-F1 scores revealed a stark drop to 0.466–0.561. Linear SVM performed best among classical baselines by preserving balanced recall across minority classes, whereas Logistic Regression completely failed to predict Anxiety. Detailed error inspection revealed that minority negative classes are frequently absorbed into the dominant Neutral category. Furthermore, the apparent high performance of the Stress category (F1 ~0.90) was traced to a text-length dataset confound rather than semantic understanding.

To move beyond single-message predictions, our temporal analysis module uses daily affective aggregation coupled with Exponential Moving Average (EMA) smoothing and a strict three-day persistence requirement. On simulated timelines, this filter ignored transient 1–2 day emotional dips while reliably flagging persistent negative distress within four days. The complete pipeline is embodied in a lightweight Streamlit dashboard backed by a local SQLite store that saves only aggregated affect scores, explicitly discarding raw input text to protect user privacy. All safety protocols escalate toward India's Tele-MANAS helpline (14416) during severe distress while remaining strictly non-diagnostic. The complete code implementation passed 49 out of 51 automated unit tests, validating the reproducibility of our baseline engineering and establishing a solid foundation for transformer fine-tuning.

Keywords: Natural language processing; affective computing; fine-grained emotion classification; temporal mood tracking; class imbalance; responsible AI; mental wellness support.
"""

ACKNOWLEDGEMENTS_TEXT = """We express our sincere gratitude to our project supervisor and the faculty members of the Department of Artificial Intelligence and Data Science, University School of Automation and Robotics (USAR), Guru Gobind Singh Indraprastha University, for their invaluable guidance, encouragement, and technical critiques throughout the development of this project.

We also thank the developers and maintainers of open-source libraries—including Scikit-Learn, PyTorch, Hugging Face Transformers, and Streamlit—whose public tools made this engineering workflow feasible. Finally, we thank our peers and families for their patience and support during long testing sessions and report revisions.
"""
