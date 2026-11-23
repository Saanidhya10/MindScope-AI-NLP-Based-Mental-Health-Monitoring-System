# repair_report_structure.py
"""
Surgically repairs structural defects in Minor_Project_Report_Updated_66_Pages.docx:
Processes sections in REVERSE order (62 down to 1) to guarantee zero index invalidation.
1. Section 62: Future Scope (10 Top Improvements from user image)
2. Section 60: 6.3.5 Aggregated Signals & 6.3.6 Threats to Validity (clean bullets)
3. Section 58: 6.3 Comparative Analysis (restores 6.3.1, 6.3.2, 6.3.3 and removes duplicates)
4. Section 42: Table 17 & 18 (Algorithms 5.1 & 5.2 formatted with clean line breaks)
5. Section 20: 3.1 Problem Definition (restores full 1, 2, 3 list and lead-in)
6. Section 7: Nomenclature (clean 2-column entries)
7. Section 6: Abbreviations (clean 2-column entries)
8. Section 5: List of Figures (clean dot-leader entries)
9. Section 4: List of Tables (clean dot-leader entries)
10. Section 3: Table of Contents Part 3 (clean dot-leader entries)
11. Section 2: Table of Contents Part 2 (clean dot-leader entries)
12. Section 1: Table of Contents Part 1 (clean dot-leader entries)

Maintains exactly 66 pages and pristine Liberation Serif / Sans styling.
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER
import shutil
import os
import win32com.client
import fitz

DOCX_IN = "Minor_Project_Report_Updated_66_Pages.docx"
DOCX_BAK = "Minor_Project_Report_Updated_66_Pages.docx.bak"
PDF_OUT = "Minor_Project_Report_Updated_66_Pages.pdf"

if os.path.exists(DOCX_BAK):
    shutil.copy(DOCX_BAK, DOCX_IN)
    print(f"Restored clean baseline from {DOCX_BAK}")
else:
    shutil.copy(DOCX_IN, DOCX_BAK)
    print(f"Created baseline backup {DOCX_BAK}")

doc = docx.Document(DOCX_IN)

def style_run(run, font_name="Liberation Serif", size_pt=9.5, bold=False, italic=False, color_rgb=(0,0,0)):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(*color_rgb)

def make_toc_entry(p, title, page_str, level=0, is_chapter=False, right_margin_in=6.25):
    p.text = ""
    p.paragraph_format.tab_stops.clear_all()
    p.paragraph_format.tab_stops.add_tab_stop(Inches(right_margin_in), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(1.5 if is_chapter else 0.8)
    p.paragraph_format.line_spacing = 1.0
    
    indent_prefix = "    " * level
    run_text = p.add_run(f"{indent_prefix}{title}")
    style_run(run_text, font_name="Liberation Serif", size_pt=9.5 if not is_chapter else 10.0, bold=is_chapter)
    
    run_tab = p.add_run(f"\t{page_str}")
    style_run(run_tab, font_name="Liberation Serif", size_pt=9.5 if not is_chapter else 10.0, bold=is_chapter)

def make_defn_entry(p, term, defn, tab_pos_in=1.5):
    p.text = ""
    p.paragraph_format.tab_stops.clear_all()
    p.paragraph_format.tab_stops.add_tab_stop(Inches(tab_pos_in), WD_TAB_ALIGNMENT.LEFT)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(1.0)
    p.paragraph_format.line_spacing = 1.0
    
    r_term = p.add_run(term)
    style_run(r_term, font_name="Liberation Serif", size_pt=9.5, bold=True)
    
    r_tab = p.add_run(f"\t{defn}")
    style_run(r_tab, font_name="Liberation Serif", size_pt=9.5, bold=False)

print("Starting reverse-order surgical repairs...")

toc_page_1 = [
    ("TABLE OF CONTENTS", None, 0, True),
    ("CERTIFICATE", "i", 0, False),
    ("DECLARATION", "ii", 0, False),
    ("ACKNOWLEDGEMENT", "iii", 0, False),
    ("LIST OF TABLES", "vii", 0, False),
    ("LIST OF FIGURES", "viii", 0, False),
    ("ABBREVIATIONS AND NOMENCLATURE", "ix", 0, False),
    ("ABSTRACT", "xi", 0, False),
    ("CHAPTER 1: INTRODUCTION", "1", 0, True),
    ("1.1 Background", "1", 1, False),
    ("1.2 Text as a Signal of Emotional State", "1", 1, False),
    ("1.3 From Polarity to Fine-Grained Emotion", "2", 1, False),
    ("1.4 Why Emotional Trends Matter", "2", 1, False),
    ("1.5 Motivation", "3", 1, False),
    ("1.6 Project Overview", "3", 1, False),
    ("1.7 Contributions of the Work", "4", 1, False),
    ("1.8 Organisation of the Report", "5", 1, False),
    ("CHAPTER 2: LITERATURE SURVEY", "6", 0, True),
    ("2.1 Overview", "6", 1, False),
    ("2.2 Mental-Health Signals in Social Media", "6", 1, False),
    ("2.3 Lexicon-Based and Classical Machine-Learning Approaches", "7", 1, False),
    ("2.4 Deep Sequence Models", "7", 1, False),
    ("2.5 Pre-Trained Transformers for Emotion Classification", "7", 1, False),
    ("2.6 Datasets", "8", 1, False),
    ("2.7 Class Imbalance and Data Augmentation", "8", 1, False),
    ("2.8 Evaluation and Ethical Practice", "9", 1, False),
    ("2.9 Summary of the Literature Surveyed", "10", 1, False),
    ("2.10 Key Observations and Research Gap", "10", 1, False),
    ("CHAPTER 3: PROBLEM STATEMENT", "12", 0, True),
    ("3.1 Problem Definition", "12", 1, False),
    ("3.2 Formal Formulation", "12", 1, False),
]

toc_page_2 = [
    ("3.3 Research Questions", "13", 1, False),
    ("3.4 Scope and Assumptions", "13", 1, False),
    ("3.5 Design Constraints", "14", 1, False),
    ("CHAPTER 4: DESCRIPTION OF VARIOUS MODULES", "15", 0, True),
    ("4.1 System Architecture", "15", 1, False),
    ("4.2 Data Acquisition and Label-Mapping Module", "16", 1, False),
    ("4.3 Pre-Processing Module", "17", 1, False),
    ("4.4 Emotion Classification Module", "18", 1, False),
    ("4.5 Class-Imbalance Mitigation Module", "19", 1, False),
    ("4.6 Evaluation Module", "20", 1, False),
    ("4.7 Temporal Aggregation Module", "21", 1, False),
    ("4.8 Recommendation and Safety Module", "22", 1, False),
    ("4.9 Storage and Privacy Module", "23", 1, False),
    ("4.10 Dashboard Module", "23", 1, False),
    ("4.11 Testing Module", "23", 1, False),
    ("CHAPTER 5: METHODOLOGY ADOPTED", "24", 0, True),
    ("5.1 Experimental Design Matrix", "24", 1, False),
    ("5.2 Project Timeline and PERT Analysis", "25", 1, False),
    ("5.3 Hardware and Software Used", "27", 1, False),
    ("5.4 Data Flow Diagrams and E-R Diagram", "28", 1, False),
    ("    5.4.1 Data Flow Diagrams", "28", 2, False),
    ("    5.4.2 Entity-Relationship Diagram", "29", 2, False),
    ("5.5 Algorithms Used", "30", 1, False),
    ("    5.5.1 TF-IDF Feature Extraction", "30", 2, False),
    ("    5.5.2 Logistic Regression", "31", 2, False),
    ("    5.5.3 Linear Support Vector Machine", "31", 2, False),
    ("    5.5.4 Random Forest", "31", 2, False),
    ("    5.5.5 Bidirectional LSTM", "31", 2, False),
    ("    5.5.6 Transformer Fine-Tuning (DistilBERT and RoBERTa)", "32", 2, False),
    ("    5.5.7 Class-Imbalance Mitigation", "33", 2, False),
    ("    5.5.8 Evaluation Metrics", "34", 2, False),
    ("    5.5.9 Training and Evaluation Protocol", "34", 2, False),
    ("    5.5.10 Temporal Aggregation and Escalation", "34", 2, False),
    ("5.6 Characterisation of the Data", "35", 1, False),
]

toc_page_3 = [
    ("CHAPTER 6: RESULTS AND DISCUSSIONS", "38", 0, True),
    ("6.1 Snapshots of Results Obtained", "38", 1, False),
    ("    6.1.1 Emotion Inference with the Trained Model", "38", 2, False),
    ("    6.1.2 Mood Aggregation and Recommendation Output", "39", 2, False),
    ("    6.1.3 Automated Test Results", "40", 2, False),
    ("    6.1.4 Dashboard", "41", 2, False),
    ("6.2 Graphs and Tables", "41", 1, False),
    ("    6.2.1 Overall Performance of the Classical Models", "41", 2, False),
    ("    6.2.2 Per-Class Performance", "43", 2, False),
    ("    6.2.3 Confusion Matrices", "44", 2, False),
    ("    6.2.4 Most Informative Terms", "45", 2, False),
    ("    6.2.5 Inference Latency", "46", 2, False),
    ("    6.2.6 Behaviour of the Temporal Aggregation and Escalation Rule", "46", 2, False),
    ("    6.2.7 Pending Result Tables", "49", 2, False),
    ("6.3 Comparative Analysis", "50", 1, False),
    ("    6.3.1 Comparison Among the Classical Models (RQ1)", "50", 2, False),
    ("    6.3.2 Why Accuracy Is Misleading Here", "50", 2, False),
    ("    6.3.3 The Stress Source Confound (RQ3)", "50", 2, False),
    ("    6.3.4 Comparison with Published Work", "51", 2, False),
    ("    6.3.5 Aggregated Versus Single-Message Signals (RQ4)", "52", 2, False),
    ("    6.3.6 Threats to Validity", "52", 2, False),
    ("CHAPTER 7: CONCLUSION AND FUTURE SCOPE", "53", 0, True),
    ("7.1 Conclusion", "53", 1, False),
    ("7.2 Future Scope", "54", 1, False),
    ("CHAPTER 8: REFERENCES", "55", 0, True),
]

abbreviations_list = [
    ("AI-ML", "Artificial Intelligence and Machine Learning"),
    ("APA", "American Psychological Association (referencing style)"),
    ("BERT", "Bidirectional Encoder Representations from Transformers"),
    ("Bi-LSTM", "Bidirectional Long Short-Term Memory"),
    ("BPE", "Byte-Pair Encoding"),
    ("CPU / GPU", "Central / Graphics Processing Unit"),
    ("CSV", "Comma-Separated Values"),
    ("DFD", "Data Flow Diagram"),
    ("EMA", "Exponential Moving Average"),
    ("E-R", "Entity-Relationship"),
    ("FK / PK", "Foreign Key / Primary Key"),
    ("GGSIPU", "Guru Gobind Singh Indraprastha University"),
    ("GloVe", "Global Vectors for Word Representation"),
    ("JSON", "JavaScript Object Notation"),
    ("LR", "Logistic Regression"),
    ("LSTM", "Long Short-Term Memory"),
    ("MAE", "Mean Absolute Error"),
    ("ML", "Machine Learning"),
    ("NIMHANS", "National Institute of Mental Health and Neuro Sciences"),
    ("NLP", "Natural Language Processing"),
    ("NMHS", "National Mental Health Survey"),
    ("PERT", "Programme Evaluation and Review Technique"),
    ("RF", "Random Forest"),
    ("RoBERTa", "Robustly Optimised BERT Pre-training Approach"),
    ("RQ", "Research Question"),
    ("SQL", "Structured Query Language"),
    ("SVM", "Support Vector Machine"),
    ("Tele-MANAS", "Tele Mental Health Assistance and Networking Across States"),
    ("TF-IDF", "Term Frequency – Inverse Document Frequency"),
    ("TP / FP / FN", "True Positives / False Positives / False Negatives"),
    ("USAR", "University School of Automation and Robotics"),
]

lot_entries = [
    ("Table 1.1", "The four layers of the proposed framework", "4"),
    ("Table 2.1", "Summary of the literature surveyed", "10"),
    ("Table 3.1", "Research questions addressed by the project", "13"),
    ("Table 4.1", "Modules of the system and their implementation", "16"),
    ("Table 4.2", "Mapping of source labels to the eight application classes", "17"),
    ("Table 4.3", "Example output of the pre-processing module (src/preprocess.py)", "18"),
    ("Table 4.4", "Classification models implemented", "19"),
    ("Table 4.5", "Supportive suggestions by dominant emotion (app/components.py)", "21"),
    ("Table 4.6", "Automated unit tests", "23"),
    ("Table 5.1", "Experimental design matrix and status of runs", "26"),
    ("Table 5.2", "PERT analysis of project activities (durations in weeks)", "27"),
    ("Table 5.3", "Hardware requirements", "27"),
    ("Table 5.4", "Software and libraries used", "28"),
    ("Table 5.5", "Schema of the PREDICTIONS table", "30"),
    ("Table 5.6", "Hyper-parameters and settings used", "35"),
    ("Table 5.7", "Class distribution across the frozen splits", "35"),
    ("Table 5.8", "Text length by class (words, transformer view)", "36"),
    ("Table 5.9", "Class weights computed from the training split (equation 5.9)", "37"),
    ("Table 6.1", "Test-set performance of the classical baselines (n = 5,112)", "42"),
    ("Table 6.2", "Validation versus test macro-F1 and accuracy", "42"),
    ("Table 6.3", "Per-class F1 on the test split", "43"),
    ("Table 6.4", "Per-class recall (diagonal of normalised CM) & share predicted Neutral", "45"),
    ("Table 6.5", "Ten highest-weighted TF-IDF terms per class in Logistic Regression", "45"),
    ("Table 6.6", "Measured inference latency of the Logistic Regression pipeline (CPU)", "46"),
    ("Table 6.7", "Escalation behaviour on simulated timelines (20 runs per scenario)", "47"),
    ("Table 6.8", "Test-set comparison of the remaining models (pending)", "49"),
    ("Table 6.9", "Per-class F1 of best transformer under each imbalance condition (pending)", "49"),
    ("Table 6.10", "Trend quality with classifier predictions on scripted timelines (pending)", "50"),
    ("Table 6.11", "Comparison with published results on related tasks", "51"),
]

lof_entries = [
    ("Figure 4.1", "Overall system architecture: online inference & offline training pipeline", "15"),
    ("Figure 4.2", "Pre-processing pipeline producing three text views for three model families", "18"),
    ("Figure 4.3", "Decision logic of the recommendation and safety layer", "21"),
    ("Figure 5.1", "Flow chart of the experimental workflow", "25"),
    ("Figure 5.2", "Gantt chart of the sixteen-week project schedule", "26"),
    ("Figure 5.3", "Level-0 data flow diagram (context diagram)", "28"),
    ("Figure 5.4", "Level-1 data flow diagram", "29"),
    ("Figure 5.5", "Entity-relationship diagram of the mood-history database", "30"),
    ("Figure 5.6", "Architecture of the Bi-LSTM emotion classifier", "32"),
    ("Figure 5.7", "Fine-tuning pipeline for DistilBERT and RoBERTa", "33"),
    ("Figure 5.8", "Class distribution of the unified dataset (all splits)", "36"),
    ("Figure 6.1", "Console snapshot: predictions of trained Logistic Regression baseline", "38"),
    ("Figure 6.2", "Console snapshot: daily summaries, escalation flags and recommendation levels", "40"),
    ("Figure 6.3", "Console snapshot: pytest run of the unit-test suite", "40"),
    ("Figure 6.4", "Streamlit dashboard: main view with consent panel, entry form, emotion card", "41"),
    ("Figure 6.5", "Streamlit dashboard: mood-over-time view with smoothed trend & threshold", "41"),
    ("Figure 6.6", "Comparison of aggregate test metrics for the three classical models", "42"),
    ("Figure 6.7", "Per-class F1 of classical models on test split (dotted line marks F1 = 0.5)", "43"),
    ("Figure 6.8", "Normalised confusion matrix of Linear SVM baseline on test split", "44"),
    ("Figure 6.9", "Normalised confusion matrices of Logistic Regression and Random Forest", "44"),
    ("Figure 6.10", "Per-entry scores, daily means, EMA trend and escalation days for simulated run", "48"),
]

# ==================== 1. REPAIR SECTION 62 (PAGE 62) ====================
top10_improvements = [
    ("1. Fine-Tune RoBERTa / DeBERTa", "Very high", "Medium", "Complete GPU benchmarking in notebooks 03 and 04 to evaluate DistilBERT and RoBERTa across multiple seeds, replacing linear bag-of-words baselines to resolve polysemy and minority class confusion."),
    ("2. Fix Stress and Anxiety Dataset Bias", "Very high", "High", "Segment Dreaddit narratives into sentence-level units and collect short stress annotations to eliminate the document-length confound and prevent classifiers from exploiting token-count shortcuts."),
    ("3. Multi-Label Emotion Detection", "High", "Medium", "Transition from mutually exclusive softmax heads to independent sigmoid classifiers, allowing the framework to capture nuanced co-occurring emotions (e.g., simultaneous sadness and anxiety)."),
    ("4. Personalized Mood Baseline", "High", "Medium", "Calibrate adaptive running mean and standard deviation thresholds over an initial two-week calibration window to tailor alert triggers to individual user expressive temperaments."),
    ("5. Intelligent Trend and Anomaly Detection", "High", "Medium", "Integrate statistical time-series decomposition (STL or Holt-Winters) to detect non-linear mood velocity and predict imminent psychological downturns prior to fixed threshold breaches."),
    ("6. Context-Aware Recommendations", "High", "Medium", "Dynamically customize coping suggestions based on circadian factors, time of day, and historical user-rated effectiveness for specific emotional states."),
    ("7. Emotion Confidence and Uncertainty", "High", "Easy–Medium", "Apply post-hoc temperature scaling and Monte Carlo Dropout to reject out-of-distribution inputs, preventing ambiguous predictions from distorting mood history."),
    ("8. Multilingual Hindi/Hinglish Support", "Medium", "High", "Fine-tune IndicBERT or MuRIL on code-mixed Hinglish corpora to accommodate everyday bilingual expressions commonly used by Indian students."),
    ("9. Interactive Dashboard Improvements", "Medium", "Easy–Medium", "Enhance user engagement with modular analytical widgets, longitudinal affective correlation charts, and responsive mobile-first UI components."),
    ("10. Privacy, Safety and Real-User Evaluation", "Very high", "Medium–High", "Conduct consented ethical pilot evaluations with mental health professionals to validate longitudinal sensitivity while maintaining strictly local, privacy-preserving client execution."),
]

# Section 62: P564 is '7.2 Future Scope', P565..P572 are 8 items, P573 is '54', P574 is sectPr
p_hdg_72 = doc.paragraphs[564]
p_hdg_72.text = "7.2 Future Scope"
p_hdg_72.paragraph_format.space_before = Pt(0)
p_hdg_72.paragraph_format.space_after = Pt(3)
style_run(p_hdg_72.runs[0], font_name="Liberation Sans", size_pt=11, bold=True)

# Replace P565..P572 with first 8 items
for i in range(8):
    p = doc.paragraphs[565 + i]
    p.text = ""
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.line_spacing = 1.02
    title, pri, diff, desc = top10_improvements[i]
    r1 = p.add_run(f"{title} (Priority: {pri} | Difficulty: {diff}): ")
    style_run(r1, font_name="Liberation Serif", size_pt=9.0, bold=True)
    r2 = p.add_run(desc)
    style_run(r2, font_name="Liberation Serif", size_pt=9.0, bold=False)

# Add items 9 and 10 before P573 (footer '54')
p_f54 = doc.paragraphs[573]
p_f54.text = "54"
p_f54.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
style_run(p_f54.runs[0], font_name="Liberation Serif", size_pt=10)

for i in [8, 9]:
    new_p = doc.add_paragraph()
    new_p.text = ""
    new_p.paragraph_format.left_indent = Inches(0.2)
    new_p.paragraph_format.space_before = Pt(0)
    new_p.paragraph_format.space_after = Pt(1.5)
    new_p.paragraph_format.line_spacing = 1.02
    title, pri, diff, desc = top10_improvements[i]
    r1 = new_p.add_run(f"{title} (Priority: {pri} | Difficulty: {diff}): ")
    style_run(r1, font_name="Liberation Serif", size_pt=9.0, bold=True)
    r2 = new_p.add_run(desc)
    style_run(r2, font_name="Liberation Serif", size_pt=9.0, bold=False)
    p_f54._p.addprevious(new_p._p)

print("Repaired Section 62 (Page 62)")

# ==================== 2. REPAIR SECTION 60 (PAGE 60) ====================
# In Section 60:
# P545: duplicate text -> replace with heading 6.3.5
# P546: duplicate text -> replace with body 6.3.5
# P547: '6.3.6 Threats to Validity'
# P548..P552: 5 bullets -> format cleanly
# P553: '52' -> format footer
p_635_h = doc.paragraphs[545]
p_635_h.text = "6.3.5 Aggregated Versus Single-Message Signals (RQ4)"
p_635_h.paragraph_format.space_before = Pt(0)
p_635_h.paragraph_format.space_after = Pt(2)
style_run(p_635_h.runs[0], font_name="Liberation Sans", size_pt=10.5, bold=True)

p_635_b = doc.paragraphs[546]
p_635_b.text = (
    "Simulation results on scripted timelines underscore the practical necessity of temporal aggregation. "
    "A single-message alarm cannot distinguish an isolated bad day from a deteriorating fortnight: it triggered "
    "across 100% of transient two-day dips, causing severe alarm fatigue. In contrast, our temporal logic—combining "
    "daily averaging, exponential moving average smoothing (α = 0.3), and a 3-day persistence requirement—suppressed "
    "all transient dips up to three days, triggered on only 15% of four-day dips, and reliably flagged 100% of sustained "
    "depressive declines with an appropriate four-day latency. For a wellness reflection tool, this balance between high "
    "sensitivity and low false-alarm frequency is ideal."
)
p_635_b.paragraph_format.space_before = Pt(0)
p_635_b.paragraph_format.space_after = Pt(4)
p_635_b.paragraph_format.line_spacing = 1.05
style_run(p_635_b.runs[0], font_name="Liberation Serif", size_pt=10.0)

p_636_h = doc.paragraphs[547]
p_636_h.text = "6.3.6 Threats to Validity"
p_636_h.paragraph_format.space_before = Pt(4)
p_636_h.paragraph_format.space_after = Pt(2)
style_run(p_636_h.runs[0], font_name="Liberation Sans", size_pt=10.5, bold=True)

threat_bullets = [
    ("•  Small minority classes. ", "With ten Anxiety and fifty-five Fear examples in the test split, per-class F1 for these classes carries wider confidence bounds; multi-seed runs are planned for transformer benchmarks."),
    ("•  Source confound. ", "The Stress class differs from all other classes in source domain and length, inflating its apparent F1 score (Section 6.3.3)."),
    ("•  Label mapping. ", "The 27 → 8 mapping involves reasoned taxonomy decisions (e.g., embarrassment mapped to Sadness, desire to Neutral) that were frozen in advance to avoid test-set overfitting."),
    ("•  Simulated timelines. ", "The temporal evaluations use scripted representative profiles to verify system behavior deterministically; human subject trials remain future work."),
    ("•  Single seed for classical models. ", "Classical results were evaluated on a single stratified seed; their deterministic optimization makes this less variable than deep neural networks."),
]

for idx, (b_bold, b_text) in enumerate(threat_bullets):
    p = doc.paragraphs[548 + idx]
    p.text = ""
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.line_spacing = 1.0
    r1 = p.add_run(b_bold)
    style_run(r1, font_name="Liberation Serif", size_pt=9.5, bold=True)
    r2 = p.add_run(b_text)
    style_run(r2, font_name="Liberation Serif", size_pt=9.5, bold=False)

p_f52 = doc.paragraphs[553]
p_f52.text = "52"
p_f52.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
style_run(p_f52.runs[0], font_name="Liberation Serif", size_pt=10)

print("Repaired Section 60 (Page 60)")

# ==================== 3. REPAIR SECTION 58 (PAGE 58) ====================
# P528 is '6.3 Comparative Analysis'
# P529..P534 are duplicate blocks -> replace with 6.3.1, 6.3.2, 6.3.3
sec58_blocks = [
    ("6.3.1 Comparison Among the Classical Models (RQ1)", True),
    ("Among the classical models, the Linear SVM is the clear choice: it achieves the best macro-F1 (0.561), the highest minority recall, and the highest per-class F1 across six of the eight affective categories, while maintaining an accuracy within 0.01 of the best baseline. Conversely, Logistic Regression, despite posting the highest test accuracy (0.693), fails completely on Anxiety (recall = 0.000), making it unsuitable for early distress detection. Random Forest falls between the two but suffers from slower training and reduced interpretability. These empirical findings demonstrate that maximum-margin linear classifiers excel on high-dimensional sparse representations under severe class skew.", False),
    ("6.3.2 Why Accuracy Is Misleading Here", True),
    ("Although all three classical baselines achieve similar overall accuracy (~69%), their practical clinical value differs drastically. A trivial classifier defaulting entirely to Neutral achieves 38.6% accuracy, and one predicting solely Joy and Neutral exceeds 74%. Because the core purpose of this system is to identify minority distress signals, macro-F1 and per-class recall represent the essential criteria, confirming that accuracy masks severe minority-class failure (Japkowicz & Stephen, 2002; Demszky et al., 2020).", False),
    ("6.3.3 The Stress Source Confound (RQ3)", True),
    ("Detailed error investigations confirm that the apparent high F1 for Stress (0.84–0.90) is primarily an empirical artifact of document length and dataset source. Stress examples from Dreaddit average six times the token length of GoEmotions comments (Table 5.8), allowing bag-of-words classifiers to exploit token count shortcuts. Furthermore, top predictive features represent generic narrative terms, and the word \"anxiety\" serves as a primary Stress feature solely because it appears frequently in the Dreaddit source subreddit.", False),
]

p_comp_hdg = doc.paragraphs[528]
p_comp_hdg.text = "6.3 Comparative Analysis"
p_comp_hdg.paragraph_format.space_before = Pt(6)
p_comp_hdg.paragraph_format.space_after = Pt(2)
style_run(p_comp_hdg.runs[0], font_name="Liberation Sans", size_pt=11, bold=True)

for idx, (text, is_hdg) in enumerate(sec58_blocks):
    p = doc.paragraphs[529 + idx]
    p.text = text
    p.paragraph_format.space_before = Pt(4 if is_hdg else 0)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.05
    style_run(p.runs[0], font_name="Liberation Sans" if is_hdg else "Liberation Serif", size_pt=10.5 if is_hdg else 10.0, bold=is_hdg)

p_f50 = doc.paragraphs[535]
p_f50.text = "50"
p_f50.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
style_run(p_f50.runs[0], font_name="Liberation Serif", size_pt=10)

print("Repaired Section 58 (Page 58)")

# ==================== 4. REPAIR SECTION 42 (TABLES 17 & 18) ====================
algo1_lines = [
    "Input : frozen splits (train, val, test); model family M; condition c; seed s",
    "1  set random seeds of Python, NumPy and PyTorch to s",
    "2  train' , weights ← PrepareTrainingData(train, c)     # mitigation on train only",
    "3  for each hyper-parameter setting h in the search grid of M do",
    "4      fit M(h) on train' (weighted loss if weights ≠ None)",
    "5      score(h) ← macro-F1 of M(h) on val",
    "6  h* ← argmax_h score(h)",
    "7  ŷ ← predictions of M(h*) on test                      # test used once",
    "8  log accuracy, macro P/R/F1, weighted F1, per-class F1, min-class F1",
    "9  save confusion matrix, predictions and model artefacts",
]

algo2_lines = [
    "Input : daily summaries D (date, S_t) for one profile; α, τ, k, max_gap",
    "1  S̃ ← EMA of S_t with factor α                    # S̃_t = α·S_t + (1−α)·S̃_(t−1)",
    "2  run ← 0; prev ← none; flags ← []",
    "3  for each day t in D (in date order) do",
    "4      gap ← days since prev (0 for the first day); prev ← t",
    "5      if S̃_t > τ then run ← (run + 1 if gap ≤ max_gap else 1)",
    "6      else run ← 0",
    "7      append (run ≥ k) to flags",
    "8  if last flag is true            → level = escalation (message + helplines)",
    "9  else if dominant(last day) ∈ N and S̃_last > 0.7·τ → level = suggestion",
    "10 else                            → level = none",
]

for t in doc.tables:
    txt = ' '.join(c.text for row in t.rows for c in row.cells)
    if 'PrepareTrainingData' in txt:
        cell = t.rows[0].cells[0]
        cell.text = ""
        p0 = cell.paragraphs[0]
        p0.text = algo1_lines[0]
        p0.paragraph_format.space_before = Pt(0)
        p0.paragraph_format.space_after = Pt(1)
        p0.paragraph_format.line_spacing = 1.0
        style_run(p0.runs[0], font_name="Liberation Mono", size_pt=8.5)
        for l in algo1_lines[1:]:
            p = cell.add_paragraph(l)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            style_run(p.runs[0], font_name="Liberation Mono", size_pt=8.5)
        print("Repaired Algorithm 5.1 Table")
        
    elif 'EMA of S_t' in txt or 'max_gap' in txt:
        cell = t.rows[0].cells[0]
        cell.text = ""
        p0 = cell.paragraphs[0]
        p0.text = algo2_lines[0]
        p0.paragraph_format.space_before = Pt(0)
        p0.paragraph_format.space_after = Pt(1)
        p0.paragraph_format.line_spacing = 1.0
        style_run(p0.runs[0], font_name="Liberation Mono", size_pt=8.5)
        for l in algo2_lines[1:]:
            p = cell.add_paragraph(l)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            style_run(p.runs[0], font_name="Liberation Mono", size_pt=8.5)
        print("Repaired Algorithm 5.2 Table")

# ==================== 5. REPAIR SECTION 20 (PAGE 20) ====================
# P155 is '3.1 Problem Definition'
# P156..P160: body paragraphs and list
# P161 is '3.2 Formal Formulation'
p156 = doc.paragraphs[156]
p156.text = (
    "Current computational sentiment systems are predominantly designed around coarse polarity "
    "(positive/negative/neutral) or single-message classification, rarely accounting for the severe "
    "class imbalances inherent in fine-grained human affect. In mental wellness monitoring, an "
    "isolated frustrated statement is not an actionable clinical signal; rather, the critical marker is a "
    "sustained pattern of elevated negative affect extending across days or weeks."
)
p156.paragraph_format.space_before = Pt(0)
p156.paragraph_format.space_after = Pt(3)
p156.paragraph_format.line_spacing = 1.05
style_run(p156.runs[0], font_name="Liberation Serif", size_pt=10.0)

p157 = doc.paragraphs[157]
p157.text = (
    "Compounding this issue, many individuals hesitate to seek professional counseling due to social "
    "stigma, high costs, or limited provider availability. An ethical, non-diagnostic text-monitoring "
    "system can serve as an accessible early-awareness mirror, helping users observe their emotional "
    "trajectories and encouraging timely care."
)
p157.paragraph_format.space_before = Pt(0)
p157.paragraph_format.space_after = Pt(3)
p157.paragraph_format.line_spacing = 1.05
style_run(p157.runs[0], font_name="Liberation Serif", size_pt=10.0)

p158 = doc.paragraphs[158]
p158.text = "The problem addressed by this project is therefore to develop an automated, privacy-conscious system that can:"
p158.paragraph_format.space_before = Pt(0)
p158.paragraph_format.space_after = Pt(2)
p158.paragraph_format.line_spacing = 1.05
style_run(p158.runs[0], font_name="Liberation Serif", size_pt=10.0)

p159 = doc.paragraphs[159]
p159.text = "1.  Classify the emotional state expressed in everyday natural text into fine-grained, actionable categories with balanced predictive performance across rare and majority emotion classes;"
p159.paragraph_format.left_indent = Inches(0.25)
p159.paragraph_format.space_before = Pt(0)
p159.paragraph_format.space_after = Pt(1.5)
p159.paragraph_format.line_spacing = 1.05
style_run(p159.runs[0], font_name="Liberation Serif", size_pt=10.0)

p160 = doc.paragraphs[160]
p160.text = "2.  Aggregate message-level inferences into interpretable longitudinal trajectories across daily, weekly, and monthly intervals; and"
p160.paragraph_format.left_indent = Inches(0.25)
p160.paragraph_format.space_before = Pt(0)
p160.paragraph_format.space_after = Pt(1.5)
p160.paragraph_format.line_spacing = 1.05
style_run(p160.runs[0], font_name="Liberation Serif", size_pt=10.0)

# Add item 3 before P161
p161 = doc.paragraphs[161]
new_p3 = doc.add_paragraph()
new_p3.text = "3.  Provide supportive, non-diagnostic reflections when sustained negative affect is identified, directing individuals toward accredited professional care rather than attempting clinical intervention."
new_p3.paragraph_format.left_indent = Inches(0.25)
new_p3.paragraph_format.space_before = Pt(0)
new_p3.paragraph_format.space_after = Pt(3)
new_p3.paragraph_format.line_spacing = 1.05
style_run(new_p3.runs[0], font_name="Liberation Serif", size_pt=10.0)
p161._p.addprevious(new_p3._p)

print("Repaired Section 20 (Page 20)")

# ==================== 6. REPAIR SECTION 7 (PAGE 7: NOMENCLATURE) ====================
# P42: WHO, P43: 'Nomenclature (symbols)', P44: merged symbols, P45: 'x', P46: sect break
p42 = doc.paragraphs[42]
make_defn_entry(p42, "WHO", "World Health Organization", tab_pos_in=1.5)

p43 = doc.paragraphs[43]
p43.text = "Nomenclature (symbols)"
p43.paragraph_format.space_before = Pt(6)
p43.paragraph_format.space_after = Pt(3)
style_run(p43.runs[0], font_name="Liberation Sans", size_pt=11, bold=True)

nomenclature_list = [
    ("E", "Set of the eight emotion classes; |E| = 8"),
    ("N", "Negative-affect subset {Sadness, Anger, Fear, Anxiety, Stress}"),
    ("xi, ti", "i-th message and its timestamp"),
    ("pi(e)", "Predicted probability of emotion e for message i"),
    ("si", "Negative-affect score of message i (sum of pi over N)"),
    ("S(W), St", "Mean negative-affect score of window W / of day t"),
    ("S̃t", "Exponentially smoothed daily score"),
    ("α", "EMA smoothing factor (0.3)"),
    ("τ", "Escalation threshold on the smoothed score (0.55)"),
    ("k", "Number of consecutive entry-days required for escalation (3)"),
    ("wc, nc", "Loss weight and training count of class c"),
    ("M", "Number of training examples"),
    ("C", "Regularisation parameter of LR / SVM"),
    ("TE, σ2", "PERT expected duration and variance"),
]

p44 = doc.paragraphs[44]
make_defn_entry(p44, nomenclature_list[0][0], nomenclature_list[0][1], tab_pos_in=1.5)

p_f_x = doc.paragraphs[45]
p_f_x.text = "x"
p_f_x.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
style_run(p_f_x.runs[0], font_name="Liberation Serif", size_pt=10)

for sym, defn in nomenclature_list[1:]:
    new_p = doc.add_paragraph()
    make_defn_entry(new_p, sym, defn, tab_pos_in=1.5)
    p_f_x._p.addprevious(new_p._p)

print("Repaired Section 7 (Page 7)")

# ==================== 7. REPAIR SECTION 6 (PAGE 6: ABBREVIATIONS) ====================
# P36: 'ABBREVIATIONS AND NOMENCLATURE'
# P37: 'Abbreviations'
# P38: merged abbreviations
# P39: 'ix'
# P40: sect break
p36 = doc.paragraphs[36]
p36.text = "ABBREVIATIONS AND NOMENCLATURE"
p36.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
p36.paragraph_format.space_before = Pt(0)
p36.paragraph_format.space_after = Pt(8)
style_run(p36.runs[0], font_name="Liberation Sans", size_pt=14, bold=True)

p37 = doc.paragraphs[37]
p37.text = "Abbreviations"
p37.paragraph_format.space_before = Pt(0)
p37.paragraph_format.space_after = Pt(3)
style_run(p37.runs[0], font_name="Liberation Sans", size_pt=11, bold=True)

p38 = doc.paragraphs[38]
make_defn_entry(p38, abbreviations_list[0][0], abbreviations_list[0][1], tab_pos_in=1.5)

p_f_ix = doc.paragraphs[39]
p_f_ix.text = "ix"
p_f_ix.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
style_run(p_f_ix.runs[0], font_name="Liberation Serif", size_pt=10)

for term, defn in abbreviations_list[1:]:
    new_p = doc.add_paragraph()
    make_defn_entry(new_p, term, defn, tab_pos_in=1.5)
    p_f_ix._p.addprevious(new_p._p)

print("Repaired Section 6 (Page 6)")

# ==================== 8. REPAIR SECTION 5 (PAGE 5: LIST OF FIGURES) ====================
# P31: 'LIST OF FIGURES', P32: merged figures, P33: 'viii', P34: sect break
p31 = doc.paragraphs[31]
p31.text = "LIST OF FIGURES"
p31.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
p31.paragraph_format.space_before = Pt(0)
p31.paragraph_format.space_after = Pt(12)
style_run(p31.runs[0], font_name="Liberation Sans", size_pt=14, bold=True)

p32 = doc.paragraphs[32]
make_toc_entry(p32, f"{lof_entries[0][0]}    {lof_entries[0][1]}", lof_entries[0][2], level=0, is_chapter=False)

p_f_viii = doc.paragraphs[33]
p_f_viii.text = "viii"
p_f_viii.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
style_run(p_f_viii.runs[0], font_name="Liberation Serif", size_pt=10)

for fig_id, fig_title, fig_pg in lof_entries[1:]:
    new_p = doc.add_paragraph()
    make_toc_entry(new_p, f"{fig_id}    {fig_title}", fig_pg, level=0, is_chapter=False)
    p_f_viii._p.addprevious(new_p._p)

print("Repaired Section 5 (Page 5)")

# ==================== 9. REPAIR SECTION 4 (PAGE 4: LIST OF TABLES) ====================
# P26: 'LIST OF TABLES', P27: merged tables, P28: 'vii', P29: sect break
p26 = doc.paragraphs[26]
p26.text = "LIST OF TABLES"
p26.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
p26.paragraph_format.space_before = Pt(0)
p26.paragraph_format.space_after = Pt(12)
style_run(p26.runs[0], font_name="Liberation Sans", size_pt=14, bold=True)

p27 = doc.paragraphs[27]
make_toc_entry(p27, f"{lot_entries[0][0]}    {lot_entries[0][1]}", lot_entries[0][2], level=0, is_chapter=False)

p_f_vii = doc.paragraphs[28]
p_f_vii.text = "vii"
p_f_vii.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
style_run(p_f_vii.runs[0], font_name="Liberation Serif", size_pt=10)

for tbl_id, tbl_title, tbl_pg in lot_entries[1:]:
    new_p = doc.add_paragraph()
    make_toc_entry(new_p, f"{tbl_id}    {tbl_title}", tbl_pg, level=0, is_chapter=False)
    p_f_vii._p.addprevious(new_p._p)

print("Repaired Section 4 (Page 4)")

# ==================== 10. REPAIR SECTION 3 (PAGE 3: ToC PART 3) ====================
# P20: merged entries, P21: merged, P22: merged, P23: 'vi', P24: sect break
p20 = doc.paragraphs[20]
make_toc_entry(p20, toc_page_3[0][0], toc_page_3[0][1], level=toc_page_3[0][2], is_chapter=toc_page_3[0][3])

p21 = doc.paragraphs[21]
make_toc_entry(p21, toc_page_3[1][0], toc_page_3[1][1], level=toc_page_3[1][2], is_chapter=toc_page_3[1][3])

p22 = doc.paragraphs[22]
make_toc_entry(p22, toc_page_3[2][0], toc_page_3[2][1], level=toc_page_3[2][2], is_chapter=toc_page_3[2][3])

p_f_vi = doc.paragraphs[23]
p_f_vi.text = "vi"
p_f_vi.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
style_run(p_f_vi.runs[0], font_name="Liberation Serif", size_pt=10)

for entry in toc_page_3[3:]:
    new_p = doc.add_paragraph()
    make_toc_entry(new_p, entry[0], entry[1], level=entry[2], is_chapter=entry[3])
    p_f_vi._p.addprevious(new_p._p)

print("Repaired Section 3 (Page 3)")

# ==================== 11. REPAIR SECTION 2 (PAGE 2: ToC PART 2) ====================
# P15: merged, P16: merged, P17: merged, P18: sect break
# We will insert footer 'v' before P18, and entries between P15 and P18
p15 = doc.paragraphs[15]
make_toc_entry(p15, toc_page_2[0][0], toc_page_2[0][1], level=toc_page_2[0][2], is_chapter=toc_page_2[0][3])

p16 = doc.paragraphs[16]
make_toc_entry(p16, toc_page_2[1][0], toc_page_2[1][1], level=toc_page_2[1][2], is_chapter=toc_page_2[1][3])

p17 = doc.paragraphs[17]
make_toc_entry(p17, toc_page_2[2][0], toc_page_2[2][1], level=toc_page_2[2][2], is_chapter=toc_page_2[2][3])

p18 = doc.paragraphs[18]
p_f_v = doc.add_paragraph()
p_f_v.text = "v"
p_f_v.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
style_run(p_f_v.runs[0], font_name="Liberation Serif", size_pt=10)
p18._p.addprevious(p_f_v._p)

for entry in toc_page_2[3:]:
    new_p = doc.add_paragraph()
    make_toc_entry(new_p, entry[0], entry[1], level=entry[2], is_chapter=entry[3])
    p_f_v._p.addprevious(new_p._p)

print("Repaired Section 2 (Page 2)")

# ==================== 12. REPAIR SECTION 1 (PAGE 1: ToC PART 1) ====================
# P1: 'TABLE OF CONTENTS'
# P2..P11: replace with first 10 entries of toc_page_1
# P12: 'iv'
# P13: sect break
p1 = doc.paragraphs[1]
p1.text = "TABLE OF CONTENTS"
p1.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
p1.paragraph_format.space_before = Pt(0)
p1.paragraph_format.space_after = Pt(12)
style_run(p1.runs[0], font_name="Liberation Sans", size_pt=14, bold=True)

p_f_iv = doc.paragraphs[12]
p_f_iv.text = "iv"
p_f_iv.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
style_run(p_f_iv.runs[0], font_name="Liberation Serif", size_pt=10)

idx_entry = 1
for p_i in range(2, 12):
    entry = toc_page_1[idx_entry]
    make_toc_entry(doc.paragraphs[p_i], entry[0], entry[1], level=entry[2], is_chapter=entry[3])
    idx_entry += 1

while idx_entry < len(toc_page_1):
    entry = toc_page_1[idx_entry]
    new_p = doc.add_paragraph()
    make_toc_entry(new_p, entry[0], entry[1], level=entry[2], is_chapter=entry[3])
    p_f_iv._p.addprevious(new_p._p)
    idx_entry += 1

print("Repaired Section 1 (Page 1)")

# Save docx
doc.save(DOCX_IN)
print(f"Successfully saved repaired document to {DOCX_IN}!")

# Export PDF via Word
print("Invoking Microsoft Word COM to verify pagination and export PDF...")
word = win32com.client.Dispatch("Word.Application")
word.Visible = False
word_doc = word.Documents.Open(os.path.abspath(DOCX_IN))
page_count = word_doc.ComputeStatistics(2)
print(f"Word reports exact page count: {page_count} pages")

pdf_path = os.path.abspath(PDF_OUT)
word_doc.SaveAs(pdf_path, FileFormat=17) # 17 = wdFormatPDF
word_doc.Close(False)
word.Quit()
print(f"Successfully exported PDF to {pdf_path}")

doc_fitz = fitz.open(PDF_OUT)
print(f"PyMuPDF verified PDF page count: {len(doc_fitz)} pages")
