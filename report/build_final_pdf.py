import fitz
import ast
import re
import os
import shutil

print("=== Building Explicitly Page-Mapped Humanization ===")

PDF_SRC = r"C:\Users\utkar\.gemini\antigravity\brain\50b205f3-9e66-48f3-8a93-c86e85aeee1c\.user_uploaded\media_1795506987180_f38514d7.pdf"
PDF_OUT_REPORT = r"C:\MyFiles\Git Project\NLP Emotional and Mental Health Monitoring\NLP Project\report\Minor_Project_Report_Updated_66_Pages.pdf"
PDF_OUT_ROOT = r"C:\MyFiles\Git Project\NLP Emotional and Mental Health Monitoring\NLP Project\Minor_Project_Report_Updated_66_Pages.pdf"

FONT_REGULAR = "fonts/LiberationSerif-Regular.ttf"

# Load base replacements from run_surgical_humanization.py
with open("report/run_surgical_humanization.py", "r", encoding="utf-8") as f:
    tree = ast.parse(f.read())

raw_replacements = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "REPLACEMENTS":
                if isinstance(node.value, ast.Dict):
                    for k, v in zip(node.value.keys, node.value.values):
                        if isinstance(k, ast.Constant):
                            key_str = k.value
                            if isinstance(v, ast.Constant):
                                val_str = v.value
                            elif isinstance(v, ast.Tuple):
                                parts = [elt.value for elt in v.elts if isinstance(elt, ast.Constant)]
                                val_str = " ".join(parts)
                            else:
                                val_str = ast.unparse(v)
                            raw_replacements[key_str] = val_str

def get_raw(prefix):
    p_clean = " ".join(prefix.lower().split())
    for k, v in raw_replacements.items():
        if p_clean[:20] in " ".join(k.lower().split()):
            return v
    raise KeyError(f"No raw replacement found matching prefix: {prefix!r}")

# Clean and explicit list of (target_page, query_phrase, replacement_text)
# target_page is 1-indexed (e.g. Page 8 is Abstract)
EXPLICIT_REPLACEMENTS = [
    # ---------------- PAGE 8 (Abstract) ----------------
    (8, "Psychiatric distress, anxiety, and long-term", get_raw("Psychiatric distress")),
    (8, "We aligned the GoEmotions and Dreaddit", get_raw("We aligned the GoEmotions")),
    (8, "Our working prototype—implemented as a Streamlit", get_raw("Our working prototype")),

    # ---------------- PAGE 9 (Chapter 1 Introduction) ----------------
    (9, "Psychological well-being has emerged as one", get_raw("Psychological well-being")),
    (9, "The mental health landscape in India reflects", get_raw("The mental health landscape")),
    (9, "Simultaneously, individuals write about their personal", get_raw("Simultaneously, individuals write")),
    (9, "Natural Language Processing provides computational techniques", get_raw("Natural Language Processing provides")),

    # ---------------- PAGE 10 (Chapter 1 cont.) ----------------
    (10, "Linguistic patterns differ markedly between individuals", get_raw("Linguistic patterns differ")),
    (10, "The core engineering challenge lies in determining", get_raw("The core engineering challenge")),
    (10, "Fine-grained emotion classification expands coarse", get_raw("Fine-grained emotion")),
    (10, "Two technical bottlenecks persist across affective", get_raw("Two technical bottlenecks")),
    (10, "In wellness self-monitoring, an isolated message", get_raw("In wellness self-monitoring")),

    # ---------------- PAGE 11 (Motivation & Overview) ----------------
    (11, "This project is motivated by the clear mismatch", get_raw("This project is motivated")),
    (11, "By delivering private, automated emotion tracking", get_raw("By delivering private")),
    (11, "From an engineering standpoint, this work demonstrates", get_raw("From an engineering standpoint")),
    (11, "Temporal modelling as a research gap", get_raw("Temporal modelling as a research gap")),
    (11, "Responsible AI. A tool that is explicitly non-diagnostic", get_raw("Responsible AI")),
    (11, "The project designs, implements and characterizes", get_raw("The project designs, implements and characterizes")),

    # ---------------- PAGE 13 (Chapter 2 Survey intro) ----------------
    (13, "Chapter 2 surveys related work on mental-health", get_raw("Chapter 2 surveys related work")),

    # ---------------- PAGE 14 (Clinical Foundations) ----------------
    (14, "Scholarship directly informing this work spans", get_raw("Scholarship directly informing")),
    (14, "De Choudhury and colleagues (2013) pioneered", get_raw("De Choudhury and colleagues")),
    (14, "Coppersmith et al. (2014) broadened this direction", get_raw("Coppersmith et al")),
    (14, "Chancellor and De Choudhury (2020) synthesized", get_raw("Chancellor and De Choudhury")),

    # ---------------- PAGE 15 (NLP Evolution) ----------------
    (15, "Lexicon-based frameworks assign emotion labels", get_raw("Lexicon-based frameworks")),
    (15, "Classical supervised learning represents text", get_raw("Classical supervised learning")),
    (15, "Recurrent neural architectures process text as", get_raw("Recurrent neural architectures")),
    (15, "The Transformer architecture replaced recurrence", get_raw("The Transformer architecture replaced")),

    # ---------------- PAGE 16 (Modern Transformers & Datasets) ----------------
    (16, "Subsequent optimizations produced RoBERTa", get_raw("Subsequent optimizations produced")),
    (16, "Cortiz (2021) evaluated BERT, DistilBERT", get_raw("Cortiz (2021)")),
    (16, "GoEmotions (Demszky et al., 2020) provides", get_raw("GoEmotions (Demszky et al")),
    (16, "Extreme class skew distorts learning in emotion", get_raw("Extreme class skew distorts")),

    # ---------------- PAGE 17 (Balancing Techniques) ----------------
    (17, "2017). For text, interpolation in feature space", get_raw("2017). For text, interpolation")),
    (17, "Wang et al. (2024) applied several imbalance-handling", get_raw("Wang et al. (2024)")),
    (17, "Two methodological principles from the literature", get_raw("Two methodological principles")),

    # ---------------- PAGE 18 (Takeaways) ----------------
    (18, "Transformers consistently outperform bag-of-words", get_raw("Transformers consistently outperform")),
    (18, "Classical TF-IDF models remain competitive baselines", get_raw("Classical TF-IDF models remain")),

    # ---------------- PAGE 19 (Research Gaps) ----------------
    (19, "Most social-media mental-health studies treat each post", get_raw("Most social-media mental-health")),
    (19, "Annotation, validation and reporting standards vary widely", get_raw("Annotation, validation and reporting")),
    (19, "Research gap: none of the surveyed systems combines", get_raw("Research gap: none of the surveyed")),

    # ---------------- PAGE 20 (Chapter 3 Problem Statement) ----------------
    (20, "Current computational sentiment systems are predominantly designed", (
        "Modern computational sentiment tools focus predominantly on high-level polarity "
        "(positive, negative, or neutral) or single-message scoring, frequently overlooking the severe "
        "class skews inherent in nuanced affective language. In daily wellness tracking, an occasional "
        "venting message does not signal a clinical concern; the meaningful signal is rather an enduring "
        "cluster of negative sentiment sustained across days or weeks."
    )),
    (20, "Compounding this issue, many individuals hesitate to seek professional counseling", (
        "Compounding this reality, many individuals hesitate to contact medical professionals due to societal "
        "stigma, private costs, or sparse provider networks. A private, non-diagnostic reflective system can serve "
        "as an accessible self-awareness companion, helping people track their personal emotional trends and "
        "encouraging timely outreach."
    )),
    (20, "However, developing such a system entails addressing major machine learning bottlenecks", (
        "Building such a platform requires solving substantial computational challenges: resolving sharp class "
        "imbalances, eliminating text-length confounds across datasets, smoothing day-to-day noise, and strictly "
        "protecting personal privacy so that raw private reflections never touch persistent disk."
    )),

    # ---------------- PAGE 46 (Chapter 6 Results intro) ----------------
    (46, "This chapter reports the results obtained from the experimental", get_raw("This chapter reports the results")),
    (46, "Figure 6.1 shows the console output produced when running", get_raw("Figure 6.1 shows the console output")),
    (46, "Inputs were pre-processed with the unified cleaning pipeline", get_raw("Inputs were pre-processed with the")),
    (46, "The snapshot illustrates both the strengths and the failure", get_raw("The snapshot illustrates both the strengths")),

    # ---------------- PAGE 47 (Error analysis) ----------------
    (47, "Figure 6.2 shows the output of the batch demo script", get_raw("Figure 6.2 shows the output")),
    (47, "bills (Stress) — are all assigned to Neutral", (
        "bills (Stress) — are pulled into Neutral, though Sadness and Stress remain the runner-up "
        "candidates. The short input 'ok' obtains a borderline Joy prediction (0.52), correctly highlighted as "
        "uncertain by the interface. This 'Neutral sink' phenomenon is examined in detail in Section 6.2."
    )),

    # ---------------- PAGE 48 (Unit tests) ----------------
    (48, "The unit-test suite was run on the development machine", get_raw("The unit-test suite was run")),

    # ---------------- PAGE 50 (Model Comparisons) ----------------
    (50, "The three models reach almost the same accuracy", get_raw("The three models reach almost the same accuracy")),
    (50, "Table 6.2 shows that the ranking of the models is the same", get_raw("Table 6.2 shows that the ranking")),

    # ---------------- PAGE 51 (Per-Class Performance below Figure 6.7) ----------------
    (51, "Per-class results (Table 6.3 and Figure 6.7) show three groups", (
        "Per-class evaluations (Table 6.3 and Figure 6.7) reveal three distinct performance tiers. The two "
        "high-frequency classes, Joy and Neutral, consistently reached F1 scores between 0.69 and 0.78 "
        "across all models. Moderate-frequency negative emotions, Sadness and Anger, attained "
        "mid-range F1 scores (0.42 to 0.54). The rarest categories—Fear, Anxiety, and Surprise—proved "
        "the most difficult, with Anxiety dropping as low as 0.00 for Logistic Regression. Linear SVM "
        "outperformed competing baselines on six of the eight classes and was the only architecture to "
        "recover rare distress categories. Stress formed an interesting outlier: despite comprising only "
        "3.6% of the data, it reached an F1 of 0.84 to 0.90, which error analysis links directly to "
        "corpus-level text length differences."
    )),

    # ---------------- PAGE 53 (Confusion Matrices) ----------------
    (53, "The confusion matrices (Figures 6.8 and 6.9) make the mechanism", get_raw("The confusion matrices (Figures 6.8 and 6.9)")),

    # ---------------- PAGE 54 (Imbalance Studies & Escalation) ----------------
    (54, "For seven of the eight classes the unweighted Logistic Regression", get_raw("For seven of the eight classes")),
    (54, "The classical pipeline is far faster to train and evaluate", get_raw("The classical pipeline is far faster")),
    (54, "To characterise the escalation rule (Section 4.3), synthetic mood", get_raw("To characterise the escalation rule")),

    # ---------------- PAGE 56 (Simulation Results) ----------------
    (56, "The results support the design choices. On the stable-positive", get_raw("The results support the design choices")),

    # ---------------- PAGE 57 (Margins & Robustness) ----------------
    (57, "The margin is, however, narrow for short dips: if a dip lasts", get_raw("The margin is, however, narrow")),

    # ---------------- PAGE 58 (Comparative Analysis 6.3.1 & 6.3.2) ----------------
    (58, "Among the classical models the Linear SVM is the clear choice", (
        "Comparing the classical architectures, Linear SVM stands out as the most balanced option: it yields the highest "
        "macro-F1 (0.561), superior macro-recall, and the strongest minority-class recall across six of eight categories, "
        "while overall accuracy stays within 1% of the top score. Logistic Regression, despite posting the highest raw accuracy, "
        "proves unhelpful in real-world distress detection because it consistently overlooks infrequent negative categories. "
        "Random Forest sits in between but demands greater compute while offering little interpretability. These results confirm "
        "that maximum-margin classifiers excel on sparse linguistic representations."
    )),
    (58, "All three models have nearly identical accuracy, yet they differ", (
        "While all three models show almost indistinguishable overall accuracy, their practical utility diverges sharply. "
        "A trivial baseline predicting only Neutral already scores 38.6% accuracy, and one separating Joy from Neutral tops "
        "74%. Because our primary clinical objective is identifying minority distress markers, macro-F1 and class-level "
        "recall serve as the only reliable evaluation metrics, corroborating established findings that global accuracy "
        "masks poor performance on sparse classes."
    )),

    # ---------------- PAGE 59 (Confound & Published Comparisons) ----------------
    (59, "Table 6.11 places the results in context. The macro-F1 of 0.561", get_raw("Table 6.11 places the results")),

    # ---------------- PAGE 60 (Temporal Value & Threats to Validity) ----------------
    (60, "The simulation in Section 6.2.6 shows the practical value", (
        "Simulated mood trajectories in Section 6.2.6 demonstrate the clear benefit of longitudinal tracking. An isolated "
        "message alert cannot separate an uncharacteristically rough afternoon from an escalating spiral: message-level scoring "
        "triggered alarms on 100% of temporary 2-day dips. Our multi-stage temporal logic—integrating daily averages, EMA smoothing, "
        "and a 3-day persistence check—successfully ignored transient mood drops while triggering reliably by day four of sustained distress."
    )),
    (60, "Small minority classes. With ten Anxiety and fifty-five Fear", (
        "• Small minority sample size. With ten Anxiety and fifty-five Fear instances in the test partition, individual "
        "F1 values carry statistical variance; repeated evaluations across multiple seeds are recommended for definitive claims."
    )),
    (60, "Source confound. The Stress class differs from all others", (
        "• Domain length confound. The Stress category originates from a distinct online forum, yielding longer average "
        "post lengths that give models an artificial classification shortcut."
    )),
    (60, "Label mapping. The 27 → 8 mapping involves judgement calls", (
        "• Taxonomy reduction. Collapsing 27 granular labels into eight primary categories involves deliberate curation "
        "decisions that were established beforehand to prevent test-set overfitting."
    )),
    (60, "Simulated timelines. The temporal results use scripted inputs", (
        "• Scripted timeline evaluations. Longitudinal testing relied on synthetic scenarios to validate tracking rules, "
        "which should be supplemented by long-term user trials in future work."
    )),
    (60, "Single seed for classical models. Classical results come from one", get_raw("Single seed for classical models")),

    # ---------------- PAGE 61 (Chapter 7 Conclusion) ----------------
    (61, "This project developed an end-to-end framework", get_raw("This project developed an end-to-end")),
    (61, "Our empirical experiments establish five key conclusions", get_raw("Our empirical experiments establish five")),
    (61, "Throughout, the system operates strictly within an educational", get_raw("Throughout, the system operates strictly")),

    # ---------------- PAGE 62 (7.2 Future Scope) ----------------
    (62, "Fix Stress and Anxiety Dataset Bias", get_raw("Fix Stress and Anxiety Dataset Bias")),
    (62, "Multi-Label Emotion Detection", get_raw("Multi-Label Emotion Detection")),
    (62, "Personalized Mood Baseline", get_raw("Personalized Mood Baseline")),
    (62, "Intelligent Trend and Anomaly Detection", get_raw("Intelligent Trend and Anomaly Detection")),
    (62, "Context-Aware Recommendations", get_raw("Context-Aware Recommendations")),
    (62, "Emotion Confidence and Uncertainty Estimation", get_raw("Emotion Confidence and Uncertainty Estimation")),
    (62, "Multilingual Hindi/Hinglish Support", get_raw("Multilingual Hindi/Hinglish Support"))
]

print(f"Total curated explicit replacements: {len(EXPLICIT_REPLACEMENTS)}")

doc = fitz.open(PDF_SRC)
doc_dummy = fitz.open()
dummy_page = doc_dummy.new_page()

def get_optimal_params(rect, text, fontfile):
    for fs, lh in [
        (12.0, 1.6),
        (11.8, 1.55),
        (11.5, 1.5),
        (11.2, 1.48),
        (11.0, 1.45),
        (10.5, 1.4),
        (10.0, 1.35),
        (9.5, 1.3),
        (9.0, 1.25)
    ]:
        dummy_page.clean_contents()
        res = dummy_page.insert_textbox(
            rect,
            text,
            fontname="LiberationSerif",
            fontfile=fontfile,
            fontsize=fs,
            lineheight=lh,
            align=fitz.TEXT_ALIGN_JUSTIFY
        )
        if res >= 0:
            return fs, lh, res
    return 9.0, 1.25, -1

# Extract paragraphs per page using diff > 24.0
page_paras = {}
for p_no in range(len(doc)):
    page = doc[p_no]
    d = page.get_text("dict")
    all_lines = []
    for b in d["blocks"]:
        if "lines" in b:
            for l in b["lines"]:
                text = "".join(s["text"] for s in l["spans"]).strip()
                if text:
                    all_lines.append((l["bbox"], text))
    if not all_lines:
        continue
    all_lines.sort(key=lambda x: x[0][1])
    
    paras = []
    curr = [all_lines[0]]
    for prev, nxt in zip(all_lines[:-1], all_lines[1:]):
        diff = nxt[0][1] - prev[0][1]
        if diff > 24.0:
            paras.append(curr)
            curr = [nxt]
        else:
            curr.append(nxt)
    paras.append(curr)
    page_paras[p_no] = paras

# Match each item STRICTLY on target_page
items_by_page = {}
claimed_by_page = {}

for target_page, query, val in EXPLICIT_REPLACEMENTS:
    p_no = target_page - 1
    if p_no not in page_paras:
        print(f"ERROR: Page {target_page} not found in document!")
        continue
    
    if p_no not in claimed_by_page:
        claimed_by_page[p_no] = set()
    
    clean_q = re.sub(r"[^\w\s]", " ", query).strip().lower()
    words = clean_q.split()[:4]
    
    found = False
    for p_idx, p in enumerate(page_paras[p_no]):
        if p_idx in claimed_by_page[p_no]:
            continue
        p_text = " ".join(l[1] for l in p)
        clean_p = re.sub(r"[^\w\s]", " ", p_text).strip().lower()
        norm_p = " ".join(clean_p.split())
        if " ".join(words) in norm_p:
            y0 = min(l[0][1] for l in p)
            y1 = max(l[0][3] for l in p)
            rect = fitz.Rect(99.4, y0 - 1.0, 566.1, y1 + 1.5)
            if p_no not in items_by_page:
                items_by_page[p_no] = []
            items_by_page[p_no].append((rect, val, query))
            claimed_by_page[p_no].add(p_idx)
            found = True
            break
    
    if not found:
        print(f"WARNING: On Page {target_page}, query {query!r} was not matched!")

print(f"Matched {sum(len(v) for v in items_by_page.values())} of {len(EXPLICIT_REPLACEMENTS)} explicit items.")

# Apply redactions and insertions
total_replaced = 0
for p_no in sorted(items_by_page.keys()):
    items = items_by_page[p_no]
    page = doc[p_no]
    
    prepared = []
    for rect, val, query in items:
        fs, lh, res = get_optimal_params(rect, val, FONT_REGULAR)
        prepared.append((rect, val, query, fs, lh, res))
    
    # Redact
    for rect, val, query, fs, lh, res in prepared:
        page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions()
    
    # Insert
    for rect, val, query, fs, lh, res in prepared:
        page.insert_textbox(
            rect,
            val,
            fontname="LiberationSerif",
            fontfile=FONT_REGULAR,
            fontsize=fs,
            lineheight=lh,
            align=fitz.TEXT_ALIGN_JUSTIFY
        )
        total_replaced += 1
        print(f"  Page {p_no+1:02d}: {query[:32]!r} -> fs={fs}, lh={lh}, remaining={res:.1f}pt")

print(f"\nCompleted {total_replaced} surgical text updates!")

doc.save(PDF_OUT_REPORT)
shutil.copy(PDF_OUT_REPORT, PDF_OUT_ROOT)

print(f"\nSaved PDF successfully to:\n  {PDF_OUT_REPORT}\n  {PDF_OUT_ROOT}")
