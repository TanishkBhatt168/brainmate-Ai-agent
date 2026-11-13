# 🧠 Brainmate — AI Research Assistant
## Phase 1: Dataset Cleaning, Machine Learning Model Training & Evaluation

---

## 1. Project Introduction

**Brainmate** is an AI Research Assistant designed to assist researchers, academics, and students in navigating, organizing, and synthesizing scientific literature. 

In this foundational phase (**Phase 1**), we implement the core machine learning classification engine: an end-to-end natural language processing (NLP) pipeline that analyzes the title and abstract of scientific research papers from arXiv and automatically categorizes them into their corresponding subject field (e.g., Computer Vision, Computation & Language / NLP, Artificial Intelligence, Robotics, Cryptography).

---

## 2. Phase 1 Objectives

1. **Exploratory Data Analysis (EDA):** Deeply analyze the raw 430 MB arXiv research paper dataset, distinguish physical file lines from true tabular records, assess class imbalance, and audit missing data.
2. **Robust Data Cleaning:** Normalize whitespace, remove non-informative retracted paper placeholders, eliminate exact duplicate publications, and preserve valid arXiv version revisions.
3. **Memory-Conscious Sampling:** Create a category-diversity-preserving stratified sample (20,000 papers) that runs efficiently on 16 GB RAM while safely handling rare categories.
4. **Feature Engineering via TF-IDF:** Transform combined paper titles and abstracts into numerical sparse feature vectors while strictly preventing target leakage.
5. **Model Training & Comparison:** Implement unified Scikit-learn pipelines to train and compare three distinct supervised multiclass classifiers:
   - **Multinomial Naive Bayes (`MultinomialNB`)**
   - **Linear Support Vector Classification (`LinearSVC`)**
   - **Logistic Regression (`LogisticRegression`)**
6. **Rigorous Evaluation:** Evaluate models on held-out test data using Accuracy, Macro/Weighted Precision, Recall, and F1-scores.
7. **Model Persistence & Real-World Inference:** Save the best pipeline using `joblib` and provide an interactive inference function for predicting new research papers.

---

## 3. Problem Statement

Given:
- **Input Features ($X$):** A research paper's `title` and `summary` (abstract).
- **Target Label ($y$):** The primary scientific subject code (`category_code`).

**Goal:** Train a supervised multiclass text classification model that maps the combined textual description $(X = \text{title} + \text{summary})$ to one of the 6 core AI and Computer Science categories ($y \in \mathcal{C}$):
1. **`cs.CV`** — Computer Vision and Pattern Recognition
2. **`cs.CL`** — Computation and Language (Natural Language Processing)
3. **`cs.LG`** — Machine Learning
4. **`cs.AI`** — Artificial Intelligence
5. **`cs.RO`** — Robotics
6. **`cs.CR`** — Cryptography and Security

> [!IMPORTANT]
> **Target Leakage Prevention:** `category` and `category_code` represent the exact same underlying label in different formats (e.g., `"cs.AI"` vs `"Artificial Intelligence"`). Neither field is ever provided as an input feature during model training or inference.

---

## 4. Dataset Description & Column Meanings

The dataset (`arXiv_scientific_dataset.csv`, ~410.25 MB on disk) contains 287,421 published scientific paper records.

| Column | Data Type | Description | Role in ML Pipeline |
| :--- | :--- | :--- | :--- |
| `id` | `str` | Unique arXiv identifier (e.g., `cs-9308101v1`, `abs-1408.1485v1`) | Record identifier & version auditing |
| `title` | `str` | Title of the research paper | **Input Feature** |
| `category` | `str` | Human-readable scientific domain (e.g., `Artificial Intelligence`) | Label interpretation & human mapping |
| `category_code` | `str` | Official arXiv category taxonomy code (e.g., `cs.AI`, `cs.CV`) | **Target Variable ($y$)** |
| `published_date` | `str` | Original submission date (e.g., `8/1/93`) | Metadata |
| `updated_date` | `str` | Last update or revision timestamp | Metadata |
| `authors` | `str` | List of author names | Metadata |
| `first_author` | `str` | Primary author name | Metadata |
| `summary` | `str` | Full research abstract written by the authors | **Input Feature** |
| `summary_word_count` | `int64` | Word count of the abstract | Data auditing & length profiling |

### Critical Finding: Physical Lines vs. Tabular Records
- **Physical Lines on Disk:** `2,316,189`
- **Actual Tabular CSV Records:** `287,421`

Scientific abstracts contain multiple paragraphs separated by newline characters (`\n`). Inside a CSV file, these multi-line paragraphs are enclosed in quotes (`"..."`). Standard text line-counting utilities (e.g. `wc -l` or iterating raw file lines) incorrectly report 2.3 million lines. Our parser uses chunked CSV parsing to properly group each paper into a single structured row.

---

## 5. Project Folder Structure

```
training model/
├── arXiv_scientific_dataset.csv     # Original 430 MB dataset (preserved & untouched)
├── arXiv_cleaned_sample_20k.csv     # 20k diversity-preserving cleaned sample (51.5 MB)
├── 01_understand_data.py            # Step 1: Exploratory Data Analysis & Schema Inspection
├── 02_clean_data.py                 # Step 2: Streaming Cleaning, Deduplication & Sampling
├── 03_train_evaluate.py             # Steps 3-7: TF-IDF Pipeline, Model Training, Evaluation & CLI
├── requirements.txt                 # Minimal project dependencies (scikit-learn, pandas, numpy, joblib)
├── README.md                        # Comprehensive project documentation
└── results/
    ├── dataset_analysis.txt         # Detailed findings from 01_understand_data.py
    ├── cleaning_report.txt          # Detailed audit log from 02_clean_data.py
    ├── model_evaluation.txt         # Full benchmark metrics & classification report
    └── trained_model.joblib         # Saved Scikit-learn Pipeline (LinearSVC + TF-IDF)
```

---

## 6. Data-Cleaning Process

The data cleaning pipeline in `02_clean_data.py` inspects every row and enforces data quality while strictly keeping the original CSV untouched:

1. **Whitespace Normalization:**
   - Strips leading and trailing whitespace.
   - Collapses newline characters (`\n`, `\r`), tabs (`\t`), and multi-spaces into single clean spaces.
   - Retains all scientific terms, punctuation, mathematical expressions, and acronyms without aggressive word dropping.
2. **Removal of Retracted / Non-Informative Placeholders (20 records):**
   - Several papers in arXiv are officially retracted or withdrawn, leaving placeholder abstracts such as:
     - `"This paper has been withdrawn"`
     - `"This paper has been retracted."`
     - `"#NAME?"`
     - `"This article is taken out."`
   - These records do not contain scientific abstracts and are safely removed.
3. **Exact Duplicate Text Removal (38 records):**
   - Papers sharing identical normalized `(title, summary)` content are deduplicated to prevent data leakage between train and test splits.
4. **ID Versioning Analysis:**
   - arXiv assigns IDs with version suffixes (e.g. `...v1`, `...v2`). Papers sharing a base ID but possessing distinct abstracts represent legitimate author revisions and are preserved.
5. **Memory Management & Stratified Sampling:**
   - Generates a balanced 20,000-record dataset (`arXiv_cleaned_sample_20k.csv`).
   - Every eligible category receives a guaranteed quota of at least 5 papers to preserve category diversity.
   - Ultra-rare categories (<5 total papers across the entire 287k dataset) are excluded so that 80/20 train/test splitting never encounters single-instance classes.

---

## 7. TF-IDF Explained with a Simple Example

Computers cannot process raw English sentences directly; text must be converted into numerical vectors. We use **TF-IDF (Term Frequency – Inverse Document Frequency)**.

TF-IDF evaluates how important a word is to a specific research paper within the entire corpus:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

1. **Term Frequency ($\text{TF}$):** How often word $t$ appears in document $d$.
   $$\text{TF}(t, d) = \frac{\text{Count of } t \text{ in } d}{\text{Total words in } d}$$
2. **Inverse Document Frequency ($\text{IDF}$):** Measures how rare the word is across all documents $D$.
   $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### Intuitive Example
Consider two papers:
- **Paper A:** "This paper introduces a convolutional neural network for image segmentation."
- **Paper B:** "This paper discusses quantum state measurement in superconducting qubits."

Words like *"this"*, *"paper"*, and *"for"* appear in almost every research paper. Their IDF score is near **zero**, giving them virtually no influence on the classification. 

In contrast:
- *"Convolutional"* and *"segmentation"* appear almost exclusively in **Computer Vision (`cs.CV`)** papers. Their IDF score is very **high**.
- *"Superconducting"* and *"qubits"* appear in **Quantum Physics (`quant-ph`)**. Their IDF score is very **high**.

### Our TF-IDF Configuration
- `max_features=25000`: Restricts vocabulary to the 25,000 most informative tokens, preventing RAM exhaustion.
- `ngram_range=(1, 2)`: Captures single words (*"transformer"*) and 2-word phrases (*"deep learning"*, *"object detection"*).
- `stop_words='english'`: Eliminates non-informative filler words.
- `sublinear_tf=True`: Uses $1 + \log(\text{TF})$ to prevent a word repeated 20 times from dominating 20x more than a word mentioned once.

---

## 8. Classification Algorithms Explained

We evaluate three foundational algorithms for multiclass text categorization:

### 1. Multinomial Naive Bayes (`MultinomialNB`)
- **How it works:** Uses Bayes' Theorem under the assumption that word occurrences are conditionally independent given the category:
  $$P(C \mid d) \propto P(C) \prod_{i=1}^{n} P(w_i \mid C)$$
- **Strengths:** Very fast training, low memory footprint, excellent baseline for text categorization.
- **Limitation:** The "naive" independence assumption ignores phrase structure and correlations between words.

### 2. Linear Support Vector Classification (`LinearSVC`)
- **How it works:** Finds the optimal hyperplane in high-dimensional space that maximizes the geometric margin (distance) between different classes:
  $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$$
- **Strengths:** Exceptionally strong on high-dimensional sparse text vectors (25,000 TF-IDF features). Regularization prevents overfitting even with a large vocabulary.
- **Why it shines here:** Linear SVM achieved the **highest accuracy (64.98%)** and the **highest Macro F1-score (0.1215)** across all 130 categories.

### 3. Logistic Regression (`LogisticRegression`)
- **How it works:** Fits a linear model with a softmax function to model the probability of each category:
  $$P(Y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$
- **Strengths:** Calibrated probabilistic outputs and smooth gradient optimization.
- **Limitation:** Optimization with L-BFGS across 130 classes is computationally heavier (~120 seconds vs. 23 seconds for LinearSVC).

---

## 9. Training and Testing Split

To evaluate true generalization to unseen scientific papers:
- **Dataset Size:** 20,000 papers across 130 scientific fields.
- **Split Ratio:** **80% Training (16,000 papers)** and **20% Testing (4,000 papers)**.
- **Stratification:** Stratified splitting guarantees that every category has exactly an 80/20 train/test representation.
- **Data Leakage Prohibition:** Preprocessing (`TfidfVectorizer.fit`) is performed strictly on the training set. The test set is only transformed (`transform`), never fitted.
- **Reproducibility:** A fixed random seed (`random_state=42`) is used across all steps.

---

## 10. Evaluation Metrics Explained

In multiclass classification with severe category imbalance, no single number tells the full story. We calculate:

1. **Accuracy:**
   $$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$
   *Limitation:* In an imbalanced dataset where the top 4 classes account for 75% of papers, a naive model predicting only the top classes could score high accuracy while completely failing on the other 126 classes.
2. **Precision (Macro):**
   $$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$
   Measures prediction quality: when the model predicts a paper belongs to `cs.RO` (Robotics), how often is it right?
3. **Recall (Macro):**
   $$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$
   Measures coverage: of all true `cs.RO` papers, what proportion did the model successfully find?
4. **Macro F1-Score (Primary Selection Metric):**
   $$\text{Macro F1} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \frac{2 \times \text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
   Calculates the harmonic mean of precision and recall for each individual category, then averages across all 130 categories equally. This penalizes models that ignore smaller scientific domains.
5. **Weighted F1-Score:**
   Averages F1-scores weighted by the number of papers in each category in the test set.

---

## 11. Actual Measured Evaluation Results

All three models were trained on identical balanced training sets (16,000 papers, ~2,667 per category) and evaluated on identical test sets (4,000 papers, ~667 per category) with `random_state=42`.

| Model | Training Time | Test Accuracy | Macro F1-Score | Weighted F1-Score | Macro Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 13.62 s | 81.38% | 0.8136 | 0.8135 | 0.8138 |
| **Linear Support Vector Machine (LinearSVC)** | 14.61 s | 81.95% | 0.8186 | 0.8185 | 0.8179 |
| **Logistic Regression** 🏆 | **18.45 s** | **82.47%** | **0.8248** | **0.8247** | **0.8249** |

### Per-Category Classification Performance (Champion: Logistic Regression)

| Category Code | Subject Domain | Precision | Recall | F1-Score | Test Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`cs.CR`** | Cryptography and Security | 90% | 92% | **0.91** | 666 |
| **`cs.RO`** | Robotics | 90% | 89% | **0.90** | 666 |
| **`cs.CV`** | Computer Vision & Pattern Recognition | 86% | 85% | **0.85** | 667 |
| **`cs.CL`** | Computation & Language (NLP) | 83% | 83% | **0.83** | 667 |
| **`cs.LG`** | Machine Learning | 74% | 75% | **0.74** | 667 |
| **`cs.AI`** | Artificial Intelligence | 71% | 71% | **0.71** | 667 |

### Why Logistic Regression was Selected
1. **Highest Accuracy & Macro F1:** Attained **82.47% Accuracy** and **0.8248 Macro F1-score**, leading across all metrics.
2. **True Class Balance:** Because the training set is balanced across all 6 domains, Macro F1 and Weighted F1 are virtually identical (0.8248 vs 0.8247).
3. **Calibrated Probabilistic Confidence:** Provides native softmax probabilities for displaying confidence percentages and ranking categories for unseen papers.

---

## 12. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Recommended environment: Anaconda / Python virtual environment

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

The contents of `requirements.txt` are:
```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

---

## 13. Step-by-Step Execution Guide

Run each script sequentially in your terminal:

### Step 1: Understand the Dataset (EDA)
Inspects the schema, counts records vs. physical lines, missing values, duplicates, and category distributions:
```bash
python 01_understand_data.py
```
*Output generated:* `results/dataset_analysis.txt`

### Step 2: Clean and Preprocess the Dataset
Cleans text, strips retracted placeholders, deduplicates records, and generates the 20k diversity sample:
```bash
python 02_clean_data.py
```
*Outputs generated:* `arXiv_cleaned_sample_20k.csv`, `results/cleaning_report.txt`

### Step 3–6: Train, Evaluate, Compare & Save Best Model
Trains all three models, generates the comparison table, saves the best model, and runs verification:
```bash
python 03_train_evaluate.py
```
*Outputs generated:* `results/trained_model.joblib`, `results/model_evaluation.txt`

---

## 14. How to Make a Prediction for a New Paper

You can test new research papers using the built-in CLI inference mode in `03_train_evaluate.py`:

```bash
python 03_train_evaluate.py --predict --title "YOLOv8: Real-Time Object Detection and Image Segmentation" --summary "Object detection and instance segmentation are foundational computer vision problems. We present YOLOv8, a cutting-edge real-time object detection model that achieves state-of-the-art accuracy on COCO benchmark."
```

### Verified Live Output:
```
==============================================================================
BRAINMATE — INFERENCE MODE: TEST A NEW RESEARCH PAPER
==============================================================================

Paper Title:     YOLOv8: Real-Time Object Detection and Image Segmentation
Predicted Code:  cs.CV
Predicted Name:  Computer Vision and Pattern Recognition
Disclaimer:      Note: This prediction is a machine learning statistical estimate and may be incorrect.
```

### Testing Another Scientific Domain (Robotics):
```bash
python 03_train_evaluate.py --predict --title "Autonomous Quadrotor Navigation in GPS-Denied Environments" --summary "We propose a vision-based navigation system for unmanned aerial vehicles flying through complex indoor environments without GPS availability using onboard stereo cameras and IMU sensors."
```
```
Predicted Code:  cs.RO
Predicted Name:  Robotics
```

---

## 15. Limitations of the Current Model

1. **Single-Label Restriction:** Research papers frequently cross multiple disciplines (e.g., a paper applying NLP to biomedical records spans `cs.CL` and `q-bio`). The current model assigns only a single primary category.
2. **Category Imbalance:** While stratified sampling preserves categories, arXiv is naturally heavily skewed toward Computer Science (`cs.LG`, `cs.CV`, `cs.AI`). Categories with very few papers inherently achieve lower individual F1-scores.
3. **Contextual Semantics:** TF-IDF represents bags of unigrams and bigrams, missing deeper contextual nuances, long-range syntax, and subtle mathematical reasoning.

---

## 16. Future Brainmate Development Plans

- **Phase 2 (Completed):** Complete modern web application with Next.js, Express.js, and Flask ML service.
- **Phase 3:** Neural embeddings (e.g., SciBERT, Sentence-Transformers) for semantic representation.
- **Phase 4:** Retrieval-Augmented Generation (RAG) system with local LLM integration to synthesize insights across scientific papers.
- **Phase 5:** PDF upload parsing and citation network graph visualization.

---

# 🚀 Phase 2: Complete AI Research Assistant Web Application

Brainmate has been expanded into a complete, modern, professional AI-powered Research Assistant web application inspired by ChatGPT.

## 17. System Architecture

```
Next.js Frontend (Port 3000)
       |  HTTP / JSON
       v
Node.js + Express Backend (Port 5000)
       |  HTTP / REST
       v
Python Flask ML Service (Port 8000)
       |  In-memory TF-IDF + Logistic Regression
       +---> Existing Trained Model (results/trained_model.joblib)
       +---> Existing Dataset (arXiv_scientific_dataset.csv)
```

### Technology Stack:
- **Frontend:** Next.js 14, React 18, Tailwind CSS, Lucide React icons, browser `localStorage` (no database).
- **Backend:** Node.js, Express.js, CORS, Axios, REST API routing.
- **Machine Learning Service:** Python, Flask, Scikit-learn, Pandas, NumPy, Joblib.
- **Trained Model Artifact:** Pre-trained Logistic Regression Pipeline (82.47% accuracy, 6 arXiv categories).
- **Dataset:** `arXiv_scientific_dataset.csv` (287,421 arXiv papers).

---

## 18. Running the Application

To run Brainmate locally, start each of the three services in its respective terminal window:

### 1. Start the Python Flask ML Service (Port 8000)
```bash
# From the project root:
python ml_service/app.py
```
*Loads the pre-trained Logistic Regression model from `results/trained_model.joblib` and builds the memory-conscious arXiv search index once on startup.*

### 2. Start the Node.js Express Backend (Port 5000)
```bash
# In a new terminal window:
cd backend
npm start
```
*Listens on http://localhost:5000 and connects to the Flask ML service on http://localhost:8000.*

### 3. Start the Next.js Frontend (Port 3000)
```bash
# In a new terminal window:
cd frontend
npm run dev
```
*Open your browser and navigate to **http://localhost:3000** to use Brainmate.*

---

## 19. API Endpoints Reference

### Flask ML Service (`http://localhost:8000`)
- `GET  /health` — Checks status of trained model and TF-IDF search index.
- `POST /predict` — Predicts arXiv category given `title` and/or `summary` using the pre-trained model. Returns predicted category, arXiv code, and calibrated class probabilities.
- `POST /search` — Searches the arXiv dataset using TF-IDF and Cosine Similarity. Returns matching papers with titles, abstracts, dates, and similarity scores.
- `GET  /paper/<id>` — Retrieves metadata for a specific paper by its arXiv ID.

### Express Backend (`http://localhost:5000`)
- `GET  /api/health` — Checks backend uptime and Flask ML service connectivity.
- `POST /api/chat` — Conversational router that detects intent (search, classification, or category exploration) and formats structured research answers.
- `POST /api/search` — Proxies paper search to the ML service.
- `POST /api/classify` — Proxies paper category prediction to the ML service.
- `GET  /api/paper/:id` — Proxies individual paper retrieval.

