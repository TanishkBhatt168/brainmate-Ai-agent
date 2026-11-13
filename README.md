# 🧠 Brainmate — AI Research Assistant

**An AI-powered platform for scientific research paper classification, discovery, and exploration.**

Brainmate is an AI Research Assistant designed to help students, researchers, and academics discover, search, and explore scientific research papers. It uses Machine Learning and Natural Language Processing (NLP) to classify research papers based on their titles and abstracts.

---

## 🚀 Project Overview

Brainmate is developed in multiple phases.

### Phase 1: Machine Learning Model Development

The first phase focuses on preparing the arXiv research paper dataset, cleaning the data, training machine learning models, and evaluating their performance.

Key activities:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Duplicate removal and missing-value handling
- TF-IDF feature extraction
- Training multiple classification algorithms
- Model comparison and evaluation
- Saving the trained model for predictions

### Phase 2: AI Research Assistant Web Application

The second phase focuses on developing a complete web application with a modern user interface and integrated machine learning backend.

Users can:
- Search scientific research papers.
- Predict research paper categories.
- Explore research papers.
- Interact with the Brainmate research assistant.
- View relevant research information.

---

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React.js, Next.js, Tailwind CSS |
| Backend | Node.js, Express.js |
| Machine Learning | Python, Scikit-learn |
| Data Processing | Pandas, NumPy |
| NLP | TF-IDF |
| ML Model Storage | Joblib |
| ML API | Flask |
| Dataset | CSV (arXiv Research Papers) |

**Database:** No PostgreSQL or other database is used. The project uses a CSV dataset for research paper data and browser localStorage for client-side storage.

---

## 🧠 Machine Learning Pipeline

The ML pipeline uses research paper titles and abstracts to predict their scientific subject categories.

### Dataset Information

- Original Dataset: `arXiv_scientific_dataset.csv`
- Original Dataset Size: Approximately 410 MB
- Total Records: 287,421 research papers
- Cleaned Sample: 20,000 research papers

The original dataset is stored locally and is not included in this GitHub repository because of its large file size.

### Input Features

- `title` — Research paper title
- `summary` — Research paper abstract

### Target Variable

- `category_code` — Scientific research category

The category fields are used as target labels and are not included as model input features.

### Classification Algorithms

Three machine learning algorithms were implemented and compared:

1. Multinomial Naive Bayes
2. Linear Support Vector Classification (LinearSVC)
3. Logistic Regression

### Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) converts textual research paper information into numerical features that can be processed by machine learning algorithms.

### Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-Score
- Weighted F1-Score

The best-performing model is saved using Joblib and used for future predictions.

---

## 🏗️ System Architecture

```text
             USER
               |
               v
       React / Next.js
        Frontend
        Port 3000
               |
               | HTTP / JSON
               v
       Node.js / Express
          Backend
         Port 5000
               |
               | REST API
               v
        Python / Flask
          ML Service
         Port 8000
               |
        +------+------+
        |             |
        v             v
   Trained ML      CSV Research
      Model          Dataset
```

### Application Components

**1. Frontend**
- Developed using React.js and Next.js.
- Provides the user interface.
- Communicates with the Express backend.

**2. Backend**
- Developed using Node.js and Express.js.
- Handles API requests.
- Connects the frontend with the ML service.

**3. Machine Learning Service**
- Developed using Python and Flask.
- Loads the trained ML model.
- Performs research paper classification.
- Provides research paper search functionality using the dataset.

---

## 📁 Project Structure

```text
training model/
│
├── frontend/
│   └── Next.js / React Application
│
├── backend/
│   └── Express.js Backend
│
├── ml_service/
│   └── app.py
│
├── results/
│   ├── trained_model.joblib
│   ├── dataset_analysis.txt
│   ├── cleaning_report.txt
│   └── model_evaluation.txt
│
├── 01_understand_data.py
├── 02_clean_data.py
├── 03_train_evaluate.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** Large CSV datasets, `node_modules`, and local environment files are excluded from the repository.

---

## ⚙️ Installation and Setup

### Prerequisites

Install the following before running the project:

- Python 3.10+
- Node.js
- npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/TanishkBhatt168/brainmate-Ai-agent.git
```

```bash
cd brainmate-Ai-agent
```

### 2. Setup Python Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Install Backend Dependencies

```bash
cd backend
npm install
```

### 4. Install Frontend Dependencies

Open another terminal:

```bash
cd frontend
npm install
```

---

## ▶️ Running the Application

Brainmate consists of three services. Run each service in a separate terminal.

### 1. Start ML Service

From the project root:

```bash
python ml_service/app.py
```

ML service runs on:

`http://localhost:8000`

### 2. Start Backend Server

```bash
cd backend
npm start
```

Backend runs on:

`http://localhost:5000`

### 3. Start Frontend

```bash
cd frontend
npm run dev
```

Frontend runs on:

`http://localhost:3000`

Open the application in your browser:

**http://localhost:3000**

---

## 🔗 API Endpoints

### Flask ML Service

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks ML service status |
| POST | `/predict` | Predicts research paper category |
| POST | `/search` | Searches research papers |
| GET | `/paper/<id>` | Retrieves paper information |

### Express Backend

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Checks backend status |
| POST | `/api/chat` | Handles assistant chat requests |
| POST | `/api/search` | Searches research papers |
| POST | `/api/classify` | Classifies research papers |
| GET | `/api/paper/:id` | Retrieves paper information |

---

## 📊 Dataset and GitHub Storage

The original dataset and generated cleaned sample are not uploaded to GitHub because of their large file sizes.

Files excluded from Git:

- `arXiv_scientific_dataset.csv`
- `arXiv_cleaned_sample_20k.csv`
- `node_modules/`
- `.venv/`
- `__pycache__/`
- `.env`

The original CSV dataset must be obtained separately and placed in the project root for local dataset processing and paper search.

The cleaned sample can be generated by running:

```bash
python 02_clean_data.py
```

---

## 🔮 Future Enhancements

- Integration of lightweight local Large Language Models (LLMs).
- Retrieval-Augmented Generation (RAG).
- Advanced semantic research paper search.
- Research paper summarization.
- PDF upload and extraction.
- Citation and reference analysis.
- Improved contextual research recommendations.

---

## 🎯 Project Goal

The main goal of Brainmate is to simplify scientific research by combining Machine Learning, Natural Language Processing, and modern web technologies into a single AI-powered research assistant.

**Brainmate — Making Research Smarter and Easier.** 🧠🚀

---

## 👨‍💻 Developer

**Tanishk Bhatt**

GitHub: https://github.com/TanishkBhatt168

Project Repository: https://github.com/TanishkBhatt168/brainmate-Ai-agent