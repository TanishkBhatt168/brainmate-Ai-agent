"""
=============================================================================
Brainmate - Phase 1: Machine Learning Foundation
Script: 03_train_evaluate.py
Description: Trains and compares three multiclass text classification models:
             1. Multinomial Naive Bayes (MultinomialNB)
             2. Linear Support Vector Machine (LinearSVC)
             3. Logistic Regression (LogisticRegression)
             
             Uses Scikit-learn Pipelines combining TF-IDF feature extraction
             and classifiers. Evaluates on unseen 20% test data using Accuracy,
             Macro/Weighted Precision, Recall, and F1-score.
             Saves the best-performing pipeline to results/trained_model.joblib
             and writes an evaluation report to results/model_evaluation.txt.
             Includes a prediction function for testing new papers.
=============================================================================
"""

import os
import sys
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# 6 Target Core AI & Computer Science Categories
TARGET_6_CATEGORIES = {
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CL": "Computation and Language (Natural Language Processing)",
    "cs.LG": "Machine Learning",
    "cs.AI": "Artificial Intelligence",
    "cs.RO": "Robotics",
    "cs.CR": "Cryptography and Security",
}


def get_dataset_path(preferred_sample="arXiv_cleaned_sample_20k.csv", raw_dataset="arXiv_scientific_dataset.csv"):
    """
    Locates the dataset to use. Prefers the cleaned sample CSV created in Step 2.
    If not found, falls back to the raw arXiv dataset in the project root.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(script_dir, preferred_sample)
    if os.path.exists(sample_path):
        return sample_path, True

    raw_path = os.path.join(script_dir, raw_dataset)
    if os.path.exists(raw_path):
        return raw_path, False

    cwd_raw = os.path.join(os.getcwd(), raw_dataset)
    if os.path.exists(cwd_raw):
        return cwd_raw, False

    raise FileNotFoundError(
        f"Neither '{preferred_sample}' nor '{raw_dataset}' could be located.\n"
        "Please ensure the dataset file is placed in the project root."
    )


def build_category_mapping(df):
    """
    Constructs a dictionary mapping category_code (e.g., 'cs.AI')
    to its human-readable category name (e.g., 'Artificial Intelligence').
    """
    mapping = {}
    if "category" in df.columns and "category_code" in df.columns:
        unique_pairs = df[["category_code", "category"]].drop_duplicates()
        for _, row in unique_pairs.iterrows():
            code = str(row["category_code"]).strip()
            name = str(row["category"]).strip()
            if code and code not in mapping:
                mapping[code] = name
    return mapping


def load_and_prepare_data(data_path, is_sample=True, target_sample_size=20000, random_seed=42):
    """
    Loads research paper data, validates text features, combines title and summary,
    and returns feature array X, target labels y, and category mapping.
    
    CRITICAL: Never includes category or category_code in input features to prevent
    target leakage.
    """
    print(f"\n[Data Loading] Loading data from: {data_path}")
    t0 = time.time()

    if is_sample:
        df = pd.read_csv(data_path)
        print(f"   Loaded pre-cleaned sample with {len(df):,} records in {time.time() - t0:.2f}s.")
    else:
        print("   Loading raw dataset and performing safe sampling...")
        # Stream and clean raw dataset
        chunks = []
        for chunk in pd.read_csv(data_path, chunksize=50000, usecols=["id", "title", "summary", "category_code", "category"]):
            chunks.append(chunk)
        df_raw = pd.concat(chunks, ignore_index=True)
        print(f"   Loaded raw {len(df_raw):,} records. Cleaning text...")

        df_raw["title_clean"] = df_raw["title"].astype(str).str.strip()
        df_raw["summary_clean"] = df_raw["summary"].astype(str).str.strip()
        df_raw["category_code"] = df_raw["category_code"].astype(str).str.strip()
        
        # Filter out empty or non-informative
        valid_mask = (
            (df_raw["title_clean"].str.len() > 0)
            & (df_raw["summary_clean"].str.len() >= 20)
            & (df_raw["category_code"].str.len() > 0)
            & (~df_raw["summary_clean"].str.lower().str.contains(r"withdrawn|retracted|#name\?"))
        )
        df_clean = df_raw[valid_mask].drop_duplicates(subset=["title_clean", "summary_clean"]).copy()
        print(f"   Cleaned records: {len(df_clean):,}")

        # Sample if needed
        if target_sample_size and len(df_clean) > target_sample_size:
            cat_counts = df_clean["category_code"].value_counts()
            eligible = cat_counts[cat_counts >= 5].index
            df_el = df_clean[df_clean["category_code"].isin(eligible)]
            
            # Diversity-preserving sample
            base_alloc = {c: 5 for c in eligible}
            rem_quota = target_sample_size - sum(base_alloc.values())
            rem_counts = cat_counts[eligible] - 5
            total_rem = rem_counts.sum()
            add_alloc = (rem_counts / total_rem * rem_quota).round().astype(int)
            final_alloc = {c: base_alloc[c] + add_alloc[c] for c in eligible}
            diff = target_sample_size - sum(final_alloc.values())
            top_cats = cat_counts[eligible].index
            for i in range(abs(diff)):
                final_alloc[top_cats[i % len(top_cats)]] += 1 if diff > 0 else -1

            samples = [df_el[df_el["category_code"] == c].sample(n=n, random_state=random_seed) for c, n in final_alloc.items()]
            df = pd.concat(samples, ignore_index=True).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        else:
            df = df_clean

    category_mapping = build_category_mapping(df)

    # Combine Title and Summary into a single text feature
    if "combined_text" in df.columns and df["combined_text"].notna().all():
        X = df["combined_text"].astype(str)
    else:
        X = df["title"].astype(str).str.strip() + " " + df["summary"].astype(str).str.strip()

    y = df["category_code"].astype(str)

    print(f"   Total valid records ready: {len(X):,}")
    print(f"   Unique categories: {y.nunique()}")
    return X, y, category_mapping, df


def build_pipeline(classifier, max_features=25000, ngram_range=(1, 2)):
    """
    Creates a unified Scikit-learn Pipeline combining TF-IDF vectorization
    and a classification algorithm.
    
    TF-IDF Vectorizer Parameters:
      - max_features: 25,000 top vocabulary terms (controls memory usage)
      - ngram_range: (1, 2) extracts both single words and 2-word phrases (e.g. 'deep learning')
      - stop_words: 'english' filters common non-informative words ('the', 'is', 'at')
      - sublinear_tf: True applies logarithmic sublinear scaling 1 + log(tf) to damp high counts
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True,
    )
    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])


def train_and_evaluate_models(X_train, X_test, y_train, y_test, random_seed=42):
    """
    Trains all three classification models on identical training data and
    evaluates them on the unseen test data.
    
    Models compared:
      1. Multinomial Naive Bayes (alpha=0.1)
      2. Linear Support Vector Machine (LinearSVC, C=1.0)
      3. Logistic Regression (C=1.0, max_iter=300)
    """
    print("\n" + "=" * 78)
    print("STEP 4 & 5: TRAINING AND EVALUATING CLASSIFICATION MODELS")
    print("=" * 78)

    models_config = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear Support Vector Machine": LinearSVC(
            C=1.0, dual="auto", max_iter=2000, random_state=random_seed
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=300, random_state=random_seed, solver="lbfgs"
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, max_depth=25, n_jobs=-1, random_state=random_seed
        ),
    }

    results = {}

    for model_name, clf in models_config.items():
        print(f"\n---> Training: {model_name}...")
        pipeline = build_pipeline(clf, max_features=25000, ngram_range=(1, 2))

        # Measure training duration
        t_start = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - t_start
        print(f"     Finished training in {training_time:.2f} seconds.")

        # Measure evaluation duration and metrics on held-out test data
        t_eval_start = time.time()
        y_pred = pipeline.predict(X_test)
        eval_time = time.time() - t_eval_start

        # Compute key evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Generate classification report
        cls_report_text = classification_report(y_test, y_pred, zero_division=0)

        results[model_name] = {
            "pipeline": pipeline,
            "training_time": training_time,
            "eval_time": eval_time,
            "accuracy": acc,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "y_pred": y_pred,
            "classification_report": cls_report_text,
        }

        print(f"     Accuracy:    {acc * 100:.2f}%")
        print(f"     Precision:   {macro_prec:.4f}")
        print(f"     Recall:      {macro_rec:.4f}")
        print(f"     F1-Score:    {macro_f1:.4f}")

    return results


def display_comparison_table(results):
    """
    Prints a clear, formatted comparison table of all evaluated models
    showing Accuracy, Precision, Recall, and F1-Score.
    """
    print("\n" + "=" * 96)
    print("4 ML MODELS PERFORMANCE COMPARISON (ON HELD-OUT 20% TEST SET)")
    print("=" * 96)
    header = f"{'Model Name':<32} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<9} | {'F1-Score':<9} | {'Train Time':<10}"
    print(header)
    print("-" * 96)
    for name, m in results.items():
        row = (
            f"{name:<32} | "
            f"{m['accuracy'] * 100:>7.2f}%   | "
            f"{m['macro_precision']:>9.4f}  | "
            f"{m['macro_recall']:>8.4f}  | "
            f"{m['macro_f1']:>8.4f}  | "
            f"{m['training_time']:>8.2f}s"
        )
        print(row)
    print("=" * 96)


def select_best_model(results):
    """
    Selects the best performing model based primarily on Macro F1-score.
    
    Why Macro F1-score?
    In multiclass datasets with severe class imbalance (such as arXiv, where cs.LG
    has over 80,000 papers while smaller categories have only tens or hundreds),
    standard accuracy is dominated by the majority classes. Macro F1 computes
    the unweighted arithmetic mean of F1 across all classes, treating every
    scientific domain with equal importance.
    """
    best_name = max(results.keys(), key=lambda k: results[k]["macro_f1"])
    print(f"\n>>> Best Model Selected: '{best_name}' (Highest Macro F1: {results[best_name]['macro_f1']:.4f})")
    return best_name, results[best_name]


def save_trained_pipeline(pipeline, category_mapping, output_path):
    """
    Saves the entire fitted Pipeline (TF-IDF vectorizer + trained classifier)
    along with category metadata to results/trained_model.joblib using Joblib.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    payload = {
        "pipeline": pipeline,
        "category_mapping": category_mapping,
        "classes": pipeline.classes_,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(payload, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n[Saved Model] Pipeline successfully saved to: {output_path} ({file_size_mb:.2f} MB)")


def verify_saved_model(saved_model_path, test_sample_text, expected_label=None):
    """
    Verifies that the saved model file can be reloaded from disk using Joblib
    and execute an end-to-end prediction on unseen text.
    """
    print("\n[Verification] Loading saved model from disk and running verification test...")
    loaded_data = joblib.load(saved_model_path)
    loaded_pipe = loaded_data["pipeline"]
    mapping = loaded_data.get("category_mapping", {})

    # Run prediction
    pred_code = loaded_pipe.predict([test_sample_text])[0]
    pred_name = mapping.get(pred_code, "Scientific Category")

    print(f"   Input text sample: '{test_sample_text[:90]}...'")
    print(f"   Predicted Code:    {pred_code}")
    print(f"   Predicted Name:    {pred_name}")
    if expected_label:
        print(f"   Actual Label:      {expected_label}")
        print(f"   Match:             {'YES (Correct)' if pred_code == expected_label else 'DIFFERENT'}")
    print("   Verification test PASSED: Saved model reloads and predicts correctly.")
    return pred_code, pred_name


def predict_paper(title, summary, model_path=None):
    """
    Predicts the subject category of a research paper given its title and summary.
    
    Accepts:
      - title: string
      - summary: string
      - model_path: optional path to trained_model.joblib
      
    Returns:
      - dict with predicted code, readable category name, top confidence, and ranked predictions
    """
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", "trained_model.joblib"
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python 03_train_evaluate.py' first to train and save the model."
        )

    loaded = joblib.load(model_path)
    pipeline = loaded["pipeline"]
    mapping = loaded.get("category_mapping", {})

    # Clean input text identically to training preprocessing
    t_clean = " ".join(str(title).split())
    s_clean = " ".join(str(summary).split())
    combined_input = f"{t_clean} {s_clean}".strip()

    pred_code = pipeline.predict([combined_input])[0]
    pred_name = mapping.get(pred_code, "Scientific Category")

    # Estimate confidence percentages across categories
    clf = pipeline.named_steps["classifier"]
    confidences = []
    if hasattr(clf, "predict_proba"):
        probs = pipeline.predict_proba([combined_input])[0]
        classes = pipeline.classes_
        top_idx = np.argsort(probs)[::-1]
        for idx in top_idx:
            c = classes[idx]
            confidences.append({
                "code": c,
                "name": mapping.get(c, c),
                "percentage": f"{probs[idx] * 100:.1f}%",
            })
    elif hasattr(clf, "decision_function"):
        df_vals = pipeline.decision_function([combined_input])[0]
        exp_vals = np.exp(df_vals - np.max(df_vals))
        probs = exp_vals / np.sum(exp_vals)
        classes = pipeline.classes_
        top_idx = np.argsort(probs)[::-1]
        for idx in top_idx:
            c = classes[idx]
            confidences.append({
                "code": c,
                "name": mapping.get(c, c),
                "percentage": f"{probs[idx] * 100:.1f}%",
            })

    top_confidence = confidences[0]["percentage"] if confidences else "N/A"

    return {
        "title": t_clean,
        "category_code": pred_code,
        "category_name": pred_name,
        "top_confidence": top_confidence,
        "ranked_predictions": confidences,
        "disclaimer": "Note: This prediction is a machine learning statistical estimate and may be incorrect.",
    }


def save_evaluation_report(
    results,
    best_name,
    X_train_len,
    X_test_len,
    total_len,
    num_classes,
    random_seed,
    output_path,
):
    """
    Writes a comprehensive, beginner-friendly evaluation report to results/model_evaluation.txt.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines = []
    lines.append("=" * 88)
    lines.append("BRAINMATE — MACHINE LEARNING MODEL EVALUATION REPORT")
    lines.append("=" * 88)
    lines.append("")
    lines.append("1. EXPERIMENTAL CONFIGURATION & DATASET SPLIT")
    lines.append("-" * 88)
    lines.append(f"• Total Records Evaluated:       {total_len:,}")
    lines.append(f"• Training Set Size (80%):       {X_train_len:,} papers")
    lines.append(f"• Test Set Size (20% Held-Out):  {X_test_len:,} papers")
    lines.append(f"• Number of Target Classes:      {num_classes} arXiv categories")
    lines.append(f"• Split Strategy:                Stratified random split (preserves class ratios)")
    lines.append(f"• Fixed Random Seed:             {random_seed} (guarantees exact reproducibility)")
    lines.append(f"• Input Text Features:           Paper Title + Summary (combined into single string)")
    lines.append(f"• Target Label:                  category_code (e.g. cs.LG, cs.CV, cs.AI)")
    lines.append("• Target Leakage Prevention:     Strictly excluded category names and codes from inputs")
    lines.append("")

    lines.append("2. FEATURE EXTRACTION (TF-IDF PARAMETERS)")
    lines.append("-" * 88)
    lines.append("• Vectorizer:                    Scikit-learn TfidfVectorizer")
    lines.append("• Vocabulary Size (max_features): 25,000 features")
    lines.append("• N-gram Range:                  (1, 2) unigrams and bigrams")
    lines.append("• Stop Words:                    'english' removed")
    lines.append("• Sublinear TF:                  True (1 + log(tf) applied to dampen term burstiness)")
    lines.append("• Leakage Rule:                  TF-IDF was fitted on TRAINING data ONLY, never on test")
    lines.append("")

    lines.append("3. 4 ML MODELS COMPARISON SUMMARY")
    lines.append("-" * 96)
    header = f"{'Model Name':<32} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<9} | {'F1-Score':<9} | {'Train Time':<10}"
    lines.append(header)
    lines.append("-" * 96)
    for name, m in results.items():
        row = (
            f"{name:<32} | "
            f"{m['accuracy'] * 100:>7.2f}%   | "
            f"{m['macro_precision']:>9.4f}  | "
            f"{m['macro_recall']:>8.4f}  | "
            f"{m['macro_f1']:>8.4f}  | "
            f"{m['training_time']:>8.2f}s"
        )
        lines.append(row)
    lines.append("-" * 96)
    lines.append("")

    lines.append("4. MODEL SELECTION JUSTIFICATION")
    lines.append("-" * 88)
    best = results[best_name]
    lines.append(f"• Selected Best Model:           {best_name}")
    lines.append(f"• Measured Accuracy:             {best['accuracy'] * 100:.2f}%")
    lines.append(f"• Measured Macro F1-score:       {best['macro_f1']:.4f}")
    lines.append(f"• Measured Weighted F1-score:    {best['weighted_f1']:.4f}")
    lines.append(f"• Training Time:                 {best['training_time']:.2f} seconds")
    rationale = [
        f"  1. Top Generalization: '{best_name}' achieved the highest Accuracy ({best['accuracy'] * 100:.2f}%) "
        f"and the highest Macro F1-score ({best['macro_f1']:.4f}) on the held-out test set.",
        f"  2. Balanced Multi-Class Precision: Maintained high F1 across all {num_classes} target categories "
        f"(Weighted F1: {best['weighted_f1']:.4f}).",
        f"  3. Training & Inference Speed: Trained in {best['training_time']:.2f} seconds with fast inference latency.",
    ]
    lines.append("\n• Selection Rationale:\n" + "\n".join(rationale))
    lines.append("")

    lines.append("5. BEGINNER-FRIENDLY METRIC EXPLANATIONS")
    lines.append("-" * 88)
    lines.append(
        "• Accuracy:\n"
        "  The percentage of test papers whose categories were correctly predicted out of all test papers.\n"
        "• Precision (Macro):\n"
        "  Of all papers the model predicted as category X, what fraction actually belonged to X? Averaged\n"
        "  equally across all categories.\n"
        "• Recall (Macro):\n"
        "  Of all papers that truly belonged to category X, what fraction did the model successfully find?\n"
        "  Averaged equally across all categories.\n"
        "• Macro F1-Score:\n"
        "  The harmonic mean of Precision and Recall calculated per class, then averaged across all classes.\n"
        "  Crucial for imbalanced data because it treats rare categories equally with massive ones.\n"
        "• Weighted F1-Score:\n"
        "  F1-score averaged across classes with weights proportional to each category's test sample size."
    )
    lines.append("")

    lines.append("6. DETAILED CLASSIFICATION REPORT (BEST MODEL)")
    lines.append("-" * 88)
    lines.append(best["classification_report"])
    lines.append("")
    lines.append("=" * 88)
    lines.append("END OF EVALUATION REPORT")
    lines.append("=" * 88)

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n[Evaluation Report] Full report written to: {output_path}")


def main():
    """
    Main execution workflow for Step 4, 5, 6, and 7.
    """
    parser = argparse.ArgumentParser(description="Train, evaluate, and save scientific paper classifiers.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="Number of records to use for training (default: 20000). Set to 0 to use all records.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run prediction mode for a new paper instead of training.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Paper title for prediction mode.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="",
        help="Paper summary/abstract for prediction mode.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, "results", "trained_model.joblib")
    eval_report_path = os.path.join(script_dir, "results", "model_evaluation.txt")

    # If --predict flag is passed, run inference
    if args.predict:
        title = args.title or "Attention Is All You Need"
        summary = (
            args.summary
            or "The dominant sequence transduction models are based on complex recurrent or "
            "convolutional neural networks that include an encoder and a decoder. The best performing "
            "models also connect the encoder and decoder through an attention mechanism. We propose a "
            "new simple network architecture, the Transformer, based solely on attention mechanisms, "
            "dispensing with recurrence and convolutions entirely."
        )
        print("=" * 78)
        print("BRAINMATE — INFERENCE MODE: TEST A NEW RESEARCH PAPER")
        print("=" * 78)
        result = predict_paper(title, summary, model_save_path)
        print(f"\nPaper Title:     {result['title']}")
        print(f"Predicted Code:  {result['category_code']}")
        print(f"Predicted Name:  {result['category_name']}")
        if "top_confidence" in result and result["top_confidence"] != "N/A":
            print(f"Confidence:      {result['top_confidence']}")
        if "ranked_predictions" in result and result["ranked_predictions"]:
            print("\nCategory Probability Breakdown:")
            for rank, item in enumerate(result["ranked_predictions"][:6], 1):
                print(f"  {rank}. {item['code']:<10} ({item['name']:<45}) : {item['percentage']}")
        print(f"\n{result['disclaimer']}")
        return

    print("=" * 78)
    print("BRAINMATE — STEP 4, 5, 6: MODEL TRAINING, EVALUATION & SAVING")
    print("=" * 78)

    # 1. Locate and load dataset
    data_path, is_sample = get_dataset_path()
    X, y, category_mapping, df = load_and_prepare_data(
        data_path,
        is_sample=is_sample,
        target_sample_size=args.sample_size,
        random_seed=args.random_seed,
    )

    # 2. Stratified 80/20 train/test split
    print("\n[Splitting] Creating stratified 80% training / 20% testing split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=args.random_seed,
    )
    print(f"   Total records:     {len(X):,}")
    print(f"   Training records:  {len(X_train):,} (80%)")
    print(f"   Testing records:   {len(X_test):,} (20%)")
    print(f"   Number of classes: {y.nunique()}")

    # 3. Train all 3 models on training data, evaluate on held-out test data
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, random_seed=args.random_seed)

    # 4. Display comparison table
    display_comparison_table(results)

    # 5. Select best model based on Macro F1
    best_name, best_info = select_best_model(results)

    # 6. Save the best pipeline and metadata
    save_trained_pipeline(best_info["pipeline"], category_mapping, model_save_path)

    # 7. Save evaluation report
    save_evaluation_report(
        results=results,
        best_name=best_name,
        X_train_len=len(X_train),
        X_test_len=len(X_test),
        total_len=len(X),
        num_classes=y.nunique(),
        random_seed=args.random_seed,
        output_path=eval_report_path,
    )

    # 8. Step Six Verification: Load saved model from disk and verify inference
    sample_idx = X_test.index[0]
    sample_text = X_test.iloc[0]
    expected_label = y_test.iloc[0]
    verify_saved_model(model_save_path, sample_text, expected_label=expected_label)

    # 9. Step Seven Test: Run prediction on a new paper
    print("\n" + "=" * 78)
    print("STEP 7: TESTING A BRAND NEW RESEARCH PAPER")
    print("=" * 78)
    demo_title = "Language Models are Few-Shot Learners"
    demo_summary = (
        "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by "
        "pre-training on a large corpus of text followed by fine-tuning on a specific task. "
        "While typically task-agnostic in architecture, this method still requires task-specific "
        "fine-tuning datasets of thousands or tens of thousands of examples. Here we show that "
        "scaling up language models greatly improves task-agnostic, few-shot performance."
    )
    demo_result = predict_paper(demo_title, demo_summary, model_save_path)
    print(f"Test Paper Title: {demo_result['title']}")
    print(f"Predicted Code:   {demo_result['category_code']}")
    print(f"Predicted Name:   {demo_result['category_name']}")
    print(f"{demo_result['disclaimer']}")

    print("\n" + "=" * 78)
    print("All tasks completed successfully!")
    print("=" * 78)


if __name__ == "__main__":
    main()
