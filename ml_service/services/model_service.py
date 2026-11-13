"""
=============================================================================
Brainmate - Machine Learning Service
Module: services/model_service.py
Description: Encapsulates loading and inference for the pre-trained arXiv
             research paper category classification model.
=============================================================================
"""

import os
import time
import joblib
import numpy as np


def normalize_text_whitespace(text):
    """
    Normalizes text identically to the original training preprocessing:
    stripping leading/trailing whitespace and collapsing whitespace sequences.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


class ModelService:
    def __init__(self, model_path=None):
        self.model_path = model_path or self._resolve_model_path()
        self.pipeline = None
        self.category_mapping = {}
        self.classes = []
        self.saved_at = None
        self.is_loaded = False
        self.load_error = None

        self._load_model()

    def _resolve_model_path(self):
        """
        Locates the pre-trained trained_model.joblib file across common
        relative and absolute directories without hardcoded paths.
        """
        env_path = os.environ.get("MODEL_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            # Relative to this script inside ml_service/services/
            os.path.join(current_dir, "..", "..", "results", "trained_model.joblib"),
            os.path.join(current_dir, "..", "results", "trained_model.joblib"),
            os.path.join(current_dir, "..", "model", "trained_model.joblib"),
            # Relative to process working directory
            os.path.join(os.getcwd(), "results", "trained_model.joblib"),
            os.path.join(os.getcwd(), "..", "results", "trained_model.joblib"),
        ]

        for cand in candidates:
            normalized = os.path.normpath(cand)
            if os.path.exists(normalized):
                return normalized

        # Return primary intended path so error message shows exact expected file
        return os.path.normpath(candidates[0])

    def _load_model(self):
        """
        Loads the pre-trained Joblib model payload. Does NOT retrain or alter.
        Reports missing file cleanly if not found.
        """
        if not os.path.exists(self.model_path):
            self.load_error = (
                f"Trained model artifact not found at: '{self.model_path}'. "
                "Expected existing pre-trained file 'results/trained_model.joblib'."
            )
            print(f"[ModelService ERROR] {self.load_error}")
            return

        try:
            print(f"[ModelService] Loading trained model from: {self.model_path}...")
            payload = joblib.load(self.model_path)

            if isinstance(payload, dict) and "pipeline" in payload:
                self.pipeline = payload["pipeline"]
                self.category_mapping = payload.get("category_mapping", {})
                self.classes = list(payload.get("classes", getattr(self.pipeline, "classes_", [])))
                self.saved_at = payload.get("saved_at")
            else:
                # Direct Pipeline object fallback
                self.pipeline = payload
                self.classes = list(getattr(payload, "classes_", []))
                self.category_mapping = {}

            self.is_loaded = True
            print(f"[ModelService] Successfully loaded model with {len(self.classes)} classes.")
        except Exception as e:
            self.load_error = f"Failed to load trained model: {str(e)}"
            print(f"[ModelService ERROR] {self.load_error}")

    def predict(self, title, summary):
        """
        Predicts the research category for a given title and summary.

        CRITICAL RULES:
        - Reuses the existing trained model and vectorizer.
        - Never fits a new vectorizer or alters model state.
        - Never takes category or category_code as an input.
        - Uses valid probability estimates from predict_proba.
        """
        if not self.is_loaded:
            raise RuntimeError(self.load_error or "Model is not loaded.")

        clean_title = normalize_text_whitespace(title)
        clean_summary = normalize_text_whitespace(summary)

        if not clean_title and not clean_summary:
            raise ValueError("Both title and summary are empty. At least one text field is required.")

        combined_input = f"{clean_title} {clean_summary}".strip()

        # Predict primary category code
        pred_code = self.pipeline.predict([combined_input])[0]
        pred_name = self.category_mapping.get(pred_code, pred_code)

        # Compute calibrated probabilities via predict_proba
        ranked_predictions = []
        top_confidence = 0.0

        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba([combined_input])[0]
            classes = self.pipeline.classes_
            sorted_indices = np.argsort(probs)[::-1]

            top_confidence = float(probs[sorted_indices[0]])

            for idx in sorted_indices:
                cls_code = str(classes[idx])
                prob = float(probs[idx])
                ranked_predictions.append({
                    "category_code": cls_code,
                    "category_name": self.category_mapping.get(cls_code, cls_code),
                    "probability": round(prob, 4),
                    "percentage": f"{prob * 100:.1f}%",
                })
        else:
            # Fallback if probability is unsupported
            ranked_predictions.append({
                "category_code": str(pred_code),
                "category_name": pred_name,
                "probability": None,
                "percentage": "N/A",
            })

        return {
            "predicted_category": pred_name,
            "category_code": str(pred_code),
            "confidence": round(top_confidence, 4) if top_confidence > 0 else None,
            "confidence_percentage": f"{top_confidence * 100:.1f}%" if top_confidence > 0 else "N/A",
            "ranked_predictions": ranked_predictions,
            "input_preview": {
                "title": clean_title,
                "summary": clean_summary[:250] + ("..." if len(clean_summary) > 250 else ""),
            },
        }

    def get_info(self):
        """Returns metadata about the loaded model."""
        return {
            "is_loaded": self.is_loaded,
            "model_path": self.model_path,
            "classes": self.classes,
            "category_mapping": self.category_mapping,
            "saved_at": self.saved_at,
            "error": self.load_error,
        }

