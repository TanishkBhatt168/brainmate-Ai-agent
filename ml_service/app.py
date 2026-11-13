"""
=============================================================================
Brainmate - Machine Learning Service
Application: ml_service/app.py
Port: 8000
Endpoints:
  - GET  /health   -> Check service, model, and search index health
  - POST /predict  -> Predict research paper category from title & summary
  - POST /search   -> Search scientific papers using TF-IDF & cosine similarity
  - GET  /paper/<id> -> Retrieve individual paper by arXiv ID
=============================================================================
"""

import os
import sys
from flask import Flask, request, jsonify

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.model_service import ModelService
from services.search_service import SearchService

app = Flask(__name__)

# Initialize services once at application startup
print("=" * 60)
print("Initializing Brainmate ML Service...")
print("=" * 60)

model_service = ModelService()
search_service = SearchService()

print("=" * 60)
print("Brainmate ML Service initialization complete.")
print("=" * 60)


@app.after_request
def add_cors_headers(response):
    """Enable CORS for all incoming requests."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Brainmate ML Service",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Service & model health",
            "/predict": "POST - Predict paper category (expects 'title' and/or 'summary')",
            "/search": "POST - Search papers (expects 'query', optional 'top_k')",
            "/paper/<id>": "GET - Paper details by ID",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    """
    Returns the comprehensive ML service status including model and dataset status.
    """
    m_info = model_service.get_info()
    s_info = search_service.get_info()

    status_code = 200 if (m_info["is_loaded"] and s_info["is_indexed"]) else 503

    return jsonify({
        "status": "healthy" if status_code == 200 else "degraded",
        "service": "Brainmate ML Service",
        "model": {
            "loaded": m_info["is_loaded"],
            "classes": m_info["classes"],
            "category_count": len(m_info["classes"]),
            "categories": m_info["category_mapping"],
            "saved_at": m_info["saved_at"],
            "error": m_info["error"],
        },
        "search": {
            "ready": s_info["is_indexed"],
            "indexed_papers": s_info["indexed_count"],
            "coverage_note": s_info["coverage_note"],
            "error": s_info["error"],
        },
    }), status_code


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicts the research paper category based on title and summary.
    Expects JSON body:
      {
        "title": "Optional paper title",
        "summary": "Optional paper summary/abstract"
      }
    """
    if not model_service.is_loaded:
        return jsonify({
            "error": "Model service unavailable",
            "details": model_service.load_error or "Trained model is not loaded.",
        }), 503

    data = request.get_json(silent=True) or {}
    title = str(data.get("title", "")).strip()
    summary = str(data.get("summary", "")).strip()

    if not title and not summary:
        return jsonify({
            "error": "Invalid input",
            "message": "Please provide a 'title' or 'summary' (or both) to predict category.",
        }), 400

    try:
        result = model_service.predict(title=title, summary=summary)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "message": str(e),
        }), 500


@app.route("/search", methods=["POST"])
def search():
    """
    Searches the arXiv dataset using TF-IDF and Cosine Similarity.
    Expects JSON body:
      {
        "query": "search terms or research topic",
        "top_k": 5 (optional, default 5)
      }
    """
    if not search_service.is_indexed:
        return jsonify({
            "error": "Search service unavailable",
            "details": search_service.load_error or "Search index is not built.",
        }), 503

    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    top_k = data.get("top_k", 5)

    if not query:
        return jsonify({
            "error": "Invalid input",
            "message": "Please provide a 'query' parameter for research paper search.",
        }), 400

    try:
        results = search_service.search(query=query, top_k=top_k)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({
            "error": "Search failed",
            "message": str(e),
        }), 500


@app.route("/paper/<path:paper_id>", methods=["GET"])
def get_paper(paper_id):
    """
    Retrieves paper details by its arXiv ID.
    """
    if not search_service.is_indexed:
        return jsonify({
            "error": "Search service unavailable",
            "details": search_service.load_error,
        }), 503

    paper = search_service.get_paper_by_id(paper_id)
    if not paper:
        return jsonify({
            "error": "Not Found",
            "message": f"No paper found matching arXiv ID: '{paper_id}'.",
        }), 404

    return jsonify({"paper": paper}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Flask ML Service on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

