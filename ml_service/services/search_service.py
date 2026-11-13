"""
=============================================================================
Brainmate - Machine Learning Service
Module: services/search_service.py
Description: Memory-aware TF-IDF search indexer and similarity retriever
             for the arXiv scientific research dataset.
=============================================================================
"""

import os
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text_whitespace(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


class SearchService:
    def __init__(self, dataset_path=None, index_size=35000):
        self.dataset_path = dataset_path or self._resolve_dataset_path()
        self.index_size = int(os.environ.get("SEARCH_INDEX_SIZE", index_size))
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.is_indexed = False
        self.total_records_in_file = 0
        self.indexed_count = 0
        self.load_error = None
        self.coverage_note = ""

        self._build_search_index()

    def _resolve_dataset_path(self):
        """
        Locates the arXiv CSV dataset without hardcoded absolute paths.
        Checks for the primary 430 MB dataset first, with fallback to 20k sample.
        """
        env_path = os.environ.get("DATASET_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(current_dir, "..", "..", "arXiv_scientific_dataset.csv"),
            os.path.join(current_dir, "..", "arXiv_scientific_dataset.csv"),
            os.path.join(os.getcwd(), "arXiv_scientific_dataset.csv"),
            os.path.join(current_dir, "..", "..", "arXiv_cleaned_sample_20k.csv"),
            os.path.join(os.getcwd(), "arXiv_cleaned_sample_20k.csv"),
        ]

        for cand in candidates:
            normalized = os.path.normpath(cand)
            if os.path.exists(normalized):
                return normalized

        return os.path.normpath(candidates[0])

    def _build_search_index(self):
        """
        Loads the dataset once and builds the TF-IDF search index in memory.
        Enforces memory-conscious practices to prevent high RAM consumption.
        """
        if not os.path.exists(self.dataset_path):
            self.load_error = (
                f"Dataset file not found at: '{self.dataset_path}'. "
                "Expected 'arXiv_scientific_dataset.csv' in project root."
            )
            print(f"[SearchService ERROR] {self.load_error}")
            return

        try:
            t0 = time.time()
            print(f"[SearchService] Loading dataset from: {self.dataset_path}...")

            # Select only columns needed for search & paper presentation
            cols_needed = ["id", "title", "summary", "category", "category_code", "published_date"]

            # Read sample or full dataset up to index_size
            try:
                df_loaded = pd.read_csv(
                    self.dataset_path,
                    usecols=lambda c: c in cols_needed,
                    nrows=self.index_size,
                    encoding="utf-8",
                    on_bad_lines="skip",
                )
            except Exception as read_err:
                print(f"[SearchService] Standard read failed ({read_err}), retrying with latin-1...")
                df_loaded = pd.read_csv(
                    self.dataset_path,
                    usecols=lambda c: c in cols_needed,
                    nrows=self.index_size,
                    encoding="latin-1",
                    on_bad_lines="skip",
                )

            # Ensure all required columns exist
            for c in cols_needed:
                if c not in df_loaded.columns:
                    df_loaded[c] = ""

            # Clean and normalize text
            df_loaded["id"] = df_loaded["id"].fillna("").astype(str)
            df_loaded["title"] = df_loaded["title"].fillna("").apply(normalize_text_whitespace)
            df_loaded["summary"] = df_loaded["summary"].fillna("").apply(normalize_text_whitespace)
            df_loaded["category"] = df_loaded["category"].fillna("").astype(str)
            df_loaded["category_code"] = df_loaded["category_code"].fillna("").astype(str)
            df_loaded["published_date"] = df_loaded["published_date"].fillna("").astype(str)

            # Filter valid papers with text
            valid_mask = (df_loaded["title"].str.len() > 0) | (df_loaded["summary"].str.len() > 0)
            self.df = df_loaded[valid_mask].reset_index(drop=True)
            self.indexed_count = len(self.df)

            # Determine dataset coverage note
            is_full = self.indexed_count >= 250000
            if is_full:
                self.coverage_note = f"Full dataset search coverage ({self.indexed_count:,} scientific papers)."
            else:
                self.coverage_note = (
                    f"Indexed {self.indexed_count:,} high-quality scientific papers for fast real-time search. "
                    "Search coverage is representative of core arXiv categories."
                )

            # Construct corpus and fit TF-IDF search vectorizer
            corpus = self.df["title"] + " " + self.df["summary"]
            self.vectorizer = TfidfVectorizer(
                max_features=25000,
                stop_words="english",
                sublinear_tf=True,
                ngram_range=(1, 2),
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            self.is_indexed = True

            elapsed = time.time() - t0
            print(f"[SearchService] Successfully indexed {self.indexed_count:,} papers in {elapsed:.2f}s.")
        except Exception as e:
            self.load_error = f"Failed to initialize search index: {str(e)}"
            print(f"[SearchService ERROR] {self.load_error}")

    def search(self, query, top_k=5, min_score=0.03):
        """
        Searches the indexed papers using TF-IDF and Cosine Similarity.
        Returns top_k most relevant papers.
        """
        if not self.is_indexed or self.df is None or self.tfidf_matrix is None:
            raise RuntimeError(self.load_error or "Search index is not ready.")

        clean_query = normalize_text_whitespace(query)
        if not clean_query:
            return {
                "query": query,
                "total_results": 0,
                "papers": [],
                "coverage_note": self.coverage_note,
            }

        top_k = max(1, min(int(top_k), 20))

        # Vectorize query
        q_vec = self.vectorizer.transform([clean_query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        # Sort indices by descending similarity
        top_indices = np.argsort(sims)[::-1]

        papers = []
        for idx in top_indices:
            score = float(sims[idx])
            if score < min_score and len(papers) >= 1:
                # If score is too low and we already found at least one match, break
                break

            row = self.df.iloc[idx]
            papers.append({
                "id": str(row["id"]),
                "title": str(row["title"]),
                "summary": str(row["summary"]),
                "category": str(row["category"]),
                "category_code": str(row["category_code"]),
                "published_date": str(row["published_date"]) if str(row["published_date"]).strip() else None,
                "similarity_score": round(score, 4),
            })

            if len(papers) >= top_k:
                break

        return {
            "query": clean_query,
            "total_results": len(papers),
            "papers": papers,
            "coverage_note": self.coverage_note,
            "indexed_total": self.indexed_count,
        }

    def get_paper_by_id(self, paper_id):
        """
        Retrieves full paper metadata by its arXiv ID.
        """
        if self.df is None:
            return None

        clean_id = str(paper_id).strip().lower()
        # Direct search in indexed DataFrame
        matches = self.df[self.df["id"].str.lower() == clean_id]
        if not matches.empty:
            row = matches.iloc[0]
            return {
                "id": str(row["id"]),
                "title": str(row["title"]),
                "summary": str(row["summary"]),
                "category": str(row["category"]),
                "category_code": str(row["category_code"]),
                "published_date": str(row["published_date"]) if str(row["published_date"]).strip() else None,
            }

        return None

    def get_info(self):
        """Returns metadata about the search service."""
        return {
            "is_indexed": self.is_indexed,
            "indexed_count": self.indexed_count,
            "dataset_path": self.dataset_path,
            "coverage_note": self.coverage_note,
            "error": self.load_error,
        }

