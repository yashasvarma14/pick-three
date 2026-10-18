"""
Activity catalog loading and the sentence-embedding model.

The raw Kaggle file has 32,000 rows but only 196 distinct activities,
so the catalog is de-duplicated before use (see notebook section 12).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .base import contains_keyword
from .outings import outing_query

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Activity environment (indoor / outdoor / mixed)
# ---------------------------------------------------------

OUTDOOR_KEYWORDS = [
    "picnic", "walk", "hike", "fishing", "camping", "bonfire", "park", "beach",
    "road trip", "bike", "biking", "running", "run", "outside", "outdoor",
    "zoo", "garden", "gardening", "kayak", "canoe", "swim", "swimming",
]
INDOOR_KEYWORDS = [
    "movie", "movies", "theater", "museum", "library", "game night", "video game",
    "karaoke", "bowling", "restaurant", "cooking", "cook", "bake", "spa", "mall",
    "arcade", "concert hall",
]
MIXED_KEYWORDS = ["concert", "festival", "photo session", "tour", "shopping", "market", "sports game"]


def classify_activity_environment(activity_name):
    """Mixed first, then outdoor, then indoor; default mixed."""
    for keyword in MIXED_KEYWORDS:
        if contains_keyword(activity_name, keyword):
            return "mixed"
    for keyword in OUTDOOR_KEYWORDS:
        if contains_keyword(activity_name, keyword):
            return "outdoor"
    for keyword in INDOOR_KEYWORDS:
        if contains_keyword(activity_name, keyword):
            return "indoor"
    return "mixed"


# ---------------------------------------------------------
# Catalog
# ---------------------------------------------------------

def load_activity_catalog(path=None):
    """
    Load the activity catalog.

    Looks for, in order:
      1. an explicit path,
      2. data/activities_catalog.csv (already de-duplicated),
      3. data/activity.json (the raw Kaggle JSON-lines file).
    """
    candidates = [Path(path)] if path else [DATA_DIR / "activities_catalog.csv", DATA_DIR / "activity.json"]
    source = next((p for p in candidates if p.exists()), None)

    if source is None:
        raise FileNotFoundError(
            "No activity dataset found. Copy activity.json from the shared "
            "'Unstructured Final/data_raw' folder into the repo's data/ folder."
        )

    if source.suffix == ".csv":
        raw = pd.read_csv(source)
    else:
        raw = pd.read_json(source, lines=True)

    required = ["activity", "type", "participants", "price"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Activity dataset is missing columns: {missing}")

    activities = raw.drop_duplicates(subset=required).reset_index(drop=True).copy()
    activities = activities[required].copy()
    activities.insert(0, "activity_id", [f"ACT_{i:04d}" for i in range(1, len(activities) + 1)])
    activities["discovery_text"] = activities["activity"].astype(str) + ". Category: " + activities["type"].astype(str)
    activities["environment"] = activities["activity"].apply(classify_activity_environment)
    # None for at-home activities ("Take a nap"); see outings.py.
    activities["place_query"] = activities["activity"].apply(outing_query)
    return activities


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------

class SentenceEmbedder:
    """Thin wrapper so the rest of the code only needs `.encode(list_of_text)`."""

    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts):
        return self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class Resources:
    """Everything that is expensive to build and shared across runs."""

    def __init__(self, embedder, activities):
        self.embedder = embedder
        self.activities = activities
        self.activity_embeddings = np.asarray(embedder.encode(activities["discovery_text"].tolist()))
        self.valid_activity_types = sorted(activities["type"].dropna().astype(str).unique().tolist())


def build_resources(dataset_path=None, embedder=None):
    activities = load_activity_catalog(dataset_path)
    return Resources(embedder or SentenceEmbedder(), activities)
