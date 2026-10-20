"""
The activity catalog (from a Kaggle activity dataset) and the
sentence-embedding model.

The raw Kaggle file has 32,000 rows but only 196 distinct activities, so
the catalog is de-duplicated before use. Most of those are things you do
at home ("Take a nap"), so OUTING_QUERIES below lists the ones that are
real outings and what to type into Google Maps for each.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd



# ---------------------------------------------------------------------------
# Loading the catalog
# ---------------------------------------------------------------------------


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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
    # None for at-home activities ("Take a nap"); see OUTING_QUERIES.
    activities["place_query"] = activities["activity"].apply(outing_query)
    return activities


# ---------------------------------------------------------------------------
# Which catalog activities are real outings
# ---------------------------------------------------------------------------


OUTING_QUERIES = {
    # Social / nightlife
    "go to a karaoke bar with some friends": "karaoke bar",
    "explore the nightlife of your city": "popular cocktail bar",
    "go to an escape room": "escape room",
    "go bowling with some friends": "bowling alley",
    "hold a video game tournament with some friends": "video game lounge",
    "play a video game": "arcade",
    "go to an arcade with friends": "arcade",
    "learn and play a new card game": "board game cafe",
    "invite some friends over for a game night": "board game cafe",
    "have a game night with friends": "board game cafe",
    "play a game of monopoly": "board game cafe",
    # Music, shows, film
    "go to a concert with some friends": "live music venue",
    "go to a concert with local artists with some friends": "live music venue local bands",
    "go to a music festival with some friends": "live music venue",
    "go see a broadway production": "theater live show",
    "go see a movie in theaters with a few friends": "movie theater",
    "watch a classic movie": "classic movie theater",
    # Food and shopping
    "patronize a local independent restaurant": "local independent restaurant",
    "catch up with a friend over a lunch date": "cafe for lunch",
    "cook something together with someone": "cooking class",
    "shop at support your local farmers market": "farmers market",
    "go to a local thrift shop": "thrift store",
    "buy a new house decoration": "home decor store",
    # Calm, relaxing
    "visit a nearby museum": "museum",
    "visit a museum": "museum",
    "go to the library and find an interesting book": "public library",
    "go to a nail salon": "nail salon",
    "take a bubble bath": "day spa",
    "do yoga": "yoga studio",
    "meditate for five minutes": "meditation center",
    "go stargazing": "observatory stargazing",
    "watch the sunset or the sunrise": "sunset viewpoint",
    "paint the first thing you see": "paint and sip studio",
    "learn calligraphy": "calligraphy class",
    "learn woodworking": "woodworking class",
    "take a class at your local community center that interests you": "community center classes",
    # Outdoors and active
    "take a hike at a local park": "hiking trail",
    "explore a park you have never been to before": "park",
    "go for a walk": "walking trail",
    "go for a run": "running trail",
    "have a picnic with some friends": "park with picnic area",
    "have a barbecue with some friends": "park with barbecue grills",
    "have a bonfire with your close friends": "fire pit beach or park",
    "have a photo session with some friends": "scenic photo spot",
    "go on a fishing trip with some friends": "fishing pier",
    "go swimming with a friend": "public swimming pool",
    "go to the gym": "gym day pass",
    "learn how to play a new sport": "recreation center",
    "learn how to iceskate or rollerskate": "skating rink",
    "play a game of tennis with a friend": "public tennis courts",
    "play basketball with a group of friends": "public basketball court",
    "play a volleyball match with some friends": "volleyball court",
    "have a football scrimmage with some friends": "park with football field",
    # Giving back
    "volunteer at a local animal shelter": "animal shelter volunteering",
    "volunteer and help out at a senior center": "senior center volunteering",
    "volunteer at your local food pantry": "food pantry volunteering",
    "volunteer at your local food bank": "food bank volunteering",
    "donate blood at a local blood center": "blood donation center",
}


def outing_query(activity):
    """The Google Maps search phrase for an outing, or None for at-home activities."""
    return OUTING_QUERIES.get(str(activity).strip().lower())


# ---------------------------------------------------------------------------
# Embedding model + everything loaded once per server
# ---------------------------------------------------------------------------


class SentenceEmbedder:
    """
    Thin wrapper so the rest of the code only needs `.encode(list_of_text)`.

    Runs all-MiniLM-L6-v2 with fastembed (ONNX Runtime) instead of PyTorch: the
    same model and the same vectors, but it loads in a few seconds and keeps the
    server image small, so a sleeping Cloud Run instance wakes up much faster.
    """

    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        from fastembed import TextEmbedding

        # FASTEMBED_CACHE_PATH is set in the Dockerfile, where the model is baked in.
        self.model = TextEmbedding(model_name, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"))

    def encode(self, texts):
        vectors = np.array(list(self.model.embed(list(texts))), dtype=np.float32)
        # Unit length, so a dot product is the cosine similarity.
        return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


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
