"""
Turn the raw Kaggle activity file (32,000 rows, JSON lines) into the
196-row catalog the app loads. Commit the output, not the raw file.

Usage:
    python scripts/build_catalog.py path/to/activity.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recommender.data import DATA_DIR, load_activity_catalog  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python scripts/build_catalog.py path/to/activity.json")

    catalog = load_activity_catalog(sys.argv[1])
    out = DATA_DIR / "activities_catalog.csv"
    catalog[["activity", "type", "participants", "price"]].to_csv(out, index=False)
    print(f"Wrote {len(catalog)} unique activities to {out}")


if __name__ == "__main__":
    main()
