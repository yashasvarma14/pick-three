"""
Make the home page's "Austin's top 10" snapshot (web/data/featured.json).

Each run costs one Google Maps search per category (6-7 searches), which
is well inside Google's free monthly allowance. Re-run it about once a
month to keep the lists fresh:

    python scripts/build_featured.py
"""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.featured import CATEGORIES, SNAPSHOT, live_list  # noqa: E402
from recommender.core import config  # noqa: E402


def main():
    if not config.google_maps_api_key():
        sys.exit("Set GOOGLE_MAPS_API_KEY (or add it to .streamlit/secrets.toml) first.")
    lists = {}
    for key, spec in CATEGORIES.items():
        if spec["search"] is None and not config.ticketmaster_api_key():
            continue
        items = live_list(key)
        lists[key] = {"label": spec["label"], "items": items}
        print(f"{spec['label']:<18} {len(items)} places")
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps({"generated": date.today().isoformat(), "lists": lists}, indent=1, default=str),
                        encoding="utf-8")
    print(f"Wrote {SNAPSHOT}")


if __name__ == "__main__":
    main()
