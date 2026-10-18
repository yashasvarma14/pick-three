# Data

`activities_catalog.csv` is the de-duplicated activity catalog (196 rows) that the
app loads. It is built from the raw Kaggle activity file (`activity.json`, 32,000
JSON-lines rows, 99.4% exact duplicates) with:

```bash
python scripts/build_catalog.py path/to/activity.json
```

Columns: `activity`, `type`, `participants` (suggested group size), `price`
(relative 0 to 1 affordability score, not dollars).
