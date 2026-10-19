# Where things stand (web front end)

Branch: `feature/web-frontend`

Run it: `.venv/Scripts/python -m uvicorn api.main:app --port 8000`, then open http://localhost:8000

Done and tested in the browser
- `api/` FastAPI backend: search (streams progress), write-up, feedback -> memory, top-10 lists, photo proxy,
  daily caps (3 searches per visitor, 10 total by default; RUNS_PER_VISITOR_PER_DAY / RUNS_PER_DAY to change)
- `web/`: home page (hero, search box with where/when/drive, Austin top 10 from a snapshot, how it works),
  live progress checklist, results (3 picks with photos, map, all places, events, agent trace, feedback)
- Top-10 lists come from `web/data/featured.json`; refresh monthly with `python scripts/build_featured.py`

To do
- API tests (tests/test_api.py) with the fakes in tests/fakes.py
- Deployment: Dockerfile for Hugging Face Spaces (backend + site together is simplest)
- Google Cloud: per-day quota caps on Places + Routes, and a $1 budget alert
- Remove the old Streamlit `app.py` + `ui/` once happy with the website; update README
- Visual polish pass on phone sizes
