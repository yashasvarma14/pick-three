# Where things stand (web front end)

Branch: `feature/web-frontend`

Done
- `api/` FastAPI backend: search (streams progress), write-up, feedback, Austin top-10 lists, photo proxy.
  Run: `.venv/Scripts/python -m uvicorn api.main:app --reload --port 8000`, then open http://localhost:8000
- `web/index.html`, `web/css/styles.css`
- `web/js/`: util, api, motion, composer, featured, memory, main (home page works: hero, composer, top-10 rail, memory panel)
- Design rules: `docs/design.md`

In progress / to do
- `web/js/search.js` (not written yet): call `search()` from api.js, show the progress checklist (#progress, #steps,
  STEP_LABELS in util.js), switch `body[data-view]` to "results", render picks (#picks, .pick cards with photos via
  photoUrl), Leaflet map (#map, CARTO light tiles), all-places list, events, trace, then call `getWriteup(run_id)`
  and fill in the write-up; feedback form -> `sendFeedback` -> `memory.set(...)`. Hook it up in main.js onSubmit.
- Not yet tested in a browser at all.
- API tests (tests/test_api.py)
- Deployment: Dockerfile for Hugging Face Spaces (backend), Vercel/Netlify for web/
- Remove the old Streamlit `app.py` + `ui/` once the website matches it
