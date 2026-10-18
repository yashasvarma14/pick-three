# Where things stand (web front end)

Branch: `feature/web-frontend`

Done
- `api/` FastAPI backend: search (streams progress), write-up, feedback, Austin top-10 lists, photo proxy.
  Run: `.venv/Scripts/python -m uvicorn api.main:app --reload --port 8000`, then open http://localhost:8000
- `web/index.html`, `web/css/styles.css`, `web/js/util.js`, `web/js/api.js`
- Design rules: `docs/design.md`

In progress / to do
- `web/js/` modules still to write: motion.js, composer.js, featured.js, progress.js, results.js, map.js, memory.js, main.js
- API tests (tests/test_api.py)
- Deployment: Dockerfile for Hugging Face Spaces (backend), Vercel/Netlify for web/
- Remove the old Streamlit `app.py` + `ui/` once the website matches it
