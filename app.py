"""
Tonight, Sorted - the Streamlit app.

Run locally:  streamlit run app.py

This file only puts the page together. The pieces live in ui/
(sidebar, result cards, detail tabs, styles) and the recommender itself
in recommender/ (start with recommender/pipeline.py).
"""

from datetime import date

import streamlit as st

from recommender import build_resources, create_empty_user_memory, run_pipeline, synthesize_results
from recommender.core import config
from recommender.services import tracker
from ui.details import render_details
from ui.labels import AGENT_LABELS, EXAMPLES
from ui.picks import render_picks
from ui.sidebar import render_sidebar
from ui.styles import apply_styles

st.set_page_config(page_title="Tonight, Sorted", page_icon="🎟️", layout="wide")
apply_styles()

RUNS_PER_SESSION = int(config.get_setting("RUNS_PER_SESSION", 5))
RUNS_PER_DAY = int(config.get_setting("RUNS_PER_DAY", 150))


# ---------------------------------------------------------------------------
# 1. Things loaded once per server, and the demo's spending limits
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the activity catalog and embedding model…")
def get_resources():
    return build_resources()


@st.cache_resource
def daily_counter():
    """Process-wide run counter: a simple cap on API spend for a public demo."""
    return {"date": date.today(), "runs": 0}


def runs_left_today():
    counter = daily_counter()
    if counter["date"] != date.today():
        counter.update(date=date.today(), runs=0)
    return RUNS_PER_DAY - counter["runs"]


# ---------------------------------------------------------------------------
# 2. Per-visitor session state
# ---------------------------------------------------------------------------

ss = st.session_state
ss.setdefault("memory", create_empty_user_memory("visitor"))
ss.setdefault("runs_used", 0)
ss.setdefault("result", None)
ss.setdefault("prompt", "")
ss.setdefault("feedback_saved", None)
ss.setdefault("needs_writeup", False)
ss.setdefault("place_text", "University of Texas at Austin")
ss.setdefault("coords", None)
ss.setdefault("want_gps", False)
ss.setdefault("location_error", None)
ss.setdefault("session_id", tracker.new_session_id())
ss.setdefault("voted_run", None)

# ---------------------------------------------------------------------------
# 3. Sidebar + the request box
# ---------------------------------------------------------------------------

plan = render_sidebar(ss, searches_left=max(0, RUNS_PER_SESSION - ss.runs_used))
location = plan["location"]

st.markdown('<div class="hero-title" role="heading" aria-level="1">What should we do tonight?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Describe the plan in your own words. A planner agent decides which specialists to call '
    "(weather, real venues, live events, drive times, venue websites), then ranks real places and explains each pick.</div>",
    unsafe_allow_html=True,
)

example_cols = st.columns(len(EXAMPLES))
for col, example in zip(example_cols, EXAMPLES):
    if col.button(example[:38] + ("…" if len(example) > 38 else ""), help=example, width="stretch"):
        ss.prompt = example

prompt = st.text_area("Your request", key="prompt", height=96, label_visibility="collapsed",
                      placeholder="e.g. Three of us want something active outdoors this evening, under $30 each")

status = config.api_status()
missing = [name for name, ok in [("Claude", status["llm"]), ("Google Maps", status["google_maps"])] if not ok]
if missing:
    st.warning(f"The app owner still needs to add API keys for: {', '.join(missing)}.")

go = st.button("Find options", type="primary", disabled=not prompt.strip() or bool(missing) or location is None)
if location is None and not missing:
    st.caption("Add where you are in the sidebar to search.")

# ---------------------------------------------------------------------------
# 4. Run the pipeline
# ---------------------------------------------------------------------------

if go:
    if ss.runs_used >= RUNS_PER_SESSION:
        st.error(f"This demo allows {RUNS_PER_SESSION} searches per visit to keep API costs down. Refresh later to try again.")
    elif runs_left_today() <= 0:
        st.error("The demo has hit today's search limit. It resets at midnight.")
    else:
        try:
            resources = get_resources()
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()

        ss.runs_used += 1
        daily_counter()["runs"] += 1
        ss.feedback_saved = None

        with st.status("Working on it…", expanded=True) as box:
            def progress(step, message):
                box.write(f"**{AGENT_LABELS.get(step, step)}:** {message}")

            result = run_pipeline(
                resources,
                prompt.strip(),
                location,
                memory=ss.memory,
                target_datetime=plan["target_datetime"],
                max_travel_minutes=plan["max_travel_minutes"],
                use_memory=plan["use_memory"],
                progress=progress,
                synthesize=False,  # picks show first; the write-up fills in after (step 6)
                search_pages=3 if plan["deep"] else 1,
            )
            ok = bool(result["final_recommendations"])
            box.update(label=f"Done in {result['runtime_seconds']}s" if ok else "Finished without results",
                       state="complete" if ok else "error", expanded=False)
        ss.result = result
        ss.needs_writeup = ok
        if not ok:
            tracker.log_run(result, ss.session_id)  # no write-up coming: log now

# ---------------------------------------------------------------------------
# 5. Results
# ---------------------------------------------------------------------------

if ss.result:
    render_picks(ss.result, ss)
    render_details(ss.result)
else:
    st.caption("Results show up here as three ranked picks, with a map, live events, and a trace of every agent decision.")

# ---------------------------------------------------------------------------
# 6. After the page is drawn: the write-up, then redraw with it
# ---------------------------------------------------------------------------

if ss.result and ss.needs_writeup:
    ss.needs_writeup = False
    synthesize_results(ss.result)
    tracker.log_run(ss.result, ss.session_id)  # background; never slows the page
    st.rerun()

# Load the embedding model while the visitor is still typing, so the first
# search doesn't pay for it. Cached: this is instant after the first visit.
try:
    get_resources()
except FileNotFoundError:
    pass
