"""
Streamlit front end for the Adaptive Agentic Activity Recommender.

Run locally:  streamlit run app.py
"""

import html
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from recommender import (
    build_resources,
    create_empty_user_memory,
    feedback_agent,
    update_memory_from_feedback,
)
from recommender.nearby import resolve_location
from recommender.nearby_workflow import run_nearby_workflow
from recommender.workflow import synthesize_results
from recommender import tracker

try:
    from streamlit_js_eval import get_geolocation
except ImportError:  # the app still works with a typed address
    get_geolocation = None
from recommender import config

st.set_page_config(page_title="Tonight, Sorted", page_icon="🎟️", layout="wide")

RUNS_PER_SESSION = int(config.get_setting("RUNS_PER_SESSION", 5))
RUNS_PER_DAY = int(config.get_setting("RUNS_PER_DAY", 150))

EXAMPLES = [
    "Four of us want something fun and social tonight, not too expensive, and we'd rather do something than just sit.",
    "Something relaxing and quiet for two this evening.",
    "Find a live music show tonight for a group of three.",
    "Indoor activity for a rainy afternoon with my roommates.",
]

AGENT_LABELS = {
    "orchestrator": "Planner",
    "memory": "Memory",
    "preference": "Preference",
    "discovery": "Discovery",
    "weather": "Weather",
    "local_places": "Local places",
    "place_relevance": "Venue check",
    "events": "Events",
    "mobility": "Travel time",
    "venue_rag": "Website evidence (RAG)",
    "photo_vibe": "Photo vibe",
    "matching": "Matching",
    "scoring": "Scoring every place",
    "synthesizer": "Write-up",
}

# ---------------------------------------------------------------------------
# Styling: quiet page, one loud element (the ticket-stub result cards)
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Atkinson+Hyperlegible:wght@400;700&display=swap');

:root {
  --paper: #F3F5F2;
  --ink: #1E2A2B;
  --teal: #0E7C7B;
  --marigold: #F2B134;
  --muted: #5F6F6E;
  --stub: #FFFFFF;
}

html, body, .stMarkdown, .stTextArea textarea, .stButton button, [data-testid="stSidebar"] {
  font-family: 'Atkinson Hyperlegible', sans-serif;
}
h1, h2, h3, .ticket-name {
  font-family: 'Bricolage Grotesque', sans-serif;
  letter-spacing: -0.01em;
}
.hero-title {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 700;
  font-size: clamp(2.2rem, 5vw, 3.6rem);
  line-height: 1.02;
  color: var(--ink);
  margin: 0.2rem 0 0.4rem 0;
}
.hero-sub {
  color: var(--muted);
  font-size: 1.05rem;
  max-width: 62ch;
  margin-bottom: 1.2rem;
}

.ticket {
  display: grid;
  grid-template-columns: 76px 1fr;
  background: var(--stub);
  border: 1.5px solid var(--ink);
  border-radius: 14px;
  margin: 0 0 1.1rem 0;
  overflow: hidden;
}
.ticket-rank {
  background: var(--marigold);
  color: var(--ink);
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 700;
  font-size: 2.4rem;
  display: flex;
  align-items: center;
  justify-content: center;
  border-right: 2px dashed var(--ink);
}
.ticket-body { padding: 1rem 1.25rem 1.1rem 1.25rem; }
.ticket-name { font-size: 1.45rem; font-weight: 700; color: var(--ink); margin: 0; }
.ticket-activity { color: var(--teal); font-weight: 700; margin: 0.15rem 0 0.6rem 0; }
.ticket-why { color: var(--ink); line-height: 1.55; max-width: 72ch; margin: 0 0 0.6rem 0; }
.ticket ul { margin: 0 0 0.6rem 1.1rem; padding: 0; color: var(--ink); }
.ticket li { margin: 0.1rem 0; }
.ticket-caveat { color: #8A4B00; font-size: 0.95rem; margin: 0 0 0.6rem 0; }
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.ticket-photos { display: flex; gap: 0.4rem; margin: 0 0 0.6rem 0; overflow-x: auto; }
.ticket-photos img { height: 96px; width: auto; max-width: 160px; object-fit: cover; border-radius: 6px; flex: 0 0 auto; }
.ticket-vibe { font-size: 0.95rem; margin: 0 0 0.6rem 0; }
.photo-credit { font-size: 0.75rem; opacity: 0.7; margin: -0.3rem 0 0.6rem 0; }
.chip {
  border: 1px solid #C9D2D0;
  border-radius: 999px;
  padding: 0.12rem 0.65rem;
  font-size: 0.88rem;
  color: var(--ink);
  background: var(--paper);
  text-decoration: none;
}
a.chip, a.chip:visited { color: var(--teal) !important; text-decoration: none !important; font-weight: 700; }
a.chip:hover, a.chip:focus-visible { border-color: var(--teal); outline: 2px solid var(--teal); outline-offset: 1px; }

@media (max-width: 640px) {
  .ticket { grid-template-columns: 52px 1fr; }
  .ticket-rank { font-size: 1.7rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resources and limits
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


def local_now(tz_name=None):
    """
    Current wall-clock time where the visitor is. Hosting servers run on
    UTC, which would make "today" wrong after 7 PM in Austin.
    """
    tz_name = tz_name or config.get_setting("APP_TIMEZONE", "America/Chicago")
    try:
        return datetime.now(ZoneInfo(tz_name)).replace(tzinfo=None)
    except Exception:
        return datetime.now()


def current_location():
    """The visitor's location from GPS or the typed text, or None."""
    try:
        if ss.coords:
            return resolve_location(latitude=ss.coords[0], longitude=ss.coords[1])
        if ss.place_text.strip():
            return resolve_location(ss.place_text.strip())
    except Exception as exc:
        ss.location_error = str(exc)
    return None


def default_time(now):
    """7 PM, or the next whole hour if it's already evening."""
    if now.hour < 18 or now.hour >= 23:
        return dtime(19, 0)
    return dtime(now.hour + 1, 0)


# ---------------------------------------------------------------------------
# Session state
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
# Sidebar: the plan's details
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Your plan")
    st.markdown("**Where are you?**")
    if get_geolocation is not None and not ss.coords:
        if st.button("📍 Use my location", width="stretch"):
            ss.want_gps = True
    if ss.want_gps and get_geolocation is not None:
        geo = get_geolocation()
        if geo and geo.get("coords"):
            ss.coords = (geo["coords"]["latitude"], geo["coords"]["longitude"], geo["coords"].get("accuracy"))
            ss.want_gps = False
            st.rerun()
        elif geo and geo.get("error"):
            st.warning("Your browser didn't share a location. Type an address instead.")
            ss.want_gps = False
        else:
            st.caption("Allow location access in your browser's pop-up…")
    if ss.coords:
        accuracy = f" (within about {ss.coords[2]:.0f} m)" if ss.coords[2] else ""
        st.success(f"Using your current location{accuracy}.")
        if st.button("Type an address instead"):
            ss.coords = None
            st.rerun()
    else:
        st.text_input("Address, neighborhood, landmark or ZIP", key="place_text",
                      help="Searches and drive times start here.")

    location = current_location()
    if ss.location_error and not location:
        st.error(ss.location_error)
    ss.location_error = None

    right_now = st.toggle("Right now", value=False)
    if right_now:
        target_dt = None
    else:
        col_d, col_t = st.columns(2)
        now_here = local_now((location or {}).get("timezone"))
        today_here = now_here.date()
        plan_date = col_d.date_input("Date", value=today_here, min_value=today_here,
                                     max_value=today_here + timedelta(days=6))
        times = [dtime(h, m) for h in range(6, 24) for m in (0, 30)]
        start_time = default_time(now_here)
        plan_time = col_t.selectbox("Time", times, index=times.index(start_time) if start_time in times else 26,
                                    format_func=lambda t: t.strftime("%I:%M %p").lstrip("0"))
        target_dt = datetime.combine(plan_date, plan_time)

    remembered_travel = ss.memory.get("max_travel_minutes")
    max_travel = st.slider("Longest drive you'd accept (minutes)", 5, 60,
                           value=int(remembered_travel or 20), step=5)
    deep = st.toggle("Search deeper", value=False,
                     help="Up to 60 places per search instead of 20 (Google's maximum). Slower.")
    use_memory = st.toggle("Use what I've told you before", value=True)

    with st.expander("What I remember about you"):
        mem = ss.memory
        shown = False
        for key, label in [("liked_activities", "Liked"), ("disliked_activities", "Not into"),
                           ("liked_places", "Liked places"), ("rejected_places", "Skipped places"),
                           ("positive_preferences", "More of"), ("negative_preferences", "Less of")]:
            if mem.get(key):
                st.markdown(f"**{label}:** {', '.join(mem[key])}")
                shown = True
        if mem.get("max_travel_minutes"):
            st.markdown(f"**Max drive:** {mem['max_travel_minutes']} min")
            shown = True
        if not shown:
            st.caption("Nothing yet. Rate a set of recommendations and I'll remember it for this session.")
        if shown and st.button("Forget all of this"):
            ss.memory = create_empty_user_memory("visitor")
            st.rerun()

    st.caption(
        f"Searches left this session: {max(0, RUNS_PER_SESSION - ss.runs_used)}. "
        "Memory lives only in this browser tab."
        + (" Searches and ratings are logged anonymously (no exact location) to evaluate the app."
           if tracker.enabled() else "")
    )

# ---------------------------------------------------------------------------
# Hero: the question itself
# ---------------------------------------------------------------------------

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
# Run
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

            result = run_nearby_workflow(
                resources,
                prompt.strip(),
                location,
                memory=ss.memory,
                target_datetime=target_dt,
                max_travel_minutes=max_travel,
                use_memory=use_memory,
                progress=progress,
                synthesize=False,  # picks show first; the write-up fills in after
                search_pages=3 if deep else 1,
            )
            ok = bool(result["final_recommendations"])
            box.update(label=f"Done in {result['runtime_seconds']}s" if ok else "Finished without results",
                       state="complete" if ok else "error", expanded=False)
        ss.result = result
        ss.needs_writeup = ok
        if not ok:
            tracker.log_run(result, ss.session_id)  # no write-up coming: log now

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def chip(text, href=None):
    text = html.escape(str(text))
    if href:
        return f'<a class="chip" href="{html.escape(str(href))}" target="_blank" rel="noopener">{text}</a>'
    return f'<span class="chip">{text}</span>'


def text(value):
    """Clean display text: missing values (None, NaN) become an empty string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def render_ticket(rank, rec, explanation):
    name = html.escape(text(rec.get("place_name")))
    if text(rec.get("primary_type")):
        activity = html.escape(f'{text(rec["primary_type"])} · found by searching "{text(rec.get("source_activity"))}"')
    else:
        activity = html.escape(text(rec.get("source_activity")))
    why = html.escape(text(explanation.get("why_it_fits")) or text(rec.get("rag_evidence_summary")))
    evidence = explanation.get("supporting_evidence")
    evidence = [e for e in evidence if text(e)] if isinstance(evidence, list) else []
    caveat = text(explanation.get("important_caveat"))

    chips = []
    if pd.notna(rec.get("rating")):
        reviews = f" from {int(rec['review_count']):,} reviews" if pd.notna(rec.get("review_count")) else ""
        chips.append(chip(f"★ {rec['rating']:.1f}{reviews}"))
    open_now = rec.get("open_at_planned_time")
    if open_now is True:
        chips.append(chip("Open at your time"))
    elif open_now is False:
        chips.append(chip("Closed at your time"))
    if pd.notna(rec.get("traffic_minutes")):
        chips.append(chip(f"{rec['traffic_minutes']:.0f} min drive"))
    if pd.notna(rec.get("distance_miles")):
        chips.append(chip(f"{rec['distance_miles']:.1f} mi"))
    if text(rec.get("google_maps_uri")):
        chips.append(chip("Open in Maps", rec["google_maps_uri"]))
    if isinstance(rec.get("website_uri"), str) and rec["website_uri"]:
        chips.append(chip("Website", rec["website_uri"]))

    # Photo Vibe: thumbnails the vision model looked at, plus what it saw.
    photo_urls = rec.get("photo_urls") if isinstance(rec.get("photo_urls"), list) else []
    photos_html = "".join(f'<img src="{html.escape(u)}" alt="Photo of {name}" loading="lazy">'
                          for u in photo_urls if isinstance(u, str) and u.startswith("https://"))
    credits = rec.get("photo_attributions") if isinstance(rec.get("photo_attributions"), list) else []
    tags = rec.get("vibe_tags") if isinstance(rec.get("vibe_tags"), list) else []
    vibe = text(rec.get("photo_summary"))
    if tags:
        chips.extend(chip(t) for t in tags)

    bullets = "".join(f"<li>{html.escape(str(b))}</li>" for b in evidence)
    # Built without indentation or blank lines: Markdown would otherwise
    # treat indented HTML as a code block.
    parts = [
        '<div class="ticket">',
        f'<div class="ticket-rank" aria-label="Rank {rank}">{rank}</div>',
        '<div class="ticket-body">',
        f'<p class="ticket-name">{name}</p>',
        f'<p class="ticket-activity">{activity}</p>',
        f'<div class="ticket-photos">{photos_html}</div>' if photos_html else "",
        (f'<p class="photo-credit">Photos: Google Maps'
         + (f' · {html.escape(", ".join(credits[:3]))}' if credits else "") + "</p>") if photos_html else "",
        f'<p class="ticket-why">{why}</p>' if why else "",
        f'<p class="ticket-vibe"><strong>From the photos:</strong> {html.escape(vibe)}</p>' if vibe else "",
        f"<ul>{bullets}</ul>" if bullets else "",
        f'<p class="ticket-caveat">Heads up: {html.escape(caveat)}</p>' if caveat else "",
        f'<div class="chips">{"".join(chips)}</div>',
        "</div></div>",
    ]
    st.markdown(
        "".join(parts),
        unsafe_allow_html=True,
    )


result = ss.result
if result:
    if result.get("clarification_question"):
        st.info(f"**The planner wanted to check:** {result['clarification_question']}  \n"
                "The results below use your current settings. Add that detail to your request to refine them.")
    for line in result.get("adaptation_log", []):
        if line.startswith("Your request mentions a time"):
            st.caption("🕒 " + line)
        if line.startswith("Your request names"):
            st.caption("📍 " + line)
    if result.get("weak_match"):
        st.warning("**No strong match nearby.** These are the closest options found; "
                   "try a longer drive limit or a different way of describing it.")

    recs = result.get("final_recommendations") or []
    synthesis = result.get("final_response") or {}
    explanations = {str(r.get("place_name")): r for r in synthesis.get("recommendations", []) if isinstance(r, dict)}

    if not recs:
        problems = " ".join(result.get("errors", []) + result.get("warnings", []))
        if "429" in problems or "RESOURCE_EXHAUSTED" in problems:
            st.error(
                "Google Maps turned the searches down: too many requests on this API key. "
                "If the message below says **per minute**, wait a minute and try again. "
                "If it says **per day**, the key's daily limit is used up; raise it in Google Cloud Console "
                "(APIs & Services → Places API (New) → Quotas) or try again tomorrow."
            )
        st.warning(
            "No places made it through this time. Try a longer drive limit, \"Search deeper\", "
            "or describing what you want differently."
        )
        for message in dict.fromkeys(result.get("errors", []) + result.get("warnings", [])):
            st.caption(message)
    else:
        writeup_slot = st.empty()
        if synthesis.get("overall_summary"):
            writeup_slot.subheader(synthesis["overall_summary"])
        elif ss.needs_writeup:
            writeup_slot.info("✍️ Here are your picks. Writing up why each one fits…")
        for i, rec in enumerate(recs, start=1):
            render_ticket(i, rec, explanations.get(str(rec.get("place_name")), {}))
        if synthesis.get("personalization_summary"):
            st.caption(synthesis["personalization_summary"])

        # ---------------- Quick rating -> tracker ----------------
        run_id = result.get("run_id")
        if ss.voted_run == run_id:
            st.caption("Thanks for rating these picks!")
        else:
            c_label, c_yes, c_no = st.columns([3, 1, 1])
            c_label.markdown("**Were these picks helpful?**")
            if c_yes.button("👍 Yes", key=f"yes_{run_id}", width="stretch"):
                tracker.log_feedback(run_id, ss.session_id, helpful=True)
                ss.voted_run = run_id
                st.rerun()
            if c_no.button("👎 No", key=f"no_{run_id}", width="stretch"):
                tracker.log_feedback(run_id, ss.session_id, helpful=False)
                ss.voted_run = run_id
                st.rerun()

        # ---------------- Feedback -> memory ----------------
        with st.form("feedback", clear_on_submit=True):
            st.markdown("**How did these land?** Your answer shapes the next search in this session.")
            feedback_text = st.text_area("Feedback", label_visibility="collapsed",
                                         placeholder="e.g. Not another movie. Something more active, and nothing over 15 minutes away.")
            if st.form_submit_button("Save feedback") and feedback_text.strip():
                fb = feedback_agent(feedback_text.strip(), recs)
                if fb["status"] == "success":
                    ss.memory = update_memory_from_feedback(ss.memory, fb["data"])
                    ss.feedback_saved = fb["data"].get("feedback_summary") or "Saved."
                    tracker.log_feedback(result.get("run_id"), ss.session_id, text=feedback_text.strip(),
                                         summary=fb["data"].get("feedback_summary"))
                else:
                    ss.feedback_saved = f"Couldn't read that feedback: {fb.get('error')}"
                st.rerun()
        if ss.feedback_saved:
            st.success(f"Saved for your next search: {ss.feedback_saved}")

    # ---------------- Detail tabs ----------------
    all_places = pd.DataFrame(result.get("all_places") or [])
    for col in ["rating", "review_count", "straight_line_miles", "traffic_minutes", "latitude", "longitude", "top_pick"]:
        if col in all_places.columns:
            all_places[col] = pd.to_numeric(all_places[col], errors="coerce")
    tab_all, tab_map, tab_events, tab_trace, tab_table = st.tabs(
        [f"All places ({len(all_places)})", "Map", "Live events", "How the agents decided", "Top-pick scores"])

    with tab_all:
        if all_places.empty:
            st.caption("No places were found.")
        else:
            searches = result.get("search_phrases") or []
            st.caption("Every place Google Maps returned for these searches near you: "
                       + ", ".join(f'"{p}"' for p in searches)
                       + ". Sorted by how well each fits your request. Click a column to sort.")
            only_open = st.checkbox("Only places open at my time", value=False)
            shown = all_places
            if only_open and "open_at_planned_time" in shown.columns:
                shown = shown[shown["open_at_planned_time"] != False]  # noqa: E712  (unknown hours stay)
            shown = shown.assign(
                open_label=shown["open_at_planned_time"].map({True: "Open", False: "Closed"}).fillna("Hours unknown"),
                searches=shown["matched_searches"].apply(lambda x: ", ".join(x) if isinstance(x, list) else ""),
                pick=shown["top_pick"].apply(lambda x: f"#{int(x)}" if pd.notna(x) else ""),
            )
            cols = [c for c in ["list_rank", "pick", "place_name", "primary_type", "rating", "review_count",
                                "straight_line_miles", "traffic_minutes", "open_label", "searches",
                                "google_maps_uri", "website_uri"] if c in shown.columns]
            st.dataframe(shown[cols], hide_index=True, width="stretch", height=520,
                         column_config={"list_rank": "Fit rank", "pick": "Top pick", "place_name": "Place",
                                        "primary_type": "Type", "rating": st.column_config.NumberColumn("Rating", format="%.1f ★"),
                                        "review_count": "Reviews",
                                        "straight_line_miles": st.column_config.NumberColumn("Miles away", format="%.1f"),
                                        "traffic_minutes": st.column_config.NumberColumn("Drive (min)", format="%.0f"),
                                        "open_label": "At your time", "searches": "Found by",
                                        "google_maps_uri": st.column_config.LinkColumn("Maps", display_text="Open"),
                                        "website_uri": st.column_config.LinkColumn("Website", display_text="Visit")})
            st.caption("Miles are straight-line from you; drive times are checked live for the shortlist only.")

    with tab_map:
        points = all_places.dropna(subset=["latitude", "longitude"]).copy() if not all_places.empty else all_places
        if not points.empty:
            points["size"] = points["top_pick"].apply(lambda r: 140 if pd.notna(r) else 45)
            points["color"] = points["top_pick"].apply(lambda r: "#F2B134" if pd.notna(r) else "#0E7C7B")
            here = result.get("location") or {}
            if here.get("latitude") is not None:
                points = pd.concat([points, pd.DataFrame([{"latitude": here["latitude"], "longitude": here["longitude"],
                                                           "size": 120, "color": "#D1495B"}])], ignore_index=True)
            st.map(points[["latitude", "longitude", "size", "color"]], latitude="latitude", longitude="longitude",
                   size="size", color="color")
            st.caption("Red: you. Gold: your top picks. Teal: every other place found.")
        else:
            st.caption("No place locations to show.")

    with tab_events:
        events = pd.DataFrame(result.get("events") or [])
        if events.empty:
            st.caption("The planner didn't call the Events agent for this request, or Ticketmaster had nothing in the time window.")
        else:
            cols = [c for c in ["event_name", "local_date", "local_time", "venue_name", "genre", "price_min", "event_url"] if c in events.columns]
            st.dataframe(events[cols], hide_index=True, width="stretch",
                         column_config={"event_url": st.column_config.LinkColumn("Tickets"),
                                        "event_name": "Event", "local_date": "Date", "local_time": "Time",
                                        "venue_name": "Venue", "genre": "Genre", "price_min": "From ($)"})

    with tab_trace:
        plan = result.get("orchestrator_plan") or {}
        if plan.get("goal_summary"):
            st.markdown(f"**Goal the planner set:** {plan['goal_summary']}")
        called = [AGENT_LABELS.get(a, a) for a in result.get("agents_called", [])]
        skipped = [AGENT_LABELS.get(a, a) for a in result.get("agents_skipped", [])]
        c1, c2 = st.columns(2)
        c1.markdown("**Ran**\n\n" + "\n".join(f"- {a}" for a in called))
        c2.markdown("**Skipped**\n\n" + ("\n".join(f"- {a}" for a in skipped) or "Nothing"))
        if plan.get("planning_reasoning_summary"):
            st.markdown("**Why**")
            for line in plan["planning_reasoning_summary"]:
                st.markdown(f"- {line}")
        if result.get("adaptation_log"):
            st.markdown("**Adjustments made during the run**")
            for line in result["adaptation_log"]:
                st.markdown(f"- {line}")
        weather = result.get("weather") or {}
        if weather:
            st.markdown(
                f"**Weather used:** {weather.get('condition')}, {weather.get('temperature_f')}°F, "
                f"{weather.get('precipitation_probability')}% chance of rain (outdoor suitability {weather.get('weather_suitability')})"
            )
        usage = result.get("usage", {})
        calls = usage.get("api_calls", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Run time", f"{result.get('runtime_seconds')}s")
        m2.metric("LLM tokens", f"{usage.get('total_tokens', 0):,}")
        m3.metric("Est. LLM cost", f"${usage.get('llm_cost', 0):.4f}")
        m4.metric("Maps calls", calls.get("google_places", 0) + calls.get("google_routes", 0)
                  + calls.get("google_place_photos", 0))
        timings = result.get("timings") or {}
        if timings:
            st.markdown("**Seconds per step** (some steps run at the same time, so these add up to more than the run time)")
            st.bar_chart(pd.Series({AGENT_LABELS.get(k, k): v for k, v in timings.items()}, name="seconds"),
                         horizontal=True, height=280)

    with tab_table:
        ranked = pd.DataFrame(result.get("ranked_results") or [])
        if ranked.empty:
            st.caption("Nothing was ranked.")
        else:
            cols = ["final_rank", "place_name", "source_activity", "rating", "traffic_minutes",
                    "activity_context_score", "place_quality_score", "mobility_score",
                    "evidence_fit_score", "photo_vibe_score", "final_recommendation_score"]
            st.dataframe(ranked[[c for c in cols if c in ranked.columns]], hide_index=True, width="stretch",
                         column_config={"final_rank": "Rank", "place_name": "Venue", "source_activity": "Activity",
                                        "rating": "Rating", "traffic_minutes": "Drive (min)",
                                        "activity_context_score": "Activity fit", "place_quality_score": "Venue quality",
                                        "mobility_score": "Travel", "evidence_fit_score": "Website evidence",
                                        "photo_vibe_score": "Photo vibe",
                                        "final_recommendation_score": "Final score"})
            st.caption("Final score weights: 40 activity fit, 25 venue quality, 15 travel, 20 website evidence, "
                       "15 photo vibe. Missing signals are dropped and the rest re-normalized to 100%.")
else:
    st.caption("Results show up here as three ranked picks, with a map, live events, and a trace of every agent decision.")

# ---------------------------------------------------------------------------
# After the page is drawn
# ---------------------------------------------------------------------------

if ss.result and ss.needs_writeup:
    # The ranked picks are already on screen; now add the explanations.
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
