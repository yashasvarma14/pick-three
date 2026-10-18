"""
The pipeline: one search, start to finish.
==========================================

run_pipeline() is the whole recommender. It calls the agents in
recommender/agents/ in this order (the step numbers match the comments
below and the agent files):

    1  Planner + Preference        LLM, both at the same time
       (Weather and Events start in the background right away)
    2  Discovery                   catalog ideas for open-ended requests
    3  Weather                     needed before scoring
    4  Place search                Google Maps, around the visitor
    5  Score every place           embeddings + rating + distance + hours
    6  Shortlist                   one per business, some variety
       (reviews, websites and drive times start downloading here)
    7  Venue check                 LLM: does each place really fit?
    8  Travel time                 Google Routes, live traffic
    9  Evidence                    9a reviews/websites (RAG) and
                                   9b photos (vision), at the same time
    10 Matching                    final weighted ranking -> top 3
    11 Write-up                    LLM explanation, run separately by the
                                   app so the picks show up first

Anything slow that doesn't depend on the step before it runs in the
background thread pool, which is why a run takes ~30 s instead of minutes.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .agents.discovery import discovery_agent
from .agents.events import events_agent
from .agents.matching import matching_agent
from .agents.memory import build_memory_context, filter_rejected_places
from .agents.photo_vibe import photo_vibe_agent
from .agents.place_scoring import score_places
from .agents.place_search import area_search_agent, search_radius_km
from .agents.planner import orchestrator_planner
from .agents.preference import preference_agent
from .agents.relevance import place_activity_relevance_agent
from .agents.review_rag import prefetch_websites, venue_rag_agent
from .agents.shortlist import diverse_selection, remove_duplicate_listings
from .agents.travel_time import enforce_drive_limit, mobility_agent
from .agents.weather import weather_agent
from .agents.writeup import recommendation_synthesizer
from .core.config import DEFAULTS
from .core.helpers import contains_keyword, distance_km
from .core.state import add_once, add_usage, create_initial_state, record, records, timed
from .services.google_places import safe_place_details
from .services.location import resolve_location


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def local_now(tz_name):
    """The current wall-clock time where the visitor is."""
    try:
        return datetime.now(ZoneInfo(tz_name)).replace(tzinfo=None) if tz_name else datetime.now()
    except Exception:
        return datetime.now()


def build_search_phrases(preferences, discovery_df, memory, n_catalog, max_searches=4):
    """
    Searches to run, best first:
      1. the specific thing asked for ("cheap burger restaurant")
      2. the LLM's own map searches for this request
      3. a catalog outing, ONLY when the LLM gave fewer than 3 ideas
         (catalog ideas are generic and used to push out better searches,
         e.g. "hiking trail" for a toddler on a rainy morning)
    Returns (searches, searches skipped because of the visitor's memory).
    """
    specific = [t.lower() for t in preferences.get("venue_search_terms") or []]
    phrases = list(dict.fromkeys(specific + (preferences.get("search_phrases") or [])))
    if len(phrases) < 3 and not specific and discovery_df is not None and not discovery_df.empty \
            and "place_query" in discovery_df:
        extra = [q for q in dict.fromkeys(discovery_df["place_query"].dropna().tolist()) if q not in phrases]
        phrases += extra[: max(n_catalog, 3 - len(phrases))]
    disliked = (memory or {}).get("disliked_activities") or []
    skipped_for_memory = [p for p in phrases if any(contains_keyword(p, d) for d in disliked)]
    kept = [p for p in phrases if p not in skipped_for_memory]
    return kept[:max_searches], skipped_for_memory


def open_or_unknown(df):
    """Drop places Google says are closed at the planned time (unknown hours stay)."""
    if "open_at_planned_time" in df.columns:
        return df[df["open_at_planned_time"] != False]  # noqa: E712
    return df


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    resources,
    user_prompt,
    location,
    memory=None,
    target_datetime=None,
    max_travel_minutes=20,
    use_memory=True,
    synthesize=True,
    progress=None,
    **overrides,
):
    """
    resources        catalog + embedding model (core/catalog.py build_resources)
    location         from services/location.py resolve_location()
    target_datetime  naive local datetime, or None for right now
    synthesize       False = skip step 11 (the app runs it after showing the picks)
    progress         optional callback(step_name, message) for the progress box
    overrides        any key from DEFAULTS in core/config.py
    """
    settings = {**DEFAULTS, **overrides}
    notify = progress or (lambda *_: None)
    started = time.time()

    state = create_initial_state(user_prompt=user_prompt, user_id=(memory or {}).get("user_id"),
                                 city=location.get("label"))
    state["execution_mode"] = "near_me"
    state["location"] = {k: location.get(k) for k in ["latitude", "longitude", "label", "timezone"]}
    memory = memory if (use_memory and memory) else {}
    memory_context = build_memory_context(memory)
    state["memory_context"] = memory_context
    state["user_profile"] = memory
    timings = state["timings"] = {}
    pool = ThreadPoolExecutor(max_workers=8)  # background work for the whole run

    def finish(reason):
        state["finish_reason"] = state["finish_reason"] or reason
        state["runtime_seconds"] = round(time.time() - started, 2)
        pool.shutdown(wait=False, cancel_futures=True)
        return state

    def start_weather_and_events():
        radius = search_radius_km(max_travel_minutes) * 0.621371
        return (pool.submit(timed, timings, "weather", weather_agent, target_datetime, location),
                pool.submit(timed, timings, "events", events_agent, target_datetime, location, radius,
                            settings["event_max_results"]))

    # ======================================================================
    # STEP 1. Planner + Preference agents, at the same time.
    #         Weather and Events only need the place and time, so they
    #         start now too.
    # ======================================================================
    notify("orchestrator", "Planning which agents this request needs and reading your request")
    when_text = target_datetime.isoformat(sep=" ", timespec="minutes") if target_datetime else None
    plan_future = pool.submit(
        timed, timings, "orchestrator", orchestrator_planner, user_prompt, memory_context=memory_context,
        city=location.get("label"), origin=location.get("label"), target_datetime=when_text,
        use_memory=bool(memory), max_agent_calls=settings["max_agent_calls"])

    preference_query = user_prompt
    if memory_context:
        preference_query = (
            f"CURRENT USER REQUEST:\n{user_prompt}\n\nRELEVANT PERSISTENT USER MEMORY:\n{memory_context}\n\n"
            "The current request has the highest priority. Use memory only as extra personalization; "
            "if they conflict, follow the current request.")
    pref_future = pool.submit(timed, timings, "preference", preference_agent, preference_query,
                              resources.valid_activity_types, app_city=location.get("label"),
                              now=local_now(location.get("timezone")), app_time=target_datetime)
    weather_future, events_future = start_weather_and_events()

    # The plan.
    plan_response = plan_future.result()
    if not record(state, "orchestrator", plan_response):
        state["errors"].append(plan_response.get("error") or "Orchestrator failed.")
        return finish("Stopped: the orchestrator could not create a plan.")
    plan = plan_response["data"]["plan"]
    state["orchestrator_plan"] = plan
    planned = plan["execution_order"]
    state["agents_requested"] = ["orchestrator"] + planned
    for agent in plan["agents_to_skip"]:
        add_once(state["agents_skipped"], agent)
    if plan.get("human_checkpoint_needed") and plan.get("clarification_question"):
        # Searching anyway beats an empty page: the question is shown as a
        # note above the results instead of stopping the run.
        state["human_checkpoint_used"] = True
        state["clarification_question"] = plan.get("clarification_question")
        state["adaptation_log"].append("The planner had a question but searched anyway with your settings.")
    if memory:
        add_once(state["agents_called"], "memory")

    # The structured request.
    pref = pref_future.result()
    if not record(state, "preference", pref):
        state["errors"].append(pref.get("error") or "Preference Agent failed.")
        return finish("Stopped: the request could not be interpreted.")
    preferences = state["preferences"] = pref["data"]

    # A time written in the request ("this afternoon") wins over the app's
    # date/time setting, which people often leave alone.
    requested = preferences.get("requested_datetime")
    if requested and (target_datetime is None or abs((requested - target_datetime).total_seconds()) > 90 * 60):
        target_datetime = requested
        state["adaptation_log"].append(
            f"Your request mentions a time, so places were checked for {requested:%A %I:%M %p}.".replace(" 0", " "))
        weather_future, events_future = start_weather_and_events()
    state["time_used"] = target_datetime.isoformat(sep=" ", timespec="minutes") if target_datetime else "right now"

    # "Best tacos in Paris, France": search where the request says, not where
    # the visitor is (drive times from here would be meaningless).
    skip_travel = False
    named_city = preferences.get("city")
    if named_city:
        try:
            elsewhere = resolve_location(str(named_city))
            km = distance_km(location["latitude"], location["longitude"], elsewhere["latitude"], elsewhere["longitude"])
        except Exception:
            elsewhere, km = None, 0
        if elsewhere and km > 50:
            location = elsewhere
            skip_travel = True
            state["location"] = {k: location.get(k) for k in ["latitude", "longitude", "label", "timezone"]}
            state["searched_elsewhere"] = location.get("label")
            state["adaptation_log"].append(
                f"Your request names {named_city}, so the search moved there (drive times skipped).")
            weather_future, events_future = start_weather_and_events()

    # ======================================================================
    # STEP 2. Discovery -> the list of Google Maps searches to run.
    # ======================================================================
    notify("discovery", "Turning your request into map searches")
    disc = timed(timings, "discovery", discovery_agent, resources, user_prompt,
                 group_size=preferences.get("group_size"),
                 budget_level=preferences.get("budget_level") or memory.get("budget_preference"),
                 preferred_types=preferences.get("preferred_types"), top_n=settings["top_n_activities"])
    disc_df = disc["data"]["results"] if record(state, "discovery", disc) else None
    if disc_df is not None:
        state["candidate_activities"] = disc_df.to_dict("records")
    phrases, dropped = build_search_phrases(preferences, disc_df, memory, settings["catalog_ideas"],
                                            settings["max_searches"])
    if dropped:
        state["adaptation_log"].append(f"Skipped searches for things you said you're not into: {', '.join(dropped)}.")
    if not phrases:
        state["errors"].append("No searches could be built from the request.")
        return finish("Stopped: nothing to search for.")
    state["search_phrases"] = phrases

    # ======================================================================
    # STEP 3. Weather (started in step 1; scoring needs it).
    # ======================================================================
    weather = weather_future.result()
    state["usage"]["api_calls"]["open_meteo"] += 1
    suitability = None
    if record(state, "weather", weather):
        state["weather"] = weather["data"]
        suitability = weather["data"]["weather_suitability"]

    # ======================================================================
    # STEP 4. Search Google Maps around the visitor.
    # ======================================================================
    radius_miles = search_radius_km(max_travel_minutes) * 0.621371
    notify("local_places", f"Searching Google Maps within about {radius_miles:.0f} miles: " + ", ".join(phrases))
    lp = timed(timings, "local_places", area_search_agent, phrases, location, max_travel_minutes,
               settings["search_pages"])
    if not record(state, "local_places", lp):
        state["errors"].append(lp.get("error") or "Place search failed.")
        return finish("Stopped: the place search failed.")
    state["usage"]["api_calls"]["google_places"] += lp["data"]["api_calls"]
    state["warnings"].extend(lp.get("warnings", []))
    if lp["data"].get("failed_searches"):
        state["adaptation_log"].append(
            f"Google turned down {len(lp['data']['failed_searches'])} of {len(phrases)} searches (too many requests); "
            "results come from the rest.")
    places_df = lp["data"]["results"]
    for key, text in [("places_too_far_removed", "outside your drive range"),
                      ("places_closed_removed", "marked permanently or temporarily closed")]:
        if lp["data"].get(key):
            state["adaptation_log"].append(f"Dropped {lp['data'][key]} place(s) {text}.")
    places_df, removed = filter_rejected_places(places_df, memory)
    state["confidence"]["memory_rejected_places_removed"] = removed
    places_df, duplicates = remove_duplicate_listings(places_df)
    state["confidence"]["duplicate_listings_removed"] = duplicates
    if duplicates:
        state["adaptation_log"].append(f"Merged {duplicates} duplicate Google listing(s) of the same venue.")
    if places_df.empty:
        return finish("Completed without recommendations: Google Maps found nothing nearby for these searches.")

    # ======================================================================
    # STEP 5. Score every place found.
    # ======================================================================
    notify("matching", f"Scoring all {len(places_df)} places found")
    scored = timed(timings, "scoring", score_places, resources.embedder, places_df, user_prompt,
                   max_travel_minutes, target_datetime, suitability)
    state["confidence"]["places_found"] = len(scored)

    # ======================================================================
    # STEP 6. Shortlist, and start the slow downloads for it in the
    #         background (reviews, websites, drive times) while step 7 runs.
    # ======================================================================
    travel_n = settings["shortlist_for_travel"]
    specific = bool(preferences.get("venue_search_terms"))

    def pick_shortlist(df):
        return pd.DataFrame(diverse_selection(df, settings["shortlist_for_relevance"], specific, max_same_type=4,
                                              request_text=user_prompt, max_bars=4))

    head = pick_shortlist(scored)
    # Reviews cost more on Google's price list, so only the likely top 8.
    detail_futures = {pid: pool.submit(safe_place_details, pid) for pid in head["place_id"].head(travel_n)}
    prefetch_websites(head, pool)

    def start_travel_times(df):
        if skip_travel or df.empty:
            return None
        return pool.submit(timed, timings, "mobility", mobility_agent, df, location["origin"],
                           max_travel_minutes, len(df), target_datetime, location.get("timezone"))

    travel_futures = [start_travel_times(head)]

    # ======================================================================
    # STEP 7. Venue check (LLM). If too few places pass (or they're closed
    #         at the planned time), check the next batch once.
    # ======================================================================
    notify("place_relevance", "Checking the top places actually fit what you asked for")
    checked, kept_parts, flagged = set(), [], set()
    batch, attempt = head, 0
    while not batch.empty:
        step_name = "place_relevance" if attempt == 0 else f"place_relevance_retry{attempt}"
        rel = timed(timings, step_name, place_activity_relevance_agent, batch,
                    settings["place_relevance_threshold"], user_prompt)
        state["agent_results"][step_name] = rel
        checked |= set(batch["place_id"])
        if rel["status"] == "success":
            add_usage(state, rel)
            add_once(state["agents_called"], "place_relevance")
            kept = rel["data"]["results"]
            flagged |= set(batch["place_id"]) - set(kept["place_id"])
            kept_parts.append(kept)
        else:
            kept_parts.append(batch)  # the check failed: keep the batch unvalidated
        total_usable = sum(len(open_or_unknown(k)) for k in kept_parts)
        if total_usable >= settings["min_usable_places"] or attempt >= settings["relevance_retries"]:
            break
        attempt += 1
        batch = pick_shortlist(scored[~scored["place_id"].isin(checked)])
        if not batch.empty:
            notify("place_relevance", "Too few places fit; checking the next batch")
            state["adaptation_log"].append(
                f"Only {total_usable} place(s) fit and were open, so {len(batch)} more were checked.")
            travel_futures.append(start_travel_times(batch))

    non_empty = [k for k in kept_parts if not k.empty]
    shortlist = pd.concat(non_empty, ignore_index=True) if non_empty else head.iloc[0:0]
    if shortlist.empty:
        shortlist = head  # nothing passed: fall back to the unvalidated shortlist
        state["adaptation_log"].append("Nothing passed the relevance check, so the closest matches are shown.")
    shortlist = shortlist.sort_values("list_rank").reset_index(drop=True)
    if flagged:
        state["adaptation_log"].append(f"The relevance check set aside {len(flagged)} place(s) that didn't fit the request.")
    still_open = open_or_unknown(shortlist)
    if len(still_open) < len(shortlist) and not still_open.empty:
        state["adaptation_log"].append(
            f"Left {len(shortlist) - len(still_open)} place(s) out of the top picks because they're closed at your time.")
        shortlist = still_open

    # ======================================================================
    # STEP 8. Travel times (started in step 6/7) + the drive limit.
    # ======================================================================
    notify("mobility", "Getting live drive times")
    finalists = shortlist.head(travel_n)
    results = [f.result() for f in travel_futures if f is not None]
    if skip_travel:
        mob = {"agent": "mobility", "status": "skipped", "data": {}, "error": None, "usage": {},
               "warnings": ["Searching another city: drive times skipped."]}
    elif not results:
        mob = {"agent": "mobility", "status": "skipped", "data": {}, "error": None, "usage": {},
               "warnings": ["No places to time."]}
    else:
        ok = [r for r in results if r["status"] == "success"]
        mob = ok[0] if ok else results[0]
        if ok:  # one result per batch checked in step 7: combine them
            mob = dict(mob, data=dict(mob["data"], api_calls=sum(r["data"]["api_calls"] for r in ok),
                                      results=pd.concat([r["data"]["results"] for r in ok], ignore_index=True)))
    if mob["status"] == "success":
        state["usage"]["api_calls"]["google_routes"] += mob["data"]["api_calls"]
        mob_df = mob["data"]["results"]
        cols = [c for c in ["place_id", "origin", "distance_miles", "traffic_minutes", "static_minutes",
                            "traffic_delay_minutes", "mobility_score"] if c in mob_df.columns]
        if not mob_df.empty:
            joined = enforce_drive_limit(shortlist.merge(mob_df[cols], on="place_id", how="inner"),
                                         max_travel_minutes, state)
            mob["data"]["results"] = joined
            if not joined.empty:
                finalists = joined.head(travel_n)
                state["mobility"] = finalists.to_dict("records")
    record(state, "mobility", mob)

    # ======================================================================
    # STEP 9. Evidence for the finalists, both agents at the same time:
    #         9a reviews + websites (RAG)   9b photos (vision LLM)
    # ======================================================================
    notify("venue_rag", "Reading Google reviews and venue websites")
    details = {pid: (detail_futures[pid].result() if pid in detail_futures else safe_place_details(pid))
               for pid in finalists["place_id"]}
    finalists = finalists.copy()
    finalists["editorial_summary"] = finalists["place_id"].map(lambda p: details.get(p, {}).get("editorial_summary"))
    finalists["google_reviews"] = finalists["place_id"].map(lambda p: details.get(p, {}).get("reviews", []))
    state["usage"]["api_calls"]["google_places"] += len(details)

    photo_mode = str(settings.get("photo_vibe", "auto")).lower()
    want_photos = photo_mode == "always" or (photo_mode == "auto" and "photo_vibe" in planned)
    photo_future = None
    if want_photos:
        notify("photo_vibe", "Looking at photos of the top places")
        photo_future = pool.submit(timed, timings, "photo_vibe", photo_vibe_agent, finalists,
                                   {pid: d.get("photos") or [] for pid, d in details.items()}, user_prompt,
                                   settings["photo_max_places"], settings["photos_per_place"])

    rag = timed(timings, "venue_rag", venue_rag_agent, resources.embedder, finalists, user_prompt,
                settings["rag_max_places"], settings["rag_top_k_chunks"] + 4)
    review_assessments = None
    if record(state, "venue_rag", rag):
        review_assessments = rag["data"]["venue_assessments"]
        state["review_evidence"] = {"venue_assessments": review_assessments,
                                    "retrieved_evidence": rag["data"]["retrieved_evidence"].to_dict("records")}
        state["confidence"]["rag_coverage_rate"] = rag["data"]["coverage_rate"]

    photo_assessments = None
    if photo_future is not None:
        photo = photo_future.result()
        if record(state, "photo_vibe", photo):
            photo_assessments = photo["data"]["photo_assessments"]
            state["photo_evidence"] = photo_assessments
            state["usage"]["api_calls"]["google_place_photos"] += photo["data"]["photos_used"]
    else:
        add_once(state["agents_skipped"], "photo_vibe")

    # ======================================================================
    # STEP 10. Matching -> final ranking -> top picks.
    # ======================================================================
    notify("matching", "Ranking the final options")
    match = timed(timings, "matching", matching_agent, None, finalists.drop(columns=["google_reviews"]),
                  review_assessments, photo_assessments)
    if record(state, "matching", match):
        final_df = match["data"]["results"]
        state["ranked_results"] = records(final_df)
        # Top picks: best scores, but never two of the same business and,
        # for open-ended requests, at most two of the same kind of place.
        picks = diverse_selection(final_df, settings["final_recommendations"], specific, max_same_type=2,
                                  request_text=user_prompt, max_bars=1)
        # Weak match: even the best pick only loosely fits (skiing in Austin
        # -> a ski shop). Say so instead of presenting it as a strong answer.
        fits = [p.get("place_activity_relevance_score") for p in picks
                if isinstance(p.get("place_activity_relevance_score"), (int, float))]
        if picks and fits and max(fits) < settings["strong_match_score"]:
            state["weak_match"] = True
            state["adaptation_log"].append("No strong match was found nearby; these are the closest options.")
        state["final_recommendations"] = records(pd.DataFrame(picks).assign(final_rank=range(1, len(picks) + 1)))
    else:
        state["errors"].append(match.get("error") or "Matching failed.")

    # Everything found, for the app's map and full list.
    top_ids = {r["place_id"]: i for i, r in enumerate(state["final_recommendations"], start=1)}
    all_places = scored.drop(columns=["opening_periods"], errors="ignore").copy()
    all_places["top_pick"] = all_places["place_id"].map(top_ids)
    all_places["set_aside_by_relevance_check"] = all_places["place_id"].isin(flagged)
    if state["mobility"]:
        travel = pd.DataFrame(state["mobility"])[["place_id", "traffic_minutes"]]
        all_places = all_places.merge(travel, on="place_id", how="left")
        all_places["traffic_minutes"] = pd.to_numeric(all_places["traffic_minutes"], errors="coerce")
    state["all_places"] = records(all_places)

    # Events (started in step 1).
    ev = events_future.result()
    if ev["status"] != "skipped":
        state["usage"]["api_calls"]["ticketmaster"] += 1
    if record(state, "events", ev):
        state["events"] = ev["data"]["results"].to_dict("records")

    # ======================================================================
    # STEP 11. Write-up (the app calls synthesize_results() itself instead,
    #          after the picks are on screen).
    # ======================================================================
    if synthesize:
        state["runtime_seconds"] = round(time.time() - started, 2)
        synthesize_results(state, settings["final_recommendations"], notify)

    if state["final_recommendations"]:
        return finish("Completed: ranked every place found and explained the top picks.")
    return finish("Completed without recommendations: no place survived the checks.")


def synthesize_results(state, max_recommendations=None, notify=None):
    """
    Step 11 on its own, so the app can show the ranked picks first and fill
    in the explanations a few seconds later.
    """
    if not state.get("final_recommendations") or state.get("final_response"):
        return state
    (notify or (lambda *_: None))("synthesizer", "Writing up why each pick fits")
    add_once(state["agents_requested"], "synthesizer")
    syn = timed(state.setdefault("timings", {}), "synthesizer", recommendation_synthesizer,
                state, max_recommendations or DEFAULTS["final_recommendations"])
    if record(state, "synthesizer", syn):
        state["final_response"] = syn["data"]["synthesis"]
    else:
        state["adaptation_log"].append("The written explanation failed validation, so the ranked results are shown on their own.")
    state["runtime_seconds"] = round((state.get("runtime_seconds") or 0) + state["timings"]["synthesizer"], 2)
    return state
