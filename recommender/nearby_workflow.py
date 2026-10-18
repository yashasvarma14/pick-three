"""
"Near me" workflow used by the app.

Planner + Preference (together)
  -> search ideas: the Preference Agent's Google Maps searches, plus
     catalog outings from the Discovery Agent for open-ended requests
  -> Area search: every search, around the visitor     (+ Weather, Events)
  -> Score every place found (fit, quality, proximity, open, weather)
  -> Relevance check on the shortlist
  -> Drive times on the shortlist (+ Google reviews prefetched)
  -> Venue RAG over reviews and websites  (+ Photo Vibe on Google photos)
  -> Matching -> grounded write-up

Every place found stays in state["all_places"] so the app can show all of them.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .base import contains_keyword, create_initial_state
from .config import DEFAULTS
from .discovery import discovery_agent
from .events import events_agent
from .matching import matching_agent
from .memory import build_memory_context, filter_rejected_places
from .mobility import mobility_agent
from .places import distance_km
from .nearby import (area_search_agent, resolve_location, diverse_selection, remove_duplicate_listings, safe_place_details,
                     score_places, search_radius_km)
from .orchestrator import orchestrator_planner
from .photos import photo_vibe_agent
from .preference import preference_agent
from .rag import prefetch_websites, venue_rag_agent
from .relevance import place_activity_relevance_agent
from .weather import weather_agent
from .workflow import _add, _add_usage, _record, _timed, enforce_drive_limit, records, synthesize_results

def local_now(tz_name):
    try:
        return datetime.now(ZoneInfo(tz_name)).replace(tzinfo=None) if tz_name else datetime.now()
    except Exception:
        return datetime.now()


NEARBY_DEFAULTS = {
    "shortlist_for_relevance": 12,
    "shortlist_for_travel": 8,
    "catalog_ideas": 1,
    # Each search uses one of Google's daily "SearchTextRequest" quota
    # (three with "Search deeper"). 4 searches x 20 places is plenty.
    "max_searches": 4,
    "min_usable_places": 6,   # fewer than this after the relevance check -> check the next batch
    "relevance_retries": 1,
    "strong_match_score": 0.75,  # best pick's relevance below this -> "no strong match" note
    "search_pages": 1,
}


def _search_ideas(preferences, discovery_df, memory, n_catalog, max_searches=4):
    """
    Searches to run, best first:
      1. the specific thing asked for ("cheap burger restaurant")
      2. the AI's own map searches for this request
      3. a Kaggle catalog outing, ONLY when the AI gave fewer than 3 ideas
         (catalog ideas are generic and earlier displaced better searches,
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


def run_nearby_workflow(
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
    location         from nearby.resolve_location(): latitude, longitude,
                     label, timezone, origin
    target_datetime  naive local datetime, or None for right now
    """
    settings = {**DEFAULTS, **NEARBY_DEFAULTS, **overrides}
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
    pool = ThreadPoolExecutor(max_workers=8)

    def finish(reason):
        state["finish_reason"] = state["finish_reason"] or reason
        state["runtime_seconds"] = round(time.time() - started, 2)
        pool.shutdown(wait=False, cancel_futures=True)
        return state

    # ---------------- 1-2. PLAN + PREFERENCE (at the same time) ----------------
    notify("orchestrator", "Planning which agents this request needs and reading your request")
    when_text = target_datetime.isoformat(sep=" ", timespec="minutes") if target_datetime else None
    plan_future = pool.submit(
        _timed, timings, "orchestrator", orchestrator_planner, user_prompt, memory_context=memory_context,
        city=location.get("label"), origin=location.get("label"), target_datetime=when_text,
        use_memory=bool(memory), max_agent_calls=settings["max_agent_calls"])
    preference_query = user_prompt
    if memory_context:
        preference_query = (
            f"CURRENT USER REQUEST:\n{user_prompt}\n\nRELEVANT PERSISTENT USER MEMORY:\n{memory_context}\n\n"
            "The current request has the highest priority. Use memory only as extra personalization; "
            "if they conflict, follow the current request.")
    now_here = local_now(location.get("timezone"))
    pref_future = pool.submit(_timed, timings, "preference", preference_agent, preference_query,
                              resources.valid_activity_types, app_city=location.get("label"),
                              now=now_here, app_time=target_datetime)

    # Weather and Events only need the location and time.
    radius_miles = search_radius_km(max_travel_minutes) * 0.621371
    weather_future = pool.submit(_timed, timings, "weather", weather_agent, None, target_datetime, location)
    events_future = pool.submit(_timed, timings, "events", events_agent, None, target_datetime,
                                settings["event_max_results"], None, location, radius_miles)

    plan_response = plan_future.result()
    if not _record(state, "orchestrator", plan_response):
        state["errors"].append(plan_response.get("error") or "Orchestrator failed.")
        return finish("Stopped: the orchestrator could not create a plan.")
    plan = plan_response["data"]["plan"]
    state["orchestrator_plan"] = plan
    order = plan["execution_order"]
    state["agents_requested"] = ["orchestrator"] + order
    for agent in plan["agents_to_skip"]:
        _add(state["agents_skipped"], agent)
    if plan.get("human_checkpoint_needed") and plan.get("clarification_question"):
        # Searching anyway beats an empty page: the question is shown as a
        # note above the results instead of stopping the run.
        state["human_checkpoint_used"] = True
        state["clarification_question"] = plan.get("clarification_question")
        state["adaptation_log"].append("The planner had a question but searched anyway with your settings.")
    if memory:
        _add(state["agents_called"], "memory")

    pref = pref_future.result()
    if not _record(state, "preference", pref):
        state["errors"].append(pref.get("error") or "Preference Agent failed.")
        return finish("Stopped: the request could not be interpreted.")
    preferences = state["preferences"] = pref["data"]

    # A time written in the request ("this afternoon", "this weekend") wins
    # over the app's date/time setting, which visitors often leave alone.
    requested = preferences.get("requested_datetime")
    if requested and (target_datetime is None or abs((requested - target_datetime).total_seconds()) > 90 * 60):
        target_datetime = requested
        state["adaptation_log"].append(
            f"Your request mentions a time, so places were checked for {requested:%A %I:%M %p}.".replace(" 0", " "))
        weather_future = pool.submit(_timed, timings, "weather", weather_agent, None, target_datetime, location)
        events_future = pool.submit(_timed, timings, "events", events_agent, None, target_datetime,
                                    settings["event_max_results"], None, location, radius_miles)
    state["time_used"] = target_datetime.isoformat(sep=" ", timespec="minutes") if target_datetime else "right now"

    # "Best tacos in Paris, France": search where the request says, not
    # where the visitor is. Drive times from here would be meaningless.
    skip_travel = False
    named = preferences.get("city")
    if named:
        try:
            elsewhere = resolve_location(str(named))
            km = distance_km(location["latitude"], location["longitude"], elsewhere["latitude"], elsewhere["longitude"])
        except Exception:
            elsewhere, km = None, 0
        if elsewhere and km > 50:
            location = elsewhere
            skip_travel = True
            state["location"] = {k: location.get(k) for k in ["latitude", "longitude", "label", "timezone"]}
            state["searched_elsewhere"] = location.get("label")
            state["adaptation_log"].append(
                f"Your request names {named}, so the search moved there (drive times skipped).")
            weather_future = pool.submit(_timed, timings, "weather", weather_agent, None, target_datetime, location)
            events_future = pool.submit(_timed, timings, "events", events_agent, None, target_datetime,
                                        settings["event_max_results"], None, location, radius_miles)

    # ---------------- 3. SEARCH IDEAS ----------------
    notify("discovery", "Turning your request into map searches")
    disc = _timed(timings, "discovery", discovery_agent, resources, user_prompt,
                  group_size=preferences.get("group_size"),
                  budget_level=preferences.get("budget_level") or memory.get("budget_preference"),
                  preferred_types=preferences.get("preferred_types"), top_n=settings["top_n_activities"])
    disc_df = disc["data"]["results"] if _record(state, "discovery", disc) else None
    if disc_df is not None:
        state["candidate_activities"] = disc_df.to_dict("records")
    phrases, dropped = _search_ideas(preferences, disc_df, memory, settings["catalog_ideas"],
                                     settings["max_searches"])
    if dropped:
        state["adaptation_log"].append(f"Skipped searches for things you said you're not into: {', '.join(dropped)}.")
    if not phrases:
        state["errors"].append("No searches could be built from the request.")
        return finish("Stopped: nothing to search for.")
    state["search_phrases"] = phrases

    # ---------------- 4. WEATHER (needed for scoring) ----------------
    weather = weather_future.result()
    state["usage"]["api_calls"]["open_meteo"] += 1
    suitability = None
    if _record(state, "weather", weather):
        state["weather"] = weather["data"]
        suitability = weather["data"]["weather_suitability"]

    # ---------------- 5. SEARCH AROUND THE VISITOR ----------------
    notify("local_places", f"Searching Google Maps within about {radius_miles:.0f} miles: " + ", ".join(phrases))
    lp = _timed(timings, "local_places", area_search_agent, phrases, location, max_travel_minutes,
                settings["search_pages"])
    if not _record(state, "local_places", lp):
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

    # ---------------- 6. SCORE EVERY PLACE ----------------
    notify("matching", f"Scoring all {len(places_df)} places found")
    scored = _timed(timings, "scoring", score_places, resources.embedder, places_df, user_prompt,
                    max_travel_minutes, target_datetime, suitability)
    state["confidence"]["places_found"] = len(scored)

    # Head start: reviews, websites and drive times for the shortlist load
    # while the Relevance check runs.
    travel_n = settings["shortlist_for_travel"]
    specific = bool(preferences.get("venue_search_terms"))
    # Shortlist: one spot per business, enough reviews to trust the rating,
    # and some variety for open-ended requests.
    head = pd.DataFrame(diverse_selection(scored, settings["shortlist_for_relevance"], specific, max_same_type=4,
                                          request_text=user_prompt, max_bars=4))
    # Reviews cost more on Google's price list, so only the likely top 8.
    detail_futures = {pid: pool.submit(safe_place_details, pid) for pid in head["place_id"].head(travel_n)}
    prefetch_websites(head, pool)
    def start_mobility(df):
        if skip_travel or df.empty:
            return None
        return pool.submit(_timed, timings, "mobility", mobility_agent, df, location["origin"],
                           max_travel_minutes, len(df), target_datetime, location.get("timezone"))

    mobility_futures = [start_mobility(head)]

    # ---------------- 7. RELEVANCE CHECK ON THE SHORTLIST ----------------
    # If the check (plus "closed at your time") leaves too few usable places,
    # check the next batch once: an agentic adapt-and-retry step.
    def usable(df):
        if "open_at_planned_time" in df.columns:
            return df[df["open_at_planned_time"] != False]  # noqa: E712
        return df

    notify("place_relevance", "Checking the top places actually fit what you asked for")
    checked, kept_parts, flagged = set(), [], set()
    batch, attempt = head, 0
    while not batch.empty:
        rel = _timed(timings, "place_relevance" if attempt == 0 else f"place_relevance_retry{attempt}",
                     place_activity_relevance_agent, batch, settings["place_relevance_threshold"], user_prompt)
        state["agent_results"]["place_relevance" if attempt == 0 else f"place_relevance_retry{attempt}"] = rel
        checked |= set(batch["place_id"])
        if rel["status"] == "success":
            _add_usage(state, rel)
            _add(state["agents_called"], "place_relevance")
            kept = rel["data"]["results"]
            flagged |= set(batch["place_id"]) - set(kept["place_id"])
            kept_parts.append(kept)
        else:
            kept_parts.append(batch)  # check failed: keep the batch unvalidated
        total_usable = sum(len(usable(k)) for k in kept_parts)
        if total_usable >= settings["min_usable_places"] or attempt >= settings["relevance_retries"]:
            break
        attempt += 1
        rest = scored[~scored["place_id"].isin(checked)]
        batch = pd.DataFrame(diverse_selection(rest, settings["shortlist_for_relevance"], specific, max_same_type=4,
                                               request_text=user_prompt, max_bars=4))
        if not batch.empty:
            notify("place_relevance", "Too few places fit; checking the next batch")
            state["adaptation_log"].append(
                f"Only {total_usable} place(s) fit and were open, so {len(batch)} more were checked.")
            mobility_futures.append(start_mobility(batch))

    non_empty = [k for k in kept_parts if not k.empty]
    shortlist = pd.concat(non_empty, ignore_index=True) if non_empty else head.iloc[0:0]
    if shortlist.empty:
        shortlist = head  # nothing passed: fall back to the unvalidated shortlist
        state["adaptation_log"].append("Nothing passed the relevance check, so the closest matches are shown.")
    shortlist = shortlist.sort_values("list_rank").reset_index(drop=True)
    if flagged:
        state["adaptation_log"].append(f"The relevance check set aside {len(flagged)} place(s) that didn't fit the request.")
    open_or_unknown = usable(shortlist)
    if len(open_or_unknown) < len(shortlist) and not open_or_unknown.empty:
        state["adaptation_log"].append(
            f"Left {len(shortlist) - len(open_or_unknown)} place(s) out of the top picks because they're closed at your time.")
        shortlist = open_or_unknown

    # ---------------- 8. DRIVE TIMES ----------------
    notify("mobility", "Getting live drive times")
    matching_input = shortlist.head(travel_n)
    results = [f.result() for f in mobility_futures if f is not None]
    if skip_travel:
        mob = {"agent": "mobility", "status": "skipped", "data": {}, "warnings": ["Searching another city: drive times skipped."],
               "error": None, "usage": {}}
    elif not results:
        mob = {"agent": "mobility", "status": "skipped", "data": {}, "warnings": ["No places to time."], "error": None, "usage": {}}
    else:
        ok = [r for r in results if r["status"] == "success"]
        mob = ok[0] if ok else results[0]
        if ok:
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
                matching_input = joined.head(travel_n)
                state["mobility"] = matching_input.to_dict("records")
    _record(state, "mobility", mob)

    # ---------------- 9. EVIDENCE: REVIEWS + WEBSITES ----------------
    notify("venue_rag", "Reading Google reviews and venue websites")
    details = {pid: (detail_futures[pid].result() if pid in detail_futures else safe_place_details(pid))
               for pid in matching_input["place_id"]}
    matching_input = matching_input.copy()
    matching_input["editorial_summary"] = matching_input["place_id"].map(lambda p: details.get(p, {}).get("editorial_summary"))
    matching_input["google_reviews"] = matching_input["place_id"].map(lambda p: details.get(p, {}).get("reviews", []))
    state["usage"]["api_calls"]["google_places"] += len(details)
    # Photos and text evidence don't depend on each other: run them together.
    photo_mode = str(settings.get("photo_vibe", "auto")).lower()
    want_photos = photo_mode == "always" or (photo_mode == "auto" and "photo_vibe" in order)
    photo_future = None
    if want_photos:
        notify("photo_vibe", "Looking at photos of the top places")
        photo_future = pool.submit(_timed, timings, "photo_vibe", photo_vibe_agent, matching_input,
                                   {pid: d.get("photos") or [] for pid, d in details.items()}, user_prompt,
                                   settings["photo_max_places"], settings["photos_per_place"])
    rag = _timed(timings, "venue_rag", venue_rag_agent, resources.embedder, matching_input, user_prompt,
                 settings["rag_max_places"], settings["rag_top_k_chunks"] + 4)
    assessments = None
    if _record(state, "venue_rag", rag):
        assessments = rag["data"]["venue_assessments"]
        state["review_evidence"] = {"venue_assessments": assessments,
                                    "retrieved_evidence": rag["data"]["retrieved_evidence"].to_dict("records")}
        state["confidence"]["rag_coverage_rate"] = rag["data"]["coverage_rate"]

    photo_assessments = None
    if photo_future is not None:
        photo = photo_future.result()
        if _record(state, "photo_vibe", photo):
            photo_assessments = photo["data"]["photo_assessments"]
            state["photo_evidence"] = photo_assessments
            state["usage"]["api_calls"]["google_place_photos"] += photo["data"]["photos_used"]
    else:
        _add(state["agents_skipped"], "photo_vibe")

    # ---------------- 10. MATCHING ----------------
    notify("matching", "Ranking the final options")
    match = _timed(timings, "matching", matching_agent, None, matching_input.drop(columns=["google_reviews"]),
                   assessments, photo_assessments)
    if _record(state, "matching", match):
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

    # Everything found, for the map and the full list.
    top_ids = {r["place_id"]: i for i, r in enumerate(state["final_recommendations"], start=1)}
    all_places = scored.drop(columns=["opening_periods"], errors="ignore").copy()
    all_places["top_pick"] = all_places["place_id"].map(top_ids)
    all_places["set_aside_by_relevance_check"] = all_places["place_id"].isin(flagged)
    if state["mobility"]:
        travel = pd.DataFrame(state["mobility"])[["place_id", "traffic_minutes"]]
        all_places = all_places.merge(travel, on="place_id", how="left")
        all_places["traffic_minutes"] = pd.to_numeric(all_places["traffic_minutes"], errors="coerce")
    state["all_places"] = records(all_places)

    # ---------------- EVENTS (ran in the background) ----------------
    ev = events_future.result()
    if ev["status"] != "skipped":
        state["usage"]["api_calls"]["ticketmaster"] += 1
    if _record(state, "events", ev):
        state["events"] = ev["data"]["results"].to_dict("records")

    # ---------------- WRITE-UP ----------------
    if synthesize:
        state["runtime_seconds"] = round(time.time() - started, 2)
        synthesize_results(state, settings["final_recommendations"], notify)

    if state["final_recommendations"]:
        return finish("Completed: ranked every place found and explained the top picks.")
    return finish("Completed without recommendations: no place survived the checks.")
