"""
Adaptive agentic workflow (notebook section 27, with review fixes).

Planner -> selected specialists -> validation -> bounded adaptation
-> Matching -> grounded synthesis.

Speed: steps that don't depend on each other run at the same time.
  * Planner and Preference (two LLM calls) start together.
  * Weather and Events start as soon as the city is known, while
    Discovery runs.
  * Once venues are found, their websites start downloading and drive
    times start computing while the Relevance check runs.
The order of decisions is unchanged: each step still only uses results
it would have had in the one-at-a-time version.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .base import create_initial_state
from .config import DEFAULTS
from .discovery import discovery_agent
from .events import events_agent
from .matching import matching_agent
from .memory import apply_memory_activity_filter, build_memory_context, filter_rejected_places
from .mobility import mobility_agent
from .orchestrator import orchestrator_planner
from .places import local_place_agent
from .preference import preference_agent
from .rag import prefetch_websites, venue_rag_agent
from .relevance import place_activity_relevance_agent
from .synthesizer import recommendation_synthesizer
from .weather import apply_weather_reranking, geocode_city, weather_agent


def _add(lst, item):
    if item and item not in lst:
        lst.append(item)


def _add_usage(state, response):
    usage = (response or {}).get("usage") or {}
    for key in ["input_tokens", "output_tokens", "total_tokens"]:
        state["usage"][key] += usage.get(key, 0) or 0
    state["usage"]["llm_cost"] += usage.get("cost", 0.0) or 0.0


def _record(state, name, response):
    state["agent_results"][name] = response
    if response["status"] == "success":
        _add(state["agents_called"], name)
        _add_usage(state, response)
    elif response["status"] == "skipped":
        _add(state["agents_skipped"], name)
        state["warnings"].extend(response.get("warnings", []))
    else:
        if response.get("error"):
            state["warnings"].append(f"{name}: {response['error']}")
    return response["status"] == "success"


DIRECT_TYPE = "direct request"


def add_direct_candidates(candidates, terms, group_size=None):
    """Put the user's own specific searches at the top of the activity list."""
    from .data import classify_activity_environment

    top = float(candidates["discovery_score"].max()) if not candidates.empty else 1.0
    rows = []
    for i, term in enumerate(terms):
        rows.append({
            "activity_id": f"DIRECT_{i + 1}", "activity": term[:1].upper() + term[1:], "type": DIRECT_TYPE,
            "participants": group_size or 1, "price": 0.0, "discovery_text": term,
            "environment": classify_activity_environment(term), "place_query": term,
            "semantic_similarity": 1.0, "participant_fit": 1.0, "budget_fit": 1.0, "category_fit": 1.0,
            # Scored just above the best catalog match: it is what was asked for.
            "discovery_score": min(1.0, top + 0.05),
        })
    out = pd.concat([pd.DataFrame(rows), candidates], ignore_index=True)
    out["discovery_rank"] = range(1, len(out) + 1)
    out["activity_rank"] = out["discovery_rank"]
    return out


def enforce_drive_limit(mob_df, max_travel_minutes, state):
    """
    Treat "longest drive you'd accept" as a real limit (with a 5-minute
    grace for traffic estimates). If nothing is close enough, keep the
    three closest and say so, instead of returning nothing.
    """
    if not max_travel_minutes or "traffic_minutes" not in mob_df.columns or mob_df.empty:
        return mob_df
    limit = max_travel_minutes + 5
    minutes = pd.to_numeric(mob_df["traffic_minutes"], errors="coerce")
    within = mob_df[minutes.isna() | (minutes <= limit)]
    removed = len(mob_df) - len(within)
    state["confidence"]["over_drive_limit_removed"] = removed
    if within.empty:
        state["adaptation_log"].append(
            f"Nothing found was within a {max_travel_minutes}-minute drive, so the closest options are shown instead.")
        return mob_df.assign(_m=minutes).sort_values("_m").drop(columns="_m").head(3)
    if removed:
        state["adaptation_log"].append(f"Removed {removed} venue(s) more than {max_travel_minutes} minutes away.")
    return within


def records(df):
    """DataFrame -> list of dicts with missing values as None (never NaN)."""
    return [{k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
            for row in df.to_dict("records")]


def _timed(timings, name, func, *args, **kwargs):
    t0 = time.time()
    try:
        return func(*args, **kwargs)
    finally:
        timings[name] = round(time.time() - t0, 2)


def run_adaptive_workflow(
    resources,
    user_prompt,
    memory=None,
    city=None,
    origin=None,
    target_datetime=None,
    max_travel_minutes=20,
    use_memory=True,
    synthesize=True,
    progress=None,
    **overrides,
):
    """
    resources        data.Resources (embedder + activity catalog)
    memory           the visitor's memory dict (session-scoped in the app)
    target_datetime  naive datetime in the city's local time, or None for "now"
    progress         optional callback(step_key, message) for the UI
    """
    settings = {**DEFAULTS, **overrides}
    notify = progress or (lambda *_: None)
    started = time.time()

    state = create_initial_state(user_prompt=user_prompt, user_id=(memory or {}).get("user_id"), city=city)
    state["execution_mode"] = "adaptive_agentic"
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

    # ---------------- 1. PLAN  (+ 2. PREFERENCE, started at the same time) ----------------
    notify("orchestrator", "Planning which agents this request needs and reading your request")
    plan_future = pool.submit(
        _timed, timings, "orchestrator", orchestrator_planner,
        user_prompt, memory_context=memory_context, city=city, origin=origin,
        target_datetime=target_datetime.isoformat(sep=" ", timespec="minutes") if target_datetime else None,
        use_memory=bool(use_memory and memory), max_agent_calls=settings["max_agent_calls"],
    )
    preference_query = user_prompt
    if memory_context:
        preference_query = (
            f"CURRENT USER REQUEST:\n{user_prompt}\n\nRELEVANT PERSISTENT USER MEMORY:\n{memory_context}\n\n"
            "The current request has the highest priority. Use memory only as extra personalization; "
            "if they conflict, follow the current request."
        )
    pref_future = pool.submit(_timed, timings, "preference", preference_agent,
                              preference_query, resources.valid_activity_types, app_city=city)

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

    if plan.get("human_checkpoint_needed"):
        state["human_checkpoint_used"] = True
        state["clarification_question"] = plan.get("clarification_question")
        return finish("Paused: the orchestrator needs a clarification.")

    if "memory" in order:
        _add(state["agents_called"], "memory")

    # ---------------- 2. PREFERENCE ----------------
    pref = pref_future.result()
    if not _record(state, "preference", pref):
        state["errors"].append(pref.get("error") or "Preference Agent failed.")
        return finish("Stopped: the request could not be interpreted.")
    preferences = pref["data"]
    state["preferences"] = preferences
    resolved_city = city or preferences.get("city")
    state["city"] = resolved_city

    if preferences.get("needs_clarification") and not resolved_city:
        state["human_checkpoint_used"] = True
        state["clarification_question"] = preferences.get("clarification_question")
        return finish("Paused: the Preference Agent needs a clarification.")

    location = geocode_city(resolved_city) if resolved_city else None
    tz_name = (location or {}).get("timezone")

    # Weather and Events only need the city and time, so they start now
    # and run while Discovery works.
    weather_future = events_future = None
    if "weather" in order and resolved_city:
        weather_future = pool.submit(_timed, timings, "weather", weather_agent, resolved_city, target_datetime)
    if "events" in order and resolved_city:
        events_future = pool.submit(_timed, timings, "events", events_agent,
                                    resolved_city, target_datetime, settings["event_max_results"])

    # ---------------- 3. DISCOVERY ----------------
    notify("discovery", f"Searching {len(resources.activities)} activity ideas by meaning")
    # Embed the user's own words. The notebook embedded the memory
    # instructions too, which diluted the similarity scores.
    disc = _timed(
        timings, "discovery", discovery_agent,
        resources, user_prompt,
        group_size=preferences.get("group_size"),
        budget_level=preferences.get("budget_level") or memory.get("budget_preference"),
        preferred_types=preferences.get("preferred_types"),
        top_n=settings["top_n_activities"],
    )
    if not _record(state, "discovery", disc):
        state["errors"].append(disc.get("error") or "Discovery Agent failed.")
        return finish("Stopped: no activity candidates could be created.")
    candidates = disc["data"]["results"].copy()
    candidates["activity_rank"] = candidates["discovery_rank"]
    state["confidence"]["discovery_semantic_strength"] = disc.get("confidence")

    # ---------------- 4. MEMORY FILTER ----------------
    if memory:
        notify("memory", "Applying what you told us earlier")
        before = len(candidates)
        candidates = apply_memory_activity_filter(resources.embedder, candidates, memory, user_prompt)
        state["confidence"]["memory_activities_removed"] = before - len(candidates)

    # ---------------- 4b. DIRECT SEARCH ----------------
    # "I want cheap burgers" names something the 196-activity catalog
    # doesn't have; the closest catalog match was "Volunteer at your local
    # food bank". When the Preference Agent pulls out a specific search
    # ("cheap burger restaurant"), Places searches for exactly that.
    direct_terms = preferences.get("venue_search_terms") or []
    if direct_terms:
        candidates = add_direct_candidates(candidates, direct_terms, preferences.get("group_size"))
        state["adaptation_log"].append(
            "You asked for something specific, so venues were searched directly for: "
            + ", ".join(f"\"{t}\"" for t in direct_terms) + ".")

    # ---------------- 5. WEATHER ----------------
    if weather_future is not None:
        notify("weather", "Checking the forecast")
        weather = weather_future.result()
        state["usage"]["api_calls"]["open_meteo"] += 2
        if _record(state, "weather", weather):
            state["weather"] = weather["data"]
            candidates = apply_weather_reranking(candidates, weather["data"]["weather_suitability"])
            state["confidence"]["weather_suitability"] = weather["data"]["weather_suitability"]
        else:
            state["adaptation_log"].append("Weather was unavailable, so activities were not weather-adjusted.")

    state["candidate_activities"] = candidates.to_dict("records")

    # ---------------- 6. LOCAL PLACES (+ one bounded retry) ----------------
    places_df = pd.DataFrame()
    mobility_future = None
    if "local_places" in order and resolved_city:
        notify("local_places", "Finding real venues")
        if direct_terms:
            # Only the named thing is searched, with more venues per search.
            to_search = candidates[candidates["type"] == DIRECT_TYPE]
            lp = _timed(timings, "local_places", local_place_agent, to_search, resolved_city, len(to_search),
                        max(settings["places_per_activity"], 6 // len(to_search)))
        else:
            lp = _timed(timings, "local_places", local_place_agent, candidates, resolved_city,
                        settings["activities_to_search"], settings["places_per_activity"])
        if _record(state, "local_places", lp):
            places_df = lp["data"]["results"]
            if lp["data"].get("places_too_far_removed"):
                state["adaptation_log"].append(
                    f"Dropped {lp['data']['places_too_far_removed']} venue(s) Google returned from outside {resolved_city}.")
            state["usage"]["api_calls"]["google_places"] += lp["data"]["api_calls"]

        if places_df.empty and settings["max_adaptation_retries"] > 0 and lp["status"] != "skipped":
            state["adaptation_log"].append("Local Places found no usable venues; the search was broadened and retried once.")
            retry = local_place_agent(candidates, resolved_city,
                                      min(len(candidates), max(settings["activities_to_search"] + 2, 5)),
                                      max(settings["places_per_activity"], 3))
            state["agent_results"]["local_places_retry"] = retry
            if retry["status"] == "success":
                places_df = retry["data"]["results"]
                state["usage"]["api_calls"]["google_places"] += retry["data"]["api_calls"]

        places_df, removed = filter_rejected_places(places_df, memory)
        state["confidence"]["memory_rejected_places_removed"] = removed

        # Head start: download venue websites and compute drive times
        # while the Relevance check runs. Mobility gets a few extra venues
        # so it still has enough after Relevance removes any.
        prefetch_websites(places_df.head(settings["max_places_for_mobility"] * 2), pool)
        if "mobility" in order and origin and not places_df.empty:
            mobility_future = pool.submit(
                _timed, timings, "mobility", mobility_agent,
                places_df.head(settings["max_places_for_mobility"] * 2), origin, max_travel_minutes,
                settings["max_places_for_mobility"] * 2, target_datetime, tz_name)

        # ---------------- 7. PLACE-ACTIVITY VALIDATION ----------------
        if not places_df.empty:
            notify("place_relevance", "Checking each venue actually fits its activity")
            rel = _timed(timings, "place_relevance", place_activity_relevance_agent, places_df, settings["place_relevance_threshold"])
            state["agent_results"]["place_relevance"] = rel
            if rel["status"] == "success":
                _add_usage(state, rel)
                _add(state["agents_called"], "place_relevance")
                state["confidence"]["place_relevance_removed"] = rel["data"]["places_removed"]
                if not rel["data"]["results"].empty:
                    places_df = rel["data"]["results"]
                    if rel["data"]["places_removed"]:
                        state["adaptation_log"].append(
                            f"Relevance validation removed {rel['data']['places_removed']} implausible venue/activity pair(s).")
                else:
                    state["adaptation_log"].append("Every venue failed relevance validation; the unvalidated set was kept as a fallback.")

        state["candidate_places"] = places_df.to_dict("records")
        state["confidence"]["places_found"] = len(places_df)

    # ---------------- 8. EVENTS ----------------
    if events_future is not None:
        notify("events", "Looking up live events")
        ev = events_future.result()
        if ev["status"] != "skipped":
            state["usage"]["api_calls"]["ticketmaster"] += 1
        if _record(state, "events", ev):
            state["events"] = ev["data"]["results"].to_dict("records")
            state["confidence"]["events_found"] = ev["data"]["events_found"]

    # ---------------- 9. MOBILITY ----------------
    # Without travel times, keep a modest number of venues so RAG and
    # Matching stay fast; mobility_score is left missing, not invented.
    matching_input = places_df.head(settings["max_places_for_mobility"] * 2)
    if "mobility" in order and not places_df.empty:
        if origin:
            notify("mobility", "Estimating drive times")
            mob = mobility_future.result() if mobility_future is not None else mobility_agent(
                places_df, origin, max_travel_minutes, settings["max_places_for_mobility"], target_datetime, tz_name)
            if mob["status"] == "success":
                state["usage"]["api_calls"]["google_routes"] += mob["data"]["api_calls"]
                # Same venues the one-at-a-time version would have used: the
                # top surviving venues after Relevance, now with drive times.
                mob_df = mob["data"]["results"]
                travel_cols = [c for c in ["place_id", "origin", "distance_miles", "traffic_minutes", "static_minutes",
                                           "traffic_delay_minutes", "mobility_score"] if c in mob_df.columns]
                if not mob_df.empty and "place_id" in mob_df.columns:
                    mob_df = places_df.merge(mob_df[travel_cols], on="place_id", how="inner")
                    mob_df = enforce_drive_limit(mob_df, max_travel_minutes, state)
                    mob_df = (mob_df.head(settings["max_places_for_mobility"])
                              .sort_values(["mobility_score", "place_quality_score"], ascending=False)
                              .reset_index(drop=True))
                mob["data"]["results"] = mob_df
            if _record(state, "mobility", mob) and not mob["data"]["results"].empty:
                matching_input = mob["data"]["results"]
                state["mobility"] = matching_input.to_dict("records")
            else:
                state["adaptation_log"].append("Mobility was unavailable, so ranking continued without travel time.")
        else:
            _add(state["agents_skipped"], "mobility")
            state["adaptation_log"].append("No starting point was given, so Mobility was skipped.")

    # ---------------- 10. VENUE RAG ----------------
    assessments = None
    if "venue_rag" in order and matching_input is not None and not matching_input.empty:
        notify("venue_rag", "Reading venue websites for evidence")
        rag = _timed(timings, "venue_rag", venue_rag_agent, resources.embedder, matching_input, user_prompt,
                              settings["rag_max_places"], settings["rag_top_k_chunks"])
        if _record(state, "venue_rag", rag):
            assessments = rag["data"]["venue_assessments"]
            state["review_evidence"] = {
                "venue_assessments": assessments,
                "retrieved_evidence": rag["data"]["retrieved_evidence"].to_dict("records"),
            }
            state["confidence"]["rag_retrieval_similarity"] = rag["data"]["average_retrieval_similarity"]
            state["confidence"]["rag_coverage_rate"] = rag["data"]["coverage_rate"]
        else:
            state["adaptation_log"].append("Venue websites gave no usable evidence, so Matching used structured signals only.")

    # ---------------- 11. MATCHING ----------------
    if "matching" in order and matching_input is not None and not matching_input.empty:
        notify("matching", "Ranking the final options")
        match = _timed(timings, "matching", matching_agent, candidates, matching_input, assessments)
        if _record(state, "matching", match):
            final_df = match["data"]["results"]
            state["ranked_results"] = records(final_df)
            state["final_recommendations"] = records(final_df.head(settings["final_recommendations"]))
        else:
            state["errors"].append(match.get("error") or "Matching failed.")

    # ---------------- 12. SYNTHESIS ----------------
    if synthesize:
        state["runtime_seconds"] = round(time.time() - started, 2)
        synthesize_results(state, settings["final_recommendations"], notify)

    if state["final_recommendations"]:
        return finish("Completed: the selected agents produced ranked, explained recommendations.")
    return finish("Completed without recommendations: no venues survived the pipeline.")


def synthesize_results(state, max_recommendations=None, notify=None):
    """
    The grounded write-up, as its own step so the app can show the ranked
    picks first and fill in the explanations a few seconds later.
    """
    if not state.get("final_recommendations") or state.get("final_response"):
        return state
    (notify or (lambda *_: None))("synthesizer", "Writing up why each pick fits")
    _add(state["agents_requested"], "synthesizer")
    syn = _timed(state.setdefault("timings", {}), "synthesizer", recommendation_synthesizer,
                 state, max_recommendations or DEFAULTS["final_recommendations"])
    if _record(state, "synthesizer", syn):
        state["final_response"] = syn["data"]["synthesis"]
    else:
        state["adaptation_log"].append("The written explanation failed validation, so the ranked results are shown on their own.")
    state["runtime_seconds"] = round((state.get("runtime_seconds") or 0) + state["timings"]["synthesizer"], 2)
    return state
