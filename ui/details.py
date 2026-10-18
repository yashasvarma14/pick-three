"""
The detail tabs under the picks: every place found, a map, live events,
the agents' reasoning, and the score table.
"""

import pandas as pd
import streamlit as st

from .labels import AGENT_LABELS


def render_details(result):
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
