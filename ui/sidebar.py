"""
The sidebar: where you are, when, how far you'll drive, and memory.
"""

from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st

from recommender import create_empty_user_memory, resolve_location
from recommender.core import config
from recommender.services import tracker

try:
    from streamlit_js_eval import get_geolocation
except ImportError:  # the app still works with a typed address
    get_geolocation = None


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


def default_time(now):
    """7 PM, or the next whole hour if it's already evening."""
    if now.hour < 18 or now.hour >= 23:
        return dtime(19, 0)
    return dtime(now.hour + 1, 0)


def current_location(ss):
    """The visitor's location from GPS or the typed text, or None."""
    try:
        if ss.coords:
            return resolve_location(latitude=ss.coords[0], longitude=ss.coords[1])
        if ss.place_text.strip():
            return resolve_location(ss.place_text.strip())
    except Exception as exc:
        ss.location_error = str(exc)
    return None


def location_picker(ss):
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

    location = current_location(ss)
    if ss.location_error and not location:
        st.error(ss.location_error)
    ss.location_error = None
    return location


def time_picker(location):
    """Returns the planned datetime, or None for 'right now'."""
    if st.toggle("Right now", value=False):
        return None
    col_d, col_t = st.columns(2)
    now_here = local_now((location or {}).get("timezone"))
    today_here = now_here.date()
    plan_date = col_d.date_input("Date", value=today_here, min_value=today_here,
                                 max_value=today_here + timedelta(days=6))
    times = [dtime(h, m) for h in range(6, 24) for m in (0, 30)]
    start_time = default_time(now_here)
    plan_time = col_t.selectbox("Time", times, index=times.index(start_time) if start_time in times else 26,
                                format_func=lambda t: t.strftime("%I:%M %p").lstrip("0"))
    return datetime.combine(plan_date, plan_time)


def memory_panel(ss):
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


def render_sidebar(ss, searches_left):
    """Draws the sidebar and returns the search settings."""
    with st.sidebar:
        st.header("Your plan")
        location = location_picker(ss)
        target_dt = time_picker(location)
        remembered_travel = ss.memory.get("max_travel_minutes")
        max_travel = st.slider("Longest drive you'd accept (minutes)", 5, 60,
                               value=int(remembered_travel or 20), step=5)
        deep = st.toggle("Search deeper", value=False,
                         help="Up to 60 places per search instead of 20 (Google's maximum). Slower.")
        use_memory = st.toggle("Use what I've told you before", value=True)
        memory_panel(ss)
        st.caption(
            f"Searches left this session: {searches_left}. Memory lives only in this browser tab."
            + (" Searches and ratings are logged anonymously (no exact location) to evaluate the app."
               if tracker.enabled() else "")
        )
    return {"location": location, "target_datetime": target_dt, "max_travel_minutes": max_travel,
            "deep": deep, "use_memory": use_memory}
