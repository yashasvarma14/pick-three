"""
The top of the results: notes from the run, the three ticket cards, and
the rating / feedback forms (feedback goes into memory for the next search).
"""

import html

import pandas as pd
import streamlit as st

from recommender import feedback_agent, update_memory_from_feedback
from recommender.services import tracker


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
        ('<p class="photo-credit">Photos: Google Maps'
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


def render_picks(result, ss):
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
