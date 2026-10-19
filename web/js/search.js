// Running a search and showing the results:
//   1. progress checklist while the agents work
//   2. the three picks, a map, every place found, events, the agents' trace
//   3. the write-up fills in a few seconds later
//   4. feedback -> memory for the next search

import { getWriteup, search, sendFeedback } from "./api.js";
import { memory } from "./memory.js";
import { countUp } from "./motion.js";
import { $, $$, STEP_LABELS, esc, icon, photoUrl, price, safeUrl, stars, toast, transition } from "./util.js";

const AGENT_NAMES = {
  orchestrator: "Planner", memory: "Memory", preference: "Preference", discovery: "Discovery", weather: "Weather",
  local_places: "Place search", place_relevance: "Venue check", events: "Events", mobility: "Travel time",
  venue_rag: "Reviews + websites", photo_vibe: "Photo vibe", matching: "Matching", synthesizer: "Write-up",
  scoring: "Scoring",
};
const EXPECTED_STEPS = 9;  // roughly how many progress messages a run sends (for the progress bar)

let current = null;   // the last result
let map = null;
const markers = new Map();  // place_id -> Leaflet marker

// ---------------------------------------------------------------------------
// 1. Progress
// ---------------------------------------------------------------------------

function startProgress() {
  const box = $("#progress");
  box.hidden = false;
  box.classList.remove("done");
  $("#steps").innerHTML = "";
  $("#progress-title").textContent = "Working on it";
  $("#progress-fill").style.transform = "scaleX(0.04)";
  const started = performance.now();
  const timer = setInterval(() => {
    $("#progress-time").textContent = `${((performance.now() - started) / 1000).toFixed(1)}s`;
  }, 100);
  let count = 0;
  let last = null;

  const finishLast = (seconds) => {
    if (!last) return;
    last.classList.remove("active");
    last.classList.add("done");
    last.querySelector(".step-icon").innerHTML = icon("check");
    if (seconds != null) last.querySelector(".step-time").textContent = `${seconds.toFixed(1)}s`;
  };

  return {
    step(event) {
      finishLast(event.seconds);
      const li = document.createElement("li");
      li.className = "step active";
      li.innerHTML = `<span class="step-icon"></span>
        <span><strong>${esc(STEP_LABELS[event.step] || event.step)}</strong><br>${esc(event.message)}</span>
        <span class="step-time"></span>`;
      $("#steps").append(li);
      last = li;
      count += 1;
      $("#progress-fill").style.transform = `scaleX(${Math.min(0.95, count / EXPECTED_STEPS)})`;
    },
    done(ok, seconds) {
      clearInterval(timer);
      finishLast(seconds);
      box.classList.add("done");
      $("#progress-fill").style.transform = "scaleX(1)";
      $("#progress-title").textContent = ok ? `Done in ${seconds}s` : "Finished without results";
    },
    fail() { clearInterval(timer); box.hidden = true; },
  };
}

// ---------------------------------------------------------------------------
// 2. Results
// ---------------------------------------------------------------------------

function factChips(p) {
  const out = [];
  if (p.rating != null) out.push(`<span class="fact star">${stars(p.rating, p.review_count)}</span>`);
  if (p.open_at_planned_time === true) out.push(`<span class="fact open">${icon("clock")}Open then</span>`);
  if (p.open_at_planned_time === false) out.push(`<span class="fact closed">${icon("clock")}Closed then</span>`);
  if (p.traffic_minutes != null) out.push(`<span class="fact">${icon("car")}${Math.round(p.traffic_minutes)} min drive</span>`);
  else if (p.straight_line_miles != null) out.push(`<span class="fact">${icon("pin")}${p.straight_line_miles.toFixed(1)} mi</span>`);
  if (price(p.price_level)) out.push(`<span class="fact">${price(p.price_level)}</span>`);
  return out.join("");
}

function scoreBars(p) {
  const rows = [["Fits the request", p.activity_context_score], ["Venue quality", p.place_quality_score],
    ["Travel", p.mobility_score], ["Reviews + website", p.evidence_fit_score], ["Photo vibe", p.photo_vibe_score],
    ["Final score", p.final_recommendation_score]].filter(([, v]) => v != null);
  if (!rows.length) return "";
  return `<details class="why-score"><summary>Why it ranked here ${icon("chev")}</summary><div class="bars">
    ${rows.map(([label, v]) => `<div class="bar"><span>${label}</span>
      <span class="bar-track"><span class="bar-fill" style="--v:${Math.max(0, Math.min(1, v)).toFixed(3)}"></span></span>
      <span>${Math.round(v * 100)}</span></div>`).join("")}</div></details>`;
}

function pickCard(p, rank, writeup) {
  const photos = (p.photo_names || []).slice(0, 2);  // each photo shown is a billed Google download
  const why = writeup?.why_it_fits || p.rag_evidence_summary;
  const caveat = writeup?.important_caveat;
  const type = [p.primary_type, p.source_activity ? `found by searching “${p.source_activity}”` : ""].filter(Boolean).join(" · ");
  return `
  <article class="pick" data-id="${esc(p.place_id)}" style="--i:${rank - 1}">
    <div class="pick-media">
      <span class="pick-rank">${rank}</span>
      ${photos.map((n, i) => `<img src="${photoUrl(n, 640)}" alt="Photo of ${esc(p.place_name)}" loading="lazy" class="${i === 0 ? "on" : ""}">`).join("")}
      ${photos.length > 1 ? `<div class="pick-dots">${photos.map((_, i) => `<button type="button" aria-label="Photo ${i + 1}" class="${i === 0 ? "on" : ""}" data-photo="${i}"></button>`).join("")}</div>` : ""}
    </div>
    <div class="pick-body">
      <div>
        <h3 class="pick-name">${esc(p.place_name)}</h3>
        <p class="pick-type">${esc(type)}</p>
      </div>
      <div class="facts">${factChips(p)}</div>
      ${why ? `<p class="pick-why">${esc(why)}</p>` : `<div class="pick-why pending skeleton"></div>`}
      ${p.photo_summary ? `<p class="pick-vibe"><strong>From the photos:</strong> ${esc(p.photo_summary)}</p>` : ""}
      ${p.vibe_tags?.length ? `<div class="tags">${p.vibe_tags.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</div>` : ""}
      ${caveat ? `<p class="caveat">Heads up: ${esc(caveat)}</p>` : ""}
      ${scoreBars(p)}
      <div class="pick-actions">
        ${safeUrl(p.google_maps_uri) ? `<a class="link-btn primary" href="${safeUrl(p.google_maps_uri)}" target="_blank" rel="noopener">${icon("pin")}Directions</a>` : ""}
        ${safeUrl(p.website_uri) ? `<a class="link-btn" href="${safeUrl(p.website_uri)}" target="_blank" rel="noopener">${icon("external")}Website</a>` : ""}
      </div>
    </div>
  </article>`;
}

function renderPicks(result) {
  const writeups = new Map((result.write_up?.recommendations || []).map((r) => [String(r.place_name), r]));
  const box = $("#picks");
  box.innerHTML = result.picks.map((p, i) => pickCard(p, i + 1, writeups.get(String(p.place_name)))).join("");
  box.classList.remove("stagger"); void box.offsetWidth; box.classList.add("stagger");

  // Photo dots switch the visible photo.
  $$(".pick", box).forEach((card) => {
    card.addEventListener("click", (e) => {
      const dot = e.target.closest("[data-photo]");
      if (!dot) return;
      const n = Number(dot.dataset.photo);
      $$(".pick-media img", card).forEach((img, i) => img.classList.toggle("on", i === n));
      $$(".pick-dots button", card).forEach((b, i) => b.classList.toggle("on", i === n));
    });
    linkHover(card, card.dataset.id);
  });
  // A photo that fails to load (expired link) just disappears.
  $$(".pick-media img", box).forEach((img) => img.addEventListener("error", () => img.remove()));
}

function renderHeader(result) {
  const placesChecked = result.places.length;
  $("#results-kicker").textContent =
    `${result.picks.length} picks · ${placesChecked} places checked · ${result.runtime_seconds}s`;
  const title = $("#results-title");
  if (result.write_up?.overall_summary) {
    title.classList.remove("pending");
    title.textContent = result.write_up.overall_summary;
  } else if (result.ok) {
    title.classList.add("pending");
    title.innerHTML = `<span class="shimmer-text">Here are your picks. Writing up why each one fits…</span>`;
  } else {
    title.classList.remove("pending");
    title.textContent = "Nothing made it through this time.";
  }

  const notes = [];
  if (result.notes.clarification_question) notes.push(`<span class="note">The planner wondered: ${esc(result.notes.clarification_question)}</span>`);
  if (result.notes.weak_match) notes.push(`<span class="note warn">No strong match nearby: these are the closest options.</span>`);
  result.notes.adaptations
    .filter((line) => /^Your request|^Nothing found|^Removed|^Skipped searches/.test(line))
    .forEach((line) => notes.push(`<span class="note">${esc(line)}</span>`));
  if (!result.ok) {
    const why = [...result.errors, ...result.warnings].join(" ");
    notes.push(`<span class="note warn">${/429|RESOURCE_EXHAUSTED/.test(why)
      ? "Google Maps is getting too many requests right now. Wait a minute and try again."
      : "Try a longer drive, “search deeper”, or describing it differently."}</span>`);
  }
  $("#notes").innerHTML = notes.join("");
}

function renderList(result) {
  const onlyOpen = $("#only-open").checked;
  const places = result.places
    .filter((p) => !onlyOpen || p.open_at_planned_time !== false)
    .sort((a, b) => (a.list_rank ?? 999) - (b.list_rank ?? 999));
  $("#all-count").textContent = `All ${result.places.length} places found`;
  $("#all-searches").textContent = result.search_phrases.length
    ? `Searched: ${result.search_phrases.map((s) => `“${s}”`).join(", ")}` : "";
  $("#place-list").innerHTML = places.map((p) => {
    const side = [p.traffic_minutes != null ? `${Math.round(p.traffic_minutes)} min` : p.straight_line_miles != null ? `${p.straight_line_miles.toFixed(1)} mi` : "",
      p.open_at_planned_time === false ? "closed then" : ""].filter(Boolean).join(" · ");
    return `<a class="place-row" data-id="${esc(p.place_id)}" href="${safeUrl(p.google_maps_uri)}" target="_blank" rel="noopener">
      <span class="num ${p.top_pick ? "pick-num" : ""}">${p.top_pick ? `#${p.top_pick}` : p.list_rank ?? ""}</span>
      <span><span class="name">${esc(p.place_name)}</span><br>
        <span class="meta">${[p.primary_type, p.rating != null ? `★ ${Number(p.rating).toFixed(1)} (${Number(p.review_count || 0).toLocaleString()})` : ""].filter(Boolean).map(esc).join(" · ")}</span></span>
      <span class="side">${esc(side)}</span></a>`;
  }).join("") || `<p class="empty">No places to show.</p>`;
  $$(".place-row").forEach((row) => linkHover(row, row.dataset.id));
}

function renderEvents(result) {
  const panel = $("#events-panel");
  panel.hidden = !result.events.length;
  $("#events-count").textContent = `${result.events.length} live events nearby`;
  $("#event-list").innerHTML = result.events.map((e, i) => `
    <a class="place-row" href="${safeUrl(e.event_url)}" target="_blank" rel="noopener">
      <span class="num">${i + 1}</span>
      <span><span class="name">${esc(e.event_name)}</span><br>
        <span class="meta">${[e.venue_name, e.genre].filter(Boolean).map(esc).join(" · ")}</span></span>
      <span class="side">${esc([e.local_date, e.local_time?.slice(0, 5)].filter(Boolean).join(" "))}</span></a>`).join("");
}

function renderTrace(result) {
  const t = result.trace;
  const names = (list) => list.map((a) => `<li>${esc(AGENT_NAMES[a] || a)}</li>`).join("") || "<li>Nothing</li>";
  const calls = result.usage.api_calls || {};
  const mapsCalls = (calls.google_places || 0) + (calls.google_routes || 0) + (calls.google_place_photos || 0);
  const timings = Object.entries(t.timings).sort((a, b) => b[1] - a[1]);
  const longest = Math.max(1, ...timings.map(([, s]) => s));
  $("#trace").innerHTML = `
    ${t.goal ? `<p><strong>Goal the planner set:</strong> ${esc(t.goal)}</p>` : ""}
    <div class="stats">
      <div class="stat"><b data-count="${result.runtime_seconds || 0}" data-dec="1">0</b><span>seconds</span></div>
      <div class="stat"><b data-count="${result.usage.llm_tokens || 0}">0</b><span>LLM tokens</span></div>
      <div class="stat"><b>$<span data-count="${result.usage.llm_cost || 0}" data-dec="4">0</span></b><span>est. LLM cost</span></div>
      <div class="stat"><b data-count="${mapsCalls}">0</b><span>Maps calls</span></div>
    </div>
    <div class="trace-grid">
      <div><h4>Ran</h4><ul>${names(t.ran)}</ul></div>
      <div><h4>Skipped</h4><ul>${names(t.skipped)}</ul></div>
    </div>
    ${t.reasoning.length ? `<h4 class="muted" style="margin:18px 0 6px;font-weight:500;font-size:13px">Why</h4>
      <ul>${t.reasoning.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : ""}
    ${timings.length ? `<h4 class="muted" style="margin:18px 0 6px;font-weight:500;font-size:13px">Seconds per step (many run at the same time)</h4>
      ${timings.map(([k, s]) => `<div class="timing"><span>${esc(AGENT_NAMES[k] || k)}</span>
        <span class="bar-track"><span class="bar-fill" style="--v:${(s / longest).toFixed(3)}"></span></span><span>${s.toFixed(1)}s</span></div>`).join("")}` : ""}`;
  const panel = $("#trace-panel");
  panel.addEventListener("toggle", () => {
    if (!panel.open) return;
    $$("[data-count]", panel).forEach((el) => countUp(el, Number(el.dataset.count), Number(el.dataset.dec || 0)));
  }, { once: true });
}

// ---------------------------------------------------------------------------
// Map (Leaflet + a light CARTO basemap)
// ---------------------------------------------------------------------------

function pinIcon(kind, label = "") {
  return L.divIcon({ className: "", html: `<div class="pin ${kind}">${label}</div>`,
    iconSize: kind === "pick" ? [28, 28] : kind === "you" ? [16, 16] : [12, 12] });
}

function renderMap(result) {
  if (!window.L) { $("#map").innerHTML = `<p class="empty" style="padding:20px">Map unavailable.</p>`; return; }
  if (!map) {
    map = L.map("map", { zoomControl: true, scrollWheelZoom: false, attributionControl: true });
    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      maxZoom: 19, subdomains: "abcd",
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
    }).addTo(map);
  }
  markers.forEach((m) => m.remove());
  markers.clear();
  map.eachLayer((layer) => { if (layer instanceof L.Marker) layer.remove(); });

  const bounds = [];
  result.places.filter((p) => !p.top_pick && p.latitude != null).forEach((p) => {
    const m = L.marker([p.latitude, p.longitude], { icon: pinIcon("other") }).addTo(map)
      .bindPopup(`<strong>${esc(p.place_name)}</strong><br>${esc(p.primary_type || "")}`);
    markers.set(p.place_id, m);
  });
  result.picks.forEach((p, i) => {
    if (p.latitude == null) return;
    const m = L.marker([p.latitude, p.longitude], { icon: pinIcon("pick", i + 1), zIndexOffset: 1000 }).addTo(map)
      .bindPopup(`<strong>${i + 1}. ${esc(p.place_name)}</strong><br>${esc(p.primary_type || "")}`);
    m.on("click", () => document.querySelector(`.pick[data-id="${CSS.escape(p.place_id)}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" }));
    markers.set(p.place_id, m);
    bounds.push([p.latitude, p.longitude]);
  });
  if (result.center) {
    L.marker([result.center.latitude, result.center.longitude], { icon: pinIcon("you"), interactive: false }).addTo(map);
    bounds.push([result.center.latitude, result.center.longitude]);
  }
  // The map's box only has its real size once the results are visible.
  requestAnimationFrame(() => {
    map.invalidateSize();
    if (bounds.length > 1) map.fitBounds(bounds, { padding: [40, 40], maxZoom: 15 });
    else if (bounds.length) map.setView(bounds[0], 13);
  });
}

// Hovering a card or list row lights up its pin (and the other way round).
function linkHover(el, placeId) {
  const toggle = (on) => {
    el.classList.toggle("linked", on);
    markers.get(placeId)?.getElement()?.querySelector(".pin")?.classList.toggle("linked", on);
  };
  el.addEventListener("mouseenter", () => toggle(true));
  el.addEventListener("mouseleave", () => toggle(false));
  el.addEventListener("focusin", () => toggle(true));
  el.addEventListener("focusout", () => toggle(false));
}

// ---------------------------------------------------------------------------
// 3. Write-up, 4. Feedback
// ---------------------------------------------------------------------------

async function loadWriteup(result) {
  try {
    const { write_up: writeUp } = await getWriteup(result.run_id);
    if (!writeUp || current !== result) return;
    result.write_up = writeUp;
    renderHeader(result);
    const byName = new Map((writeUp.recommendations || []).map((r) => [String(r.place_name), r]));
    $$(".pick").forEach((card, i) => {
      const w = byName.get(String(result.picks[i]?.place_name));
      const slot = $(".pick-why", card);
      if (!w || !slot) return;
      const p = document.createElement("p");
      p.className = "pick-why fade-in";
      p.textContent = w.why_it_fits || "";
      slot.replaceWith(p);
      if (w.important_caveat && !$(".caveat", card)) {
        const c = document.createElement("p");
        c.className = "caveat fade-in";
        c.textContent = `Heads up: ${w.important_caveat}`;
        p.after(c);
      }
    });
  } catch {
    $("#results-title").classList.remove("pending");
    $("#results-title").textContent = "Here are your picks.";
    $$(".pick-why.pending").forEach((el) => el.remove());
  }
}

export function setupResults() {
  $("#only-open").addEventListener("change", () => current && renderList(current));

  $$(".thumb").forEach((b) => b.addEventListener("click", () => {
    $$(".thumb").forEach((t) => t.classList.toggle("chosen", t === b));
    toast(b.dataset.vote === "up" ? "Glad it helped!" : "Thanks. Tell me what to change below.");
    if (b.dataset.vote === "down") $("#feedback-text").focus();
  }));

  $("#feedback-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("#feedback-text");
    const text = input.value.trim();
    if (text.length < 2 || !current) return;
    input.disabled = true;
    try {
      const res = await sendFeedback({ text, picks: current.picks.map((p, i) => ({ ...p, final_rank: i + 1 })), memory: memory.get() });
      memory.set(res.memory);
      input.value = "";
      toast(`Got it: ${res.summary || "saved for your next search."}`, 4000);
    } catch (err) {
      toast(err.message);
    } finally {
      input.disabled = false;
    }
  });
}

// ---------------------------------------------------------------------------
// The whole flow
// ---------------------------------------------------------------------------

export async function runSearch(body, { onDone }) {
  const progress = startProgress();
  $("#progress").scrollIntoView({ behavior: "smooth", block: "center" });
  try {
    let result = null;
    let failure = null;
    await search({ ...body, memory: memory.get(), use_memory: memory.useIt() }, (event) => {
      if (event.type === "progress") progress.step(event);
      if (event.type === "result") result = event.result;
      if (event.type === "error") failure = event.message;
    });
    if (failure || !result) throw new Error(failure || "The search stopped unexpectedly.");

    progress.done(result.ok, result.runtime_seconds);
    current = result;
    transition(() => {
      document.body.dataset.view = "results";
      $("#results").hidden = false;
      renderHeader(result);
      renderPicks(result);
      renderList(result);
      renderEvents(result);
      renderTrace(result);
    });
    setTimeout(() => {
      renderMap(result);
      $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
    }, 60);
    if (result.ok) loadWriteup(result);
  } catch (err) {
    progress.fail();
    throw err;
  } finally {
    onDone();
  }
}

export function goHome() {
  transition(() => {
    document.body.dataset.view = "home";
    $("#results").hidden = true;
    $("#progress").hidden = true;
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
  $("#prompt").focus({ preventScroll: true });
}
