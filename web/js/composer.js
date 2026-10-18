// The search box: the request text plus the where / when / drive chips.

import { locate } from "./api.js";
import { $, $$ } from "./util.js";
import { cyclePlaceholder, moveGlider } from "./motion.js";

export const EXAMPLES = [
  "Four of us want something fun and social tonight, not too expensive",
  "Something relaxing and quiet for two this evening",
  "Find a live music show tonight for a group of three",
  "Cozy, romantic spot for a date night, not too loud",
  "Indoor activity for a rainy afternoon with my roommates",
  "Best tacos near me, cheap and fast",
];

const state = {
  where: { text: "University of Texas at Austin" },
  whereLabel: "Near UT Austin",
  when: "tonight",
  pickedTime: null,
  drive: 20,
};

// ---------------- when ----------------
function tonightAt7() {
  const d = new Date();
  if (d.getHours() >= 19) d.setHours(d.getHours() + 1, 0, 0, 0);
  else d.setHours(19, 0, 0, 0);
  return d;
}

function whenDate() {
  if (state.when === "now") return null;
  if (state.when === "tonight") return tonightAt7();
  if (state.when === "tomorrow") { const d = new Date(); d.setDate(d.getDate() + 1); d.setHours(19, 0, 0, 0); return d; }
  return state.pickedTime ? new Date(state.pickedTime) : tonightAt7();
}

const timeText = (d) => d.toLocaleTimeString([], { hour: "numeric", minute: d.getMinutes() ? "2-digit" : undefined });

function whenLabel() {
  const d = whenDate();
  if (!d) return "Right now";
  if (state.when === "tonight") return `Tonight, ${timeText(d)}`;
  if (state.when === "tomorrow") return `Tomorrow, ${timeText(d)}`;
  return d.toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" });
}

// Local wall-clock time without a timezone, which is what the backend expects.
function localIso(d) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:00`;
}

// ---------------- popovers ----------------
function setupPopover(chip, pop, done) {
  const close = () => { pop.hidden = true; chip.setAttribute("aria-expanded", "false"); };
  chip.addEventListener("click", () => {
    const opening = pop.hidden;
    $$(".popover").forEach((p) => { p.hidden = true; });
    $$(".chip").forEach((c) => c.setAttribute("aria-expanded", "false"));
    if (opening) {
      pop.hidden = false;
      chip.setAttribute("aria-expanded", "true");
      pop.querySelectorAll(".segmented").forEach(syncSegmented);
      pop.querySelector("input[type=text]")?.focus();
    }
  });
  done.addEventListener("click", close);
  document.addEventListener("click", (e) => { if (!pop.hidden && !pop.contains(e.target) && !chip.contains(e.target)) close(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !pop.hidden) { close(); chip.focus(); } });
  return close;
}

function syncSegmented(seg) {
  moveGlider(seg, seg.querySelector('[aria-checked="true"]'), seg.querySelector(".seg-glider"));
}

function setupSegmented(seg, onPick) {
  seg.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b === btn)));
    syncSegmented(seg);
    onPick(btn);
  });
}

// ---------------- setup ----------------
export function setupComposer({ onSubmit }) {
  const form = $("#composer");
  const prompt = $("#prompt");
  const go = $("#go");

  // Grow the textarea with its text; Enter submits, Shift+Enter adds a line.
  const resize = () => {
    prompt.style.height = "auto";
    prompt.style.height = `${prompt.scrollHeight}px`;
    form.classList.toggle("has-text", prompt.value.length > 0);
    go.disabled = prompt.value.trim().length < 3;
  };
  prompt.addEventListener("input", resize);
  prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!go.disabled) form.requestSubmit(); }
  });
  cyclePlaceholder($("#placeholder"), EXAMPLES);

  // Example chips fill the box.
  const examples = $("#examples");
  EXAMPLES.slice(0, 4).forEach((text, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "example fade-in";
    b.style.setProperty("--d", `${700 + i * 60}ms`);
    b.textContent = text;
    b.addEventListener("click", () => { prompt.value = text; resize(); prompt.focus(); });
    examples.append(b);
  });

  // Where
  const whereInput = $("#where-input");
  const hint = $("#where-hint");
  const setWhere = (where, label) => {
    state.where = where;
    state.whereLabel = label;
    $("#where-label").textContent = label;
  };
  const closeWhere = setupPopover($("#where-chip"), $("#where-pop"), $("#where-done"));
  $("#where-done").addEventListener("click", async () => {
    const text = whereInput.value.trim();
    if (!text || state.where.latitude) return;
    setWhere({ text }, `Near ${text.length > 22 ? text.slice(0, 22) + "…" : text}`);
    try { await locate({ text }); hint.textContent = "Searches and drive times start here."; }
    catch (err) { hint.textContent = err.message; }
  });
  whereInput.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $("#where-done").click(); } });
  whereInput.addEventListener("input", () => { state.where = { text: whereInput.value.trim() }; });
  $("#use-gps").addEventListener("click", () => {
    if (!navigator.geolocation) { hint.textContent = "This browser can't share a location."; return; }
    hint.textContent = "Asking your browser…";
    navigator.geolocation.getCurrentPosition(
      (pos) => { setWhere({ latitude: pos.coords.latitude, longitude: pos.coords.longitude }, "Near you"); hint.textContent = "Using your current location."; closeWhere(); },
      () => { hint.textContent = "Location wasn't shared. Type an address instead."; },
      { enableHighAccuracy: true, timeout: 10000 },
    );
  });

  // When
  const whenInput = $("#when-input");
  setupPopover($("#when-chip"), $("#when-pop"), $("#when-done"));
  setupSegmented($("#when-seg"), (btn) => {
    state.when = btn.dataset.when;
    whenInput.hidden = state.when !== "pick";
    if (state.when === "pick") {
      if (!whenInput.value) whenInput.value = localIso(tonightAt7()).slice(0, 16);
      state.pickedTime = whenInput.value;
      whenInput.focus();
    }
    $("#when-label").textContent = whenLabel();
  });
  whenInput.addEventListener("change", () => { state.pickedTime = whenInput.value; $("#when-label").textContent = whenLabel(); });
  $("#when-label").textContent = whenLabel();

  // Drive
  setupPopover($("#drive-chip"), $("#drive-pop"), $("#drive-done"));
  setupSegmented($("#drive-seg"), (btn) => {
    state.drive = Number(btn.dataset.drive);
    $("#drive-label").textContent = `${state.drive} min drive`;
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    $$(".popover").forEach((p) => { p.hidden = true; });
    const when = whenDate();
    onSubmit({
      prompt: prompt.value.trim(),
      where: state.where,
      when: when ? localIso(when) : null,
      max_travel_minutes: state.drive,
      deep: $("#deep").checked,
    });
  });

  resize();
  return { setBusy: (busy) => { go.disabled = busy || prompt.value.trim().length < 3; go.querySelector("span").textContent = busy ? "Searching…" : "Find options"; } };
}
