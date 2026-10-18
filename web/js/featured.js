// "Austin's top 10": category tabs and a swipeable rail of cards.

import { getFeatured } from "./api.js";
import { $, esc, icon, photoUrl, price, safeUrl, stars } from "./util.js";
import { moveGlider } from "./motion.js";

const cache = new Map();  // category -> items (so switching tabs back is instant)

function skeletons(n = 5) {
  return Array.from({ length: n }, () => `
    <div class="card" aria-hidden="true">
      <div class="card-media skeleton"></div>
      <div class="skeleton line" style="width:70%"></div>
      <div class="skeleton line" style="width:45%"></div>
    </div>`).join("");
}

function placeCard(item, i) {
  const photo = item.photo_names?.[0];
  const meta = [stars(item.rating, item.review_count), price(item.price_level)].filter(Boolean).join(" · ");
  return `
    <a class="card" style="--i:${i}" href="${safeUrl(item.google_maps_uri)}" target="_blank" rel="noopener">
      <div class="card-media">
        ${photo ? `<img src="${photoUrl(photo, 520)}" alt="" loading="lazy" onload="this.classList.add('loaded')">` : ""}
        <span class="card-rank">${item.rank}</span>
      </div>
      <div>
        <p class="card-name">${esc(item.place_name)}</p>
        <p class="card-meta">${meta}${item.primary_type ? ` · ${esc(item.primary_type)}` : ""}</p>
      </div>
    </a>`;
}

function eventCard(item, i) {
  const when = [item.local_date, item.local_time?.slice(0, 5)].filter(Boolean).join(" · ");
  return `
    <a class="card event" style="--i:${i}" href="${safeUrl(item.event_url)}" target="_blank" rel="noopener">
      <div class="card-media"><span class="card-rank">${item.rank}</span></div>
      <div>
        <p class="card-name">${esc(item.place_name)}</p>
        <p class="card-meta">${esc(item.address || "")}${when ? ` · ${esc(when)}` : ""}</p>
      </div>
    </a>`;
}

async function show(category) {
  const rail = $("#rail");
  rail.scrollTo({ left: 0 });
  if (!cache.has(category)) rail.innerHTML = skeletons();
  try {
    const items = cache.get(category) || (await getFeatured(category)).items;
    cache.set(category, items);
    if (!items.length) { rail.innerHTML = `<p class="empty">Nothing to show here right now.</p>`; return; }
    rail.innerHTML = items.map(category === "events" ? eventCard : placeCard).join("");
    rail.classList.remove("stagger"); void rail.offsetWidth; rail.classList.add("stagger");
  } catch (err) {
    rail.innerHTML = `<p class="empty">${esc(err.message)}</p>`;
  }
  updateArrows();
}

function updateArrows() {
  const rail = $("#rail");
  $(".rail-btn.prev").classList.toggle("show", rail.scrollLeft > 8);
  $(".rail-btn.next").classList.toggle("show", rail.scrollLeft + rail.clientWidth < rail.scrollWidth - 8);
}

export function setupFeatured(categories) {
  const tabs = $("#tabs");
  const glider = tabs.querySelector(".tab-glider");
  categories.forEach((c, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tab";
    b.role = "tab";
    b.dataset.category = c.id;
    b.setAttribute("aria-selected", String(i === 0));
    b.textContent = c.label;
    tabs.append(b);
  });
  const select = (btn) => {
    tabs.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t === btn)));
    moveGlider(tabs, btn, glider);
    show(btn.dataset.category);
  };
  tabs.addEventListener("click", (e) => { const b = e.target.closest(".tab"); if (b) select(b); });
  window.addEventListener("resize", () => moveGlider(tabs, tabs.querySelector('[aria-selected="true"]'), glider));

  const rail = $("#rail");
  rail.addEventListener("scroll", updateArrows, { passive: true });
  const page = (dir) => rail.scrollBy({ left: dir * rail.clientWidth * 0.85, behavior: "smooth" });
  $(".rail-btn.prev").addEventListener("click", () => page(-1));
  $(".rail-btn.next").addEventListener("click", () => page(1));

  const first = tabs.querySelector(".tab");
  if (first) requestAnimationFrame(() => select(first));
  else $("#featured").hidden = true;
}
