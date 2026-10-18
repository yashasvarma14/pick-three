// The small effects. Each one is opt-in and respects reduced motion (the CSS handles that).

import { $$ } from "./util.js";

// Hero headline: wrap every word so it can rise out of a blur, one after another.
export function splitWords(el) {
  let i = 0;
  const walk = (node) => {
    [...node.childNodes].forEach((child) => {
      if (child.nodeType === Node.TEXT_NODE) {
        const frag = document.createDocumentFragment();
        child.textContent.split(/(\s+)/).forEach((part) => {
          if (!part.trim()) { frag.append(part); return; }
          const span = document.createElement("span");
          span.className = "word";
          span.style.setProperty("--i", i++);
          span.textContent = part;
          frag.append(span);
        });
        child.replaceWith(frag);
      } else if (child.nodeType === Node.ELEMENT_NODE) {
        if (child.tagName === "EM") {           // keep the serif word as one unit
          child.classList.add("word");
          child.style.setProperty("--i", i++);
        } else walk(child);
      }
    });
  };
  walk(el);
  return i;
}

// Sections fade up the first time they scroll into view.
export function revealOnScroll() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("in");
        observer.unobserve(entry.target);
      }
    });
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
  $$(".reveal").forEach((el, n) => {
    el.style.setProperty("--d", `${(n % 4) * 60}ms`);
    observer.observe(el);
  });
}

// The header turns frosted once anything scrolls under it.
export function frostedHeader(header) {
  const update = () => header.classList.toggle("scrolled", window.scrollY > 8);
  update();
  window.addEventListener("scroll", update, { passive: true });
}

// Slide a highlight ("glider") under the selected button of a group.
export function moveGlider(container, selected, glider) {
  if (!selected || !glider) return;
  const box = container.getBoundingClientRect();
  const item = selected.getBoundingClientRect();
  glider.style.width = `${item.width}px`;
  glider.style.transform = `translateX(${item.left - box.left + container.scrollLeft}px)`;
}

// Cycle the composer placeholder through example requests.
export function cyclePlaceholder(el, lines, every = 3600) {
  let n = 0;
  el.textContent = lines[0];
  return setInterval(() => {
    el.classList.add("swap");
    setTimeout(() => {
      n = (n + 1) % lines.length;
      el.textContent = lines[n];
      el.classList.remove("swap");
    }, 380);
  }, every);
}

// Count a number up (used for stats in the agent trace).
export function countUp(el, to, decimals = 0, ms = 700) {
  const start = performance.now();
  const tick = (now) => {
    const t = Math.min(1, (now - start) / ms);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = (to * eased).toFixed(decimals);
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
