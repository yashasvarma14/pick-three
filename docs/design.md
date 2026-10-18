# Design direction

**Feel:** clean white, calm, a little editorial. The photos of real places
carry the color; the interface stays out of the way.

## Look

| Token | Value | Use |
|---|---|---|
| Background | `#FFFFFF` | page |
| Surface | `#F7F7F5` | cards at rest, inputs |
| Ink | `#111111` | text, primary button |
| Muted | `#6E6E73` | secondary text |
| Hairline | `#E8E8E6` | borders, dividers |
| Accent | `#FF5A36` (Austin-sunset orange) | rank numbers, live dots, focus ring; used sparingly |

- **Type:** Inter for everything; Instrument Serif italic for one or two words
  in a headline ("What should we do *tonight*?").
- **Shape:** 16 px card radius, 999 px pills, soft layered shadows that only
  appear on hover.
- **Layout:** mobile first; one column on phones, results + map side by side
  on wide screens.

## Motion rules (from the research)

1. Most transitions 150–250 ms, `cubic-bezier(.2,.8,.2,1)` (fast start, soft
   landing). Nothing over 400 ms except the page-level hero entrance.
2. Only `transform`, `opacity` and `filter`: never width/height/margins.
3. Hover/press use CSS transitions (interruptible), not keyframes.
4. No motion on things done over and over (typing, toggles), just instant
   feedback.
5. Every motion explains something: what loaded, what changed, what's
   connected (hovering a card lights up its pin on the map).
6. `prefers-reduced-motion`: movement is removed, fades stay.

## The small effects

- Hero headline words fade up from a slight blur, staggered.
- Search box placeholder cycles through example requests.
- Category pills have a sliding "active" background.
- Cards lift 2 px with a soft shadow on hover; the photo zooms 3 %.
- Skeleton shimmer while lists load, never a blank gap.
- Search progress is a live checklist of agents, each ticking off as it
  finishes, with an elapsed-time counter.
- Results slide in with a short stagger; hovering a result pulses its map pin.
- The header turns frosted (backdrop blur) once you scroll.
- Buttons scale to 0.97 while pressed.
