"""
Venue Evidence RAG Agent.

retrieval (official websites) -> chunking -> embeddings -> semantic
search -> top-k evidence -> grounded LLM assessment.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .base import failed, make_agent_response, skipped, ttl_cache
from .llm import call_llm_json

USER_AGENT = "Mozilla/5.0 (compatible; MSBA-Academic-RAG-Project/1.0)"


@ttl_cache(seconds=6 * 60 * 60)
def fetch_venue_website_text(url, timeout=4, max_characters=30000):
    """
    Visible text from a venue's public website, or None.
    A 4-second limit: a site slower than that rarely has useful text,
    and it was the single slowest wait in a run. Cached for 6 hours.
    """
    if url is None or (isinstance(url, float) and np.isnan(url)) or not str(url).strip():
        return None
    try:
        response = requests.get(str(url).strip(), headers={"User-Agent": USER_AGENT}, timeout=timeout)
        response.raise_for_status()
        if "text/html" not in response.headers.get("Content-Type", "").lower():
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "header", "footer"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
        return text[:max_characters] if len(text) >= 100 else None
    except Exception:
        return None


def chunk_venue_text(text, chunk_size=700, overlap=120):
    if not text:
        return []
    text = str(text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) >= 80:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def prefetch_websites(places_df, pool):
    """Start downloading venue websites in the background (results land in the cache)."""
    if places_df is None or places_df.empty or "website_uri" not in places_df.columns:
        return []
    urls = [u for u in places_df["website_uri"].dropna().unique() if str(u).strip()]
    return [pool.submit(fetch_venue_website_text, u) for u in urls]


def build_venue_evidence_corpus(places_df, max_places=5):
    subset = places_df.head(max_places)
    rows = [row for _, row in subset.iterrows()]

    with ThreadPoolExecutor(max_workers=5) as pool:
        texts = list(pool.map(lambda r: fetch_venue_website_text(r.get("website_uri")), rows))

    evidence = []
    for row, text in zip(rows, texts):
        # Google reviews and the editorial summary ("near me" mode) are
        # short, so each one is a single chunk.
        extra = []
        if isinstance(row.get("editorial_summary"), str) and row["editorial_summary"].strip():
            extra.append(("Google editorial summary", row["editorial_summary"].strip()))
        for review in row.get("google_reviews") if isinstance(row.get("google_reviews"), list) else []:
            stars = f"{review.get('rating')}-star " if review.get("rating") else ""
            extra.append((f"Google review ({stars}{review.get('when') or 'recent'})", review["text"][:1200]))
        for n, (source, chunk) in enumerate(extra, start=1):
            evidence.append({
                "place_id": row.get("place_id"),
                "place_name": row.get("place_name"),
                "source_activity": row.get("source_activity"),
                "website_uri": source,
                "chunk_id": f"{row.get('place_id')}_REVIEW_{n}",
                "evidence_text": chunk,
            })
        for n, chunk in enumerate(chunk_venue_text(text), start=1):
            evidence.append({
                "place_id": row.get("place_id"),
                "place_name": row.get("place_name"),
                "source_activity": row.get("source_activity"),
                "website_uri": row.get("website_uri"),
                "chunk_id": f"{row.get('place_id')}_CHUNK_{n}",
                "evidence_text": chunk,
            })
    return pd.DataFrame(evidence)


def retrieve_venue_evidence(embedder, evidence_df, user_query, top_k=10, max_chunks_per_place=3):
    """
    Coverage-aware retrieval: the best chunk for every venue is guaranteed,
    then remaining slots go to the strongest extra chunks (max 3 per venue).
    """
    if evidence_df.empty:
        return evidence_df.copy()

    chunk_emb = np.asarray(embedder.encode(evidence_df["evidence_text"].astype(str).tolist()))
    query_emb = np.asarray(embedder.encode([user_query]))
    scored = evidence_df.copy()
    scored["rag_similarity"] = (query_emb @ chunk_emb.T)[0]
    scored = scored.sort_values("rag_similarity", ascending=False).reset_index(drop=True)

    best = scored.groupby("place_id", group_keys=False).head(1)
    remaining = scored[~scored["chunk_id"].isin(best["chunk_id"])]
    remaining = remaining.groupby("place_id", group_keys=False).head(max(0, max_chunks_per_place - 1))
    slots = max(0, max(top_k, len(best)) - len(best))
    extra = remaining.sort_values("rag_similarity", ascending=False).head(slots)

    return pd.concat([best, extra], ignore_index=True).sort_values("rag_similarity", ascending=False).reset_index(drop=True)


SYSTEM_PROMPT = """
You are the Venue Evidence RAG Agent for a local activity recommendation system.

Evidence comes from venue websites and, when available, Google reviews and
Google's editorial summary. Reviews are individual opinions: weigh a
pattern across several reviews more than a single one.

Reason ONLY from the retrieved evidence supplied. Do not invent venue
characteristics or use outside knowledge. If the evidence does not strongly
support a conclusion, lower evidence_fit_score and set evidence_sufficient
to false.

Return ONLY valid JSON:
{
  "venue_assessments": [
    {
      "place_id": "",
      "place_name": "",
      "evidence_fit_score": 0.0,
      "supported_strengths": [],
      "supported_concerns": [],
      "evidence_summary": "",
      "evidence_sufficient": true
    }
  ]
}

Rules:
1. evidence_fit_score is between 0 and 1 and measures how strongly the
   evidence supports the user's stated preferences.
2. Google star rating and review count are NOT textual evidence (review TEXT is).
3. Weak evidence means a lower score, not an omitted venue.
4. Return exactly one assessment for every venue in EXPECTED VENUES.
5. Copy each place_id EXACTLY as listed. Do not add venues.
6. Keep evidence_summary to one or two sentences.
"""


def venue_rag_agent(embedder, places_df, user_query, max_places=5, top_k_chunks=10):
    start = time.time()
    if places_df is None or places_df.empty:
        return skipped("venue_rag", "No venues available for RAG.")

    try:
        evidence_df = build_venue_evidence_corpus(places_df, max_places=max_places)
        if evidence_df.empty:
            return skipped("venue_rag", "No official website evidence could be retrieved.")

        retrieved = retrieve_venue_evidence(embedder, evidence_df, user_query, top_k=top_k_chunks)
        venues = retrieved[["place_id", "place_name"]].drop_duplicates("place_id")

        expected = "\n".join(f"- place_id: {r.place_id} | name: {r.place_name}" for r in venues.itertuples())
        blocks = "\n".join(
            f"\nEVIDENCE {i + 1}\nplace_id: {r.place_id}\nPlace Name: {r.place_name}\n"
            f"Source URL: {r.website_uri}\nRetrieval Similarity: {r.rag_similarity:.4f}\nEvidence:\n{r.evidence_text}\n"
            for i, r in enumerate(retrieved.itertuples())
        )
        message = f"USER REQUEST:\n{user_query}\n\nEXPECTED VENUES:\n{expected}\n\nRETRIEVED VENUE EVIDENCE:\n{blocks}"

        parsed, usage = call_llm_json(SYSTEM_PROMPT, message)
        valid_ids = set(venues["place_id"])
        assessments = [a for a in parsed.get("venue_assessments", []) if a.get("place_id") in valid_ids]
        assessed = {a["place_id"] for a in assessments}
        missing = venues[~venues["place_id"].isin(assessed)]["place_name"].tolist()

        avg_similarity = float(retrieved["rag_similarity"].mean())
        return make_agent_response(
            "venue_rag",
            data={
                "evidence_chunks_created": len(evidence_df),
                "evidence_chunks_retrieved": len(retrieved),
                "corpus_venues": evidence_df["place_id"].nunique(),
                "assessed_venues": len(assessed),
                "coverage_rate": round(len(assessed) / evidence_df["place_id"].nunique(), 4),
                "missing_assessment_venues": missing,
                "average_retrieval_similarity": avg_similarity,
                "retrieved_evidence": retrieved,
                "venue_assessments": assessments,
                "runtime_seconds": time.time() - start,
            },
            confidence=avg_similarity,
            sources=retrieved["website_uri"].dropna().unique().tolist(),
            warnings=[f"RAG assessment missing for: {', '.join(missing)}"] if missing else [],
            usage=usage,
        )
    except Exception as exc:
        return failed("venue_rag", exc)
