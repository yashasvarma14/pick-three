"""
Step 6 - Shortlist rules (plain Python)
=======================================
Turns a ranked list into a sensible shortlist:
  * one listing per business (Google sometimes lists a venue twice, and
    chains have several branches nearby)
  * no smoke shops, kratom bars, etc. as top picks
  * some variety for open-ended requests (not five bars in a row)
"""

import re
import unicodedata

import pandas as pd

from ..core.helpers import contains_keyword, distance_km



# ---------------------------------------------------------------------------
# One business, one spot
# ---------------------------------------------------------------------------


_FILLER = {"the", "a", "an", "and", "&", "of", "at", "on", "in", "co", "llc", "inc"}


def name_tokens(name):
    """'The Eureka Room - Downtown' -> ['eureka', 'room']"""
    text = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode().lower()
    text = re.split(r"\s[-|@]\s|\(|,|:", text)[0]
    return [t for t in re.findall(r"[a-z0-9']+", text) if t not in _FILLER]


_SHARED_HOSTS = {"facebook.com", "instagram.com", "google.com", "sites.google.com", "yelp.com", "linktr.ee",
                 "square.site", "squareup.com", "toasttab.com", "wixsite.com", "business.site", "godaddysites.com",
                 "weebly.com", "squarespace.com", "order.online", "doordash.com", "ubereats.com", "tripadvisor.com",
                 "eventbrite.com", "austintexas.gov", "tpwd.texas.gov", "nps.gov"}


def site_domain(url):
    """'https://www.mainevent.com/austin' -> 'mainevent.com' (None for shared hosts like facebook.com)."""
    if not isinstance(url, str) or "." not in url:
        return None
    host = re.sub(r"^https?://", "", url.strip().lower()).split("/")[0].split(":")[0]
    host = host[4:] if host.startswith("www.") else host
    if host in _SHARED_HOSTS or any(host.endswith("." + h) for h in _SHARED_HOSTS):
        return None
    return host or None


def same_business(name_a, name_b, website_a=None, website_b=None):
    """
    True for branches or duplicate listings of one business:
      * same website domain        Main Event Austin / Main Event Pflugerville (mainevent.com)
      * one name contains the other  Mohawk / Mohawk Austin, Eureka Room / The Eureka Room - Downtown
    Different businesses that only share words (Austin Pizza Garden /
    Austin Pizza Kitchen) are kept apart.
    """
    domain_a, domain_b = site_domain(website_a), site_domain(website_b)
    if domain_a and domain_a == domain_b:
        return True
    a, b = name_tokens(name_a), name_tokens(name_b)
    if not a or not b:
        return False
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return set(short) <= set(long_)


def remove_duplicate_listings(df, max_miles_apart=0.1):
    """
    Google sometimes lists one venue twice (different IDs, near-identical
    names, same spot). Keep the listing with more reviews.
    """
    if df.empty:
        return df, 0
    df = df.assign(_reviews=pd.to_numeric(df["review_count"], errors="coerce").fillna(0))
    df = df.sort_values("_reviews", ascending=False).reset_index(drop=True)
    keep, kept_rows = [], []
    for i, row in df.iterrows():
        dup = False
        for j in kept_rows:
            other = df.loc[j]
            same_address = row.get("address") and row.get("address") == other.get("address")
            close = None not in (row["latitude"], other["latitude"]) and distance_km(
                row["latitude"], row["longitude"], other["latitude"], other["longitude"]) <= max_miles_apart * 1.609
            if (close or same_address) and same_business(row["place_name"], other["place_name"],
                                                         row.get("website_uri"), other.get("website_uri")):
                # Credit the kept listing with the searches that found the duplicate.
                df.at[j, "matched_searches"] = list(dict.fromkeys(other["matched_searches"] + row["matched_searches"]))
                dup = True
                break
        if not dup:
            kept_rows.append(i)
        keep.append(not dup)
    out = df[keep].drop(columns="_reviews").reset_index(drop=True)
    return out, int(len(df) - len(out))


# ---------------------------------------------------------------------------
# Picking a varied shortlist
# ---------------------------------------------------------------------------


# Places most people wouldn't want suggested in a general recommender.
# They stay in the "All places" list; they just never become a top pick.
NOT_FOR_TOP_PICKS = ["kratom", "kava", "cbd", "hemp", "vape", "smoke shop", "tobacco", "head shop", "dispensary",
                     "liquor store", "adult entertainment", "adult store", "sex shop", "strip club", "gentlemen's club", "gentlemens club", "pawn"]


BAR_WORDS = ["bar", "pub", "brewery", "brewpub", "lounge", "night club", "nightclub", "tavern", "saloon",
             "taproom", "winery", "speakeasy"]


DRINK_WORDS = ["bar", "bars", "drink", "drinks", "cocktail", "cocktails", "beer", "beers", "wine", "brewery",
               "pub", "nightlife", "happy hour", "speakeasy", "margarita", "margaritas", "whiskey"]


def avoid_reason(row):
    """Why a place shouldn't be a top pick, or None."""
    text = f"{row.get('place_name') or ''} {row.get('primary_type') or ''} " + " ".join(
        t.replace("_", " ") for t in (row.get("place_types") or []) if isinstance(t, str))
    for word in NOT_FOR_TOP_PICKS:
        if contains_keyword(text, word):
            return word
    return None


def is_bar(row):
    kind = str(row.get("primary_type") or "")
    return any(contains_keyword(kind, w) for w in BAR_WORDS)


def asks_for_drinks(request):
    return any(contains_keyword(request or "", w) for w in DRINK_WORDS)


def diverse_selection(df, n, specific_request=False, max_same_type=2, min_reviews=10, request_text=None, max_bars=None):
    """
    Walk down a ranked list and pick n places:
      * never two branches/listings of the same business
      * never a place on the NOT_FOR_TOP_PICKS list (kratom bars, smoke shops...)
      * places with fewer than `min_reviews` reviews only if nothing better
      * for open-ended requests, at most `max_same_type` of one kind of place
        and at most `max_bars` bars/pubs/lounges, unless the request is about
        drinks; specific requests ("burgers") are exempt
    """
    rows = df.to_dict("records")
    limit_bars = max_bars is not None and not specific_request and not asks_for_drinks(request_text)
    for strict in (True, False):
        chosen, types, bars = [], {}, 0
        for row in rows:
            if avoid_reason(row):
                continue
            if any(same_business(row.get("place_name"), c.get("place_name"), row.get("website_uri"), c.get("website_uri"))
                   for c in chosen):
                continue
            reviews = row.get("review_count")
            if strict and (reviews is None or reviews != reviews or reviews < min_reviews):
                continue
            kind = str(row.get("primary_type") or "").lower()
            if strict and not specific_request and kind and types.get(kind, 0) >= max_same_type:
                continue
            bar = is_bar(row)
            if strict and limit_bars and bar and bars >= max_bars:
                continue
            chosen.append(row)
            types[kind] = types.get(kind, 0) + 1
            bars += bar
            if len(chosen) == n:
                return chosen
        if len(chosen) == n or not strict:
            return chosen
    return chosen
