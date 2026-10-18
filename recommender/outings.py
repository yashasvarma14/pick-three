"""
Which catalog activities are real outings, and what to search Google for.

The Kaggle catalog mixes outings ("Go to an escape room") with things you
do at home ("Take a nap", "Organize your pantry", "Learn Kotlin"). The app
recommends real venues, so Discovery only ranks activities listed here.
Each one maps to a Google Maps search phrase; the city is added later.

Hand-checked against all 196 activities in data/activities_catalog.csv.
To add one: copy the activity text exactly (any capitalization) and give
the kind of place you'd type into Google Maps.
"""

OUTING_QUERIES = {
    # Social / nightlife
    "go to a karaoke bar with some friends": "karaoke bar",
    "explore the nightlife of your city": "popular cocktail bar",
    "go to an escape room": "escape room",
    "go bowling with some friends": "bowling alley",
    "hold a video game tournament with some friends": "video game lounge",
    "play a video game": "arcade",
    "go to an arcade with friends": "arcade",
    "learn and play a new card game": "board game cafe",
    "invite some friends over for a game night": "board game cafe",
    "have a game night with friends": "board game cafe",
    "play a game of monopoly": "board game cafe",
    # Music, shows, film
    "go to a concert with some friends": "live music venue",
    "go to a concert with local artists with some friends": "live music venue local bands",
    "go to a music festival with some friends": "live music venue",
    "go see a broadway production": "theater live show",
    "go see a movie in theaters with a few friends": "movie theater",
    "watch a classic movie": "classic movie theater",
    # Food and shopping
    "patronize a local independent restaurant": "local independent restaurant",
    "catch up with a friend over a lunch date": "cafe for lunch",
    "cook something together with someone": "cooking class",
    "shop at support your local farmers market": "farmers market",
    "go to a local thrift shop": "thrift store",
    "buy a new house decoration": "home decor store",
    # Calm, relaxing
    "visit a nearby museum": "museum",
    "visit a museum": "museum",
    "go to the library and find an interesting book": "public library",
    "go to a nail salon": "nail salon",
    "take a bubble bath": "day spa",
    "do yoga": "yoga studio",
    "meditate for five minutes": "meditation center",
    "go stargazing": "observatory stargazing",
    "watch the sunset or the sunrise": "sunset viewpoint",
    "paint the first thing you see": "paint and sip studio",
    "learn calligraphy": "calligraphy class",
    "learn woodworking": "woodworking class",
    "take a class at your local community center that interests you": "community center classes",
    # Outdoors and active
    "take a hike at a local park": "hiking trail",
    "explore a park you have never been to before": "park",
    "go for a walk": "walking trail",
    "go for a run": "running trail",
    "have a picnic with some friends": "park with picnic area",
    "have a barbecue with some friends": "park with barbecue grills",
    "have a bonfire with your close friends": "fire pit beach or park",
    "have a photo session with some friends": "scenic photo spot",
    "go on a fishing trip with some friends": "fishing pier",
    "go swimming with a friend": "public swimming pool",
    "go to the gym": "gym day pass",
    "learn how to play a new sport": "recreation center",
    "learn how to iceskate or rollerskate": "skating rink",
    "play a game of tennis with a friend": "public tennis courts",
    "play basketball with a group of friends": "public basketball court",
    "play a volleyball match with some friends": "volleyball court",
    "have a football scrimmage with some friends": "park with football field",
    # Giving back
    "volunteer at a local animal shelter": "animal shelter volunteering",
    "volunteer and help out at a senior center": "senior center volunteering",
    "volunteer at your local food pantry": "food pantry volunteering",
    "volunteer at your local food bank": "food bank volunteering",
    "donate blood at a local blood center": "blood donation center",
}


def outing_query(activity):
    """The Google Maps search phrase for an outing, or None for at-home activities."""
    return OUTING_QUERIES.get(str(activity).strip().lower())
