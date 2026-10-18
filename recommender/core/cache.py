"""
A small time-limited cache (my own version of functools.lru_cache with an
expiry time).

Why: the same Google search, forecast or website often gets requested
twice in a few minutes (someone clicks the same example twice, or two
agents need the same venue). Caching those makes repeat runs instant and
saves API money.
"""

import copy
import functools
import threading
import time
from concurrent.futures import Future



_ALL_CACHES = []


def clear_all_caches():
    for store in _ALL_CACHES:
        store.clear()


def ttl_cache(seconds, maxsize=512):
    """
    Remember a function's result for `seconds`. Repeat searches (same venue
    website, same Places query, same forecast) come back instantly.

    Two threads asking for the same thing at once share one request
    instead of both making it. Errors are never cached. Callers get a
    copy, so changing a result can't corrupt the cache.
    """
    def decorator(func):
        store = {}
        lock = threading.Lock()
        _ALL_CACHES.append(store)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            with lock:
                hit = store.get(key)
                if hit and hit[0] > now:
                    future, owner = hit[1], False
                else:
                    future, owner = Future(), True
                    store[key] = (now + seconds, future)
                    if len(store) > maxsize:
                        for old in sorted(store, key=lambda k: store[k][0])[: len(store) - maxsize]:
                            if store[old][1].done():
                                store.pop(old, None)
            if owner:
                try:
                    future.set_result(func(*args, **kwargs))
                except BaseException as exc:
                    with lock:
                        if store.get(key, (None, None))[1] is future:
                            store.pop(key, None)
                    future.set_exception(exc)
            return copy.deepcopy(future.result())

        wrapper.cache_clear = store.clear
        return wrapper

    return decorator
