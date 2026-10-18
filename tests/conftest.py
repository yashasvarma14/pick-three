import pytest

from recommender.base import clear_all_caches


@pytest.fixture(autouse=True)
def fresh_caches():
    """Each test sees its own fake APIs, not answers cached by an earlier test."""
    clear_all_caches()
    yield
    clear_all_caches()
