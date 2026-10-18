"""
One place for every LLM call (Claude, via the Anthropic SDK): client
creation, JSON parsing, and token/cost accounting. The notebook repeated this block in
every agent; here each agent calls `call_llm_json`.
"""

import json
import re
import threading

from . import config
from .base import ttl_cache

_client = None


def get_client():
    global _client
    if _client is None:
        import anthropic

        key = config.anthropic_api_key()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
        _client = anthropic.Anthropic(api_key=key)
    return _client


def calculate_llm_cost(input_tokens=0, output_tokens=0):
    pricing = config.model_pricing()
    input_cost = input_tokens / 1_000_000 * pricing["input_per_1m"]
    output_cost = output_tokens / 1_000_000 * pricing["output_per_1m"]
    return {"input_cost": input_cost, "output_cost": output_cost, "total_cost": input_cost + output_cost}


def parse_json_text(raw_text):
    """Parse model output as JSON, tolerating code fences or stray prose."""
    text = raw_text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


_local = threading.local()


@ttl_cache(seconds=15 * 60, maxsize=256)
def _cached_call(system_prompt, user_message, model, image_urls=()):
    _local.made_request = True
    content = user_message
    if image_urls:
        # Photos first, then the question about them. Google serves them at
        # 640 px wide, a few hundred input tokens each.
        content = [{"type": "image", "source": {"type": "url", "url": url}} for url in image_urls] + [
            {"type": "text", "text": user_message}]
    response = get_client().messages.create(
        model=model,
        max_tokens=16000,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
        # Every agent asks a short, well-specified question: low effort keeps
        # thinking (and latency) small. Raise LLM_EFFORT if answers get sloppy.
        output_config={"effort": str(config.llm_effort())},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined this request.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The model's answer was cut off (max_tokens).")
    usage = response.usage
    input_tokens = (usage.input_tokens or 0) + (getattr(usage, "cache_read_input_tokens", 0) or 0)
    output_tokens = usage.output_tokens or 0
    text = "".join(block.text for block in response.content if block.type == "text")
    # Parse inside the cached function so a bad answer is never cached.
    return parse_json_text(text), (input_tokens, output_tokens, input_tokens + output_tokens)


def call_llm_json(system_prompt, user_message, model=None, image_urls=None):
    """
    Call the model and return (parsed_json, usage_dict).
    Raises on API or parsing failure so the calling agent can report it.
    image_urls: optional public image URLs sent alongside the text.

    An identical prompt within 15 minutes (the same example search clicked
    twice, the same venues to check) reuses the earlier answer, costs
    nothing, and is reported with cached=True.
    """
    model = model or config.llm_model()
    _local.made_request = False
    parsed, (input_tokens, output_tokens, total_tokens) = _cached_call(system_prompt, user_message, model,
                                                                       tuple(image_urls or ()))
    if not _local.made_request:
        input_tokens = output_tokens = total_tokens = 0
    cost = calculate_llm_cost(input_tokens, output_tokens)

    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost": cost["input_cost"],
        "output_cost": cost["output_cost"],
        "cost": cost["total_cost"],
        "model": model,
        "cached": not _local.made_request,
    }
    return parsed, usage


def merge_usage(usages):
    """Add up the usage of several calls made for one agent (e.g. one per venue)."""
    usages = [u for u in usages if u]
    merged = {key: sum(u.get(key, 0) or 0 for u in usages)
              for key in ["input_tokens", "output_tokens", "total_tokens", "input_cost", "output_cost", "cost"]}
    merged["model"] = next((u.get("model") for u in usages if u.get("model")), None)
    merged["cached"] = bool(usages) and all(u.get("cached") for u in usages)
    merged["calls"] = len(usages)
    return merged


def run_concurrently(func, items, max_workers=6):
    """
    func(item) for every item at the same time; returns a list of
    (result, None) or (None, exception) in input order. LLM latency is
    mostly output length, so several small calls in parallel finish much
    sooner than one call that writes every answer.
    """
    from concurrent.futures import ThreadPoolExecutor

    def safe(item):
        try:
            return func(item), None
        except Exception as exc:  # one failed venue must not sink the rest
            return None, exc

    items = list(items)
    if len(items) <= 1:
        return [safe(i) for i in items]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(safe, items))
