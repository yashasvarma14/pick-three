"""
One place for every LLM call: client creation, JSON parsing,
and token/cost accounting. The notebook repeated this block in
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
        from openai import OpenAI

        key = config.openai_api_key()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        _client = OpenAI(api_key=key)
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
def _cached_call(system_prompt, user_message, model):
    _local.made_request = True
    kwargs = {}
    effort = config.get_setting("OPENAI_REASONING_EFFORT")
    if effort:
        # Optional speed knob for reasoning models ("low" or "minimal").
        kwargs["reasoning"] = {"effort": str(effort)}

    response = get_client().responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        **kwargs,
    )
    usage_obj = getattr(response, "usage", None)
    input_tokens = getattr(usage_obj, "input_tokens", 0) or 0
    output_tokens = getattr(usage_obj, "output_tokens", 0) or 0
    total_tokens = getattr(usage_obj, "total_tokens", 0) or (input_tokens + output_tokens)
    # Parse inside the cached function so a bad answer is never cached.
    return parse_json_text(response.output_text), (input_tokens, output_tokens, total_tokens)


def call_llm_json(system_prompt, user_message, model=None):
    """
    Call the model and return (parsed_json, usage_dict).
    Raises on API or parsing failure so the calling agent can report it.

    An identical prompt within 15 minutes (the same example search clicked
    twice, the same venues to check) reuses the earlier answer, costs
    nothing, and is reported with cached=True.
    """
    model = model or config.llm_model()
    _local.made_request = False
    parsed, (input_tokens, output_tokens, total_tokens) = _cached_call(system_prompt, user_message, model)
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
