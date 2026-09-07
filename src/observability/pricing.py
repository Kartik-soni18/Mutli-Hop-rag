"""USD pricing, independent of the telemetry backend."""

import os
from decimal import Decimal


def price(usage, requested_model, actual_model, reported=None):
    rates = {
        name: Decimal(os.getenv(env, default))
        for name, env, default in (
            ("input", "INPUT_COST_PER_1M", "0.021"),
            ("output", "OUTPUT_COST_PER_1M", "0.063"),
            ("cached", "CACHED_COST_PER_1M", "0.0042"),
        )
    }
    inp, out, cached = (
        usage.get(k) for k in ("input_tokens", "output_tokens", "cached_tokens")
    )
    # Gateways may omit the organization prefix in the response model.
    matches = actual_model in (requested_model, requested_model.split("/")[-1])
    estimated = None
    if (
        matches
        and inp is not None
        and out is not None
        and 0 <= (cached or 0) <= inp
        and out >= 0
        and all(r.is_finite() and r >= 0 for r in rates.values())
    ):
        estimated = (
            Decimal(inp - (cached or 0)) * rates["input"]
            + Decimal(cached or 0) * rates["cached"]
            + Decimal(out) * rates["output"]
        ) / Decimal(1_000_000)
    reported = reported or {}
    authoritative = None
    if reported.get("currency") == "USD" and reported.get("amount") is not None:
        amount = Decimal(str(reported["amount"]))
        if amount.is_finite() and amount >= 0:
            authoritative = amount
    selected = authoritative if authoritative is not None else estimated
    return {
        "cost_usd": str(selected) if selected is not None else None,
        "estimated_cost_usd": str(estimated) if estimated is not None else None,
        "provider_reported_cost": reported or None,
        "cost_source": (
            "provider"
            if authoritative is not None
            else "estimated"
            if estimated is not None
            else "unavailable"
        ),
        "currency": "USD",
        "rates_per_1m": {k: str(v) for k, v in rates.items()},
        "cached_tokens_assumed_zero": cached is None,
        "pricing_model_mismatch": not matches,
    }
