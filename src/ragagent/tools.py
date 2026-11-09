"""Tools the agent can call, plus their JSON schemas for function calling."""

from __future__ import annotations

import math
from typing import Any

# Mirrors data/docs/pto-policy.md. Kept in code so the arithmetic is exact and testable.
PTO_TIERS: list[tuple[float, int]] = [(5, 25), (2, 20), (0, 15)]  # (min years, days per year)
PTO_CARRYOVER_CAP = 5


def calculate_pto(
    years_of_service: float, months_worked_this_year: int = 12, days_used: float = 0
) -> dict[str, Any]:
    """PTO for one calendar year: annual tier, proration (round down to 0.5), remaining."""
    if years_of_service < 0:
        raise ValueError("years_of_service must be >= 0")
    if not 0 <= months_worked_this_year <= 12:
        raise ValueError("months_worked_this_year must be between 0 and 12")
    if days_used < 0:
        raise ValueError("days_used must be >= 0")
    annual = next(days for min_years, days in PTO_TIERS if years_of_service >= min_years)
    accrued = math.floor(annual * months_worked_this_year / 12 * 2) / 2
    remaining = max(accrued - days_used, 0.0)
    return {
        "years_of_service": years_of_service,
        "annual_days": annual,
        "months_worked_this_year": months_worked_this_year,
        "accrued_days": accrued,
        "days_used": days_used,
        "remaining_days": remaining,
        "max_carryover_days": min(remaining, PTO_CARRYOVER_CAP),
        "source": "pto-policy",
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculate_pto",
            "description": "Calculate an employee's PTO days for this calendar year.",
            "parameters": {
                "type": "object",
                "properties": {
                    "years_of_service": {"type": "number", "description": "Completed years."},
                    "months_worked_this_year": {
                        "type": "integer",
                        "description": "Months worked this calendar year (1-12). Default 12.",
                    },
                    "days_used": {"type": "number", "description": "PTO days already used."},
                },
                "required": ["years_of_service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the HR policy documents and return a cited answer.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]
