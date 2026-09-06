"""MCP-facing ride pricing tool."""

from typing import Any, Dict, Union

from ride_agent.services.pricing import PricingService


def get_ride_price(
    distance_km: Union[float, int, str],
    duration_minutes: Union[float, int, str],
) -> Dict[str, Any]:
    """Calculate an estimated ride fare from distance and duration.

    This is a demo pricing estimate for demonstration purposes only.
    NOT a live provider quote (Uber, Ola, etc).

    Uses a simple model:
        base_fare (₹50) + (distance_km × ₹15) + (duration_minutes × ₹2)

    Args:
        distance_km: Distance traveled in kilometers.
        duration_minutes: Duration of the ride in minutes.

    Returns:
        dict: A structured dictionary containing either:
            Success:
                {
                    "success": True,
                    "estimated_fare": float,
                    "currency": "INR",
                    "pricing_model": "demo_estimate"
                }
            Failure:
                {
                    "success": False,
                    "error": str
                }
    """
    try:
        result = PricingService.calculate_fare(distance_km, duration_minutes)
        return result
    except Exception as exc:
        return {
            "success": False,
            "error": f"Unexpected error during fare calculation: {exc}",
        }
