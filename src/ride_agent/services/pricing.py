"""Pricing calculation service for ride estimates."""

from typing import Any, Dict

# Pricing model constants (in INR)
BASE_FARE = 50.0  # ₹50
PER_KM_RATE = 15.0  # ₹15 per km
PER_MINUTE_RATE = 2.0  # ₹2 per minute


class PricingService:
    """Service to calculate estimated ride fares using a deterministic model."""

    @staticmethod
    def calculate_fare(distance_km: float, duration_minutes: float) -> Dict[str, Any]:
        """Calculate an estimated ride fare.

        Uses a demo pricing model:
            base_fare (₹50) + (distance_km × ₹15) + (duration_minutes × ₹2)

        This is an estimate for demonstration purposes, NOT a live provider quote.

        Args:
            distance_km: Distance traveled in kilometers.
            duration_minutes: Duration of the ride in minutes.

        Returns:
            dict: Structured success or failure response:
                Success: {
                    "success": True,
                    "estimated_fare": float,
                    "currency": "INR",
                    "pricing_model": "demo_estimate"
                }
                Failure: {
                    "success": False,
                    "error": str
                }
        """
        # 1. Validate inputs are numeric
        try:
            distance = float(distance_km)
            duration = float(duration_minutes)
        except (TypeError, ValueError) as exc:
            return {
                "success": False,
                "error": f"Invalid input: distance and duration must be numeric. {exc}",
            }

        # 2. Validate non-negative values
        if distance < 0:
            return {
                "success": False,
                "error": "Distance cannot be negative",
            }

        if duration < 0:
            return {
                "success": False,
                "error": "Duration cannot be negative",
            }

        # 3. Calculate fare
        estimated_fare = BASE_FARE + (distance * PER_KM_RATE) + (duration * PER_MINUTE_RATE)

        # 4. Round to 2 decimal places
        estimated_fare = round(estimated_fare, 2)

        return {
            "success": True,
            "estimated_fare": estimated_fare,
            "currency": "INR",
            "pricing_model": "demo_estimate",
        }
