"""MCP-facing geocoding tool."""

from typing import Any, Dict, Optional

from ride_agent.services.nominatim import NominatimService

_service_instance: Optional[NominatimService] = None


def get_nominatim_service() -> NominatimService:
    """Get or create the global NominatimService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = NominatimService()
    return _service_instance


def set_nominatim_service(service: Optional[NominatimService]) -> None:
    """Override the global NominatimService instance (useful for testing)."""
    global _service_instance
    _service_instance = service


def geocode_location(location: str) -> Dict[str, Any]:
    """Convert a human-readable place or address into geographic coordinates.

    Uses OpenStreetMap Nominatim to look up the location and return its
    latitude, longitude, and formatted display name.

    Args:
        location: Human-readable place name, landmark, or address (e.g., 'IIT Patna').

    Returns:
        dict: A structured dictionary containing either:
            Success:
                {
                    "success": True,
                    "latitude": float,
                    "longitude": float,
                    "display_name": str
                }
            Failure:
                {
                    "success": False,
                    "error": str
                }
    """
    try:
        service = get_nominatim_service()
        result = service.geocode(location)
        return result
    except Exception as exc:
        return {
            "success": False,
            "error": f"Unexpected error during geocoding: {exc}",
        }
