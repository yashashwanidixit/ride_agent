"""MCP-facing ride time tool."""

from typing import Any, Dict, Optional, Union

from ride_agent.services.osrm import OSRMService

_service_instance: Optional[OSRMService] = None


def get_osrm_service() -> OSRMService:
    """Get or create the global OSRMService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = OSRMService()
    return _service_instance


def set_osrm_service(service: Optional[OSRMService]) -> None:
    """Override the global OSRMService instance (useful for testing)."""
    global _service_instance
    _service_instance = service


def get_ride_time(
    pickup_latitude: Union[float, int, str],
    pickup_longitude: Union[float, int, str],
    destination_latitude: Union[float, int, str],
    destination_longitude: Union[float, int, str],
) -> Dict[str, Any]:
    """Calculate estimated driving distance and duration between two coordinates.

    Uses OSRM (Open Source Routing Machine) to compute the route and extract
    driving distance and duration.

    Args:
        pickup_latitude: Latitude of pickup location (numeric or string).
        pickup_longitude: Longitude of pickup location (numeric or string).
        destination_latitude: Latitude of destination location (numeric or string).
        destination_longitude: Longitude of destination location (numeric or string).

    Returns:
        dict: A structured dictionary containing either:
            Success:
                {
                    "success": True,
                    "distance_km": float,
                    "duration_minutes": float
                }
            Failure:
                {
                    "success": False,
                    "error": str
                }
    """
    try:
        service = get_osrm_service()
        result = service.get_route(
            pickup_latitude,
            pickup_longitude,
            destination_latitude,
            destination_longitude,
        )
        return result
    except Exception as exc:
        return {
            "success": False,
            "error": f"Unexpected error during route calculation: {exc}",
        }
