"""Ride Concierge AI Agent MCP Server exposing tools for location and routing."""

from typing import Any, Dict, Union

from mcp.server import MCPServer

from ride_agent.tools.geocoding import geocode_location as _geocode_location
from ride_agent.tools.ride_time import get_ride_time as _get_ride_time
from ride_agent.tools.ride_price import get_ride_price as _get_ride_price

# Initialize MCP Server instance
mcp = MCPServer(
    name="ride-agent",
    description="Ride Concierge AI Agent MCP Server",
)


@mcp.tool(
    name="geocode_location",
    description=(
        "Converts a human-readable place, landmark, or address into geographic coordinates "
        "(latitude, longitude) using OpenStreetMap Nominatim."
    ),
)
def geocode_location(location: str) -> Dict[str, Any]:
    """Convert a human-readable place or address into geographic coordinates.

    Args:
        location: Human-readable place name or address string (e.g., 'IIT Patna').

    Returns:
        Structured dictionary with latitude, longitude, and display_name on success,
        or an error message on failure.
    """
    return _geocode_location(location)


@mcp.tool(
    name="get_ride_time",
    description=(
        "Calculates the estimated driving distance (in kilometers) and duration (in minutes) "
        "between two geographic coordinates using OSRM routing."
    ),
)
def get_ride_time(
    pickup_latitude: Union[float, int, str],
    pickup_longitude: Union[float, int, str],
    destination_latitude: Union[float, int, str],
    destination_longitude: Union[float, int, str],
) -> Dict[str, Any]:
    """Calculate estimated driving distance and duration between two coordinates.

    Args:
        pickup_latitude: Latitude of pickup location.
        pickup_longitude: Longitude of pickup location.
        destination_latitude: Latitude of destination location.
        destination_longitude: Longitude of destination location.

    Returns:
        Structured dictionary with distance_km and duration_minutes on success,
        or an error message on failure.
    """
    return _get_ride_time(
        pickup_latitude,
        pickup_longitude,
        destination_latitude,
        destination_longitude,
    )


@mcp.tool(
    name="get_ride_price",
    description=(
        "Calculates an estimated ride fare from distance and duration. "
        "This is a demo pricing model (not a live provider quote). "
        "Uses: base ₹50 + (distance_km × ₹15) + (duration_minutes × ₹2)."
    ),
)
def get_ride_price(
    distance_km: Union[float, int, str],
    duration_minutes: Union[float, int, str],
) -> Dict[str, Any]:
    """Calculate an estimated ride fare.

    Args:
        distance_km: Distance traveled in kilometers.
        duration_minutes: Duration of the ride in minutes.

    Returns:
        Structured dictionary with estimated_fare on success,
        or an error message on failure.
    """
    return _get_ride_price(distance_km, duration_minutes)


def main() -> None:
    """Run the MCP server using standard I/O (stdio) transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
