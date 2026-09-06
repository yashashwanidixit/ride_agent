"""OSRM routing service with error handling."""

from typing import Any, Dict, Optional

import httpx

OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving"


class OSRMService:
    """Service to calculate driving distance and duration using OSRM API."""

    def __init__(self, client: Optional[httpx.Client] = None) -> None:
        """Initialize OSRM service.

        Args:
            client: Optional httpx.Client for testing. If None, one will be created.
        """
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.Client:
        """Return the active HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=10.0)
            self._owns_client = True
        return self._client

    def get_route(
        self,
        pickup_latitude: float,
        pickup_longitude: float,
        destination_latitude: float,
        destination_longitude: float,
    ) -> Dict[str, Any]:
        """Calculate driving distance and duration between two coordinates.

        Args:
            pickup_latitude: Latitude of pickup location.
            pickup_longitude: Longitude of pickup location.
            destination_latitude: Latitude of destination location.
            destination_longitude: Longitude of destination location.

        Returns:
            dict: Structured success or failure response:
                Success: {
                    "success": True,
                    "distance_km": float,
                    "duration_minutes": float
                }
                Failure: {
                    "success": False,
                    "error": str
                }
        """
        # 1. Validate coordinates are numeric
        try:
            pickup_lat = float(pickup_latitude)
            pickup_lon = float(pickup_longitude)
            dest_lat = float(destination_latitude)
            dest_lon = float(destination_longitude)
        except (TypeError, ValueError) as exc:
            return {
                "success": False,
                "error": f"Invalid coordinates: coordinates must be numeric. {exc}",
            }

        # 2. Validate coordinate ranges (latitude: -90 to 90, longitude: -180 to 180)
        for name, value in [
            ("pickup_latitude", pickup_lat),
            ("pickup_longitude", pickup_lon),
            ("destination_latitude", dest_lat),
            ("destination_longitude", dest_lon),
        ]:
            if name.endswith("latitude") and not (-90 <= value <= 90):
                return {
                    "success": False,
                    "error": f"Invalid {name}: must be between -90 and 90",
                }
            if name.endswith("longitude") and not (-180 <= value <= 180):
                return {
                    "success": False,
                    "error": f"Invalid {name}: must be between -180 and 180",
                }

        # 3. Build OSRM URL
        # OSRM expects: longitude,latitude;longitude,latitude
        coordinates = f"{pickup_lon},{pickup_lat};{dest_lon},{dest_lat}"
        url = f"{OSRM_ROUTE_URL}/{coordinates}"

        # 4. Make HTTP request
        params = {"overview": "false"}

        try:
            client = self._get_client()
            response = client.get(url, params=params)

            # Check HTTP status
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"OSRM service HTTP error: {response.status_code}",
                }

            # Parse JSON
            try:
                data = response.json()
            except Exception:
                return {
                    "success": False,
                    "error": "Malformed response from OSRM service: invalid JSON",
                }

            # Check for OSRM-specific error codes
            if not isinstance(data, dict):
                return {
                    "success": False,
                    "error": "Unexpected response format from OSRM service",
                }

            # Check if OSRM returned an error in the response
            if data.get("code") != "Ok":
                error_msg = data.get("code", "Unknown error")
                return {
                    "success": False,
                    "error": f"OSRM service error: {error_msg}",
                }

            # Verify routes array exists and has at least one route
            routes = data.get("routes", [])
            if not isinstance(routes, list) or len(routes) == 0:
                return {
                    "success": False,
                    "error": "No route available between the specified coordinates",
                }

            # Parse first route
            first_route = routes[0]
            try:
                distance_meters = float(first_route.get("distance", 0))
                duration_seconds = float(first_route.get("duration", 0))
            except (KeyError, ValueError, TypeError) as exc:
                return {
                    "success": False,
                    "error": f"Malformed route data in OSRM response: {exc}",
                }

            # Convert to useful units
            distance_km = distance_meters / 1000.0
            duration_minutes = duration_seconds / 60.0

            return {
                "success": True,
                "distance_km": round(distance_km, 2),
                "duration_minutes": round(duration_minutes, 2),
            }

        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "OSRM service request timed out",
            }
        except httpx.NetworkError as exc:
            return {
                "success": False,
                "error": f"Network error connecting to OSRM service: {exc}",
            }
        except httpx.HTTPError as exc:
            return {
                "success": False,
                "error": f"OSRM service error: {exc}",
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"OSRM service unexpected error: {exc}",
            }

    def close(self) -> None:
        """Close the underlying HTTP client if owned by this service."""
        if self._owns_client and self._client is not None and not self._client.is_closed:
            self._client.close()
