"""Nominatim geocoding service with caching, rate limiting, and error handling."""

import os
import threading
import time
from typing import Any, Dict, Optional

import httpx

DEFAULT_USER_AGENT = "ride-agent-hackathon/0.1"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
DEFAULT_MIN_INTERVAL_SECONDS = 1.0


class NominatimService:
    """Service to geocode addresses/places using OpenStreetMap Nominatim."""

    def __init__(
        self,
        user_agent: Optional[str] = None,
        min_request_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.user_agent = (
            user_agent
            or os.environ.get("RIDE_AGENT_USER_AGENT")
            or DEFAULT_USER_AGENT
        )
        self.min_request_interval = min_request_interval
        self._client = client
        self._owns_client = client is None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_request_time: float = 0.0
        self._lock = threading.Lock()

    def _get_client(self) -> httpx.Client:
        """Return the active HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=10.0,
                headers={"User-Agent": self.user_agent},
            )
            self._owns_client = True
        return self._client

    def _enforce_rate_limit(self) -> None:
        """Enforce at least min_request_interval seconds between outgoing requests."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
            self._last_request_time = time.monotonic()

    def geocode(self, location: str) -> Dict[str, Any]:
        """Geocode a human-readable location into coordinates.

        Returns:
            dict: Structured success or failure response:
                Success: {"success": True, "latitude": float, "longitude": float, "display_name": str}
                Failure: {"success": False, "error": str}
        """
        # 1. Validate input
        if not isinstance(location, str) or not location.strip():
            return {
                "success": False,
                "error": "Location cannot be empty",
            }

        cleaned_location = location.strip()
        cache_key = cleaned_location.lower()

        # 2. Check in-memory cache
        with self._lock:
            if cache_key in self._cache:
                return dict(self._cache[cache_key])

        # 3. Enforce rate limit (1 request per second for Nominatim usage policy)
        self._enforce_rate_limit()

        # 4. Make HTTP request
        params = {
            "q": cleaned_location,
            "format": "jsonv2",
            "limit": 1,
        }

        try:
            client = self._get_client()
            response = client.get(
                NOMINATIM_SEARCH_URL,
                params=params,
                headers={"User-Agent": self.user_agent},
            )

            # Check HTTP status
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Geocoding service HTTP error: {response.status_code}",
                }

            # Parse JSON
            try:
                data = response.json()
            except Exception:
                return {
                    "success": False,
                    "error": "Malformed response from geocoding service: invalid JSON",
                }

            if not isinstance(data, list):
                return {
                    "success": False,
                    "error": "Unexpected response format from geocoding service",
                }

            # Handle location not found
            if not data:
                return {
                    "success": False,
                    "error": f"Location not found: {cleaned_location}",
                }

            # Parse first result
            first_match = data[0]
            try:
                latitude = float(first_match["lat"])
                longitude = float(first_match["lon"])
                display_name = str(first_match.get("display_name", cleaned_location))
            except (KeyError, ValueError, TypeError) as exc:
                return {
                    "success": False,
                    "error": f"Malformed coordinate data in geocoding response: {exc}",
                }

            result = {
                "success": True,
                "latitude": latitude,
                "longitude": longitude,
                "display_name": display_name,
            }

            # Cache successful result
            with self._lock:
                self._cache[cache_key] = result

            return dict(result)

        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "Geocoding service request timed out",
            }
        except httpx.NetworkError as exc:
            return {
                "success": False,
                "error": f"Network error connecting to geocoding service: {exc}",
            }
        except httpx.HTTPError as exc:
            return {
                "success": False,
                "error": f"Geocoding service error: {exc}",
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"Geocoding service unexpected error: {exc}",
            }

    def close(self) -> None:
        """Close the underlying HTTP client if owned by this service."""
        if self._owns_client and self._client is not None and not self._client.is_closed:
            self._client.close()

    def clear_cache(self) -> None:
        """Clear the in-memory cache (useful for testing)."""
        with self._lock:
            self._cache.clear()
