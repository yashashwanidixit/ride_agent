"""Unit tests for the geocoding service and MCP tool."""

import json
from typing import Any, Dict, List
import httpx
import pytest

from ride_agent.services.nominatim import NominatimService, DEFAULT_USER_AGENT
from ride_agent.tools.geocoding import (
    geocode_location,
    set_nominatim_service,
)
from ride_agent.server import mcp


@pytest.fixture(autouse=True)
def reset_geocoding_service():
    """Ensure each test runs with a fresh service instance and clean cache."""
    service = NominatimService(min_request_interval=0.0)
    set_nominatim_service(service)
    yield service
    service.close()
    set_nominatim_service(None)


def create_mock_service(
    mock_response_data: Any,
    status_code: int = 200,
    headers: Dict[str, str] = None,
    exception: Exception = None,
    min_request_interval: float = 0.0,
) -> tuple[NominatimService, List[httpx.Request]]:
    """Helper to create a NominatimService with a mock HTTP transport."""
    recorded_requests: List[httpx.Request] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        if exception:
            raise exception
        if isinstance(mock_response_data, str):
            content = mock_response_data.encode("utf-8")
        else:
            content = json.dumps(mock_response_data).encode("utf-8")
        resp_headers = {"Content-Type": "application/json"}
        if headers:
            resp_headers.update(headers)
        return httpx.Response(status_code=status_code, content=content, headers=resp_headers)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.Client(transport=transport)
    service = NominatimService(client=client, min_request_interval=min_request_interval)
    return service, recorded_requests


def test_successful_geocoding_iit_patna():
    """Test 1: Successful geocoding for 'IIT Patna'."""
    mock_data = [
        {
            "place_id": 123456,
            "lat": "25.5424381",
            "lon": "84.8516072",
            "display_name": "Indian Institute of Technology Patna, Bihta, Patna, Bihar, 801106, India",
            "type": "university",
        }
    ]
    service, requests = create_mock_service(mock_data)
    set_nominatim_service(service)

    result = geocode_location("IIT Patna")

    assert result["success"] is True
    assert isinstance(result["latitude"], (int, float))
    assert isinstance(result["longitude"], (int, float))
    assert pytest.approx(result["latitude"], 0.0001) == 25.5424381
    assert pytest.approx(result["longitude"], 0.0001) == 84.8516072
    assert "IIT" in result["display_name"] or "Patna" in result["display_name"]
    assert len(requests) == 1
    assert requests[0].url.params["q"] == "IIT Patna"
    assert requests[0].url.params["format"] == "jsonv2"
    assert requests[0].url.params["limit"] == "1"


def test_successful_geocoding_airport_patna():
    """Test 2: Successful geocoding for another valid location."""
    mock_data = [
        {
            "place_id": 789012,
            "lat": "25.5912444",
            "lon": "85.0880227",
            "display_name": "Jay Prakash Narayan International Airport, Patna, Bihar, India",
            "type": "aerodrome",
        }
    ]
    service, _ = create_mock_service(mock_data)
    set_nominatim_service(service)

    result = geocode_location("Jay Prakash Narayan International Airport, Patna")

    assert result["success"] is True
    assert isinstance(result["latitude"], (int, float))
    assert isinstance(result["longitude"], (int, float))
    assert pytest.approx(result["latitude"], 0.0001) == 25.5912444
    assert pytest.approx(result["longitude"], 0.0001) == 85.0880227
    assert "Airport" in result["display_name"]


def test_empty_input():
    """Test 3: Empty input handling (empty string and whitespace-only)."""
    # Empty string
    result = geocode_location("")
    assert result["success"] is False
    assert "Location cannot be empty" in result["error"]

    # Whitespace only
    result_ws = geocode_location("     ")
    assert result_ws["success"] is False
    assert "Location cannot be empty" in result_ws["error"]


def test_unknown_location():
    """Test 4: Location not found handling for nonsensical location."""
    service, _ = create_mock_service([])
    set_nominatim_service(service)

    unknown_query = "kjasdhfkjashdfkjahsdf918273918237"
    result = geocode_location(unknown_query)

    assert result["success"] is False
    assert "Location not found" in result["error"]
    assert unknown_query in result["error"]


def test_http_error_handling():
    """Test 5a: HTTP error handling (e.g. 500 Internal Server Error)."""
    service, _ = create_mock_service({}, status_code=500)
    set_nominatim_service(service)

    result = geocode_location("Patna")

    assert result["success"] is False
    assert "Geocoding service HTTP error: 500" in result["error"]
    # Ensure no raw traceback
    assert "Traceback" not in result["error"]


def test_network_connection_error():
    """Test 5b: Network connection failure raises structured error."""
    service, _ = create_mock_service(
        None,
        exception=httpx.ConnectError("Connection refused by target machine"),
    )
    set_nominatim_service(service)

    result = geocode_location("Patna")

    assert result["success"] is False
    assert "Network error connecting to geocoding service" in result["error"]
    assert "Traceback" not in result["error"]


def test_timeout_error():
    """Test 5c: Request timeout raises structured error."""
    service, _ = create_mock_service(
        None,
        exception=httpx.TimeoutException("Read timed out"),
    )
    set_nominatim_service(service)

    result = geocode_location("Patna")

    assert result["success"] is False
    assert "Geocoding service request timed out" in result["error"]
    assert "Traceback" not in result["error"]


def test_malformed_json_response():
    """Test 5d: Malformed response handling."""
    service, _ = create_mock_service("Not a valid json {{{", status_code=200)
    set_nominatim_service(service)

    result = geocode_location("Patna")

    assert result["success"] is False
    assert "Malformed response from geocoding service" in result["error"]


def test_cache_behavior():
    """Test 6: Repeated identical location queries are served from cache."""
    mock_data = [
        {
            "lat": "25.5424381",
            "lon": "84.8516072",
            "display_name": "Indian Institute of Technology Patna",
        }
    ]
    service, requests = create_mock_service(mock_data)
    set_nominatim_service(service)

    # First call - should trigger HTTP request
    result1 = geocode_location("IIT Patna")
    assert result1["success"] is True
    assert len(requests) == 1

    # Second call with identical query - should hit cache
    result2 = geocode_location("IIT Patna")
    assert result2["success"] is True
    assert len(requests) == 1  # No additional HTTP request made

    # Third call with case-insensitive variation - should also hit cache
    result3 = geocode_location("  iit patna  ")
    assert result3["success"] is True
    assert len(requests) == 1  # Still 1 request

    assert result1 == result2 == result3


def test_user_agent_header():
    """Test: User-Agent header is set correctly according to OSM policy."""
    service, requests = create_mock_service([{"lat": "0.0", "lon": "0.0"}])
    set_nominatim_service(service)

    geocode_location("Test Place")

    assert len(requests) == 1
    assert requests[0].headers["user-agent"] == DEFAULT_USER_AGENT


@pytest.mark.anyio
async def test_mcp_server_tool_exposure_and_call():
    """Test: geocode_location tool is properly registered and callable on the MCPServer."""
    mock_data = [
        {
            "lat": "25.5424381",
            "lon": "84.8516072",
            "display_name": "Indian Institute of Technology Patna",
        }
    ]
    service, _ = create_mock_service(mock_data)
    set_nominatim_service(service)

    # 1. Verify tool exists in server's tool listing
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "geocode_location" in tool_names

    tool_obj = next(t for t in tools if t.name == "geocode_location")
    assert "geographic coordinates" in tool_obj.description.lower()

    # 2. Call tool through MCP server interface
    mcp_result = await mcp.call_tool("geocode_location", {"location": "IIT Patna"})
    assert not mcp_result.is_error
    assert len(mcp_result.content) > 0
    parsed_payload = json.loads(mcp_result.content[0].text)
    assert parsed_payload["success"] is True
    assert parsed_payload["latitude"] == 25.5424381
    assert parsed_payload["longitude"] == 84.8516072


# ============================================================
# STAGE 2: RIDE TIME TESTS
# ============================================================


def create_mock_osrm_service(
    mock_response_data: Any,
    status_code: int = 200,
    headers: Dict[str, str] = None,
    exception: Exception = None,
) -> tuple:
    """Helper to create an OSRMService with a mock HTTP transport."""
    from ride_agent.services.osrm import OSRMService
    from ride_agent.tools.ride_time import set_osrm_service

    recorded_requests: List[httpx.Request] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        if exception:
            raise exception
        if isinstance(mock_response_data, str):
            content = mock_response_data.encode("utf-8")
        else:
            content = json.dumps(mock_response_data).encode("utf-8")
        resp_headers = {"Content-Type": "application/json"}
        if headers:
            resp_headers.update(headers)
        return httpx.Response(status_code=status_code, content=content, headers=resp_headers)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.Client(transport=transport)
    service = OSRMService(client=client)
    set_osrm_service(service)
    return service, recorded_requests


def test_successful_route_calculation():
    """Test 1: Successful route calculation with distance and duration conversion."""
    from ride_agent.tools.ride_time import get_ride_time

    # Real OSRM response format
    mock_response = {
        "code": "Ok",
        "routes": [
            {
                "distance": 12500.5,  # meters
                "duration": 1234.2,  # seconds
                "geometry": "...",  # not used with overview=false
            }
        ],
    }
    service, requests = create_mock_osrm_service(mock_response)

    # Use coordinates from IIT Patna to JP Airport
    result = get_ride_time(25.5424381, 84.8516072, 25.5912444, 85.0880227)

    assert result["success"] is True
    assert isinstance(result["distance_km"], float)
    assert isinstance(result["duration_minutes"], float)
    # 12500.5 meters = 12.50 km
    assert pytest.approx(result["distance_km"], 0.01) == 12.5
    # 1234.2 seconds = 20.57 minutes
    assert pytest.approx(result["duration_minutes"], 0.01) == 20.57
    assert len(requests) == 1
    # Verify OSRM URL format: longitude,latitude;longitude,latitude
    assert "84.8516072,25.5424381" in requests[0].url.path
    assert "85.0880227,25.5912444" in requests[0].url.path


def test_route_with_string_coordinates():
    """Test: String coordinates are properly converted to floats."""
    from ride_agent.tools.ride_time import get_ride_time

    mock_response = {
        "code": "Ok",
        "routes": [
            {
                "distance": 5000.0,
                "duration": 300.0,
            }
        ],
    }
    service, _ = create_mock_osrm_service(mock_response)

    # Pass coordinates as strings
    result = get_ride_time("25.5424381", "84.8516072", "25.5912444", "85.0880227")

    assert result["success"] is True
    assert result["distance_km"] == 5.0
    assert result["duration_minutes"] == 5.0


def test_no_route_available():
    """Test 2: No route available between coordinates (empty routes array)."""
    from ride_agent.tools.ride_time import get_ride_time

    mock_response = {
        "code": "Ok",
        "routes": [],
    }
    service, _ = create_mock_osrm_service(mock_response)

    result = get_ride_time(0, 0, 90, 90)

    assert result["success"] is False
    assert "No route available" in result["error"]


def test_osrm_error_response():
    """Test: OSRM service returns an error code."""
    from ride_agent.tools.ride_time import get_ride_time

    mock_response = {
        "code": "NoRoute",
        "message": "No suitable edges found near coordinates",
    }
    service, _ = create_mock_osrm_service(mock_response)

    result = get_ride_time(0, 0, 90, 90)

    assert result["success"] is False
    assert "OSRM service error: NoRoute" in result["error"]


def test_http_error_handling_ride_time():
    """Test 3: HTTP error handling for ride time."""
    from ride_agent.tools.ride_time import get_ride_time

    service, _ = create_mock_osrm_service({}, status_code=500)

    result = get_ride_time(25.5424381, 84.8516072, 25.5912444, 85.0880227)

    assert result["success"] is False
    assert "OSRM service HTTP error: 500" in result["error"]
    assert "Traceback" not in result["error"]


def test_network_error_ride_time():
    """Test 4: Network error handling for ride time."""
    from ride_agent.tools.ride_time import get_ride_time

    service, _ = create_mock_osrm_service(
        None,
        exception=httpx.ConnectError("Connection refused"),
    )

    result = get_ride_time(25.5424381, 84.8516072, 25.5912444, 85.0880227)

    assert result["success"] is False
    assert "Network error connecting to OSRM service" in result["error"]
    assert "Traceback" not in result["error"]


def test_timeout_error_ride_time():
    """Test: Request timeout handling for ride time."""
    from ride_agent.tools.ride_time import get_ride_time

    service, _ = create_mock_osrm_service(
        None,
        exception=httpx.TimeoutException("Read timed out"),
    )

    result = get_ride_time(25.5424381, 84.8516072, 25.5912444, 85.0880227)

    assert result["success"] is False
    assert "OSRM service request timed out" in result["error"]
    assert "Traceback" not in result["error"]


def test_malformed_json_response_ride_time():
    """Test 5: Malformed JSON response handling."""
    from ride_agent.tools.ride_time import get_ride_time

    service, _ = create_mock_osrm_service("Not valid json {{{", status_code=200)

    result = get_ride_time(25.5424381, 84.8516072, 25.5912444, 85.0880227)

    assert result["success"] is False
    assert "Malformed response from OSRM service" in result["error"]


def test_malformed_route_data():
    """Test: Malformed route data in OSRM response."""
    from ride_agent.tools.ride_time import get_ride_time

    mock_response = {
        "code": "Ok",
        "routes": [
            {
                "distance": "not a number",  # Should be numeric
                "duration": 300.0,
            }
        ],
    }
    service, _ = create_mock_osrm_service(mock_response)

    result = get_ride_time(25.5424381, 84.8516072, 25.5912444, 85.0880227)

    assert result["success"] is False
    assert "Malformed route data" in result["error"]


def test_invalid_coordinates_non_numeric():
    """Test: Non-numeric coordinates are rejected."""
    from ride_agent.tools.ride_time import get_ride_time

    result = get_ride_time("not a number", 84.8516072, 25.5912444, 85.0880227)

    assert result["success"] is False
    assert "Invalid coordinates" in result["error"]


def test_invalid_latitude_range():
    """Test: Latitude outside valid range is rejected."""
    from ride_agent.tools.ride_time import get_ride_time

    result = get_ride_time(95.0, 84.8516072, 25.5912444, 85.0880227)

    assert result["success"] is False
    assert "Invalid pickup_latitude" in result["error"]


def test_invalid_longitude_range():
    """Test: Longitude outside valid range is rejected."""
    from ride_agent.tools.ride_time import get_ride_time

    result = get_ride_time(25.5424381, 200.0, 25.5912444, 85.0880227)

    assert result["success"] is False
    assert "Invalid pickup_longitude" in result["error"]


@pytest.mark.anyio
async def test_mcp_server_get_ride_time_exposure():
    """Test: get_ride_time tool is properly registered on the MCPServer."""
    from ride_agent.tools.ride_time import set_osrm_service

    mock_response = {
        "code": "Ok",
        "routes": [
            {
                "distance": 10000.0,
                "duration": 600.0,
            }
        ],
    }
    service, _ = create_mock_osrm_service(mock_response)

    # 1. Verify tool exists in server's tool listing
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "get_ride_time" in tool_names
    assert "geocode_location" in tool_names  # Stage 1 tool still present

    tool_obj = next(t for t in tools if t.name == "get_ride_time")
    assert "driving distance" in tool_obj.description.lower()

    # 2. Call tool through MCP server interface
    mcp_result = await mcp.call_tool(
        "get_ride_time",
        {
            "pickup_latitude": 25.5424381,
            "pickup_longitude": 84.8516072,
            "destination_latitude": 25.5912444,
            "destination_longitude": 85.0880227,
        },
    )
    assert not mcp_result.is_error
    assert len(mcp_result.content) > 0
    parsed_payload = json.loads(mcp_result.content[0].text)
    assert parsed_payload["success"] is True
    assert parsed_payload["distance_km"] == 10.0
    assert parsed_payload["duration_minutes"] == 10.0


@pytest.mark.anyio
async def test_mcp_server_ride_time_error_handling():
    """Test: get_ride_time error is properly handled via MCP."""
    from ride_agent.tools.ride_time import set_osrm_service

    mock_response = {
        "code": "Ok",
        "routes": [],
    }
    service, _ = create_mock_osrm_service(mock_response)

    mcp_result = await mcp.call_tool(
        "get_ride_time",
        {
            "pickup_latitude": 0,
            "pickup_longitude": 0,
            "destination_latitude": 90,
            "destination_longitude": 90,
        },
    )
    assert not mcp_result.is_error
    parsed_payload = json.loads(mcp_result.content[0].text)
    assert parsed_payload["success"] is False
    assert "No route available" in parsed_payload["error"]


# ============================================================
# STAGE 3: RIDE PRICE TESTS
# ============================================================


def test_successful_price_calculation():
    """Test 1: Successful fare calculation with standard inputs."""
    from ride_agent.tools.ride_price import get_ride_price

    # Standard case: 10 km, 25 minutes
    # Expected: 50 + (10 × 15) + (25 × 2) = 50 + 150 + 50 = 250
    result = get_ride_price(10.0, 25.0)

    assert result["success"] is True
    assert result["estimated_fare"] == 250.0
    assert result["currency"] == "INR"
    assert result["pricing_model"] == "demo_estimate"


def test_price_calculation_with_string_inputs():
    """Test: String inputs are properly converted to floats."""
    from ride_agent.tools.ride_price import get_ride_price

    result = get_ride_price("10", "25")

    assert result["success"] is True
    assert result["estimated_fare"] == 250.0


def test_price_calculation_with_decimal_inputs():
    """Test: Decimal distance and duration are handled correctly."""
    from ride_agent.tools.ride_price import get_ride_price

    # 5.5 km, 12.5 minutes
    # Expected: 50 + (5.5 × 15) + (12.5 × 2) = 50 + 82.5 + 25 = 157.5
    result = get_ride_price(5.5, 12.5)

    assert result["success"] is True
    assert result["estimated_fare"] == 157.5


def test_price_calculation_zero_distance():
    """Test: Zero distance is valid (e.g., delivery at same location)."""
    from ride_agent.tools.ride_price import get_ride_price

    # 0 km, 5 minutes (base + time only)
    # Expected: 50 + 0 + 10 = 60
    result = get_ride_price(0.0, 5.0)

    assert result["success"] is True
    assert result["estimated_fare"] == 60.0


def test_price_calculation_zero_duration():
    """Test: Zero duration is valid (instantaneous trip)."""
    from ride_agent.tools.ride_price import get_ride_price

    # 10 km, 0 minutes (base + distance only)
    # Expected: 50 + 150 + 0 = 200
    result = get_ride_price(10.0, 0.0)

    assert result["success"] is True
    assert result["estimated_fare"] == 200.0


def test_price_calculation_both_zero():
    """Test: Both distance and duration zero (base fare only)."""
    from ride_agent.tools.ride_price import get_ride_price

    # Expected: 50
    result = get_ride_price(0.0, 0.0)

    assert result["success"] is True
    assert result["estimated_fare"] == 50.0


def test_price_calculation_large_values():
    """Test: Large distance and duration are handled correctly."""
    from ride_agent.tools.ride_price import get_ride_price

    # 500 km, 600 minutes
    # Expected: 50 + (500 × 15) + (600 × 2) = 50 + 7500 + 1200 = 8750
    result = get_ride_price(500.0, 600.0)

    assert result["success"] is True
    assert result["estimated_fare"] == 8750.0


def test_price_calculation_rounding():
    """Test: Fare is rounded to 2 decimal places."""
    from ride_agent.tools.ride_price import get_ride_price

    # 3.333 km, 7.777 minutes
    # Calculation: 50 + (3.333 × 15) + (7.777 × 2) = 50 + 49.995 + 15.554 = 115.549
    # Should round to 115.55
    result = get_ride_price(3.333, 7.777)

    assert result["success"] is True
    assert result["estimated_fare"] == pytest.approx(115.55, 0.01)


def test_negative_distance():
    """Test 4: Negative distance is rejected."""
    from ride_agent.tools.ride_price import get_ride_price

    result = get_ride_price(-10.0, 25.0)

    assert result["success"] is False
    assert "Distance cannot be negative" in result["error"]


def test_negative_duration():
    """Test 5: Negative duration is rejected."""
    from ride_agent.tools.ride_price import get_ride_price

    result = get_ride_price(10.0, -25.0)

    assert result["success"] is False
    assert "Duration cannot be negative" in result["error"]


def test_non_numeric_distance():
    """Test 6: Non-numeric distance is rejected."""
    from ride_agent.tools.ride_price import get_ride_price

    result = get_ride_price("not a number", 25.0)

    assert result["success"] is False
    assert "Invalid input" in result["error"]
    assert "must be numeric" in result["error"]


def test_non_numeric_duration():
    """Test 6: Non-numeric duration is rejected."""
    from ride_agent.tools.ride_price import get_ride_price

    result = get_ride_price(10.0, "not a number")

    assert result["success"] is False
    assert "Invalid input" in result["error"]
    assert "must be numeric" in result["error"]


def test_price_calculation_deterministic():
    """Test 7: Pricing calculation is deterministic."""
    from ride_agent.tools.ride_price import get_ride_price

    # Call the same calculation multiple times
    result1 = get_ride_price(10.0, 25.0)
    result2 = get_ride_price(10.0, 25.0)
    result3 = get_ride_price(10.0, 25.0)

    # All should return identical results
    assert result1 == result2 == result3
    assert result1["estimated_fare"] == 250.0


@pytest.mark.anyio
async def test_mcp_server_get_ride_price_exposure():
    """Test: get_ride_price tool is properly registered on the MCPServer."""
    # 1. Verify tool exists in server's tool listing
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "get_ride_price" in tool_names
    assert "geocode_location" in tool_names  # Stage 1 still present
    assert "get_ride_time" in tool_names  # Stage 2 still present

    tool_obj = next(t for t in tools if t.name == "get_ride_price")
    assert "fare" in tool_obj.description.lower()
    assert "demo" in tool_obj.description.lower()

    # 2. Call tool through MCP server interface
    mcp_result = await mcp.call_tool(
        "get_ride_price",
        {
            "distance_km": 10.0,
            "duration_minutes": 25.0,
        },
    )
    assert not mcp_result.is_error
    assert len(mcp_result.content) > 0
    parsed_payload = json.loads(mcp_result.content[0].text)
    assert parsed_payload["success"] is True
    assert parsed_payload["estimated_fare"] == 250.0
    assert parsed_payload["currency"] == "INR"
    assert parsed_payload["pricing_model"] == "demo_estimate"


@pytest.mark.anyio
async def test_mcp_server_ride_price_error_handling():
    """Test: get_ride_price error is properly handled via MCP."""
    mcp_result = await mcp.call_tool(
        "get_ride_price",
        {
            "distance_km": -10.0,
            "duration_minutes": 25.0,
        },
    )
    assert not mcp_result.is_error
    parsed_payload = json.loads(mcp_result.content[0].text)
    assert parsed_payload["success"] is False
    assert "Distance cannot be negative" in parsed_payload["error"]
