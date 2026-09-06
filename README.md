# Ride Concierge AI Agent - Stage 1 (MCP Geocoding Server)

This repository contains **Stage 1** of the Ride Concierge AI Agent: a working Model Context Protocol (MCP) server exposing the `geocode_location` tool backed by OpenStreetMap Nominatim.

---

## 1. What Stage 1 Does

Stage 1 provides a reliable geocoding tool via the Model Context Protocol (MCP). It converts a human-readable location, landmark, or address into structured geographic coordinates (latitude and longitude) along with the formatted display name.

- Exposes the tool `geocode_location(location: str)` via an MCP server.
- Uses OpenStreetMap Nominatim API for geocoding.
- Automatically handles query normalization, in-memory caching, rate limiting, and network/HTTP errors without exposing internal tracebacks.

---

## 2. Architecture

```
User / Caller
      ↓
[MCP Host / Client]
      ↓ (stdio JSON-RPC)
[MCP Server (src/ride_agent/server.py)]
      ↓
[MCP Tool: geocode_location (src/ride_agent/tools/geocoding.py)]
      ↓
[Service: NominatimService (src/ride_agent/services/nominatim.py)]
   ├── In-Memory Cache (hits return immediately)
   ├── Rate Limiter (max 1 request/second)
   └── HTTP Client (identifiable User-Agent)
      ↓
OpenStreetMap Nominatim API (https://nominatim.openstreetmap.org/search)
```

### Component Separation
- **`src/ride_agent/tools/`**: MCP-facing tool registration and parameter/error shielding.
- **`src/ride_agent/services/`**: External API integration, rate limiting, in-memory query caching, and HTTP communication.
- **`src/ride_agent/server.py`**: MCP server initialization and protocol entry point.

---

## 3. Installation

Requires Python **>= 3.10**.

1. Clone or navigate to the repository:
   ```bash
   cd mycode/ride_agent
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```
   Or install requirements directly:
   ```bash
   pip install "mcp[cli]>=2.1.0" "httpx>=0.28.0" "python-dotenv>=1.0.0" pytest anyio
   ```

---

## 4. Environment Variables

Create a `.env` file or export the environment variable:

```bash
# Nominatim User-Agent identifier (required by OSM Nominatim usage policy)
RIDE_AGENT_USER_AGENT=ride-agent-hackathon/0.1
```

If not provided, the service defaults to `ride-agent-hackathon/0.1`.

---

## 5. How to Start the MCP Server

To run the MCP server over standard input/output (`stdio` transport):

```bash
python -m ride_agent.server
```

Or execute directly with Python:

```bash
python src/ride_agent/server.py
```

---

## 6. How to Open MCP Inspector

You can inspect and interact with the server in the browser using the official MCP CLI development inspector:

```bash
mcp dev src/ride_agent/server.py
```

This launches the web-based MCP Inspector where you will see the `geocode_location` tool listed.

---

## 7. How to Test `geocode_location`

### In the MCP Inspector UI:
1. Open the **Tools** tab in MCP Inspector.
2. Select `geocode_location`.
3. Provide the input:
   ```json
   {
     "location": "IIT Patna"
   }
   ```
4. Click **Run Tool** and inspect the returned coordinates.

### Via Python:
```python
from ride_agent.tools.geocoding import geocode_location

result = geocode_location("IIT Patna")
print(result)
```

---

## 8. How to Run Pytest

Run unit tests (all tests mock the external HTTP layer to prevent network dependency):

```bash
pytest -v
```

---

## 9. Example Input and Output

### Successful Geocode
**Input:**
```json
{
  "location": "IIT Patna"
}
```

**Output:**
```json
{
  "success": true,
  "latitude": 25.5424381,
  "longitude": 84.8516072,
  "display_name": "Indian Institute of Technology Patna, Bihta-Lai road, Bihta, Patna, Bihar, 801106, India"
}
```

### Unknown Location
**Input:**
```json
{
  "location": "kjsdhfkjsdhf92837498273498"
}
```

**Output:**
```json
{
  "success": false,
  "error": "Location not found: kjsdhfkjsdhf92837498273498"
}
```

### Empty Input
**Input:**
```json
{
  "location": ""
}
```

**Output:**
```json
{
  "success": false,
  "error": "Location cannot be empty"
}
```

---

## 10. Nominatim Usage & Rate-Limit Policy Note

This service strictly adheres to the [OpenStreetMap Nominatim Usage Policy](https://operations.osmfoundation.org/policies/nominatim/):
1. **Identifiable User-Agent**: Every request includes a distinct `User-Agent` header configured via `RIDE_AGENT_USER_AGENT`.
2. **Rate Limit**: Outgoing requests to Nominatim are throttled to a maximum rate of 1 request per second.
3. **Caching**: In-process memory caching prevents duplicate requests for identical location queries.
