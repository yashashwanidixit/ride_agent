# Ride Agent

A local-first AI ride agent built around an **LLM + MCP architecture**. The agent interprets natural-language ride requests, selects the required tools, executes them through an MCP server.

## Architecture

```text
User / CLI
    ↓
Conversation Manager / Orchestrator
    ├── LLM Provider
    │     ├── Ollama + Qwen 2.5:3B (local)
    │     └── OpenAI-compatible provider
    │
    └── MCP Client
          ↓
       MCP Server
       ├── geocode_location → OpenStreetMap Nominatim
       ├── get_ride_time    → OSRM
       └── get_ride_price   → Deterministic fare calculation
```

- **Orchestrator / Conversation Manager:** owns conversation history, the agent loop, and multi-turn continuity.
- **LLM Provider:** handles model-specific API calls and message/tool formatting. The agent logic stays provider-independent.
- **MCP Client:** communicates with the MCP server through MCP instead of calling tools directly.
- **MCP Server:** exposes the ride capabilities as MCP tools.

## Ride Information APIs

A direct live ride-provider API for fare/booking (such as Uber/Ola) was not available for this implementation because the required API access/authentication was not accessible.

Therefore:

- **Location coordinates:** OpenStreetMap Nominatim
- **Distance & travel time:** OSRM routing
- **Fare:** deterministic demo calculation based on distance and duration

```text
Fare = ₹50 + (distance_km × ₹15) + (duration_minutes × ₹2)
```

The fare is an **estimate**, not a live ride-provider quote.

## Agent Behavior

The agent can:

- select only the tools required for a request,
- chain multiple tools sequentially,
- use previous tool results to determine the next action,


## Example

```text
User: I want to go to Patna airport.
Agent: What is the pickup location?

User: IIT Patna

LLM
 ↓
geocode_location(IIT Patna)
 ↓
geocode_location(Patna Airport)
 ↓
get_ride_time(...)
 ↓
get_ride_price(...)
 ↓
Final natural-language answer
```

## Run Locally

Start the MCP server/Inspector:

```bash
mcp dev src/ride_agent/server.py
```

Run with local Qwen through Ollama:

```bash
python -m ride_agent.cli --llm --provider ollama --model "qwen2.5:3b" "I want to go from IIT Patna to Patna airport. Tell me the approximate distance, travel time, and fare."
```

Run with an OpenAI-compatible provider:

```bash
python -m ride_agent.cli --llm --provider openai --model "YOUR_MODEL_NAME" "I want to go from IIT Patna to Patna airport. Tell me the approximate distance, travel time, and fare."
```

Keep API credentials in a local `.env` file and never commit secrets.

## Project Goal

Demonstrate **model-driven tool selection, sequential tool use, MCP-based execution** while keeping the LLM provider replaceable and the underlying ride tools independent of the model.
