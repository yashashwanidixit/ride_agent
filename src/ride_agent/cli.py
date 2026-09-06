"""CLI smoke-test interface for ride_agent tools."""

import argparse
import json
import logging
import sys
from typing import Optional, Sequence

from ride_agent.tools.geocoding import geocode_location

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the CLI for ride_agent tools."""
    parser = argparse.ArgumentParser(
        description="Ride Agent CLI - Test geocoding or LLM+MCP integration."
    )
    parser.add_argument(
        "location",
        nargs="?",
        default="",
        help="Location or place name to geocode (e.g., 'IIT Patna'). Or natural language request for LLM mode.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use the Stage 6 sequential LLM+MCP agent loop instead of direct geocoding.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:3b",
        help="Ollama model to use (default: qwen2.5:3b).",
    )

    args = parser.parse_args(argv)

    if args.llm:
        # Stage 6: sequential LLM+MCP agent loop
        _run_llm_integration(args.location, args.model)
    else:
        # Original Stage 1: Direct geocoding
        result = geocode_location(args.location)
        print(json.dumps(result, indent=2))


def _run_llm_integration(user_request: str, model_name: str) -> None:
    """Run the Stage 6 sequential LLM + MCP agent loop."""

    from ride_agent.agent import Stage5Orchestrator
    from ride_agent.agent.llm import OllamaProvider

    print("\n" + "=" * 70)
    print("STAGE 6: SEQUENTIAL AGENT LOOP (LLM + MCP)")
    print("=" * 70)
    print(f"\nUser Request:\n{user_request}\n")

    # Concrete provider is created at the application/CLI boundary.
    provider = OllamaProvider(model_name=model_name)

    orchestrator = Stage5Orchestrator(
        llm_provider=provider
    )

    try:
        if not orchestrator.initialize():
            print("\nFailed to initialize. Check that Ollama is running.")
            sys.exit(1)

        tools = orchestrator.mcp_client.discover_tools()

        print(f"\nMCP Tools Discovered: ({len(tools)})")
        for tool in tools:
            print(f"  - {tool.name}")

        print(f"\nProcessing request through LLM ({model_name})...")
        print("(Watch the log lines below for each tool call in the sequence.)\n")

        final_answer = orchestrator.process_user_request(user_request)

        print("\n" + "-" * 70)
        print("Final answer:")
        print("-" * 70)
        print(final_answer)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    finally:
        orchestrator.cleanup()
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()