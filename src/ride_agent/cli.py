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
        help="Use Stage 5 LLM+MCP integration instead of direct geocoding.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:3b",
        help="Ollama model to use (default: qwen2.5:3b).",
    )
    
    args = parser.parse_args(argv)

    if args.llm:
        # Stage 5: LLM+MCP integration
        _run_llm_integration(args.location, args.model)
    else:
        # Original Stage 1: Direct geocoding
        result = geocode_location(args.location)
        print(json.dumps(result, indent=2))


def _run_llm_integration(user_request: str, model_name: str) -> None:
    """Run the Stage 5 LLM+MCP integration.

    Args:
        user_request: Natural language request from the user.
        model_name: Name of the Ollama model to use.
    """
    from ride_agent.agent import Stage5Orchestrator

    print("\n" + "=" * 70)
    print("STAGE 5: LLM + MCP INTEGRATION")
    print("=" * 70)
    print(f"\nUser Request:\n{user_request}\n")

    orchestrator = Stage5Orchestrator(model_name=model_name)

    try:
        # Initialize
        if not orchestrator.initialize():
            print("\n Failed to initialize. Check that Ollama is running.")
            sys.exit(1)

        # Print discovered tools
        tools = orchestrator.mcp_client.discover_tools()
        print(f"\nMCP Tools Discovered: ({len(tools)})")
        for tool in tools:
            print(f"  - {tool.name}")

        # Process request
        print(f"\nProcessing request through LLM ({orchestrator.llm.model_name})...\n")
        result = orchestrator.process_user_request(user_request)

        if result:
            print("\n" + "-" * 70)
            print("MCP Tool Result:")
            print("-" * 70)
            print(json.dumps(result, indent=2))
        else:
            print("\n(No tool was invoked)")

    except Exception as e:
        print(f"\n Error: {e}")
        sys.exit(1)
    finally:
        orchestrator.cleanup()
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
