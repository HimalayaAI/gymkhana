"""Main entry point for Gymkhana REPL Server."""

from gymkhana.core.services.sandboxes.server import run_server
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gymkhana REPL Server")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host")
    parser.add_argument("--port", type=int, default=5003, help="Binding port")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, debug=args.debug)
