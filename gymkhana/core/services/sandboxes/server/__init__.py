"""Gymkhana REPL Server package.

Provides a REST API for executing Python code in a sandboxed environment.
"""

import uvicorn
from gymkhana.core.services.sandboxes.server.app import app

def run_server(host: str = "0.0.0.0", port: int = 5003, debug: bool = False):
    """Run the REPL server."""
    print(f"Starting Gymkhana REPL Server on http://{host}:{port}")
    # Flask app.run is used in app.py, but we can also use uvicorn
    # if we wanted to transition to FastAPI later.
    # For now, let's keep it compatible with the existing Flask app.
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Gymkhana REPL Server")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host")
    parser.add_argument("--port", type=int, default=5003, help="Binding port")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, debug=args.debug)
