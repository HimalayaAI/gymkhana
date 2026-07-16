#!/bin/bash
# Start the Gymkhana REPL server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "Starting Gymkhana REPL server in Docker..."

    # Build and run with docker-compose if available
    if command -v docker-compose &> /dev/null; then
        docker-compose up --build -d
        echo "Server started. View logs with: docker-compose logs -f"
    else
        # Fallback to plain docker
        docker build -t gymkhana-repl .
        docker run -d --name gymkhana-repl -p 5003:5003 --memory=2g gymkhana-repl
        echo "Server started. View logs with: docker logs -f gymkhana-repl"
    fi

    echo ""
    echo "REPL server running at http://localhost:5003"
    echo "Health check: curl http://localhost:5003/health"
else
    echo "Docker not found. Running server directly..."
    echo "Note: Make sure numpy, pandas, sympy, scipy are installed."
    python server.py --port 5003
fi
