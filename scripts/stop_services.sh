#!/bin/bash

# Gymkhana Service Stop Script
# Stops Sandbox REPL Server and Dashboard

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Stopping Gymkhana Services ==="
echo ""

# Stop services by PID files
STOPPED=0
for pidfile in run/sandbox.pid run/dashboard.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        service_name=$(basename "$pidfile" .pid)

        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $service_name (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 1

            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Force stopping $service_name..."
                kill -9 "$pid" 2>/dev/null || true
            fi

            STOPPED=$((STOPPED + 1))
        else
            echo "Service $service_name not running (stale PID file)"
        fi

        rm -f "$pidfile"
    fi
done

# Also check ports and kill anything still bound
SANDBOX_PORT=${SANDBOX_PORT:-5003}
DASHBOARD_PORT=${DASHBOARD_PORT:-8000}

for port in $SANDBOX_PORT $DASHBOARD_PORT; do
    pid=$(lsof -nP -iTCP:$port -sTCP:LISTEN -t 2>/dev/null | head -1)
    if [ -n "$pid" ]; then
        echo "Stopping process on port $port (PID: $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
        STOPPED=$((STOPPED + 1))
    fi
done

echo ""
if [ $STOPPED -gt 0 ]; then
    echo "✓ Stopped $STOPPED service(s)"
else
    echo "No services were running"
fi

echo ""
echo "Note: Database (Docker) is still running. To stop it:"
echo "  docker compose -f gymkhana/core/services/storage/docker/docker-compose.yaml down"
