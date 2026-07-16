#!/bin/bash

# Gymkhana Service Management Script
# Starts Docker DB, Sandbox REPL Server, and Dashboard
# NOTE: Activate your conda environment before running this script

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Create directories for logs and runtime files
mkdir -p logs run

# Load .env if exists
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

DB_PORT=${DB_PORT:-5433}
SANDBOX_PORT=${SANDBOX_PORT:-5003}
DASHBOARD_PORT=${DASHBOARD_PORT:-8000}
export DB_PORT SANDBOX_PORT DASHBOARD_PORT

echo "=== Gymkhana Service Startup ==="
echo ""

echo "--- Stopping existing sandbox and dashboard ---"
# Stop by saved PIDs
for pidfile in run/sandbox.pid run/dashboard.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping PID $pid ($pidfile)..."
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
done
# Stop anything still bound to sandbox/dashboard ports
for port in $SANDBOX_PORT $DASHBOARD_PORT; do
    pid=$(lsof -nP -iTCP:$port -sTCP:LISTEN -t 2>/dev/null | head -1)
    if [ -n "$pid" ]; then
        echo "Stopping process $pid on port $port..."
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
    fi
done
sleep 1
echo ""
echo "--- Checking Services ---"

# Helper function to check if a port is in use
is_port_in_use() {
    lsof -nP -iTCP:$1 -sTCP:LISTEN > /dev/null 2>&1
}

# Check if Docker daemon is running (required for DB)
docker_available() {
    docker info >/dev/null 2>&1
}

compose() {
    if docker compose version >/dev/null 2>&1; then
        docker compose "$@"
    else
        docker-compose "$@"
    fi
}

# 1. Start Docker Containers (Database)
DB_AVAILABLE=false
if is_port_in_use $DB_PORT; then
    echo "PostgreSQL is already running on port $DB_PORT."
    DB_AVAILABLE=true
elif docker_available; then
    echo "Starting PostgreSQL (Docker)..."
    if compose -f gymkhana/core/services/storage/docker/docker-compose.yaml up -d 2>/dev/null; then
        echo "Waiting for DB to be ready..."
        sleep 2
        DB_AVAILABLE=true
    else
        echo "Warning: docker-compose failed. Database may not be available."
    fi
else
    echo "Docker is not running. Database and dashboard data will be unavailable."
    echo "  Start Docker Desktop (or the Docker daemon), then run this script again for DB + dashboard data."
fi

# 2. Initialize Database Schema (only meaningful when DB is or was started)
if [ "$DB_AVAILABLE" = true ]; then
    echo "Initializing Database Schema..."
    if python scripts/init_db.py 2>&1 | tee -a logs/init_db.log; then
        echo "✓ Database schema initialized successfully"
    else
        echo "⚠ Database schema init returned non-zero (might already be initialized)"
    fi
else
    echo "⚠ Skipping database initialization (Docker not available)"
fi
echo ""

# 3. Start Sandbox REPL Server
echo "Starting Sandbox REPL Server on port $SANDBOX_PORT..."
python -m gymkhana.core.services.sandboxes.server --port "$SANDBOX_PORT" > logs/sandbox_server.log 2>&1 &
echo $! > run/sandbox.pid
sleep 2

# Verify sandbox started
if lsof -nP -iTCP:$SANDBOX_PORT -sTCP:LISTEN > /dev/null 2>&1; then
    echo "✓ Sandbox REPL Server running on port $SANDBOX_PORT"
else
    echo "✗ Failed to start Sandbox REPL Server (check logs/sandbox_server.log)"
fi
echo ""

# 4. Start Gymboard Dashboard
echo "Starting Gymboard on port $DASHBOARD_PORT..."
# Use the source-tree import path so this also works before editable install.
python -m uvicorn gymboard.gymboard.app:app --host 0.0.0.0 --port "$DASHBOARD_PORT" > logs/dashboard.log 2>&1 &
echo $! > run/dashboard.pid
sleep 2

# Verify dashboard started
if lsof -nP -iTCP:$DASHBOARD_PORT -sTCP:LISTEN > /dev/null 2>&1; then
    echo "✓ Dashboard running on port $DASHBOARD_PORT"
else
    echo "✗ Failed to start Dashboard (check logs/dashboard.log)"
fi
echo ""

echo "=== Services Ready ==="
echo "Database:  localhost:$DB_PORT $([ "$DB_AVAILABLE" = true ] && echo '✓' || echo '✗ (start Docker)')"
echo "Sandbox:   localhost:$SANDBOX_PORT"
echo "Dashboard: http://localhost:$DASHBOARD_PORT"
echo ""
echo "Logs: logs/sandbox_server.log, logs/dashboard.log, logs/init_db.log"
echo "To stop: kill \$(cat run/sandbox.pid run/dashboard.pid 2>/dev/null)"
echo ""

# 5. Open Dashboard in browser (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Opening Dashboard in browser..."
    open "http://localhost:$DASHBOARD_PORT"
fi
