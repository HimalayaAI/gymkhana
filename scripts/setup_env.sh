#!/bin/bash

# Gymkhana Environment Setup Script
# Installs dependencies and local packages

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "--- Setting up Gymkhana Environment ---"

# 1. Install Gymkhana in editable mode
echo "Installing gymkhana..."
python -m pip install -e '.[dev]'
python -m pip install -e ./gymboard

# 2. Check for .env file
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Copying .env.example if available..."
    if [ -f "example.env" ]; then
        cp example.env .env
    elif [ -f ".env.example" ]; then
        cp .env.example .env
    fi
    echo "Add provider credentials to .env before running live inference."
fi

echo "--- Setup Complete ---"
