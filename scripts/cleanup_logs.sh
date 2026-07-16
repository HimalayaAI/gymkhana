#!/bin/bash

# Gymkhana Log Cleanup Script
# Archives or removes old log files

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Gymkhana Log Cleanup ==="
echo ""

# Check if logs directory exists
if [ ! -d "logs" ]; then
    echo "No logs directory found"
    exit 0
fi

# Count log files
LOG_COUNT=$(find logs -name "*.log" -type f 2>/dev/null | wc -l | tr -d ' ')

if [ "$LOG_COUNT" -eq 0 ]; then
    echo "No log files found"
    exit 0
fi

echo "Found $LOG_COUNT log file(s)"
echo ""

# Ask user what to do
echo "Options:"
echo "  1) Archive logs (create .tar.gz and remove originals)"
echo "  2) Delete logs (remove all .log files)"
echo "  3) Cancel"
echo ""
read -p "Choose option [1-3]: " choice

case $choice in
    1)
        # Archive logs
        ARCHIVE_NAME="logs/archive_$(date +%Y%m%d_%H%M%S).tar.gz"
        echo "Creating archive: $ARCHIVE_NAME"

        tar -czf "$ARCHIVE_NAME" logs/*.log 2>/dev/null

        if [ $? -eq 0 ]; then
            echo "✓ Archive created successfully"
            echo "Removing original log files..."
            rm -f logs/*.log
            echo "✓ Log files removed"
            echo ""
            echo "Archive location: $ARCHIVE_NAME"
        else
            echo "✗ Failed to create archive"
            exit 1
        fi
        ;;
    2)
        # Delete logs
        echo "Deleting all log files..."
        rm -f logs/*.log
        echo "✓ Log files deleted"
        ;;
    3)
        echo "Cancelled"
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "Cleanup complete"
