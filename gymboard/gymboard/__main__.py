"""CLI entry point for Gymboard dashboard."""

import argparse
import os
import sys


def main():
    """Run the Gymboard dashboard."""
    parser = argparse.ArgumentParser(
        description="Gymboard - Web dashboard for Gymkhana training data"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--db-host",
        default=os.getenv("DB_HOST", "localhost"),
        help="Database host (default: localhost or DB_HOST env var)"
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=int(os.getenv("DB_PORT", "5432")),
        help="Database port (default: 5432 or DB_PORT env var)"
    )
    parser.add_argument(
        "--db-name",
        default=os.getenv("DB_NAME", "gymkhana"),
        help="Database name (default: gymkhana or DB_NAME env var)"
    )
    parser.add_argument(
        "--db-user",
        default=os.getenv("DB_USER", "postgres"),
        help="Database user (default: postgres or DB_USER env var)"
    )
    parser.add_argument(
        "--db-password",
        default=os.getenv("DB_PASSWORD", ""),
        help="Database password (default: empty or DB_PASSWORD env var)"
    )
    parser.add_argument(
        "--flatten-json",
        action="store_true",
        help="Flatten nested JSON in API responses"
    )

    args = parser.parse_args()

    # Set environment variables for the app
    os.environ["DB_HOST"] = args.db_host
    os.environ["DB_PORT"] = str(args.db_port)
    os.environ["DB_NAME"] = args.db_name
    os.environ["DB_USER"] = args.db_user
    os.environ["DB_PASSWORD"] = args.db_password
    if args.flatten_json:
        os.environ["DASHBOARD_FLATTEN_JSON"] = "true"

    print(f"Starting Gymboard on http://{args.host}:{args.port}")
    print(f"Connecting to database: {args.db_name}@{args.db_host}:{args.db_port}")

    try:
        import uvicorn
        from gymboard.app import app

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install gymboard", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error starting Gymboard: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
