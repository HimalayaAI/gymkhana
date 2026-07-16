#!/usr/bin/env python3
"""Drop and recreate all tables with updated schema.

This script drops all tables and recreates them from schema.sql.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import asyncpg


async def recreate_tables(
    host: str = "localhost",
    port: int = 5433,
    database: str = "gymkhana",
    user: str = "db_user",
    password: str = "db_pwd@123",
):
    """Drop and recreate all tables."""

    print(f"Connecting to database '{database}' on {host}:{port}...")
    conn = await asyncpg.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )

    try:
        print("✓ Connected to database")

        # Drop all tables in reverse dependency order
        print("\n🗑️  Dropping existing tables...")
        tables_to_drop = [
            "sub_agent_calls",
            "executions",
            "turns",
            "sandbox_sessions",
            "rollouts",
            "trajectories",
            "rollout_groups",
            "sharegpt_datasets",
            "requests",
        ]

        for table in tables_to_drop:
            try:
                await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                print(f"  ✓ Dropped {table}")
            except Exception as e:
                print(f"  ⚠️  Error dropping {table}: {e}")

        # Read and apply schema
        schema_path = project_root / "gymkhana" / "core" / "services" / "storage" / "schema.sql"
        print(f"\n📝 Applying schema from {schema_path}...")

        with open(schema_path, "r") as f:
            schema_sql = f.read()

        await conn.execute(schema_sql)
        print("✓ Schema applied successfully")

        # Verify tables were created
        print("\n✅ Verification:")
        result = await conn.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)

        for row in result:
            print(f"  ✓ {row['table_name']}")

        # Check for reasoning_content column in requests table
        result = await conn.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'requests'
            ORDER BY ordinal_position
        """)

        print("\n📊 Requests table columns:")
        has_reasoning_tokens = False
        has_reasoning_content = False
        for row in result:
            if row['column_name'] == 'reasoning_tokens':
                has_reasoning_tokens = True
                print(f"  ✓ {row['column_name']} ({row['data_type']}) ← NEW!")
            elif row['column_name'] == 'reasoning_content':
                has_reasoning_content = True
                print(f"  ✓ {row['column_name']} ({row['data_type']}) ← NEW!")
            else:
                print(f"    {row['column_name']} ({row['data_type']})")

        if has_reasoning_tokens and has_reasoning_content:
            print("\n✅ reasoning_tokens and reasoning_content columns successfully added to requests!")
        else:
            print(f"\n⚠️  WARNING: Missing columns in requests - reasoning_tokens: {has_reasoning_tokens}, reasoning_content: {has_reasoning_content}")

        # Check for reasoning_content column in turns table
        result = await conn.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'turns'
            ORDER BY ordinal_position
        """)

        print("\n📊 Turns table columns:")
        has_reasoning_content_turns = False
        for row in result:
            if row['column_name'] == 'reasoning_content':
                has_reasoning_content_turns = True
                print(f"  ✓ {row['column_name']} ({row['data_type']}) ← NEW!")
            else:
                print(f"    {row['column_name']} ({row['data_type']})")

        if has_reasoning_content_turns:
            print("\n✅ reasoning_content column successfully added to turns!")
        else:
            print("\n⚠️  WARNING: reasoning_content column not found in turns!")

    finally:
        await conn.close()
        print("\n✓ Database connection closed")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Drop and recreate database tables")
    parser.add_argument(
        "--host",
        default=os.getenv("DB_HOST", "localhost"),
        help="Database host (default: from DB_HOST env or localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("DB_PORT", "5433")),
        help="Database port (default: from DB_PORT env or 5433)"
    )
    parser.add_argument(
        "--database",
        default=os.getenv("DB_NAME", "gymkhana"),
        help="Database name (default: from DB_NAME env or gymkhana)"
    )
    parser.add_argument(
        "--user",
        default=os.getenv("DB_USER", "db_user"),
        help="Database user (default: from DB_USER env or db_user)"
    )
    parser.add_argument(
        "--password",
        default=os.getenv("DB_PASSWORD", "db_pwd@123"),
        help="Database password (default: from DB_PASSWORD env)"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    if not args.confirm:
        print("⚠️  WARNING: This will DROP ALL TABLES and recreate them!")
        print(f"Database: {args.database} on {args.host}:{args.port}")

        response = input("\nAre you sure you want to continue? (yes/no): ")
        if response.lower() not in ("yes", "y"):
            print("Aborted.")
            sys.exit(0)

    asyncio.run(recreate_tables(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
    ))


if __name__ == "__main__":
    main()
