#!/usr/bin/env python3
"""Check what columns exist in key tables."""

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


async def check_columns():
    """Check columns in requests, turns, and trajectories tables."""

    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5433"))
    database = os.getenv("DB_NAME", "gymkhana")
    user = os.getenv("DB_USER", "db_user")
    password = os.getenv("DB_PASSWORD", "db_pwd@123")

    conn = await asyncpg.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )

    try:
        for table in ["requests", "turns", "trajectories"]:
            print(f"\n{'='*60}")
            print(f"Table: {table}")
            print('='*60)

            # Get columns
            result = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = $1
                ORDER BY ordinal_position
            """, table)

            if not result:
                print(f"  ⚠️  Table '{table}' not found!")
                continue

            for row in result:
                nullable = "NULL" if row['is_nullable'] == 'YES' else "NOT NULL"
                print(f"  {row['column_name']:30} {row['data_type']:20} {nullable}")

            # Get row count
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            print(f"\n  Total rows: {count}")

            # If requests table, check for reasoning_content data
            if table == "requests" and count > 0:
                reasoning_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM requests WHERE reasoning_content IS NOT NULL"
                )
                print(f"  Rows with reasoning_content: {reasoning_count}")

            # If turns table, check for reasoning_content data
            if table == "turns" and count > 0:
                reasoning_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM turns WHERE reasoning_content IS NOT NULL"
                )
                print(f"  Rows with reasoning_content: {reasoning_count}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(check_columns())
