"""Utilities for database management and testing.

Contains setup scripts and validation utilities migrated from the legacy db/ directory.
"""
from __future__ import annotations

import asyncio
import logging
import os
import json
from typing import Any, Dict, List, Optional
from uuid import UUID

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

from gymkhana.core.services.storage.env_storage import EnvStorageService, STORAGE_AVAILABLE


async def setup_gymkhana_db(
    host: str = "localhost",
    port: int = 5432,
    user: str = "postgres",
    password: str = "",
    db_name: str = "gymkhana"
) -> bool:
    """Initialize the Gymkhana database and tables.

    Uses EnvStorageService to ensure all schema definitions are applied.
    """
    if not STORAGE_AVAILABLE:
        print("Error: agent_storage or asyncpg not available. Cannot setup database.")
        return False

    storage = EnvStorageService(
        host=host,
        port=port,
        user=user,
        password=password,
        db_name=db_name
    )

    try:
        await storage.initialize()
        print(f"Successfully initialized Gymkhana database: {db_name}")
        await storage.close()
        return True
    except Exception as e:
        print(f"Failed to setup database: {e}")
        return False


async def check_database_status(
    host: str = "localhost",
    port: int = 5432,
    user: str = "postgres",
    password: str = "",
    db_name: str = "gymkhana"
) -> None:
    """Print status and counts for Gymkhana database tables."""
    if not STORAGE_AVAILABLE:
        print("Error: Storage dependencies not available.")
        return

    storage = EnvStorageService(
        host=host,
        port=port,
        user=user,
        password=password,
        db_name=db_name
    )

    try:
        await storage.initialize()
        db = storage._db

        tables = ['requests', 'trajectories', 'turns', 'executions', 'sharegpt_datasets']
        print(f"\n--- Gymkhana Database Table Counts ({db_name}) ---")
        for table in tables:
            try:
                res = await db.fetch_one(f"SELECT COUNT(*) as count FROM {table}")
                count = res['count'] if res else 0
                print(f"{table:20}: {count}")
            except Exception:
                print(f"{table:20}: TABLE NOT FOUND")

        print("\n--- Latest Trajectory ---")
        try:
            res = await db.fetch_one("SELECT id, task_id, environment, success, created_at FROM trajectories ORDER BY created_at DESC LIMIT 1")
            if res:
                print(dict(res))
            else:
                print("No trajectories found.")
        except Exception as e:
            print(f"Error fetching latest trajectory: {e}")

        await storage.close()
    except Exception as e:
        print(f"Database check failed: {e}")


async def test_pg_vector_support(
    host: str = "localhost",
    port: int = 5432,
    user: str = "postgres",
    password: str = "",
    db_name: str = "gymkhana"
) -> bool:
    """Test if pgvector extension is available and working."""
    if not HAS_ASYNCPG:
        print("Error: asyncpg not installed.")
        return False

    try:
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_name
        )

        # Check for extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Simple vector operation
        res = await conn.fetchval("SELECT '[1,2,3]'::vector <=> '[3,2,1]'::vector")
        print(f"pgvector check successful. Sample distance: {res}")

        await conn.close()
        return True
    except Exception as e:
        print(f"pgvector test failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage for CLI setup
    logging.basicConfig(level=logging.ERROR)
    host = os.environ.get("DB_HOST", "localhost")
    port = int(os.environ.get("DB_PORT", 5432))
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "")
    db_name = os.environ.get("DB_NAME", "gymkhana")

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        asyncio.run(check_database_status(host, port, user, password, db_name))
    else:
        print(f"Setting up database at {host}:{port}...")
        asyncio.run(setup_gymkhana_db(host, port, user, password, db_name))
