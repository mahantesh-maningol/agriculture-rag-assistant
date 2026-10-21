
# backend/database.py

import os
from contextlib import contextmanager

from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "agriculture_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "10"))

if not DB_PASSWORD:
    raise RuntimeError("DB_PASSWORD is missing from environment variables")

DATABASE_URL = (
    f"host={DB_HOST} "
    f"port={DB_PORT} "
    f"dbname={DB_NAME} "
    f"user={DB_USER} "
    f"password={DB_PASSWORD}"
)

pool = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=POOL_MIN_SIZE,
    max_size=POOL_MAX_SIZE,
    kwargs={
        "row_factory": dict_row,
        "connect_timeout": 5,
    },
    open=False,
)


def open_database_pool() -> None:
    """Open the pool and wait for an initial connection."""
    if pool.closed:
        pool.open(wait=True, timeout=10)


def close_database_pool() -> None:
    """Close pooled connections during application shutdown."""
    if not pool.closed:
        pool.close()


@contextmanager
def get_connection():
    """
    Borrow a connection from the pool.

    The connection is returned to the pool when the context exits.
    Use parameterized SQL queries for user/model-provided values.
    """
    with pool.connection() as conn:
        yield conn


def check_database_connection() -> bool:
    """Check whether PostgreSQL can execute a simple query."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS connected")
            result = cur.fetchone()

    return result["connected"] == 1
