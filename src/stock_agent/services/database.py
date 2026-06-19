from contextlib import contextmanager

import psycopg

from stock_agent.config import get_settings


@contextmanager
def database_connection():
    with psycopg.connect(get_settings().database_url) as connection:
        yield connection
