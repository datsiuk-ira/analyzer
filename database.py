import sqlite3
import pandas as pd
from datetime import datetime
import json
from logger import logger
import os

class DatabaseManager:
    """
    Handles SQLite database connections and schema for paper trading.
    Uses Repository Pattern for data access.
    """
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self._conn = None
        self._init_db()

    def _get_connection(self):
        # SQLite connection must be created per thread if not using check_same_thread=False
        # Since Streamlit is multi-threaded, we handle it carefully.
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn

    def _init_db(self):
        """Initializes the database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Portfolios table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                initial_balance REAL NOT NULL,
                current_balance REAL NOT NULL,
                risk_per_trade REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER,
                symbol TEXT NOT NULL,
                entry_price REAL NOT NULL,
                position_size_usdt REAL NOT NULL,
                quantity REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                direction TEXT NOT NULL,
                status TEXT NOT NULL,
                entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                exit_time TIMESTAMP,
                exit_price REAL,
                pnl REAL,
                notes TEXT,
                score_breakdown TEXT,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
        ''')
        
        # Trade logs for "Near Miss" analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                event_type TEXT NOT NULL,
                price_reached REAL NOT NULL,
                distance_pct REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully.")

    def execute_query(self, query: str, params: tuple = ()):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor
        except Exception as e:
            logger.error(f"Database query error: {e}")
            raise e

    def fetch_all(self, query: str, params: tuple = ()) -> pd.DataFrame:
        conn = self._get_connection()
        try:
            return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"Database fetch error: {e}")
            return pd.DataFrame()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
