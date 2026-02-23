import sqlite3
import pandas as pd
from datetime import datetime
import json
from logger import logger
import os

import time
import random
import functools

def retry_db_transaction(max_retries=5, initial_delay=0.05, max_delay=0.2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e).lower():
                        last_exception = e
                        delay = random.uniform(initial_delay, max_delay)
                        logger.warning(f"Database locked, retrying {i+1}/{max_retries} after {delay:.3f}s...")
                        time.sleep(delay)
                    else:
                        raise e
            logger.error(f"Max retries reached for database transaction. Last error: {last_exception}")
            raise last_exception
        return wrapper
    return decorator

# Global flag to ensure we only log "initialized" once per process
_DB_INITIALIZED = False

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
        # check_same_thread=False is used to allow sharing between UI and background scanner.
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        return self._conn

    def _init_db(self):
        """Initializes the database schema."""
        global _DB_INITIALIZED
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        
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
                leverage REAL DEFAULT 1.0,
                max_drawdown_price REAL,
                max_profit_price REAL,
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

        # Signal History for Confidence Scoring
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                score REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # FIX #16: Add indexes for frequently queried columns
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol_status ON trades(symbol, status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_portfolio ON trades(portfolio_id, status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_history_symbol ON signal_history(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_logs_trade ON trade_logs(trade_id, timestamp)')
        
        # HOTFIX 2.1: Global symbol lock table.
        # Prevents the bot from opening multiple trades on the same symbol/timeframe
        # pair within a cooldown window (e.g. after a Stop-Loss or Partial TP).
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_state (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol              TEXT    NOT NULL,
                timeframe           TEXT    NOT NULL,
                lock_until_timestamp REAL   NOT NULL,  -- Unix epoch seconds
                reason              TEXT,
                UNIQUE(symbol, timeframe) ON CONFLICT REPLACE
            )
        ''')
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_trading_state_lookup '
            'ON trading_state(symbol, timeframe, lock_until_timestamp)'
        )
        
        conn.commit()
        # Only log if it's the first time we're initializing in this process
        if not _DB_INITIALIZED:
            logger.info("Database initialized successfully.")
            _DB_INITIALIZED = True

    @retry_db_transaction()
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

    # ─────────────────────────────────────────────────────────────────
    # HOTFIX 2.1: Symbol lock helpers (global overtrading protection).
    # Used by live bot AND by the backtester's in-memory equivalent.
    # ─────────────────────────────────────────────────────────────────

    @retry_db_transaction()
    def lock_symbol(self, symbol: str, timeframe: str,
                    minutes: float, reason: str = '') -> None:
        """
        Locks ``symbol/timeframe`` for ``minutes`` minutes.
        Any new lock on the same pair replaces the old one (upsert via UNIQUE
        ON CONFLICT REPLACE defined on the table).

        Call this after a Stop-Loss exit or when taking a Partial TP.
        """
        import time as _time
        lock_until = _time.time() + minutes * 60.0
        self.execute_query(
            '''
            INSERT INTO trading_state (symbol, timeframe, lock_until_timestamp, reason)
            VALUES (?, ?, ?, ?)
            ''',
            (symbol, timeframe, lock_until, reason)
        )
        logger.info(
            f"[LOCK] {symbol}/{timeframe} locked for {minutes:.1f}m. Reason: {reason}"
        )

    def is_symbol_locked(self, symbol: str, timeframe: str) -> bool:
        """
        Returns True if the symbol/timeframe pair is currently locked
        (i.e. the lock has not yet expired).

        Call this early in the pipeline to skip data fetching entirely.
        """
        import time as _time
        now = _time.time()
        result = self.fetch_all(
            '''
            SELECT lock_until_timestamp FROM trading_state
            WHERE symbol = ? AND timeframe = ? AND lock_until_timestamp > ?
            LIMIT 1
            ''',
            (symbol, timeframe, now)
        )
        locked = not result.empty
        if locked:
            remaining = result.iloc[0]['lock_until_timestamp'] - now
            logger.debug(
                f"[LOCK] {symbol}/{timeframe} is locked for {remaining/60:.1f}m more."
            )
        return locked

    @retry_db_transaction()
    def unlock_symbol(self, symbol: str, timeframe: str) -> None:
        """Manually remove a lock (e.g. on graceful shutdown or strategy reset)."""
        self.execute_query(
            'DELETE FROM trading_state WHERE symbol = ? AND timeframe = ?',
            (symbol, timeframe)
        )
        logger.info(f"[LOCK] {symbol}/{timeframe} manually unlocked.")
