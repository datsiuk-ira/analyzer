import logging
import sys
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name: str = "TradingAssistant", level: int = None):
    """
    Sets up a professional logger with console + rotating file handler.
    Log level controlled by LOG_LEVEL env variable (DEBUG, INFO, WARNING, ERROR).
    """
    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-5s | %(name)s | %(filename)s:%(lineno)d | %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
        
        # Rotating file handler (5MB per file, keep 3 backups)
        try:
            log_dir = os.path.dirname(os.path.abspath(__file__))
            log_file = os.path.join(log_dir, "trading.log")
            file_handler = RotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)  # File always captures DEBUG
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create file handler: {e}")
        
    return logger


# Global logger instance
logger = setup_logger()
