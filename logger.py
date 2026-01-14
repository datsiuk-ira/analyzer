import logging
import sys

def setup_logger(name: str = "TradingAssistant", level: int = logging.INFO):
    """
    Sets up a professional logger for the application.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

# Global logger instance
logger = setup_logger()
