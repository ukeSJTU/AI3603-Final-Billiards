"""
logger.py - Simple logging configuration

Usage:
    from logger import logger

    logger.info("Information")
    logger.error("Error")
"""

import sys
from pathlib import Path

from loguru import logger

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Remove default handler
logger.remove()

# Console output
logger.add(sys.stderr, level="INFO")

# File output (named by timestamp)
logger.add(
    "logs/{time:YYYY-MM-DD_HH-mm-ss}.log",
    level="DEBUG",
    rotation="10 MB",  # Rotate log file when it reaches 10 MB
    encoding="utf-8",
    retention="30 days",  # Keep logs for 30 days
)

if __name__ == "__main__":
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
