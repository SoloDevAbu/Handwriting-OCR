import sys
from pathlib import Path
from loguru import logger
from config import settings, LOG_DIR

def setup_logger():
    """Configure the logging system"""
    # Remove default logger
    logger.remove()
    
    # Add console logger with custom format
    logger.add(
        sys.stderr,
        format=settings.LOG_FORMAT,
        level=settings.LOG_LEVEL,
        colorize=True
    )
    
    # Add file logger for errors
    logger.add(
        LOG_DIR / "error.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="ERROR",
        rotation="1 day",
        compression="zip"
    )
    
    # Add file logger for all messages
    logger.add(
        LOG_DIR / "app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level=settings.LOG_LEVEL,
        rotation="1 day",
        retention="7 days"
    )

def log_request(method: str, endpoint: str, params: dict = None):
    """Log incoming API requests"""
    logger.info(f"Request: {method} {endpoint} - Params: {params or {}}")

def log_error(exc: Exception, context: str = None):
    """Log errors with context"""
    error_msg = f"{type(exc).__name__}: {str(exc)}"
    if context:
        error_msg = f"{context} - {error_msg}"
    logger.error(error_msg, exc_info=True)

def log_model_prediction(image_shape: tuple, num_regions: int, confidence: float = None):
    """Log model prediction metrics"""
    logger.info(
        f"Prediction - Image shape: {image_shape}, "
        f"Detected regions: {num_regions}, "
        f"Confidence: {confidence or 'N/A'}"
    )