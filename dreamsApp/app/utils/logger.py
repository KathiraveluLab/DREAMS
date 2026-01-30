import logging
import os
from datetime import datetime

def setup_logger(name, log_dir='logs'):
    """
    Create a production-ready logger with file and console output.
    
    Args:
        name: Logger name (usually module name like 'fl_worker')
        log_dir: Directory to store log files
    
    Returns:
        Configured logger instance
    """
    
    # Ensure logs directory exists at project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    log_path = os.path.join(base_dir, log_dir)
    os.makedirs(log_path, exist_ok=True)
    
    # Create log file with date suffix
    log_file = os.path.join(log_path, f'{name}_{datetime.now().strftime("%Y-%m-%d")}.log')
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers on multiple calls
    if logger.handlers:
        return logger
    
    # File Handler - All logs
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Console Handler - INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Format
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
