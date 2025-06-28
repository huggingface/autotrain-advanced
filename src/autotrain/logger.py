import logging
import sys
from datetime import datetime
import os

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Name of the logger
        log_file: Path to log file (optional)
        level: Logging level
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    
    if log_file:
        
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_training_logger(output_dir):
    """
    Get a logger specifically for training progress.
    
    Args:
        output_dir: Directory to save logs
    """
    
    log_dir = os.path.join(output_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    return setup_logger("training", log_file)


logger = setup_logger("autotrain") 