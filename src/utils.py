"""
Utility functions for logging and helpers
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class PipelineLogger:
    """Centralized logging for the entire pipeline"""
    
    def __init__(self, log_file: str = "logs/pipeline.log", level: int = logging.INFO):
        """
        Initialize logger with file and console handlers
        
        Args:
            log_file: Path to log file
            level: Logging level (default: INFO)
        """
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger("AIDocumentPipeline")
        self.logger.setLevel(level)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
        
        # File handler
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def success(self, message: str):
        """Log success message (as info with emoji)"""
        self.logger.info(f"✓ {message}")
    
    def section(self, title: str):
        """Log section header"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"  {title}")
        self.logger.info(f"{'='*60}")


def get_logger() -> PipelineLogger:
    """Get or create pipeline logger instance"""
    return PipelineLogger()
