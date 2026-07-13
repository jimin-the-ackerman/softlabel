"""
Utility functions for ProGen framework.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler


def get_default_output_dir(data: str, llm_model: str) -> str:
    """Generate default output directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"./results_progen/{data}/{llm_model}/{timestamp}"


def save_config(args, output_dir: Path):
    """Save configuration arguments to a JSON file."""
    config = vars(args)
    config_file = output_dir / "config.json"
    
    # Convert Path objects to strings for JSON serialization
    config_serializable = {}
    for key, value in config.items():
        if isinstance(value, Path):
            config_serializable[key] = str(value)
        else:
            config_serializable[key] = value
    
    with open(config_file, 'w') as f:
        json.dump(config_serializable, f, indent=2, default=str)
    
    logging.info(f"Configuration saved to: {config_file}")


def setup_logging(output_dir: str = None):
    """Setup logging to both console and file."""
    # Create formatters
    console_formatter = logging.Formatter('%(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                      datefmt='%Y-%m-%d %H:%M:%S')
    
    # Create handlers
    console_handler = RichHandler()
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(console_handler)
    
    # Add file handler if output_dir is provided
    if output_dir:
        log_file = os.path.join(output_dir, 'progen.log')
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler) 