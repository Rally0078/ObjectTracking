import logging
from pathlib import Path
import os

def setup_logger(name: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    log_path = Path(os.path.abspath('./logs/latest.log'))
    log_path.parent.mkdir(exist_ok=True, parents=True)                    
    file_handler = logging.FileHandler(filename=log_path, mode='a+')

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s]: (%(name)s-%(levelname)s) - %(message)s')
    if verbose:
        console_handler.setLevel(logging.DEBUG)
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger