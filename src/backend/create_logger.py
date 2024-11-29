import logging
from ...config import log_file_path

def create_logger(module_name: str) -> logging.Logger:
    """ Creates a logger for the specified module
    """
    # Create a logger object
    logger = logging.getLogger(module_name)

    # Create a file handler for persisting logs by writing them to a file
    handler = logging.FileHandler(log_file_path, mode="a")

    # Create a formatter to write logs in the same consistent format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Assign the formatter to the handler
    handler.setFormatter(formatter)

    return logger