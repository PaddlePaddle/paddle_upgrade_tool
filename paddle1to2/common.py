import sys
import logging


__all__ = [
        'logger',
        'log_to_file',
        ]

class ColorFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)

        grey = "\x1b[38;21m"
        yellow = "\x1b[33;21m"
        red = "\x1b[31;21m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"

        self.FORMATS = {
            logging.DEBUG: grey + fmt + reset,
            logging.INFO: grey + fmt + reset,
            logging.WARNING: yellow + fmt + reset,
            logging.ERROR: red + fmt + reset,
            logging.CRITICAL: bold_red + fmt + reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def log_to_file(log_filepath="report.log"):
    log_filepath = log_filepath or "report.log"
    file_handler = logging.FileHandler(log_filepath)
    log_format = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

def _build_default_logger():
    logger = logging.getLogger('paddle1to2')
    logger.setLevel("INFO")
    
    console_handler = logging.StreamHandler(stream=sys.stdout) # default stream is sys.stderr
    log_format = ColorFormatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger

logger = _build_default_logger()
