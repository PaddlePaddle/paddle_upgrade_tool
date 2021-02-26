import sys
import logging
import threading
from multiprocessing import Manager


__all__ = [
        'logger',
        'log_to_file',
        ]

class ColorFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        self.fmt = fmt

        light_gray='\033[0;37m'
        dark_gray='\033[1;30m'
        yellow = "\033[0;33m"
        red = "\033[0;31m"
        reset = "\033[0m"

        self.FORMATS = {
            logging.DEBUG: dark_gray + fmt + reset,
            logging.INFO: light_gray + fmt + reset,
            logging.WARNING: yellow + fmt + reset,
            logging.ERROR: red + fmt + reset,
        }

    def format(self, record):
        log_fmt = None
        # if not windows, add color info
        if sys.platform.lower() != 'win32':
            log_fmt = self.FORMATS.get(record.levelno)
        if log_fmt is None:
            log_fmt = self.fmt
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def log_to_file(log_filepath="report.log"):
    log_filepath = log_filepath or "report.log"
    file_handler = logging.FileHandler(log_filepath)
    log_format = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    headless_file_handler = logging.FileHandler(log_filepath)
    headless_log_format = logging.Formatter('%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    headless_file_handler.setFormatter(headless_log_format)
    headless_logger.addHandler(headless_file_handler)

def _build_default_logger():
    logger = logging.getLogger('paddle_upgrade_tool')
    logger.setLevel("INFO")

    console_handler = logging.StreamHandler(stream=sys.stdout) # default stream is sys.stderr
    log_format = ColorFormatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger

def _build_headless_logger():
    logger = logging.getLogger('statistic')
    logger.setLevel("INFO")

    console_handler = logging.StreamHandler(stream=sys.stdout) # default stream is sys.stderr
    log_format = ColorFormatter('%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger

logger = _build_default_logger()
headless_logger = _build_headless_logger()

# record refactor log.
# example:
# statistic = {
#   '/path/to/file1.py':{
#       'info':  ['rename "paddle.api1" to "paddle.api2"', 'rename "paddle.api3" to "paddle.api4"'],
#       'warn':  ['delete "paddle.api5"'],
#       'error': ['parse "paddle.api6" error'],
#   },
# }
manager = Manager()
statistic = manager.dict()
statistic_lock = threading.Lock()
