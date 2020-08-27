import sys
import logging

def log_to_file(log_filepath="report.log"):
    log_filepath = log_filepath or "report.log"
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

def _build_default_logger():
    logger = logging.getLogger('paddle1to2')
    logger.setLevel("INFO")
    
    console_handler = logging.StreamHandler(stream=sys.stdout) # default stream is sys.stderr
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger

log_format = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = _build_default_logger()
