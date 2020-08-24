from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
from common import *
from framework import Query

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", dest="log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="set log level")
    parser.add_argument("--no-log-file", dest="no_log_file", action='store_true', default=False, help="don't log to file")
    parser.add_argument("--log-filepath", dest="log_filepath", type=str, help='set log file path, default is "report.log"')
    parser.add_argument("--inpath", dest="inpath", required=True, type=str, help='the file or directory path you want to upgrade.')
    parser.add_argument("--outpath", dest="outpath", type=str, help='the file or directory path you want to upgrade.')

    
    args = parser.parse_args()
    if args.log_level:
        logger.setLevel(args.log_level)
    if not args.no_log_file:
        log_to_file(args.log_filepath)
    print(args)
    logger.debug("this is debug")
    logger.info("this is info")
    logger.warning("this is warning")
    logger.error("this is error")

    q = Query(args.inpath)
    print(q)

if __name__ == "__main__":
    main()
