from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import argparse
from common import *
from bowler import Query
from refactor import *
from spec import change_spec

def should_convert():
    """
    check if convert should be run.
    convert should be interrupted in the following cases:
    1. directory is not a git repo, and there are something not committed.
    2. file has been converted.
    """
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", dest="log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="set log level")
    parser.add_argument("--no-log-file", dest="no_log_file", action='store_true', default=False, help="don't log to file")
    parser.add_argument("--log-filepath", dest="log_filepath", type=str, help='set log file path, default is "report.log"')
    parser.add_argument("--inpath", dest="inpath", required=True, type=str, help='the file or directory path you want to upgrade.')
    parser.add_argument("--write", dest="write", action='store_true', default=False, help='modify files in place.')

    args = parser.parse_args()
    if args.log_level:
        logger.setLevel(args.log_level)
    if not args.no_log_file:
        log_to_file(args.log_filepath)

    print(args)

    if not should_convert():
        print("convert abort!")
        sys.exit(1)

    # refactor code via "Query" step by step.
    q = Query(args.inpath)
    refactor_demo(q, change_spec)
    # refactor import statement 
    logger.debug("run refactor step: refactor_import")
    refactor_import(q, change_spec)
    # rename all alias to main alias(also old api).
    logger.debug("run refactor step: norm_api_alias")
    norm_api_alias(q, change_spec)
    # transform args to kwargs.
    logger.debug("run refactor step: args_to_kwargs")
    args_to_kwargs(q, change_spec)
    # print warning if specified args are used.
    logger.debug("run refactor step: args_warning")
    args_warning(q, change_spec)
    # rename, add or remove kwargs.
    logger.debug("run refactor step: refactor_kwargs")
    refactor_kwargs(q, change_spec)
    # print warning if specified api are used.
    logger.debug("run refactor step: api_warning")
    api_warning(q, change_spec)
    # rename old api to new api.
    logger.debug("run refactor step: api_rename")
    api_rename(q, change_spec)
    # refactor syntax, such as removing "with" statement.
    logger.debug("run refactor step: refactor_syntax")
    refactor_syntax(q, change_spec)
    # post refactor after all prior refactor steps.
    logger.debug("run refactor step: post_refactor")
    post_refactor(q, change_spec)

    if args.write:
        # print diff to stdout, and modify file in place.
        q.execute(interactive=False, write=True, silent=False)
        logger.info("refactor finished, and source files are modified")
    else:
        # print diff to stdout
        q.execute(interactive=False, write=False, silent=False)
        logger.info("refactor finished without touching source files")

if __name__ == "__main__":
    sys.exit(main())
