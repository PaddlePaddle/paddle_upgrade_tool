from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import argparse
import click

from bowler import Query

from paddle1to2.common import *
from paddle1to2 import refactor, filters, utils
from paddle1to2.refactor import *
from paddle1to2.spec import change_spec
from paddle1to2.utils import backup_inpath

def should_convert(inpath):
    """
    check if convert should be run.
    """
    # check if inpath exists, and python files in inpath are valid.
    if not utils.valid_path(inpath):
        return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", dest="log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set log level, default is INFO")
    parser.add_argument("--no-log-file", dest="no_log_file", action='store_true', default=False, help="Don't log to file")
    parser.add_argument("--log-filepath", dest="log_filepath", type=str, help='Set log file path, default is "report.log"')
    parser.add_argument("--inpath", required=True, type=str, help='The file or directory path you want to upgrade.')
    parser.add_argument("--backup", type=str, nargs='?', default=None, const=None, help='backup directory, default is the "~/.paddle1to2/".')
    parser.add_argument("--write", action='store_true', default=False, help='Modify files in place.')
    parser.add_argument("--refactor", action='append', choices=refactor.__all__, help='This is a debug option. Specify refactor you want to run. If none, all refactors will be run.')
    parser.add_argument("--print-match", action='store_true', default=False, help='This is a debug option. Print matched code and node for each file.')

    args = parser.parse_args()
    if args.refactor:
        args.refactor = set(args.refactor)
    if args.backup is None:
        home = os.path.expanduser('~')
        args.backup = os.path.join(home, '.paddle1to2')
    else:
        args.backup = os.path.expanduser(args.backup)

    if args.log_level:
        logger.setLevel(args.log_level)
    if not args.no_log_file:
        log_to_file(args.log_filepath)
    if not should_convert(args.inpath):
        logger.error("convert abort!")
        sys.exit(1)

    # refactor code via "Query" step by step.
    q = Query(args.inpath)
    for fn in refactor.__all__:
        refactor_func = getattr(refactor, fn)
        if args.refactor and fn not in args.refactor:
            continue
        assert callable(refactor_func), "{} is not callable.".format(fn)
        logger.debug("run refactor: {}".format(fn))
        if args.print_match:
            refactor_func(q, change_spec).filter(filters.print_match)
        else:
            refactor_func(q, change_spec)

    if args.write:
        if click.confirm(click.style("Files will be modified in-place, but don't worry, we will backup your files to {} automatically. do you want to continue?".format(args.backup), fg='red', bold=True)):
            # backup args.inpath
            backup_inpath(args.inpath, args.backup)
            # print diff to stdout, and modify file in place.
            q.execute(interactive=False, write=True, silent=False)
            logger.info("Refactor finished, and source files are modified in-place. Recover your files from {} if anything is wrong.".format(args.backup))
        else:
            click.secho('Refactor abort!', fg='red', bold=True)
    else:
        # print diff to stdout
        q.execute(interactive=False, write=False, silent=False)
        logger.warning('Refactor finished without touching source files, add "--write" to modify source files in-place if everything is ok.')

if __name__ == "__main__":
    sys.exit(main())
