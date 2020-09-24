#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import difflib
import logging
import multiprocessing
import os
import time
from queue import Empty
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

from tools import click
from fissix.pgen2.parse import ParseError
from fissix.refactor import RefactoringTool

from .helpers import filename_endswith
from .types import (
    BadTransform,
    BowlerException,
    BowlerQuit,
    Filename,
    FilenameMatcher,
    Fixers,
    Hunk,
    Node,
    Processor,
    RetryFile,
)

PROMPT_HELP = {
    "y": "apply this hunk",
    "n": "skip this hunk",
    "a": "apply this hunk and all remaining hunks for this file",
    "d": "skip this hunk and all remaining hunks for this file",
    "q": "quit; do not apply this hunk or any remaining hunks",
    "?": "show help",
}

log = logging.getLogger(__name__)


def diff_texts(a, b, filename):
    lines_a = a.splitlines()
    lines_b = b.splitlines()
    return difflib.unified_diff(lines_a, lines_b, filename, filename, lineterm="")


def prompt_user(question, options, default = ""):
    options = options.lower()
    default = default.lower()
    assert len(default) < 2 and default in options

    if "?" not in options:
        options += "?"

    prompt_options = ",".join(o.upper() if o == default else o for o in options)
    prompt = "{} [{}]? ".format(question, prompt_options)
    result = ""

    while True:
        result = input(prompt).strip().lower()
        if result == "?":
            for option in PROMPT_HELP:
                click.secho("{} - {}".format(option, PROMPT_HELP[option]), fg="red", bold=True)

        elif len(result) == 1 and result in options:
            return result

        elif result:
            click.echo('invalid response "{}"'.format(result))

        elif default:
            return default


class BowlerTool(RefactoringTool):
    NUM_PROCESSES = os.cpu_count() or 1
    IN_PROCESS = False  # set when run DEBUG mode from command line

    def __init__(
        self,
        fixers,
        *args,
        need_confirm = False,
        parallel = None,
        write = False,
        silent = False,
        in_process = False,
        hunk_processor = None,
        filename_matcher = None,
        **kwargs):
        self.backup = kwargs.pop('backup', None)
        self.print_hint = kwargs.pop('print_hint', True)
        options = kwargs.pop("options", {})
        options["print_function"] = True
        super().__init__(fixers, *args, options=options, **kwargs)
        self.need_confirm = need_confirm
        self.parallel = parallel
        self.queue_count = 0
        self.queue = multiprocessing.JoinableQueue()  # type: ignore
        # if need_confirm, refactor files in one process one by one to avoid log disorder.
        if self.need_confirm:
            self.results = multiprocessing.Queue(maxsize=1)  # type: ignore
            self.NUM_PROCESSES = 1
            self.semaphore_confirm = multiprocessing.Semaphore(1)
            self.parallel = None
        else:
            if self.parallel is not None:
                self.NUM_PROCESSES = max(1, min(self.parallel, 100))
            self.results = multiprocessing.Queue()  # type: ignore
        self.semaphore = multiprocessing.Semaphore(self.NUM_PROCESSES)
        self.write = write
        self.silent = silent
        # pick the most restrictive of flags
        self.in_process = in_process or self.IN_PROCESS
        self.exceptions = []
        if hunk_processor is not None:
            self.hunk_processor = hunk_processor
        else:
            self.hunk_processor = lambda f, h: True
        self.filename_matcher = filename_matcher or filename_endswith(".py")

    def log_error(self, msg, *args, **kwds):
        self.logger.error(msg, *args, **kwds)

    def get_fixers(self):
        fixers = [f(self.options, self.fixer_log) for f in self.fixers]
        pre = [f for f in fixers if f.order == "pre"]
        post = [f for f in fixers if f.order == "post"]
        return pre, post

    def processed_file(
        self, new_text, filename, old_text = "", *args, **kwargs
    ):
        self.files.append(filename)
        hunks = []
        if old_text != new_text:
            a, b, *lines = list(diff_texts(old_text, new_text, filename))

            hunk = []
            for line in lines:
                if line.startswith("@@"):
                    if hunk:
                        hunks.append([a, b, *hunk])
                        hunk = []
                hunk.append(line)

            if hunk:
                hunks.append([a, b, *hunk])

            try:
                new_tree = self.driver.parse_string(new_text)
                if new_tree is None:
                    raise AssertionError("Re-parsed CST is None")
            except Exception as e:
                raise BadTransform(
                    "Transforms generated invalid CST for {}".format(filename),
                    filename=filename,
                    hunks=hunks,
                ) from e

        return hunks

    def refactor_file(self, filename, *a, **k):
        try:
            hunks = []
            input, encoding = self._read_python_source(filename)
            if input is None:
                # Reading the file failed.
                return hunks
        except (OSError, UnicodeDecodeError) as e:
            log.error("Skipping {}: failed to read because {}".format(filename, e))
            return hunks

        try:
            if not input.endswith("\n"):
                input += "\n"
            tree = self.refactor_string(input, filename)
            if tree:
                hunks = self.processed_file(str(tree), filename, input)
        except ParseError as e:
            log.exception("Skipping {filename}: failed to parse ({e})")

        return hunks, str(tree).encode(encoding)

    def refactor_dir(self, dir_name, *a, **k):
        """Descends down a directory and refactor every Python file found.

        Python files are those for which `self.filename_matcher(filename)`
        returns true, to allow for custom extensions.

        Files and subdirectories starting with '.' are skipped.
        """
        for dirpath, dirnames, filenames in os.walk(dir_name):
            self.log_debug("Descending into %s", dirpath)
            dirnames.sort()
            filenames.sort()
            for name in filenames:
                fullname = os.path.join(dirpath, name)
                if not name.startswith(".") and self.filename_matcher(
                    Filename(fullname)
                ):
                    self.queue_work(Filename(fullname))
            # Modify dirnames in-place to remove subdirs with leading dots
            dirnames[:] = [dn for dn in dirnames if not dn.startswith(".")]

    def refactor_queue(self):
        self.semaphore.acquire()
        while True:
            filename = self.queue.get()

            if filename is None:
                break

            try:
                if self.need_confirm:
                    self.semaphore_confirm.acquire()
                hunks, new_text = self.refactor_file(filename)
                self.results.put((filename, hunks, None, new_text))

            except RetryFile:
                self.log_debug("Retrying {} later...".format(filename))
                self.queue.put(filename)
            except BowlerException as e:
                log.exception("Bowler exception during transform of {}: {}".format(filename, e))
                self.results.put((filename, e.hunks, e, None))
            except Exception as e:
                log.exception("Skipping {}: failed to transform because {}".format(filename, e))
                self.results.put((filename, [], e, None))

            finally:
                self.queue.task_done()
        self.semaphore.release()

    def queue_work(self, filename):
        self.queue.put(filename)
        self.queue_count += 1

    def refactor(self, items, *a, **k):
        """Refactor a list of files and directories."""

        for dir_or_file in sorted(items):
            if os.path.isdir(dir_or_file):
                self.refactor_dir(dir_or_file)
            else:
                self.queue_work(Filename(dir_or_file))

        children = []
        if self.in_process:
            self.queue.put(None)
            self.refactor_queue()
        else:
            child_count = max(1, min(self.NUM_PROCESSES, self.queue_count))
            self.log_debug("starting {} processes".format(child_count))
            for i in range(child_count):
                child = multiprocessing.Process(target=self.refactor_queue)
                child.start()
                children.append(child)
                self.queue.put(None)

        results_count = 0

        while True:
            try:
                filename, hunks, exc, new_text = self.results.get_nowait()
                results_count += 1

                if exc:
                    self.log_error("{}: {}".format(type(exc).__name__, exc))
                    if exc.__cause__:
                        self.log_error(
                            "  {}: {}".format(type(exc.__cause__).__name__, exc.__cause__)
                        )
                    if isinstance(exc, BowlerException) and exc.hunks:
                        diff = "\n".join("\n".join(hunk) for hunk in exc.hunks)
                        self.log_error("Generated transform:\n{}".format(diff))
                    self.exceptions.append(exc)
                else:
                    self.log_debug("results: got {} hunks for {}".format(len(hunks), filename))
                    self.print_hunks(filename, hunks)
                    if hunks and self.write:
                        if self.need_confirm:
                            if click.confirm(click.style('"{}" will be modified in-place, and it has been backed up to "{}". Do you want to continue?'.format(filename, self.backup), fg='red', bold=True)):
                                self.write_result(filename, new_text)
                                if self.print_hint:
                                    click.secho('"{}" refactor done! Recover your files from "{}" if anything is wrong.'.format(filename, self.backup))
                            else:
                                if self.print_hint:
                                    click.secho('"{}" refactor cancelled!'.format(filename), fg='red', bold=True)
                        else:
                            self.write_result(filename, new_text)
                            if self.print_hint:
                                click.secho('"{}" refactor done! Recover your files from "{}" if anything is wrong.'.format(filename, self.backup))
                if self.need_confirm:
                    self.semaphore_confirm.release()

            except Empty:
                if self.queue.empty() and results_count == self.queue_count:
                    break

                elif not self.in_process and not any(
                    child.is_alive() for child in children
                ):
                    self.log_debug("child processes stopped without consuming work")
                    break

                else:
                    time.sleep(0.05)

            except BowlerQuit:
                for child in children:
                    child.terminate()
                break

        self.log_debug("all children stopped and all diff hunks processed")

    def print_hunks(self, filename, hunks):
        auto_yes = False
        result = ""
        # print same filename header only once.
        hunks_header = set()
        for hunk in hunks:
            header = "{} {}".format(hunk[0], hunk[1])
            if self.hunk_processor(filename, hunk) is False:
                continue
            if not self.silent:
                # print header, e.g.
                # --- ./model.py
                # +++ ./model.py
                if header not in hunks_header:
                    for line in hunk[:2]:
                        if line.startswith("---"):
                            click.secho(line, fg="red", bold=True)
                        elif line.startswith("+++"):
                            click.secho(line, fg="green", bold=True)
                hunks_header.add(header)

                # print diff content
                for line in hunk[2:]:
                    if line.startswith("-"):
                        click.secho(line, fg="red")
                    elif line.startswith("+"):
                        click.secho(line, fg="green")
                    else:
                        click.echo(line)

    def write_result(self, filename, new_text):
        if isinstance(new_text, bytes):
            with open(filename, 'wb') as f:
                f.write(new_text)

    def run(self, paths):
        if not self.errors:
            self.refactor(paths)
            self.summarize()

        return int(bool(self.errors or self.exceptions))
