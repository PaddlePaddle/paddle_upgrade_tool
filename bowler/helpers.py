#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional, Sequence, Union

import click
from fissix.pgen2.token import tok_name
from fissix.pytree import Leaf, Node, type_repr

from .types import LN, SYMBOL, TOKEN, Capture, Filename, FilenameMatcher

log = logging.getLogger(__name__)

INDENT_STR = ".  "


def print_selector_pattern(
    node, results = None, filename = None
):
    key = ""
    if results:
        for k, v in results.items():
            if node == v:
                key = k + "="
            elif isinstance(v, list) and node in v:  # v is a list?
                key = k + "="

    if isinstance(node, Leaf):
        click.echo("{}{} ".format(key, repr(node.value)), nl=False)
    else:
        click.echo("{}{} ".format(key, type_repr(node.type)), nl=False)
        if node.children:
            click.echo("< ", nl=False)
            for child in node.children:
                print_selector_pattern(child, results, filename)
            click.echo("> ", nl=False)


def print_tree(
    node,
    results = None,
    filename = None,
    indent = 0,
    recurse = -1,
):
    filename = filename or Filename("")
    tab = INDENT_STR * indent
    if filename and indent == 0:
        click.secho(filename, fg="red", bold=True)

    if isinstance(node, Leaf):
        click.echo(
            click.style(tab, fg="black", bold=True)
            + click.style(
                "[{}] {} {}".format(tok_name[node.type], repr(node.prefix), repr(node.value)),
                fg="yellow",
            )
        )
    else:
        click.echo(
            click.style(tab, fg="black", bold=True)
            + click.style("[{}] {}".format(type_repr(node.type), repr(node.prefix)), fg="blue")
        )

    if node.children:
        if recurse:
            for child in node.children:
                # N.b. do not pass results here since we print them once
                # at the end.
                print_tree(child, indent=indent + 1, recurse=recurse - 1)
        else:
            click.echo(INDENT_STR * (indent + 1) + "...")

    if results is None:
        return

    for key in results:
        if key == "node":
            continue

        value = results[key]
        if isinstance(value, (Leaf, Node)):
            click.secho("results[{}] =".format(repr(key)), fg="red")
            print_tree(value, indent=1, recurse=1)
        else:
            # TODO: Improve display of multi-match here, see
            # test_print_tree_captures test.
            click.secho("results[{}] = {}".format(repr(key), value), fg="red")


def dotted_parts(name):
    pre, dot, post = name.partition(".")
    if post:
        post_parts = dotted_parts(post)
    else:
        post_parts = []
    result = []
    if pre:
        result.append(pre)
    if pre and dot:
        result.append(dot)
    if post_parts:
        result.extend(post_parts)
    return result


def quoted_parts(name):
    return ["'{}'".format(part) for part in dotted_parts(name)]


def power_parts(name):
    parts = quoted_parts(name)
    index = 0
    while index < len(parts):
        if parts[index] == "'.'":
            parts.insert(index, "trailer<")
            parts.insert(index + 3, ">")
            index += 1
        index += 1
    return parts


def is_method(node):
    return (
        node.type == SYMBOL.funcdef
        and node.parent is not None
        and node.parent.type == SYMBOL.suite
        and node.parent.parent is not None
        and node.parent.parent.type == SYMBOL.classdef
    )


def is_call_to(node, func_name):
    """Returns whether the node represents a call to the named function."""
    return (
        node.type == SYMBOL.power
        and node.children[0].type == TOKEN.NAME
        and node.children[0].value == func_name
    )


def find_first(node, target, recursive = False):
    queue = [node]
    queue.extend(node.children)
    while queue:
        child = queue.pop(0)
        if child.type == target:
            return child
        if recursive:
            queue = child.children + queue
    return None


def find_previous(node, target, recursive = False):
    while node.prev_sibling is not None:
        node = node.prev_sibling
        result = find_last(node, target, recursive)
        if result:
            return result
    return None


def find_next(node, target, recursive = False):
    while node.next_sibling is not None:
        node = node.next_sibling
        result = find_first(node, target, recursive)
        if result:
            return result
    return None


def find_last(node, target, recursive = False):
    queue = []
    queue.extend(reversed(node.children))
    while queue:
        child = queue.pop(0)
        if recursive:
            result = find_last(child, target, recursive)
            if result:
                return result
        if child.type == target:
            return child
    return None


def get_class(node):
    while node.parent is not None:
        if node.type == SYMBOL.classdef:
            return node
        node = node.parent
    raise ValueError("classdef node not found")


class Once:
    """Simple object that evaluates to True once, and then always False."""

    def __init__(self):
        self.done = False

    def __bool__(self):
        if self.done:
            return False
        else:
            self.done = True
            return True


def filename_endswith(ext):
    if isinstance(ext, str):
        ext = [ext]

    def inner(filename):
        return any(filename.endswith(e) for e in ext)

    return inner
