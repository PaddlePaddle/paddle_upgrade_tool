from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import argparse
from six import StringIO
from lib2to3.patcomp import PatternCompiler
from lib2to3 import pytree
from lib2to3.pgen2 import driver
from lib2to3.pygram import python_symbols, python_grammar

def main():
    parser = argparse.ArgumentParser()
    g1 = parser.add_mutually_exclusive_group(required=True)
    g1.add_argument("-pf", "--pattern-file", dest="pattern_file", type=str, help='Read pattern from the specified file')
    g1.add_argument("-ps", "--pattern-string", dest="pattern_string", type=str, help='A pattern string')
    g2 = parser.add_mutually_exclusive_group(required=True)
    g2.add_argument("-sf", "--source-file", dest="source_file", type=str, help="Read code snippet from the specified file")
    g2.add_argument("-ss", "--source-string", dest="source_string", type=str, help="A code snippet string")
    parser.add_argument("--print-results", dest="print_results", action='store_true', default=False, help="Print match results")
    parser.add_argument("--print-lineno", dest="print_lineno", action='store_true', default=False, help="Print match code with line number")
    # Parse command line arguments
    args = parser.parse_args()

    # parse source snippet to CST tree
    driver_ = driver.Driver(python_grammar, convert=pytree.convert)
    if args.source_file:
        tree = driver_.parse_file(args.source_file)
    else:
        tree = driver_.parse_stream(StringIO(args.source_string + "\n"))
    # compile pattern
    if args.pattern_file:
        with open(args.pattern_file, 'r') as f:
            pattern = f.read()
    else:
        pattern = args.pattern_string
    PC = PatternCompiler()
    pattern, pattern_tree = PC.compile_pattern(pattern, with_tree=True)
    for node in tree.post_order():
        results = {'node':node}
        if pattern.match(node, results):
            match_node = results['node']
            src_lines = str(match_node).splitlines()
            if args.print_lineno:
                # calculate lineno_list according to the right most leaf node.
                # because some node includes prefix, which is not a node, and we can't get it's lineno.
                right_most_leaf = match_node
                while not isinstance(right_most_leaf, pytree.Leaf):
                    right_most_leaf = right_most_leaf.children[-1]
                last_lineno = right_most_leaf.get_lineno()
                lineno_list = list(range(last_lineno - len(src_lines) + 1, last_lineno + 1))
                src_lines = [str(lineno) + ' ' + line for lineno, line in zip(lineno_list, src_lines)]
            for line in src_lines:
                print(line)
            if args.print_results:
                print(results)
            print('-' * 20)

if __name__ == "__main__":
    sys.exit(main())
