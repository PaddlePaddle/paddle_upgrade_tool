import os
import shutil
from io import StringIO
from datetime import datetime
from fnmatch import filter

from fissix.pgen2 import driver
from fissix import pytree
from fissix.pygram import python_grammar, python_symbols
from fissix.pytree import Leaf, Node

from paddle1to2.common import logger

def node2code(nodes, with_first_prefix=False):
    """
    convert a node or a list of nodes to code str, e.g.
    "    import paddle" -> return "import paddle"
    "    import     paddle" -> return "import     paddle"
    """
    if not isinstance(nodes, list):
        nodes = [nodes]
    code = ''
    is_first_leaf = True
    for node in nodes:
        for leaf in node.leaves():
            if is_first_leaf:
                code = code + leaf.value
                is_first_leaf = False
            else:
                code = code + str(leaf)
    return code

def code_repr(src: str):
    """
    convert source code to node tree, e.g.
    src = 'path.to.api'
    output = "Node(power, [Leaf(1, 'path'), Node(trailer, [Leaf(23, '.'), Leaf(1, 'to')]), Node(trailer, [Leaf(23, '.'), Leaf(1, 'api')])])"
    """
    driver_ = driver.Driver(python_grammar, convert=pytree.convert)
    tree = driver_.parse_stream(StringIO(src + '\n'))
    ret = tree.children[0].children[0]
    ret.parent = None
    return ret

def replace_module_path(node, before, after):
    """
    replace before with after in node.children, e.g.
    before = 'path1.to1.api1'
    after = 'path2.to2.api2'

    before replace, node is:
    Node(power, [Leaf(1, 'path1'), Node(trailer, [Leaf(23, '.'), Leaf(1, 'to1')]), Node(trailer, [Leaf(23, '.'), Leaf(1, 'api1')])])

    after replace, node is:
    Node(power, [Leaf(1, 'path2'), Node(trailer, [Leaf(23, '.'), Leaf(1, 'to2')]), Node(trailer, [Leaf(23, '.'), Leaf(1, 'api2')])])

    """
    before_parts = before.split('.')
    after_node = code_repr(after)
    # reserve prefix
    after_node.children[0].prefix = node.children[0].prefix
    del node.children[:len(before_parts)]
    for i in range(len(after_node.children)):
        node.insert_child(i, after_node.children[i])

def startswith(module_path1, module_path2):
    """
    check module_path1 includes module_path2, e.g.
    "path.to.api.func_name" includes "path.to.api",
    but not include "path.api.func"

    return True if module_path1 includes module_path2, False otherwise.
    """
    # convert "path.to.api()" to "path.to.api"
    idx = module_path1.find('(')
    if idx != -1:
        module_path1 = module_path1[:idx]
    idx = module_path2.find('(')
    if idx != -1:
        module_path2 = module_path2[:idx]
    dotted_parts1 = module_path1.split('.')
    dotted_parts2 = module_path2.split('.')
    if len(dotted_parts1) < len(dotted_parts2):
        return False
    for i in range(len(dotted_parts2)):
        if dotted_parts1[i] != dotted_parts2[i]:
            return False
    return True

def log_debug(filename, lineno, msg):
    _msg = "{}:{} {}".format(filename, lineno, msg)
    logger.debug(_msg)

def log_info(filename, lineno, msg):
    _msg = "{}:{} {}".format(filename, lineno, msg)
    logger.info(_msg)

def log_warning(filename, lineno, msg):
    _msg = "{}:{} {}".format(filename, lineno, msg)
    logger.warning(_msg)

def log_error(filename, lineno, msg):
    _msg = "{}:{} {}".format(filename, lineno, msg)
    logger.error(_msg)

def _include_patterns(*patterns):
    """Factory function that can be used with copytree() ignore parameter.

    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().
    """
    def _ignore_patterns(path, names):
        keep = set(name for pattern in patterns
                            for name in filter(names, pattern))
        ignore = set(name for name in names
                        if name not in keep and not os.path.isdir(os.path.join(path, name)))
        return ignore
    return _ignore_patterns

def backup_inpath(inpath, backup):
    inpath = os.path.abspath(inpath)
    backup = os.path.abspath(backup)
    if not os.path.exists(backup):
        os.makedirs(backup)
    basename = os.path.basename(inpath)
    bak_basename = basename + "_backup_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    dst_path = os.path.join(backup, bak_basename)
    if os.path.isfile(inpath):
        shutil.copy(inpath, dst_path)
    else:
        shutil.copytree(inpath, dst_path, ignore=_include_patterns("*.py"))

