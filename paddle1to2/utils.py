import os
import sys
import shutil
from io import StringIO
from datetime import datetime
from fnmatch import filter

from fissix.pgen2 import driver
from fissix import pytree
from fissix.pygram import python_grammar, python_symbols
from fissix.pgen2 import token
from fissix.pytree import Leaf, Node
from fissix.fixer_util import Attr, Comma, Dot, LParen, Name, Newline, RParen, KeywordArg, Number, ArgList, Newline

from paddle1to2.common import logger

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
    return tree

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
    _node = code_repr(after).children[0].children[0]
    _node.parent = None
    after_node = _node
    # reserve prefix
    after_node.children[0].prefix = node.children[0].prefix
    del node.children[:len(before_parts)]
    for i in range(len(after_node.children)):
        node.insert_child(i, after_node.children[i])


def norm_arglist(trailer_node):
    """
    normilize argument list node, e.g.
    Node(trailer, [Leaf(7, '('), Leaf(2, '3'), Leaf(8, ')')])
    to
    Node(trailer, [Leaf(7, '('), Node(arglist, [Node(argument, [Leaf(2, '1')])]), Leaf(8, ')')])
    """
    if trailer_node.type != python_symbols.trailer or len(trailer_node.children) < 2 or len(trailer_node.children) > 3:
        logger.warning("node type is not trailer, or len(children) < 2 or len(children) > 3")
        return
    second_node = trailer_node.children[1]
    # add arglist node if needed
    if second_node.type != python_symbols.arglist:
        if second_node.type == token.RPAR:
            arglist_node = Node(python_symbols.arglist, [])
            trailer_node.insert_child(1, arglist_node)
        else:
            _second_node = second_node.clone()
            arglist_node = Node(python_symbols.arglist, [_second_node])
            second_node.remove()
            trailer_node.insert_child(1, arglist_node)
    else:
        arglist_node = second_node

    # add argument node if needed
    for i in range(len(arglist_node.children)):
        node = arglist_node.children[i]
        # skip "," node
        if node.type == token.COMMA:
            continue
        if node.type != python_symbols.argument:
            _node = node.clone()
            arg_node = Node(python_symbols.argument, [_node])
            node.replace(arg_node)

def add_argument(filename, trailer_node, key, value):
    """
    add "key=value" to arglist in trailer_node,
    if arglist already contains key, reassign key to value
    """
    if trailer_node.type != python_symbols.trailer and len(trailer_node.children) != 3:
        log_warning(filename, trailer_node.get_lineno(), "node type is not trailer or len(children) != 3. you may need to call norm_arglist first.")
        return
    arglist_node = trailer_node.children[1]
    if arglist_node.type != python_symbols.arglist:
        log_warning(filename, trailer_node.get_lineno(), "trailer_node.children[1] is not arglist.")
        return
    found_key = False
    for node in arglist_node.children:
        if node.type != python_symbols.argument:
            continue
        _key_node = node.children[0]
        if _key_node.value == key:
            found_key = True
            _value_node = node.children[2]
            if _value_node.value != value:
                _value_node_copy = _value_node.clone()
                _value_node_copy.type = token.NAME
                _value_node_copy.value = value
                _value_node.replace(_value_node_copy)
                log_warning(filename, arglist_node.get_lineno(), 'argument "{}" is reassigned to "{}"'.format(key, value))
            break
    if not found_key:
        key_node = Name(key)
        value_node = Name(value)
        if arglist_node.children:
            arglist_node.append_child(Comma())
            key_node.prefix = " "
        arg_node = KeywordArg(key_node, value_node)
        arglist_node.append_child(arg_node)
        log_warning(filename, arglist_node.get_lineno(), 'add argument "{}={}"'.format(key, value))


def remove_argument(filename, trailer_node, key):
    """
    remove "key" from arglist in trailer_node,
    """
    if trailer_node.type != python_symbols.trailer and len(trailer_node.children) != 3:
        log_warning(filename, trailer_node.get_lineno(), "node type is not trailer or len(children) != 3. you may need to call norm_arglist first.")
        return
    #removed_value = None # record removed key=value and arglist_node
    arglist_node = trailer_node.children[1]
    if arglist_node.type != python_symbols.arglist:
        log_warning(filename, trailer_node.get_lineno(), "trailer_node.children[1] is not arglist.")
        return
    found_key = False
    for node in arglist_node.children:
        if node.type != python_symbols.argument:
            continue
        _key_node = node.children[0]
        if _key_node.value == key:
            found_key = True
            #removed_value = node.children[2].value if len(node.children) >= 3 else None
            if node.prev_sibling is not None and node.prev_sibling.type is token.COMMA:
                node.prev_sibling.remove()
            elif node.next_sibling is not None and node.next_sibling.type is token.COMMA:
                node.next_sibling.remove()
            node.remove()
            log_warning(filename, arglist_node.get_lineno(), 'argument "{}" is removed.'.format(key))
            break
    if not found_key:
        log_warning(filename, arglist_node.get_lineno(), 'argument "{}" not found.'.format(key))
    #return removed_value


def rename_argument(filename, trailer_node, old_key, new_key):
    """
    rename "old_key" to "new_key" in arglist in trailer_node,
    """
    if trailer_node.type != python_symbols.trailer and len(trailer_node.children) != 3:
        log_warning(filename, trailer_node.get_lineno(), "node type is not trailer or len(children) != 3. you may need to call norm_arglist first.")
        return
    arglist_node = trailer_node.children[1]
    if arglist_node.type != python_symbols.arglist:
        log_warning(filename, trailer_node.get_lineno(), "trailer_node.children[1] is not arglist.")
        return
    found_key = False
    for node in arglist_node.children:
        if node.type != python_symbols.argument:
            continue
        _key_node = node.children[0]
        if _key_node.value == old_key:
            found_key = True
            _key_node.value = new_key
            log_warning(filename, arglist_node.get_lineno(), 'rename argument "{}" to "{}".'.format(old_key, new_key))
            break
    if not found_key:
        log_warning(filename, arglist_node.get_lineno(), 'argument "{}" not found.'.format(old_key))


def apply_argument(filename, trailer_node, fun):
    """
    call fun(argument_node) for each argument
    """
    if trailer_node.type != python_symbols.trailer and len(trailer_node.children) != 3:
        log_warning(filename, trailer_node.get_lineno(), "node type is not trailer or len(children) != 3. you may need to call norm_arglist first.")
        return
    arglist_node = trailer_node.children[1]
    if arglist_node.type != python_symbols.arglist:
        log_warning(filename, trailer_node.get_lineno(), "trailer_node.children[1] is not arglist.")
        return
    for node in arglist_node.children:
        if node.type != python_symbols.argument:
            continue
        fun(node)


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

def dec_indent(indent, count=1):
    """
    decrease indent, e.g.
    if indent = "        ", and count = 1, return "    ",
    if indent = "        ", and count = 2, return "",
    """
    if indent.endswith('\t'):
        indent = indent[:len(indent) - 1 * count]
    elif indent.endswith('    '):
        indent = indent[:len(indent) - 4 * count]
    return indent

def get_indent(node):
    """
    get indent string of current node
    """
    while node is not None:
        if node.type == token.INDENT:
            return node.value
        node = node.prev_sibling
    return ""

def _is_windows():
    """
    check if current operating system is windows
    """
    return sys.platform.lower() == 'win32'

def _is_windows_file(filepath):
    """
    if file contains "\r\n", treat it as windows file, and return True, else False
    """
    with open(filepath, 'rb') as f:
        content = f.read()
    if content.find(b'\r\n') != -1:
        return True
    return False

def _is_utf8(filepath):
    """
    check if file is utf8 encoding
    """
    with open(filepath, 'rb') as f:
        content = f.read()
    try:
        content_utf8 = content.decode('utf-8')
    except UnicodeDecodeError as e:
        return False
    return True

def valid_path(inpath):
    # check if inpath is valid
    if not os.path.exists(inpath):
        logger.error("{} doesn't exist.".format(inpath))
        return False
    valid = True
    if os.path.isfile(inpath):
        """
        refactor windows files on linux or mac os is not supported.
        """
        if not _is_windows():
            if _is_windows_file(inpath):
                logger.error('{} is a windows file, you can use "dos2unix" command to convert it to linux file.'.format(inpath))
                valid = False
            if not _is_utf8(inpath):
                logger.error('{} encoding is not utf-8, you can use "iconv" command to convert it to utf-8.'.format(inpath))
                valid = False
    elif os.path.isdir(inpath):
        for dirpath, dirnames, filenames in os.walk(inpath):
            for filename in filenames:
                if not filename.endswith('.py'):
                    continue
                filepath = os.path.join(dirpath, filename)
                if not _is_windows() and _is_windows_file(filepath):
                    logger.error('{} is a windows file, you can use "dos2unix" command to convert it to linux file.'.format(filepath))
                    valid = False
                if not _is_utf8(filepath):
                    logger.error('{} encoding is not utf-8, you can use "iconv" command to convert it to utf-8.'.format(filepath))
                    valid = False
    else:
        logger.error('{} is neither a file nor a directory.'.format(inpath))
        valid = False

    return valid

