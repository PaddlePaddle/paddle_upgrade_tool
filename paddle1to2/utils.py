from io import StringIO

from fissix.pgen2 import driver
from fissix import pytree
from fissix.pygram import python_grammar, python_symbols

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
