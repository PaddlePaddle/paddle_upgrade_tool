from bowler.types import LN, Capture, Filename, SYMBOL, TOKEN
from fissix.fixer_util import Name, Call, Number, KeywordArg, Comma, Newline
from fissix.pytree import Leaf, Node, type_repr
from fissix.pygram import python_grammar, python_symbols
from fissix.pgen2 import token
from fissix import patcomp

from paddle1to2 import utils
from paddle1to2.utils import log_debug, log_info, log_warning, log_error


def default_transformer(node: LN, capture: Capture, filename: Filename):
    fp = capture.get("function_parameters")
    if fp and fp.children[1].type == SYMBOL.arglist:
        arg_node = KeywordArg(Name("trans_arg"), Number("1"))
        fp.children[1].append_child(Comma())
        fp.children[1].append_child(arg_node)


def act_transformer(filename, trailer_node, removed_value):
    """
    add act to forward function, after delete act arg from api
    """
    if removed_value == "None":
        return
    # parent must be a power node
    power_node = trailer_node.parent
    if not isinstance(power_node, Node) and power_node.type != python_symbols.power:
        return
    # parent of parent must be an expression
    expr_node = power_node.parent
    if not isinstance(expr_node, Node) and expr_node.type != python_symbols.expr_stmt:
        return
    assign_idx = -1 # "=" index
    for idx in range(len(expr_node.children)):
        if expr_node.children[idx].type == token.EQUAL:
            assign_idx = idx
            break
    if assign_idx == -1:
        return
    layer_name = utils.node2code(expr_node.children[0:assign_idx])
    # Layer Class
    if 'self.' in layer_name:
        _forward_act_transformer(filename, expr_node, layer_name, removed_value)
    # invoke activation function directly
    else:
        _function_act_transformer(filename, expr_node, removed_value)

_pattern_funcdef_forward = "funcdef< 'def' 'forward' any* >"
_pattern_funcdef_forward = patcomp.compile_pattern(_pattern_funcdef_forward)
_pattern_expr_stmt = "simple_stmt< expr_stmt< left=(any*) '=' right=(any*) > any* >"
_pattern_expr_stmt = patcomp.compile_pattern(_pattern_expr_stmt)

def _forward_act_transformer(filename, expr_node, layer_name, removed_value):
    # find funcdef node
    funcdef_node = None
    node = expr_node
    while node is not None:
        if node.type == python_symbols.funcdef:
            funcdef_node = node
            break
        node = node.parent
    if funcdef_node is None:
        return
    # find def forward function node
    forward_node = None
    for node in funcdef_node.parent.children:
        results = {'node': node}
        if _pattern_funcdef_forward.match(node, results) and results["node"] is node:
            forward_node = node
            break
    if forward_node is None:
        return
    for node in forward_node.post_order():
        results = {'node': node}
        if _pattern_expr_stmt.match(node, results) and results["node"] is node:
            right=utils.node2code(results['right']).strip()
            if not utils.startswith(right, layer_name):
                continue
            left = utils.node2code(results['left']).strip()
            # if removed_value type is str
            if '"' in removed_value or "'" in removed_value:
                act = removed_value
                act = act.strip('"')
                act = act.strip("'")
                act = act.strip()
                # create statement like "x = paddle.nn.function.act(x)"
                code = left + ' = ' + 'paddle.nn.functional.' + act + '(' + left + ')'
                _create_simple_stmt_node_and_insert_behind(code, node)
            # removed_value is a variable
            else:
                # add "self._act = act" after expr_node to make it visible to other methods
                act_var_name = "self._" + removed_value
                code = act_var_name + " = " + removed_value
                _create_simple_stmt_node_and_insert_behind(code, expr_node.parent)
                # create statement like "x = getattr(paddle.nn.function, act)(x) if act else x"
                code = left + " = getattr(paddle.nn.function, " + act_var_name + ")(" + left + ") if " + act_var_name + " else " + left
                _create_simple_stmt_node_and_insert_behind(code, node)
                log_warning(filename, expr_node.get_lineno(), 'variable "{}" may not be visible here.'.format(removed_value))


def _function_act_transformer(filename, expr_node, removed_value):
    simple_stmt_node = expr_node.parent
    results = {'node': simple_stmt_node}
    if _pattern_expr_stmt.match(simple_stmt_node, results) and results["node"] is simple_stmt_node:
        left = utils.node2code(results['left']).strip()
        # if removed_value type is str
        if '"' in removed_value or "'" in removed_value:
            act = removed_value
            act = act.strip('"')
            act = act.strip("'")
            act = act.strip()
            # create statement like "x = paddle.nn.function.act(x)"
            code = left + ' = ' + 'paddle.nn.functional.' + act + '(' + left + ')'
            _create_simple_stmt_node_and_insert_behind(code, simple_stmt_node)
        # removed_value is a variable
        else:
            # create statement like "x = getattr(paddle.nn.function, act)(x) if act else x"
            code = left + " = getattr(paddle.nn.function, " + removed_value + ")(" + left + ") if " + removed_value + " else " + left
            _create_simple_stmt_node_and_insert_behind(code, simple_stmt_node)


def _create_simple_stmt_node_and_insert_behind(code, node):
    if node is None or node.type != python_symbols.simple_stmt:
        return
    simple_stmt_node = Node(python_symbols.simple_stmt, [Newline()])
    _node = utils.code_repr(code).children[0].children[0]
    _node.parent = None
    simple_stmt_node.insert_child(0, _node)
    simple_stmt_node.prefix = utils.get_indent(node)
    utils.insert_node_behind(node, simple_stmt_node)
