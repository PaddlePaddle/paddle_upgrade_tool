from bowler.types import LN, Capture, Filename, SYMBOL, TOKEN
from fissix.fixer_util import Name, Call, Number, KeywordArg, Comma


def default_transformer(node: LN, capture: Capture, filename: Filename):
    fp = capture.get("function_parameters")
    if fp and fp.children[1].type == SYMBOL.arglist:
        arg_node = KeywordArg(Name("trans_arg"), Number("1"))
        fp.children[1].append_child(Comma())
        fp.children[1].append_child(arg_node)

def act_transformer(trailer_node, removed_value):
    """
    add act to forward function, after delete act arg from api
    """
    print("removed_value:", removed_value)
    pass
