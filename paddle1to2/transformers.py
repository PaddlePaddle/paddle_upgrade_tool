from bowler.types import LN, Capture, Filename, SYMBOL, TOKEN
from fissix.fixer_util import Name, Call, Number, KeywordArg, Comma


def default_transformer(node: LN, capture: Capture, filename: Filename) -> None:
    fp = capture.get("function_parameters")
    if fp and fp.children[1].type == SYMBOL.arglist:
        arg_node = KeywordArg(Name("trans_arg"), Number("1"))
        fp.children[1].append_child(Comma())
        fp.children[1].append_child(arg_node)
