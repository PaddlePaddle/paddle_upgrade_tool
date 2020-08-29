from bowler.types import LN, Capture, Filename, SYMBOL, TOKEN
from fissix.fixer_util import Name, Call, Number

def  _default_transformer(node: LN, capture: Capture, filename: Filename) -> None:
    fp = capture.get("function_parameters")
    if fp and fp.children[1].type == SYMBOL.arglist:
        arg_node = KeywordArg(Name("trans_arg"), Number("1"))
        fp.children[1].children.append(arg_node)
