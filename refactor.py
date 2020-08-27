from bowler import Query
from bowler.helpers import power_parts, quoted_parts, dotted_parts
from bowler.types import LN, Capture, Filename, SYMBOL, TOKEN
from fissix.pytree import Leaf, Node, type_repr
from fissix.fixer_util import Attr, Comma, Dot, LParen, Name, Newline, RParen, KeywordArg

from common import logger
import processors
import fixers

# don't change the order if you don't know what you are doing.
__all__ = [
    'refactor_demo',
    'refactor_import',
    'norm_api_alias',
    'args_to_kwargs',
    'args_warning',
    'refactor_kwargs',
    'api_warning',
    'api_rename',
    'refactor_syntax',
    'post_refactor',
    ]

def refactor_demo(q: Query, change_spec) -> "Query":
    q.select_function("old_api").is_call().rename("new_api").process(processors.demo_post_processor)

    #q.fixer(fixers.FixerDemo)
    return q

def refactor_import(q: Query, change_spec) -> "Query":
    """
    1. add "import paddle" if needed.
    2. remove "import paddle.mod" if needed.
    3. remove "import paddle.module as mod", and convert "mod.api" to "paddle.module.api"
    4. remove "from paddle.module import api", and convert "api" to "paddle.module.api"
    """

    # select import_name and import_from
    pattern = """
        (
            import_name< 'import'
                (
                    module_name='{name}'
                |
                    module_name=dotted_name< {dotted_name} any* >
                |
                    dotted_as_name<
                        (
                            module_name='{name}'
                        |
                            module_name=dotted_name< {dotted_name} any* >
                        )
                        'as' module_nickname=any
                    >
                )
            >
        |
            import_from< 'from'
                (
                    module_name='{name}'
                |
                    module_name=dotted_name< {dotted_name} any* >
                )
                'import' ['(']
                (
                    import_as_name<
                        module_import=any
                        'as'
                        module_nickname=any
                    >*
                |
                    import_as_names<
                        module_imports=any*
                    >
                |
                    module_import=any
                )
             [')'] >
        |
             module_name=power<
                [TOKEN]
                any
                module_access=trailer< any* >*
            >
        )
    """
    _kwargs = {}
    _kwargs['name'] = 'paddle'
    _kwargs["dotted_name"] = " ".join(quoted_parts(_kwargs["name"]))
    _kwargs["power_name"] = " ".join(power_parts(_kwargs["name"]))
    pattern = pattern.format(**_kwargs)
    _imports_full_name = {}

    def _find_imports(node: LN, capture: Capture, filename: Filename) -> bool:
        if capture and ('module_imports' in capture or 'module_nickname' in capture):
            if filename not in _imports_full_name:
                _imports_full_name[filename] = {}
            if 'module_imports' in capture:
                for leaf in capture['module_imports']:
                    if leaf.type == TOKEN.NAME:
                        old_name = leaf.value.strip()
                        new_name = str(capture['module_name']).strip() + '.' + old_name
                        _imports_full_name[filename][old_name] = new_name
            if 'module_nickname' in capture:
                old_name = str(capture['module_nickname']).strip()
                new_name = str(capture['module_name']).strip()
                _imports_full_name[filename][old_name] = new_name
            return False
        return True

    q.select(pattern).filter(_find_imports)

    def _rename(node: LN, capture: Capture, filename: Filename) -> None:
        if filename not in _imports_full_name:
            return
        logger.debug(f"{filename} [{list(capture)}]: {node}")

        rename_dict = _imports_full_name[filename]
        # If two keys reference the same underlying object, do not modify it twice
        visited: List[LN] = []
        for _key, value in capture.items():
            logger.debug(f"{_key}: {value}")
            if value in visited:
                continue
            visited.append(value)

            if isinstance(value, Leaf) and value.type == TOKEN.NAME:
                if value.value in rename_dict:
                    # find old_name and new_name
                    old_name = value.value
                    new_name = rename_dict[old_name]
                    if value.parent is not None:
                        value.replace(Name(new_name, prefix=value.prefix))
                        break
            elif isinstance(value, Node):
                # find old_name and new_name
                code = str(value).strip()
                old_name = None
                for k, _ in rename_dict.items():
                    if not code.startswith(k):
                        continue
                    if old_name is None:
                        old_name = k
                    if len(k) > len(old_name):
                        old_name = k
                if old_name is None:
                    continue
                new_name = rename_dict[old_name]

                if type_repr(value.type) == "dotted_name":
                    dp_old = dotted_parts(old_name)
                    dp_new = dotted_parts(new_name)
                    parts = zip(dp_old, dp_new, value.children)
                    for old, new, leaf in parts:
                        if old != leaf.value:
                            break
                        if old != new:
                            leaf.replace(Name(new, prefix=leaf.prefix))

                    if len(dp_new) < len(dp_old):
                        # if new path is shorter, remove excess children
                        del value.children[len(dp_new) : len(dp_old)]
                    elif len(dp_new) > len(dp_old):
                        # if new path is longer, add new children
                        children = [
                            Name(new) for new in dp_new[len(dp_old) : len(dp_new)]
                        ]
                        value.children[len(dp_old) : len(dp_old)] = children

                elif type_repr(value.type) == "power":
                    # We don't actually need the '.' so just skip it
                    dp_old = old_name.split(".")
                    dp_new = new_name.split(".")

                    for old, new, leaf in zip(dp_old, dp_new, value.children):
                        if isinstance(leaf, Node):
                            name_leaf = leaf.children[1]
                        else:
                            name_leaf = leaf
                        if old != name_leaf.value:
                            break
                        name_leaf.replace(Name(new, prefix=name_leaf.prefix))

                    if len(dp_new) < len(dp_old):
                        # if new path is shorter, remove excess children
                        del value.children[len(dp_new) : len(dp_old)]
                    elif len(dp_new) > len(dp_old):
                        # if new path is longer, add new trailers in the middle
                        for i in range(len(dp_old), len(dp_new)):
                            value.insert_child(
                                i, Node(SYMBOL.trailer, [Dot(), Name(dp_new[i])])
                            )
    q.modify(_rename)
    # change module_access
    pass
    # add "import paddle" if needed
    pass
    # remove import_name and import_from
    pass
    #q.select_module('paddle')
    return q

def norm_api_alias(q: Query, change_spec) -> "Query":
    """
    rename all alias to main alias. e.g.
    origin code snippet:
       ```
       a = path1.to1.alias1()
       ```
    refactored code snippet:
       ```
       a = path2.to2.main_alias()
       ```
    """
    return q

def args_to_kwargs(q:Query, change_spec) -> "Query":
    """
    convert args to kwargs. e.g.
    origin code snippet:
        ```
        a = path.to.api(1, 2)
        ```
    refactored code snippet:
        ```
        a = path.to.api(x=1, y=2)
        ```
    """
    pattern = """
    (
        power< name=any* trailer<  '(' arglist=any* ')' > >
    )
    """
    def _get_func_name(lns: list):
        func_name = ""
        for ln in lns:
            if isinstance(ln, Leaf):
                func_name = func_name + ln.value
            elif isinstance(ln, Node):
                for l in ln.leaves():
                    func_name = func_name + l.value
        return func_name

    def _filter_func(node, capture, fu):
        name = capture["name"]
        func_name = _get_func_name(name)
        return func_name in change_spec

    def _modify_args_to_kwargs(node, capture, fn):
        args = capture["arglist"]
        name = capture["name"]
        func_name = _get_func_name(name)

        print(func_name)
        arg_list = change_spec[func_name]['args_list']
        if args and args[0].type == SYMBOL.arglist:
            child = args[0].children

        index = 0
        for x in child:
            if x.type == SYMBOL.argument:
                index = index + 1
            elif x.type != TOKEN.COMMA:
                x.replace(KeywordArg(Name(arg_list[index]), x.clone()))
                index = index + 1

    q.select(pattern).filter(_filter_func).modify(_modify_args_to_kwargs)

    return q

def args_warning(q:Query, change_spec) -> "Query":
    """
    print warning if specified args are used.
    """
    return q

def refactor_kwargs(q:Query, change_spec) -> "Query":
    """
    rename, remove or add kwargs. e.g.
    origin code snippet:
        ```
        a = path.to.api(k1='v1', k2='v2')
        ```
    refactor rule is: [('k1', 'k2_rename'), ('k2', ''), ('', 'k3', 'v3')]
    refactored code snippet:
        ```
        a = path.to.api(k1_rename='v1', k3='v3')
        ```
    """
    return q

def api_warning(q:Query, change_spec) -> "Query":
    """
    print warning if specified api are used.
    """
    return q

def api_rename(q:Query, change_spec) -> "Query":
    """
    rename old api to new api. e.g.
    origin code snippet:
        ```
        a = old_path.old_to.old_api(1, 2)
        ```
    refactored code snippet:
        ```
        a = new_path.new_to.new_api(1, 2)
        ```
    """
    return q

def refactor_syntax(q:Query, change_spec) -> "Query":
    """
    refactor syntax, such as removing "with" statement. e.g.
    origin code snippet:
        ```
        with paddle.fluid.dygraph.guard(place):
            path.to.api()
        ```
    refactored code snippet:
        ```
        paddle.disable_static(place)
        path.to.api()
        ```
    """
    return q

def post_refactor(q:Query, change_spec) -> "Query":
    """
    post refactor after all prior refactor steps.
    """
    return q

