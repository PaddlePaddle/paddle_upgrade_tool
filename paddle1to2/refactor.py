from bowler import Query
from bowler.helpers import power_parts, quoted_parts, dotted_parts
from bowler.types import LN, Capture, Filename, SYMBOL, TOKEN

from fissix.pytree import Leaf, Node, type_repr
from fissix.fixer_util import Attr, Comma, Dot, LParen, Name, Newline, RParen, KeywordArg, Number, ArgList, Newline
from fissix.fixer_util import is_import, touch_import, find_root
from fissix.pygram import python_grammar, python_symbols
from fissix.patcomp import PatternCompiler

from paddle1to2.common import logger
from paddle1to2 import processors, fixers, utils, transformers
from paddle1to2.utils import log_debug, log_info, log_warning, log_error


# don't change the order if you don't know what you are doing.
__all__ = [
    'refactor_demo',
    'refactor_import',
    'norm_api_alias',
    'args_to_kwargs',
    'refactor_kwargs',
    'api_rename',
    'refactor_with',
    'post_refactor',
    ]

def refactor_demo(q: Query, change_spec) -> "Query":
    #q.select_function("old_api").is_call().rename("new_api").process(processors.demo_post_processor)
    q.select_function("old_api").rename("new_api")

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
            file_input< any* >
         |
            name_import=import_name< 'import' '{name}' >
         |
            as_import=import_name< 'import'
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
            from_import=import_from< 'from'
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
             leaf_node=NAME
        )
    """
    _kwargs = {}
    _kwargs['name'] = 'paddle'
    _kwargs["dotted_name"] = " ".join(quoted_parts(_kwargs["name"]))
    _kwargs["power_name"] = " ".join(power_parts(_kwargs["name"]))
    pattern = pattern.format(**_kwargs)

    imports_map = {}
    paddle_imported = False
    paddle_found = False

    def _find_imports(node: LN, capture: Capture, filename: Filename) -> bool:
        nonlocal paddle_imported, paddle_found
        if not is_import(node):
            return True
        if capture and 'name_import' in capture:
            paddle_imported = True
            paddle_found = True
        if capture and ('module_import' in capture or 'module_imports' in capture or 'module_nickname' in capture):
            paddle_found = True
            if filename not in imports_map:
                imports_map[filename] = {}
            if 'module_import' in capture:
                leaf = capture['module_import']
                if leaf.type == TOKEN.NAME:
                    old_name = leaf.value.strip()
                    new_name = str(capture['module_name']).strip() + '.' + old_name
                    imports_map[filename][old_name] = new_name
            if 'module_imports' in capture:
                for leaf in capture['module_imports']:
                    if leaf.type == TOKEN.NAME:
                        old_name = leaf.value.strip()
                        new_name = str(capture['module_name']).strip() + '.' + old_name
                        imports_map[filename][old_name] = new_name
            if 'module_nickname' in capture:
                old_name = str(capture['module_nickname']).strip()
                new_name = str(capture['module_name']).strip()
                imports_map[filename][old_name] = new_name
        return True

    q.select(pattern).filter(_find_imports)
    # convert to full module path
    def _full_module_path(node: LN, capture: Capture, filename: Filename) -> None:
        if not (isinstance(node, Leaf) and node.type == TOKEN.NAME):
            return
        if filename not in imports_map:
            return
        logger.debug("{} [{}]: {}".format(filename, list(capture), node))

        # skip import statement
        p = node.parent
        while p is not None:
            if p.type in {SYMBOL.import_name, SYMBOL.import_from}:
                return
            p = p.parent
        # skip if it's already a full module path
        if node.prev_sibling is not None and node.prev_sibling.type == TOKEN.DOT:
            return

        rename_dict = imports_map[filename]
        if node.value in rename_dict:
            # find old_name and new_name
            old_name = node.value
            new_name = rename_dict[old_name]
            if node.parent is not None:
                new_node = utils.code_repr(new_name)
                new_node.children[0].prefix = node.prefix
                if node.parent.type == SYMBOL.power:
                    node.replace(new_node.children)
                else:
                    node.replace(new_node)
                log_info(filename, node.get_lineno(), "{} -> {}".format(utils.node2code(node), utils.node2code(new_node)))
    q.modify(_full_module_path)

    # remove as_import and from_import
    def _remove_import(node: LN, capture: Capture, filename: Filename) -> None:
        if not is_import(node):
            return
        _node = capture.get('as_import', None) or capture.get('from_import', None)
        if _node is not None:
            prefix = _node.prefix
            p = _node.parent
            _node.remove()
            log_warning(filename, p.get_lineno(), 'remove "{}"'.format(utils.node2code(_node)))
            # delete NEWLINE node after delete as_import or from_import
            if p and p.children and len(p.children) == 1 and p.children[0].type == TOKEN.NEWLINE:
                p.children[0].remove()
                # restore comment
                p.next_sibling.prefix = prefix + p.next_sibling.prefix
    q.modify(_remove_import)

    # add "import paddle" if needed
    def _add_import(node: LN, capture: Capture, filename: Filename) -> None:
        nonlocal paddle_imported, paddle_found
        if node.type != SYMBOL.file_input:
            return
        if paddle_imported:
            return
        if paddle_found:
            touch_import(None, 'paddle', node)
            log_info(filename, node.get_lineno(), 'add "import paddle"')
            paddle_imported = True
    q.modify(_add_import)

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
    # construct alias mapping
    alias_map = {}
    for main_alias, v in change_spec.items():
        for alias in v.get('alias', []):
            alias_map[alias] = main_alias

    pattern = """ power< 'paddle' trailer< any* >* > """
    def _norm(node: LN, capture: Capture, filename: Filename) -> None:
        code = ''
        for leaf in node.leaves():
            code = code + leaf.value
        found_alias = False
        alias = None
        for _alias in alias_map.keys():
            if utils.startswith(code, _alias):
                found_alias = True
                alias = _alias
                break
        if not found_alias:
            return
        main_alias = alias_map[alias]
        update_to = change_spec[main_alias].get('update_to', None)
        # if main_alias contains "update_to" field, rename alias to "update_to" directly
        utils.replace_module_path(node, alias, main_alias)
        log_warning(filename, node.get_lineno(), '{} -> {}'.format(alias, main_alias))
    q.select(pattern).modify(_norm)

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
    # find all func call start with paddle
    pattern = """
    (
        power< name=('paddle' any*) trailer<  '(' arglist=any* ')' > >
    )
    """

    def _modify_args_to_kwargs(node, capture, fn):
        args = capture["arglist"]

        #get paddle func full name
        func_name = ""
        for node in capture["name"]:
            for l in node.leaves():
                func_name = func_name + l.value

        if func_name not in change_spec or 'args_list' not in change_spec[func_name]:
            return

        args_list = change_spec[func_name]['args_list']
        if len(args_list) == 0:
            return

        if isinstance(args[0], Leaf):
            if 1 != len(args_list):
                warning_msg = "argument list length not equal, raw func argument list length is 1, but expected length is {}".format(len(args_list))
                log_warning(fn, node.get_lineno(), warning_msg)
                return

            args[0].replace(KeywordArg(Name(args_list[0]), args[0].clone()))

        elif isinstance(args[0], Node):
            if args[0].type == SYMBOL.arglist:
                if len(args[0].children) != (len(args_list) *2-1):
                    warning_msg = "argument list length not equal, raw func argument list length is {}, but expected length is {}".format(int((len(args[0].children) +1 )/2), len(args_list))
                    log_warning(fn, node.get_lineno(), warning_msg)
                    return

                child = args[0].children
                index = 0
                for ln in child:
                    if ln.type == SYMBOL.argument:
                        index = index + 1
                    elif ln.type != TOKEN.COMMA:
                        ln.replace(KeywordArg(Name(args_list[index]), ln.clone()))
                        index = index + 1
            elif args[0].type == SYMBOL.argument:
                if 1 != len(args_list):
                    warning_msg = "argument list length not equal, raw func argument list length is 1, but expected length is {}".format(len(args_list))
                    log_warning(fn, node.get_lineno(), warning_msg)
                    return

                raw_arg_name = args[0].children[0].value
                if raw_arg_name != args_list[0]:
                    warning_msg = "exist function argument name ({}) not equal expected argument name ({})".format(raw_arg_name, args_list[0])
                    log_warning(fn, node.get_lineno(), warning_msg)
                    return


    q.select(pattern).modify(_modify_args_to_kwargs)

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
    # find all func call start with paddle
    pattern = """
    (
        power< name=('paddle' any*) function_parameters=trailer<  '(' any* ')' > >
    )
    """
    def _get_leaf(arg_val)->Leaf:
        if arg_val.lstrip('-').replace(".", "").isnumeric():
            return Leaf(TOKEN.NUMBER, arg_val)
        else:
            return Leaf(TOKEN.NAME, arg_val)

    def _refector_args(node: LN, capture: Capture, fn: Filename) -> None:
        func_para_node = capture["function_parameters"]

        #get paddle func full name
        func_name = ""
        for node in capture["name"]:
            for l in node.leaves():
                func_name = func_name + l.value

        if func_name not in change_spec:
            return
        
        args_change = change_spec[func_name].get('args_change', [])

        for arg_tuple in args_change:
            # add new keyword argument
            if len(arg_tuple) == 3:
                old_arg = arg_tuple[0]
                new_arg = arg_tuple[1]
                arg_val = arg_tuple[2]
                # old_arg is not empty, do nothing
                if old_arg != "":
                    continue

                arg_node = KeywordArg(Name(new_arg, prefix=" "), _get_leaf(arg_val))
                # f() -> f(new_arg = arg_val)
                if func_para_node.children[0].type == TOKEN.LPAR and func_para_node.children[1].type == TOKEN.RPAR:
                    arg_node = KeywordArg(Name(new_arg), _get_leaf(arg_val))
                    func_para_node.insert_child(1, arg_node)
                    log_info(fn, node.get_lineno(), "add keyword argument: {} = {}".format(new_arg, arg_val))
                    continue

                # f(1) -> f(1, new_arg = arg_val)
                if isinstance(func_para_node.children[1], Leaf):
                    # arguent -> arglist
                    func_para_node.children[1] = ArgList([func_para_node.children[1].clone(), Comma(), arg_node]).children[1]
                    log_info(fn, node.get_lineno(), "add keyword argument: {} = {}".format(new_arg, arg_val))
                    continue

                # f(x=1) -> f(x=1, new_arg = arg_val)
                if func_para_node.children[1].type == SYMBOL.argument:
                    if func_para_node.children[1].children[0].value == new_arg:
                        warning_msg = "can not add the exist arg_name = {} ".format(new_arg)
                        log_warning(fn, node.get_lineno(), warning_msg)
                    else:
                        # arguent -> arglist
                        func_para_node.children[1] = ArgList([func_para_node.children[1].clone(), Comma(), arg_node]).children[1]
                        log_info(fn, node.get_lineno(), "add keyword argument: {} = {}".format(new_arg, arg_val))
                    continue

                # f(x=1, y=2) -> f(x=1, y=2, new_arg= arg_val)
                if func_para_node.children[1].type == SYMBOL.arglist:
                    is_exist = False
                    for ln in func_para_node.children[1].children:
                        # kewword arg like x=1
                        if isinstance(ln, Node) and ln.type == SYMBOL.argument:
                            if ln.children[0].value == new_arg:
                                warning_msg = "can not add the exist arg_name = {} ".format(new_arg)
                                log_warning(fn, node.get_lineno(), warning_msg)
                                is_exist = True
                                break
                    # next tuple
                    if is_exist:
                        continue

                    #insert new_arg_node to the end
                    func_para_node.children[1].append_child(Comma())
                    func_para_node.children[1].append_child(arg_node)
                    log_info(fn, node.get_lineno(), "add keyword argument: {} = {}".format(new_arg, arg_val))
            # delete or rename keyword argument 
            elif len(arg_tuple) == 2:
                old_arg = arg_tuple[0]
                new_arg = arg_tuple[1]

                #f() can not do rename or delete operation
                if func_para_node.children == [LParen(), RParen()]:
                    log_warning(fn, node.get_lineno(), "can not rename or delete argument for empty function parameters")
                    continue

                #f(1) can not do rename or delete operation
                if isinstance(func_para_node.children[1], Leaf):
                    log_warning(fn, node.get_lineno(), "can not rename or delete argument for none keyword parameters")
                    continue

                # f(x=1)
                if func_para_node.children[1].type == SYMBOL.argument:
                    if func_para_node.children[1].children[0].value != old_arg:
                        log_warning(fn, node.get_lineno(), "can not find argument '{}' for delete or rename argument".format(old_arg))
                    else:
                        # f(x=1) -> f()
                        if new_arg == "":
                            func_para_node.children = [LParen(), RParen()]
                            log_info(fn, node.get_lineno(), "delete keyword argument: {}".format(old_arg))
                        # f(x=1) -> f(x_new = 1)
                        else:
                            func_para_node.children[1].children[0] = Name(new_arg, func_para_node.children[1].children[0].prefix)
                            log_info(fn, node.get_lineno(), 'rename keyword argument from {} to {}'.format(old_arg, new_arg))
                    continue

                # f(x=1, y=1)
                if func_para_node.children[1].type == SYMBOL.arglist:
                    is_exist = False
                    for ln in func_para_node.children[1].children:
                        # kewword arg like x=1
                        if isinstance(ln, Node) and ln.type == SYMBOL.argument:
                            if ln.children[0].value == old_arg:
                                #delete argument
                                if new_arg == "":
                                    if ln.next_sibling == Comma():
                                        ln.next_sibling.remove()
                                        ln.remove()
                                    else:
                                        if ln.prev_sibling == Comma():
                                            ln.prev_sibling.remove()
                                        ln.remove()
                                    log_info(fn, node.get_lineno(), 'delete keyword argument: {}'.format(old_arg))
                                #rename argument
                                else:
                                    ln.children[0] = Name(new_arg, ln.children[0].prefix)
                                    log_info(fn, node.get_lineno(), 'rename keyword argument from {} to {}'.format(old_arg, new_arg))
                                
                                is_exist = True
                                break
                    if not is_exist:
                        log_warning(fn, node.get_lineno(), "can not find argument '{}' for delete or rename argument".format(old_arg))

            else:
                log_warning(fn, node.get_lineno(), "the length of args_change tuple is not equal 2 or 3, api name ={}, tuple= {}".format(func_name, arg_tuple))

        # if api in args_warning, print warning info
        if "args_warning" in change_spec[func_name]:
            args_warning = change_spec[func_name]["args_warning"]
            if func_para_node.children[1].type == SYMBOL.argument:
                arg_name = func_para_node.children[1].children[0].value
                if arg_name in args_warning:
                    warning_info = args_warning[arg_name]
                    log_warning(fn, node.get_lineno(), warning_info)

            if func_para_node.children[1].type == SYMBOL.arglist:
                for n in func_para_node.children[1].children:
                    if isinstance(n, Node) and n.type == SYMBOL.argument:
                        arg_name = n.children[0].value
                        if arg_name in args_warning:
                            warning_info = args_warning[arg_name]
                            log_warning(fn, node.get_lineno(), warning_info)

        if "args_transformer" in change_spec[func_name]:
            transformer_func = eval("transformers." + change_spec[func_name]["args_transformer"])
            transformer_func(node, capture, fn)

    q.select(pattern).modify(_refector_args)
    return q

def api_rename(q:Query, change_spec) -> "Query":
    """
    1. rename old api to new api. e.g.
        origin code snippet:
            ```
            a = old_path.old_to.old_api(1, 2)
            ```
        refactored code snippet:
           ```
           a = new_path.new_to.new_api(1, 2)
           ```
    2. print warning if specified api are used.
    """
    # construct api rename mapping and api warning mapping
    rename_map = {}
    warning_map = {}
    for main_alias, v in change_spec.items():
        new_api_name = v.get('update_to', None)
        if new_api_name is not None:
            rename_map[main_alias] = new_api_name
        warning = v.get('warning', None)
        if warning is not None:
            warning_map[main_alias] = warning

    pattern = """ power< 'paddle' trailer< any* >* > """
    def _api_rename(node: LN, capture: Capture, filename: Filename) -> None:
        code = ''
        for leaf in node.leaves():
            code = code + leaf.value
        found_rename = False
        found_warning = False
        api = None
        for _api in rename_map.keys():
            if utils.startswith(code, _api):
                found_rename = True
                api = _api
                break
        for _api in warning_map.keys():
            if utils.startswith(code, _api):
                found_warning = True
                api = _api
                break
        if not found_rename and not found_warning:
            return
        # if found rename, replace old_api with new_api
        if found_rename:
            utils.replace_module_path(node, api, rename_map[api])
        # if not found rename and found warning, print warning
        elif found_warning:
            log_warning(filename, node.get_lineno(), warning_map[api])
    q.select(pattern).modify(_api_rename)

    return q

def refactor_with(q:Query, change_spec) -> "Query":
    """
    refactor with syntax, e.g.
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
    pattern = "with=with_stmt< 'with' guard=(power< api=(( 'paddle' | 'fluid' | 'dygraph' ) trailer< '.' NAME >* trailer< '.' 'guard' > ) arg_list=trailer< '(' any* ')' > >) any* suite=suite< any* > any* >"
    def _remove_with_dygraph_guard(node: LN, capture: Capture, filename: Filename) -> None:
        # index of with_node, with_node will be replaced with simple_stmt node
        with_node = capture['with']
        parent = with_node.parent
        idx = None
        for i, child in enumerate(parent.children):
            if child is with_node:
                idx = i
                break

        # create simple_stmt node for "paddle.disable_static"
        arg_list_nodes = capture['arg_list']
        simple_stmt = Node(SYMBOL.simple_stmt, [Newline()])
        simple_stmt.insert_child(0, utils.code_repr('paddle.disable_static' + str(arg_list_nodes)))
        simple_stmt.prefix = with_node.prefix

        suite_node = capture['suite']
        # remove first newline
        for node in suite_node.children:
            if not isinstance(node, Leaf):
                continue
            if node.type == TOKEN.NEWLINE:
                node.remove()
                break
        # remove first indent node, and add indent prefix to sibling node.
        indent = None
        for node in suite_node.children:
            if not isinstance(node, Leaf):
                continue
            if node.type == TOKEN.INDENT:
                indent = node.value
                if node.next_sibling is not None:
                    node.next_sibling.prefix = node.prefix + indent
                node.remove()
                break
        # remove last dedent node
        for node in suite_node.children[::-1]:
            if not isinstance(node, Leaf):
                continue
            if node.type == TOKEN.DEDENT:
                if with_node.next_sibling is not None:
                    with_node.next_sibling.prefix = node.prefix
                node.remove()
                break

        # unindent all code in suite
        for node in suite_node.leaves():
            if node.type == TOKEN.INDENT:
                node.value = utils.dec_indent(node.value)
            else:
                node.prefix = utils.dec_indent(node.prefix)

        with_node.remove()
        parent.insert_child(idx, simple_stmt)
        idx += 1
        for node in suite_node.children:
            parent.insert_child(idx, node)
            idx += 1

    q.select(pattern).modify(_remove_with_dygraph_guard)
    return q

def post_refactor(q:Query, change_spec) -> "Query":
    """
    post refactor after all prior refactor steps.
    """
    return q

