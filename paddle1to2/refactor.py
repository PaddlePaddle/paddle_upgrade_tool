from bowler import Query
from bowler.helpers import power_parts, quoted_parts, dotted_parts
from bowler.types import LN, Capture, Filename

from fissix.pytree import Leaf, Node, type_repr
from fissix.fixer_util import Attr, Comma, Dot, LParen, Name, Newline, RParen, KeywordArg, Number, ArgList
from fissix.fixer_util import is_import, touch_import, find_root
from fissix.pygram import python_grammar, python_symbols
from fissix.patcomp import PatternCompiler
from fissix.pgen2 import token

from paddle1to2.common import logger
from paddle1to2 import processors, fixers, utils, transformers
from paddle1to2.utils import log_debug, log_info, log_warning, log_error


# don't change the order if you don't know what you are doing.
__all__ = [
    'refactor_import',
    'norm_api_alias',
    'args_to_kwargs',
    'refactor_kwargs',
    'api_rename',
    'refactor_with',
    'post_refactor',
    ]

def refactor_demo(q: Query, change_spec):
    #q.select_function("old_api").is_call().rename("new_api").process(processors.demo_post_processor)
    #q.select_function("old_api").rename("new_api")

    #q.fixer(fixers.FixerDemo)
    return q

def refactor_import(q: Query, change_spec):
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
    paddle_imported = set()
    paddle_found = set()

    def _find_imports(node: LN, capture: Capture, filename: Filename):
        if not is_import(node):
            return True
        if capture and 'name_import' in capture:
            paddle_imported.add(filename)
            paddle_found.add(filename)
        if capture and ('module_import' in capture or 'module_imports' in capture or 'module_nickname' in capture):
            paddle_found.add(filename)
            if filename not in imports_map:
                imports_map[filename] = {}
            if 'module_import' in capture:
                leaf = capture['module_import']
                if leaf.type == token.NAME:
                    old_name = leaf.value.strip()
                    new_name = str(capture['module_name']).strip() + '.' + old_name
                    imports_map[filename][old_name] = new_name
            if 'module_imports' in capture:
                for leaf in capture['module_imports']:
                    if leaf.type == token.NAME:
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
    def _full_module_path(node: LN, capture: Capture, filename: Filename):
        if not (isinstance(node, Leaf) and node.type == token.NAME):
            return
        if filename not in imports_map:
            return
        logger.debug("{} [{}]: {}".format(filename, list(capture), node))

        # skip import statement
        p = node.parent
        while p is not None:
            if p.type in {python_symbols.import_name, python_symbols.import_from}:
                return
            p = p.parent
        # skip if it's already a full module path
        if node.prev_sibling is not None and node.prev_sibling.type == token.DOT:
            return

        rename_dict = imports_map[filename]
        if node.value in rename_dict:
            # find old_name and new_name
            old_name = node.value
            new_name = rename_dict[old_name]
            if node.parent is not None:
                _node = utils.code_repr(new_name).children[0].children[0]
                _node.parent = None
                new_node = _node
                new_node.children[0].prefix = node.prefix
                if node.parent.type == python_symbols.power:
                    node.replace(new_node.children)
                else:
                    node.replace(new_node)
                log_info(filename, node.get_lineno(), "{} -> {}".format(utils.node2code(node), utils.node2code(new_node)))
    q.modify(_full_module_path)

    # remove as_import and from_import
    def _remove_import(node: LN, capture: Capture, filename: Filename):
        if not is_import(node):
            return
        _node = capture.get('as_import', None) or capture.get('from_import', None)
        if _node is not None:
            prefix = _node.prefix
            p = _node.parent
            _node.remove()
            log_warning(filename, p.get_lineno(), 'remove "{}"'.format(utils.node2code(_node)))
            # delete NEWLINE node after delete as_import or from_import
            if p and p.children and len(p.children) == 1 and p.children[0].type == token.NEWLINE:
                p.children[0].remove()
                # restore comment
                p.next_sibling.prefix = prefix + p.next_sibling.prefix
    q.modify(_remove_import)

    # add "import paddle" if needed
    def _add_import(node: LN, capture: Capture, filename: Filename):
        if node.type != python_symbols.file_input:
            return
        if filename in paddle_imported:
            return
        if filename in paddle_found:
            touch_import(None, 'paddle', node)
            log_info(filename, node.get_lineno(), 'add "import paddle"')
            paddle_imported.add(filename)
    q.modify(_add_import)

    return q

def norm_api_alias(q: Query, change_spec):
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
    def _norm(node: LN, capture: Capture, filename: Filename):
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
        log_info(filename, node.get_lineno(), '{} -> {}'.format(alias, main_alias))
    q.select(pattern).modify(_norm)

    return q

def args_to_kwargs(q:Query, change_spec):
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
        power< api=('paddle' any*) trailer_node=trailer< '(' any* ')' > >
    )
    """

    def _modify_args_to_kwargs(node, capture, filename):
        #get full api, e.g. paddle.fluid.layers.Layer
        api_name = utils.node2code(capture["api"]).strip()
        if api_name not in change_spec:
            return
        trailer_node = capture["trailer_node"]
        utils.norm_arglist(trailer_node)
        args_list = change_spec[api_name].get('args_list', None)

        encounter_kwarg = False
        idx = 0
        def _add_arg_name(argument_node):
            nonlocal encounter_kwarg
            nonlocal idx
            if args_list is None:
                return
            if encounter_kwarg:
                return
            if idx >= len(args_list):
                msg = 'args_list: "{}" is shorter than positional arguments.'.format(args_list)
                log_error(filename, argument_node.get_lineno(), msg)
                return
            if len(argument_node.children) >= 3:
                encounter_kwarg = True
                msg = 'args_list: "{}" is longer than positional arguments, redundant arguments will be skipped.'.format(args_list)
                log_info(filename, argument_node.get_lineno(), msg)
                return
            key = args_list[idx]
            argument_node.insert_child(0, Leaf(token.EQUAL, "="))
            argument_node.insert_child(0, Name(key))
            argument_node.children[0].prefix = argument_node.children[2].prefix
            argument_node.children[2].prefix = ""
            idx += 1
            msg = 'add argument name "{}" for {}-th argument.'.format(key, idx)
            log_debug(filename, argument_node.get_lineno(), msg)
        utils.apply_argument(filename, trailer_node, _add_arg_name)

    q.select(pattern).modify(_modify_args_to_kwargs)
    return q

def refactor_kwargs(q:Query, change_spec):
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
        power< api=('paddle' any*) trailer_node=trailer<  '(' any* ')' > >
    )
    """
    def _refector_args(node: LN, capture: Capture, filename: Filename):
        #get full api, e.g. paddle.fluid.layers.Layer
        api_name = utils.node2code(capture["api"]).strip()
        if api_name not in change_spec:
            return
        trailer_node = capture["trailer_node"]
        utils.norm_arglist(trailer_node)
        args_change = change_spec[api_name].get('args_change', [])

        for change in args_change:
            # add new keyword argument
            if len(change) == 3:
                old_arg = change[0].strip()
                new_arg = change[1].strip()
                arg_val = change[2].strip()
                # old_arg is not empty, do nothing
                if old_arg != "" or new_arg == "":
                    logger.error('add argument error. api: "{}", args_change: "{}", format should be ["", "new_arg", "default_value"]'.format(api_name, change))
                    continue

                utils.add_argument(filename, trailer_node, new_arg, arg_val)
            # delete or rename keyword argument 
            elif len(change) == 2:
                old_arg = change[0].strip()
                new_arg = change[1].strip()
                if old_arg == "" and new_arg == "":
                    logger.error('api: "{}", args_change: "{}", format should be ["arg", ""] or ["old_arg", "new_arg"]'.format(api_name, change))
                    continue

                if new_arg == '':
                    removed_value = utils.remove_argument(filename, trailer_node, old_arg)
                    if old_arg == 'act' and removed_value is not None:
                        transformers.act_transformer(filename, trailer_node, removed_value)
                else:
                    utils.rename_argument(filename, trailer_node, old_arg, new_arg)
            else:
                logger.error('api: "{}", args_change: "{}", format should be ["arg", ""] or ["old_arg", "new_arg"] or ["", "new_arg", "default_value"]'.format(api_name, change))

        # if api in args_warning, print warning info
        args_warning = change_spec[api_name].get("args_warning", {})
        def _print_warning(argument_node):
            if argument_node.type != python_symbols.argument:
                return
            if len(argument_node.children) == 3:
               key = argument_node.children[0].value
               if key in args_warning:
                   warning_msg = args_warning[key]
                   log_warning(filename, argument_node.get_lineno(), warning_msg)
        utils.apply_argument(filename, trailer_node, _print_warning)
                
        # run customized transformer
        if "args_transformer" in change_spec[api_name]:
            transformer_func = eval("transformers." + change_spec[api_name]["args_transformer"])
            transformer_func(node, capture, filename)

    q.select(pattern).modify(_refector_args)
    return q

def api_rename(q:Query, change_spec):
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
    def _api_rename(node: LN, capture: Capture, filename: Filename):
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

def refactor_with(q:Query, change_spec):
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
    def _remove_with_dygraph_guard(node: LN, capture: Capture, filename: Filename):
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
        simple_stmt_disable_static = Node(python_symbols.simple_stmt, [Newline()])
        _node = utils.code_repr('paddle.disable_static' + str(arg_list_nodes)).children[0].children[0]
        _node.parent = None
        simple_stmt_disable_static.insert_child(0, _node)
        simple_stmt_disable_static.prefix = with_node.prefix

        # create simple_stmt node for "paddle.enable_static"
        simple_stmt_enable_static = Node(python_symbols.simple_stmt, [Newline()])
        simple_stmt_enable_static
        _node = utils.code_repr('paddle.enable_static()').children[0].children[0]
        _node.parent = None
        simple_stmt_enable_static.insert_child(0, _node)
        simple_stmt_enable_static.prefix = utils.get_indent(with_node)

        suite_node = capture['suite']
        # remove first newline
        for node in suite_node.children:
            if not isinstance(node, Leaf):
                continue
            if node.type == token.NEWLINE:
                node.remove()
                break
        # remove first indent node, and add indent prefix to sibling node.
        indent = None
        for node in suite_node.children:
            if not isinstance(node, Leaf):
                continue
            if node.type == token.INDENT:
                indent = node.value
                if node.next_sibling is not None:
                    node.next_sibling.prefix = node.prefix + indent
                node.remove()
                break

        # transfer post leading dedent node prefix to sibling of with node
        leaves = [leaf for leaf in suite_node.leaves()]
        # visit all leaves in reversed order
        last_dedent_leaf_idx = len(leaves)
        for leaf in leaves[::-1]:
            if leaf.type == token.DEDENT:
                with_node.next_sibling.prefix = leaf.prefix + with_node.next_sibling.prefix
                leaf.prefix = ""
            else:
                break

        # remove dedenet node corresponding to with node
        for node in suite_node.children[::-1]:
            if not isinstance(node, Leaf):
                continue
            if node.type == token.DEDENT:
                node.remove()
                break

        # unindent all code in suite
        for node in suite_node.leaves():
            if node.type == token.INDENT:
                node.value = utils.dec_indent(node.value)
            else:
                node.prefix = utils.dec_indent(node.prefix)

        with_node.remove()
        parent.insert_child(idx, simple_stmt_disable_static)
        idx += 1
        for node in suite_node.children:
            parent.insert_child(idx, node)
            idx += 1
        parent.insert_child(idx, simple_stmt_enable_static)

    q.select(pattern).modify(_remove_with_dygraph_guard)
    return q

def post_refactor(q:Query, change_spec):
    """
    post refactor after all prior refactor steps.
    """
    return q

