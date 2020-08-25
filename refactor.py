from bowler import Query
from common import logger

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
    q.select_function("old_api").is_call().rename("new_api")
    return q

def refactor_import(q: Query, change_spec) -> "Query":
    """
    1. add "import paddle" if needed.
    2. remove "import paddle.mod" if needed.
    3. remove "import paddle.module as mod", and convert "mod.api" to "paddle.mod.api"
    4. remove "from paddle.module import api", and convert "api" to "paddle.module.api"
    """
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

