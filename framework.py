import os
import inspect
from functools import wraps
from attr import Factory, dataclass
from typing import Callable, List, Optional, Type, TypeVar, Union, cast
from typing import Any, Callable, Dict, List, NewType, Optional, Type, Union

__all__ = ['Query']


class FileNotFoundException(Exception):
    def __init__(self, message):
        super(FileNotFoundException, self).__init__(message)

def dotted_parts(name: str) -> List[str]:
    """
    'path.to.api' -> ['path', '.', 'to', '.', 'api']
    """
    pre, dot, post = name.partition(".")
    if post:
        post_parts = dotted_parts(post)
    else:
        post_parts = []
    result = []
    if pre:
        result.append(pre)
    if pre and dot:
        result.append(dot)
    if post_parts:
        result.extend(post_parts)
    return result


def quoted_parts(name: str) -> List[str]:
    """
    'path.to.api' -> ["'path'", "'.'", "'to'", "'.'", "'api'"]
    """
    return [f"'{part}'" for part in dotted_parts(name)]


def power_parts(name: str) -> List[str]:
    """
    'path.to.api' -> ["'path'", 'trailer<', "'.'", "'to'", '>', 'trailer<', "'.'", "'api'", '>']
    """
    parts = quoted_parts(name)
    index = 0
    while index < len(parts):
        if parts[index] == "'.'":
            parts.insert(index, "trailer<")
            parts.insert(index + 3, ">")
            index += 1
        index += 1
    return parts

@dataclass
class Transform:
    selector: str = ""
    kwargs: Dict[str, Any] = Factory(dict)
    filters: List[Filter] = Factory(list)
    callbacks: List[Callback] = Factory(list)
    fixer: Optional[Type[BaseFix]] = None

SELECTORS = {}
Q = TypeVar("Q", bound="Query")
QM = Callable[..., Q]
# selector decorator
def selector(pattern: str) -> Callable[[QM], QM]:
    def wrapper(fn: QM) -> QM:
        selector = fn.__name__.replace("select_", "").lower()
        SELECTORS[selector] = pattern.strip()

        signature = inspect.signature(fn)
        arg_names = list(signature.parameters)[1:]

        @wraps(fn)
        def wrapped(self: Q, *args, **kwargs) -> Q:
            for arg, value in zip(arg_names, args):
                if hasattr(value, "__name__"):
                    kwargs["source"] = value
                    kwargs[arg] = value.__name__
                else:
                    kwargs[arg] = str(value)

            if "name" in kwargs:
                kwargs["dotted_name"] = " ".join(quoted_parts(kwargs["name"]))
                kwargs["power_name"] = " ".join(power_parts(kwargs["name"]))
            self.transforms.append(Transform(selector, kwargs))
            return self

        return wrapped

    return wrapper

class Query:
    def __init__(self, path):
        self.path = path
        self.transforms: List[Transform] = []
        if not os.path.exists(self.path):
            raise FileNotFoundException('path: "{}" does not exist.'.format(self.path))

    def __str__(self):
        return 'Query("{}")'.format(self.path)

    @selector(
        """
        file_input< any* >
    """
    )
    def select_root(self) -> "Query":
        pass 

    @selector(
        """
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
                {power_name}
                module_access=trailer< any* >*
            >
        )
    """
    )
    def select_module(self, name: str) -> "Query":
        pass
