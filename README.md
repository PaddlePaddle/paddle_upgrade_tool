Upgrade your python model from paddle-1.x to paddle-2.

[![Build Status](https://travis-ci.org/PaddlePaddle/paddle_upgrade_tool.svg?branch=master)](https://travis-ci.org/PaddlePaddle/paddle_upgrade_tool)
[![Coverage Status](https://coveralls.io/repos/github/PaddlePaddle/paddle_upgrade_tool/badge.svg?branch=master&kill_cache=1)](https://coveralls.io/github/PaddlePaddle/paddle_upgrade_tool?branch=master)
[![Version](https://img.shields.io/pypi/v/paddle_upgrade_tool)](https://pypi.org/project/paddle_upgrade_tool)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

### Attention
`paddle_upgrade_tool` aims to convert python files from paddle-1.x to paddle-2 one by one, it won't handle indirect imports. e.g.

```python
# filename "a.py"
import paddle.fluid as fluid
pass

# filename "b.py"
from a import *
class MyLayer(fluid.layers.Layer):
    pass
```

`fluid.layers.Layer` in "b.py" won't get converted.
**So you have to make sure you have import all used paddle modules, classes, objects directly for every python file before running `paddle_upgrade_tool`.**

### Install
paddle_upgrade_tool support Linux, Mac OS, Windows([Git Bash](https://gitforwindows.org/) is recommended), **but it requires Python 3.5.4 or higher to run**. Multi-Process is supported for Linux and Mac OS, Single-Process is support for Windows, this will lead to performance difference.

1. install with pip

```bash
pip install -U paddle_upgrade_tool
paddle_upgrade_tool --help # show help
paddle_upgrade_tool --inpath /path/to/model.py # upgrade your model from paddle-1.x to paddle-2.0
```

**ATTENTION**: If your device contains multiple versions of python, you may need to run the following commands instead:
```bash
python3 -m pip install -U paddle_upgrade_tool
python3 -m paddle_upgrade_tool -h
python3 -m paddle_upgrade_tool --inpath /path/to/model.py
```

2. install from source

```bash
git clone https://github.com/T8T9/paddle_upgrade_tool.git
cd paddle_upgrade_tool
python setup.py sdist bdist_wheel
pip install -U ./dist/paddle_upgrade_tool-*.whl
paddle_upgrade_tool --help # show help
paddle_upgrade_tool --inpath /path/to/model.py # upgrade your model from paddle-1.x to paddle-2.0
```

### Develop
If you are a develop, and you want to test your code quickly, you can run the following command in project directory:

```bash
python -m paddle_upgrade_tool --inpath /path/to/model.py

#or 

python paddle_upgrade_tool/main.py --inpath /path/to/model.py
```

Moreover, if you want to run a specific refactor, you can use the following command:

```bash
python -m paddle_upgrade_tool --inpath /path/to/model.py --refactor <refactor_name>
```

use `python -m paddle_upgrade_tool -h` to see full list of all refactors.

if you want to run all unittest, use command:

```bash
python -m unittest discover paddle_upgrade_tool/tests/
# or
python setup.py test
```
or use command:

```bash
python -m unittest paddle_upgrade_tool/tests/test_refactor.py
```
to run specific test file.

### Change Spec
`change_spec` is a python dict defined in spec.py, it defines the rules to refactor your code.

```python
change_spec = {
    "path.to.old_api": {
        "alias": [
            "path.to.old_api_alias1",
            "path.to1.to2.old_api_alias2",
            ],
        "update_to": "path.to.new_api",
        "warning": "this api is deprecated.",
        "args_list": ["arg1", "arg2"],
        "args_change": [
                ["arg2", "arg2_rename"],
                ["arg3", ""],
                ["", "new_arg", "default_value"],
            ],
        "args_warning": {"arg1":"warning message"},
        "args_transformer": "_default_transformer",
    },
}
```

- `alias`: a list of alias of main alias `path.to.old_api`, all alias will be replaced with main alias.
- `update_to`: `path.to.old_api` will be replaced with this new api if specified.
- `warning`: print specified warning message when `path.to.old_api` is found. This field will be ignored if `update_to` is specified.
- `args_list`: is argument list of `path.to.old_api`.
- `args_change`: a list of list. It contains following format:
  - `["arg", "new_arg"]`: rename a argument, e.g. `func(arg=value)` -> `func(new_arg=value)`
  - `["arg", ""]`: remove a argument, e.g. `func(arg=value)` -> `func()`
  - `["", "new_arg", "default_value"]`: add a new argument, e.g. `func(arg=value)` -> `func(arg=value, new_arg=default_value)`
- `args_warning`: print specified warning message for specified argument after apply `args_change`.
- `args_transformer`: execute customized transformer on an [AST node](https://github.com/python/cpython/blob/75c80b0bda89debf312f075716b8c467d411f90e/Lib/lib2to3/pytree.py#L207), it will be called after applying `args_change` to do further refactor.

### Other Tools
1. find pattern of specific code snippet, usage:

```bash
find_pattern 'import paddle'
```
`find_pattern` command will traverse all nodes in AST, if you see code snippet you want, type in 'y' to get pattern.

2. find match node in specific code for specific pattern, usage:

```bash
find_match_node -ss 'import paddle' -ps 'any'
```

you can also specify "--print-results" option to got representation of matched node, specify "--print-lineno" to got line number of matched code.


### Acknowledgements
- [Bowler](https://github.com/facebookincubator/Bowler/): Safe code refactoring for modern Python projects.
- [lib2to3](https://github.com/python/cpython/tree/master/Lib/lib2to3): A built-in python library to refactor python code.
- [fissix](https://github.com/jreese/fissix/): A backport of latest lib2to3, with enhancements.
