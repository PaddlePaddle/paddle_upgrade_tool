Upgrade your python model from paddle-1.x to paddle-2.0

### Change Spec
```
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
            ("arg2", "arg2_rename"),
            ("arg3", ""),
            ("", "new_arg", "default_value"),
            ],
        "args_warning": {"arg1":"warning message"},
        "args_transformer": "_default_transformer",
    },
}
```

### Install
1. install with pip

```bash
pip install paddle1to2
paddle1to2 --help # show help
paddle1to2 --inpath /path/to/model.py # upgrade your model from paddle-1.x to paddle-2.0
```

2. install from source

```bash
git clone https://github.com/T8T9/paddle1to2.git
cd paddle1to2
python setup.py sdist bdist_wheel
pip install -U ./dist/paddle1to2-*.whl
paddle1to2 --help # show help
paddle1to2 --inpath /path/to/model.py # upgrade your model from paddle-1.x to paddle-2.0
```

### Develop
If you are a develop, and you want to test your code quickly, you can run the following command in project directory:

```bash
python -m paddle1to2 --inpath /path/to/model.py

#or 

python paddle1to2/main.py --inpath /path/to/model.py
```

Moreover, if you want to run a specific refactor, you can use the following command:

```bash
python -m paddle1to2 --inpath /path/to/model.py --refactor <refactor_name>
```

use `python -m paddle1to2 -h` to see full list of all refactors.


### Acknowledgements
- [Bowler](https://github.com/facebookincubator/Bowler/): Safe code refactoring for modern Python projects.
- [lib2to3](https://github.com/python/cpython/tree/master/Lib/lib2to3): A built-in python library to refactor python code.
