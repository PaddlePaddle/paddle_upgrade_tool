### Change Spec:
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

### Acknowledgements
- [Bowler](https://github.com/facebookincubator/Bowler/): Safe code refactoring for modern Python projects.
- [lib2to3](https://github.com/python/cpython/tree/master/Lib/lib2to3): A built-in python library to refactor python code.
