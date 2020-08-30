change_spec = {
    "paddle.to.old_api": {
        "alias": [
            "paddle.to.old_api_alias1",
            "paddle.to1.to2.old_api_alias2",
            ],
        "update_to": "paddle.to.new_api",
        "warning": "this api is deprecated.",
        "args_list": ["arg1", "arg2"],
        "args_change": [
            ("arg2", "arg2_rename"),
            ("arg3", ""),
            ("", "new_arg", "default_value"),
            ],
        "args_warning": {"arg1":"warning message"},
        "args_transformer": "default_transformer",
    },
    "paddle.optimizer.AdamOptimizer": {
        "alias": [
            "paddle.fluid.optimizer.AdamOptimizer",
            ],
        "update_to": "paddle.AdamOptimizer",
    },
    "paddle.optimizer.TestOptimizer": {
        "warning": "this api is deprecated, use paddle.TestOptimizer please.",
    },

}
