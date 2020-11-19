change_spec = {
    "paddle.fluid.layers.clip": {
        "alias": [
            "paddle.fluid.layers.nn.clip"
        ],
        "update_to": "paddle.clip"
    },
    "paddle.fluid.data": {
        "update_to": "paddle.static.data"
    },
    "paddle.fluid.layers.nn.pow": {
        "alias": [
            "paddle.fluid.layers.pow"
        ],
        "update_to": "paddle.pow",
        "args_list": [
            "x",
            "factor",
            "name"
        ],
        "args_change": [
            [
                "factor",
                "y"
            ],
        ]
    },
    "paddle.fluid.layers.ops.ceil": {
        "alias": [
            "paddle.fluid.layers.ceil"
        ],
        "update_to": "paddle.ceil"
    },
    "paddle.fluid.layers.control_flow.while_loop": {
        "alias": [
            "paddle.fluid.layers.while_loop"
        ],
        "update_to": "paddle.static.nn.while_loop"
    },
    "paddle.fluid.layers.tensor.trace": {
        "alias": [
            "paddle.fluid.layers.trace"
        ],
        "update_to": "paddle.trace",
        "args_list": [
            "input",
            "offset",
            "dim1",
            "dim2",
            "out",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "dim1",
                "axis1"
            ],
            [
                "dim2",
                "axis2"
            ],
            [
                "out",
                ""
            ]
        ],
        "args_warning": {
            "out": "this arg is deleted in this version"
        }
    },
    "paddle.fluid.load": {
        "alias": [
            "paddle.fluid.io.load"
        ],
        "update_to": "paddle.static.load"
    },
    "paddle.fluid.layers.tensor.full_like": {
        "alias": [
            "paddle.fluid.layers.full_like"
        ],
        "update_to": "paddle.full_like",
        "args_list": [
            "input",
            "fill_value",
            "out",
            "dtype",
            "device",
            "stop_gradient",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "out",
                ""
            ],
            [
                "device",
                ""
            ],
            [
                "stop_gradient",
                ""
            ]
        ],
        "args_warning": {
            "out": ""
        }
    },
    "paddle.fluid.layers.tensor.argmax": {
        "alias": [
            "paddle.fluid.layers.argmax"
        ],
        "update_to": "paddle.argmax"
    },
    "paddle.fluid.dygraph.nn.PRelu": {
        "alias": [
            "paddle.fluid.dygraph.PRelu"
        ],
        "warning": "this api is update to paddle.nn.PReLU"
    },
    "paddle.fluid.layers.ops.tanh_shrink": {
        "alias": [
            "paddle.fluid.layers.tanh_shrink"
        ],
        "update_to": "paddle.nn.functional.tanhshrink"
    },
    "paddle.fluid.layers.tensor.linspace": {
        "alias": [
            "paddle.fluid.layers.linspace"
        ],
        "update_to": "paddle.linspace",
    },
    "paddle.fluid.layers.expand": {
        "alias": [
            "paddle.fluid.layers.nn.expand"
        ],
        "warning": "this api is update to paddle.expand"
    },
    "paddle.fluid.backward.append_backward": {
        "update_to": "paddle.static.append_backward"
    },
    "paddle.fluid.layers.diag_embed": {
        "alias": [
            "paddle.fluid.layers.nn.diag_embed"
        ],
        "update_to": "paddle.nn.functional.diag_embed"
    },
    "paddle.fluid.layers.square_error_cost": {
        "alias": [
            "paddle.fluid.layers.loss.square_error_cost"
        ],
        "update_to": "paddle.nn.functional.square_error_cost"
    },
    "paddle.fluid.layers.detection.multi_box_head": {
        "alias": [
            "paddle.fluid.layers.multi_box_head"
        ],
        "update_to": "paddle.static.nn.multi_box_head"
    },
    "paddle.fluid.dygraph.InverseTimeDecay": {
        "alias": [
            "paddle.fluid.dygraph.learning_rate_scheduler.InverseTimeDecay"
        ],
        "warning": "this api is update to paddle.expand"
    },
    "paddle.fluid.layers.asin": {
        "alias": [
            "paddle.fluid.layers.ops.asin"
        ],
        "update_to": "paddle.asin"
    },
    "paddle.fluid.Program": {
        "alias": [
            "paddle.fluid.framework.Program"
        ],
        "update_to": "paddle.static.Program"
    },
    "paddle.fluid.metrics.Auc": {
        "warning": "this api is update to paddle.metric.Auc"
    },
    "paddle.fluid.layers.nn.reshape": {
        "alias": [
            "paddle.fluid.layers.reshape"
        ],
        "warning": "this api is update to paddle.reshape"
    },
    "paddle.fluid.layers.control_flow.greater_than": {
        "alias": [
            "paddle.fluid.layers.greater_than"
        ],
        "update_to": "paddle.greater_than",
        "args_list": [
            "x",
            "y",
            "cond"
        ],
        "args_change": [
            [
                "cond",
                ""
            ]
        ],
        "args_warning": {
            "cond": "this args is deleted in this version"
        }
    },
    "paddle.fluid.compiler.CompiledProgram": {
        "alias": [
            "paddle.fluid.CompiledProgram"
        ],
        "update_to": "paddle.static.CompiledProgram"
    },
    # TODO transformer
    "paddle.fluid.layers.flatten": {
        "alias": [
            "paddle.fluid.layers.nn.flatten"
        ],
        "warning": "this api is update to paddle.flatten"
    },
    "paddle.fluid.dygraph.nn.BCELoss": {
        "alias": [
            "paddle.fluid.dygraph.BCELoss"
        ],
        "update_to": "paddle.nn.BCELoss",
    },
    "paddle.fluid.dygraph.learning_rate_scheduler.ExponentialDecay": {
        "alias": [
            "paddle.fluid.dygraph.ExponentialDecay"
        ],
        # "update_to": "paddle.optimizer.lr.ExponentialDecay",
        "warning": "this api is update to paddle.optimizer.lr.ExponentialDecay"
    },
    "paddle.fluid.layers.detection.ssd_loss": {
        "alias": [
            "paddle.fluid.layers.ssd_loss"
        ],
        "update_to": "paddle.nn.functional.ssd_loss"
    },
    "paddle.fluid.layers.nn.logical_not": {
        "alias": [
            "paddle.fluid.layers.logical_not"
        ],
        "update_to": "paddle.logical_not"
    },
    # manual check
    "paddle.fluid.layers.nn.randint": {
        "alias": [
            "paddle.fluid.layers.randint"
        ],
        "update_to": "paddle.randint",
        "args_list": [
            "low",
            "high",
            "shape",
            "out",
            "dtype",
            "device",
            "stop_gradient",
            "seed",
            "name"
        ],
        "args_change": [
            [
                "out",
                ""
            ],
            [
                "device",
                ""
            ],
            [
                "stop_gradient",
                ""
            ],
            [
                "seed",
                ""
            ]
        ],
        "args_warning": {
            "out": "this args is deleted in this version",
            "device": "this args is deleted in this version",
            "stop_gradient": "this args is deleted in this version",
            "seed": "this args is deleted in this version"
        }
    },
    "paddle.fluid.enable_imperative": {
        "alias": [
            "paddle.fluid.dygraph.base.enable_imperative",
            "paddle.fluid.dygraph.enable_imperative"
        ],
        "update_to": "paddle.disable_static"
    },
    "paddle.fluid.layers.softplus": {
        "alias": [
            "paddle.fluid.layers.ops.softplus"
        ],
        "update_to": "paddle.nn.functional.softplus",
    },
    "paddle.fluid.layers.ops.softshrink": {
        "alias": [
            "paddle.fluid.layers.softshrink"
        ],
        "update_to": "paddle.nn.functional.softshrink",
        "args_list": [
            "x",
            "alpha"
        ],
        "args_change": [
            [
                "alpha",
                "threshold"
            ]
        ]
    },
    "paddle.fluid.dygraph.Pool2D": {
        "alias": [
            "paddle.fluid.dygraph.nn.Pool2D"
        ],
        "warning": "please use paddle.nn.AvgPool2D or paddle.nn.MaxPool2D"
    },
    "paddle.fluid.layers.mean_iou": {
        "alias": [
            "paddle.fluid.layers.nn.mean_iou"
        ],
        "update_to": "paddle.metric.mean_iou"
    },
    "paddle.fluid.layers.multiplex": {
        "alias": [
            "paddle.fluid.layers.nn.multiplex"
        ],
        "update_to": "paddle.multiplex"
    },
    "paddle.fluid.embedding": {
        "alias": [
            "paddle.fluid.input.embedding",
            "paddle.fluid.layers.embedding",
            "paddle.fluid.layers.nn.embedding"
        ],
        "update_to": "paddle.static.nn.embedding"
    },
    "paddle.fluid.io.load_inference_model": {
        "update_to": "paddle.static.load_inference_model"
    },
    "paddle.fluid.layers.nn.scale": {
        "alias": [
            "paddle.fluid.layers.scale"
        ],
        "update_to": "paddle.scale"
    },
    # manual check
    "paddle.fluid.layers.full": {
        "alias": [
            "paddle.fluid.layers.tensor.full"
        ],
        "update_to": "paddle.full",
        "args_list": [
            "shape",
            "fill_value",
            "out",
            "dtype",
            "device",
            "stop_gradient",
            "name"
        ],
        "args_change": [
            [
                "out",
                ""
            ],
            [
                "device",
                ""
            ],
            [
                "stop_gradient",
                ""
            ]
        ],
        "args_warning": {
            "out": "this args is deleted in this version",
            "device": "this args is deleted in this version",
            "stop_gradient": "this args is deleted in this version"
        }
    },
    "paddle.fluid.layers.create_parameter": {
        "alias": [
            "paddle.fluid.layers.tensor.create_parameter"
        ],
        "update_to": "paddle.create_parameter"
    },
    # manual check
    "paddle.fluid.layers.nn.softmax": {
        "alias": [
            "paddle.fluid.layers.softmax"
        ],
        "update_to": "paddle.nn.functional.softmax",
        "args_list": [
            "input",
            "use_cudnn",
            "name",
            "axis"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "use_cudnn",
                ""
            ]
        ],
        "args_warning": {
            "use_cudnn": "This args is deleted in this version"
        }
    },
    "paddle.fluid.layers.is_empty": {
        "alias": [
            "paddle.fluid.layers.control_flow.is_empty"
        ],
        "update_to": "paddle.is_empty",
        "args_list": [
            "x",
            "cond"
        ],
        "args_change": [
            [
                "cond",
                ""
            ]
        ],
        "args_warning": {
            "cond": "this arg is deleted in this version"
        }
    },
    "paddle.fluid.dygraph.Conv3DTranspose": {
        "alias": [
            "paddle.fluid.dygraph.nn.Conv3DTranspose"
        ],
        "update_to": "paddle.nn.Conv3DTranspose",
        "args_list": [
            "num_channels",
            "num_filters",
            "filter_size",
            "padding",
            "stride",
            "dilation",
            "groups",
            "param_attr",
            "bias_attr",
            "use_cudnn",
            "act",
            "dtype"
        ],
        "args_change": [
            [
                "num_channels",
                "in_channels"
            ],
            [
                "num_filters",
                "out_channels"
            ],
            [
                "filter_size",
                "kernel_size"
            ],
            [
                "param_attr",
                "weight_attr"
            ],
            [
                "use_cudnn",
                ""
            ],
            [
                "act",
                ""
            ],
            [
                "dtype",
                ""
            ]
        ],
        "args_warning": {
            "use_cudnn": "this args is deleted in paddle.nn.Conv3DTranspose",
            "act": "this args is deleted in paddle.nn.Conv3DTranspose",
            "dtype": "this args is deleted in paddle.nn.Conv3DTranspose"
        }
    },
    "paddle.fluid.layers.nn.crop_tensor": {
        "alias": [
            "paddle.fluid.layers.crop_tensor"
        ],
        "update_to": "paddle.crop"
    },
    "paddle.fluid.layers.tensor.zeros": {
        "alias": [
            "paddle.fluid.layers.zeros"
        ],
        "update_to": "paddle.zeros",
        "args_list": [
            "shape",
            "dtype",
            "force_cpu"
        ],
        "args_change": [
            [
                "force_cpu",
                ""
            ]
        ],
        "args_warning": {
            "force_cpu": "this arg is deleted in this version"
        }
    },
    "paddle.fluid.layers.reduce_prod": {
        "alias": [
            "paddle.fluid.layers.nn.reduce_prod"
        ],
        "update_to": "paddle.prod",
        "args_list": [
            "input",
            "dim",
            "keep_dim",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "dim",
                "axis"
            ],
            [
                "keep_dim",
                "keepdim"
            ]
        ]
    },
    "paddle.fluid.layers.elementwise_floordiv": {
        "alias": [
            "paddle.fluid.layers.nn.elementwise_floordiv"
        ],
        "update_to": "paddle.floor_divide",
        "args_list": [
            "x",
            "y",
            "axis",
            "act",
            "name"
        ],
        "args_change": [
            [
                "axis",
                ""
            ],
            [
                "act",
                ""
            ]
        ],
        "args_warning": {
            "axis": "axis is deleted in paddle.floor_divide",
            "act": "act is deleted in paddle.floor_divide"
        }
    },
    "paddle.fluid.layers.elementwise_add": {
        "alias": [
            "paddle.fluid.layers.nn.elementwise_add"
        ],
        "update_to": "paddle.add",
        "args_list": [
            "x",
            "y",
            "axis",
            "act",
            "name"
        ],
        "args_change": [
            [
                "axis",
                ""
            ],
            [
                "act",
                ""
            ]
        ],
        "args_warning": {
            "axis": "axis is deleted in paddle.add",
            "act": "act is deleted in paddle.add"
        }
    },
    "paddle.fluid.layers.nn.elementwise_sub": {
        "alias": [
            "paddle.fluid.layers.elementwise_sub"
        ],
        "warning": "please use paddle.add to do subtract."
    },
    # manual check
    "paddle.fluid.layers.elementwise_mul": {
        "alias": [
            "paddle.fluid.layers.nn.elementwise_mul"
        ],
        "update_to": "paddle.multiply",
        "args_list": [
            "x",
            "y",
            "axis",
            "act",
            "name"
        ],
        "args_change": [
            [
                "act",
                ""
            ]
        ],
        "args_warning": {
            "act": "act is deleted in paddle.multiply"
        }
    },
    "paddle.fluid.layers.elementwise_equal": {
        "alias": [
            "paddle.fluid.layers.nn.elementwise_equal"
        ],
        "update_to": "paddle.equal"
    },
    # TODO transformer
    "paddle.fluid.layers.hsigmoid": {
        "alias": [
            "paddle.fluid.layers.loss.hsigmoid"
        ],
        "warning": "this api is update to paddle.nn.functional.hsigmoid_loss",
    },
    "paddle.fluid.layers.scatter_nd": {
        "alias": [
            "paddle.fluid.layers.nn.scatter_nd"
        ],
        "update_to": "paddle.scatter_nd"
    },
    "paddle.fluid.dygraph.LayerList": {
        "alias": [
            "paddle.fluid.dygraph.container.LayerList"
        ],
        "update_to": "paddle.nn.LayerList"
    },
    "paddle.fluid.dygraph.learning_rate_scheduler.PiecewiseDecay": {
        "alias": [
            "paddle.fluid.dygraph.PiecewiseDecay"
        ],
        "warning": "this api is update to paddle.optimizer.lr.PiecewiseDecay"
    },
    "paddle.fluid.CPUPlace": {
        "update_to": "paddle.CPUPlace"
    },
    "paddle.fluid.layers.pixel_shuffle": {
        "alias": [
            "paddle.fluid.layers.nn.pixel_shuffle"
        ],
        "update_to": "paddle.nn.functional.pixel_shuffle",
    },
    "paddle.fluid.optimizer.MomentumOptimizer": {
        "alias": [
            "paddle.fluid.optimizer.Momentum"
        ],
        "update_to": "paddle.optimizer.Momentum",
        "args_list": [
            "learning_rate",
            "momentum",
            "parameter_list",
            "use_nesterov",
            "regularization",
            "grad_clip",
            "name"
        ],
        "args_change": [
            [
                "parameter_list",
                "parameters"
            ],
            [
                "regularization",
                "weight_decay"
            ]
        ]
    },
    "paddle.fluid.layers.tensor.ones_like": {
        "alias": [
            "paddle.fluid.layers.ones_like"
        ],
        "update_to": "paddle.ones_like",
        "args_list": [
            "x",
            "out"
        ],
        "args_change": [
            [
                "out",
                ""
            ]
        ],
        "args_warning": {
            "out": "this args is deleted in this version"
        }
    },
    "paddle.fluid.layers.nn.strided_slice": {
        "alias": [
            "paddle.fluid.layers.strided_slice"
        ],
        "update_to": "paddle.strided_slice",
        "args_list": [
            "input",
            "axes",
            "starts",
            "ends",
            "strides"
        ],
        "args_change": [
            [
                "input",
                "x"
            ]
        ]
    },
    "paddle.fluid.dygraph.learning_rate_scheduler.StepDecay": {
        "alias": [
            "paddle.fluid.dygraph.StepDecay"
        ],
        "warning": "this api is update to paddle.optimizer.lr.StepDecay"
    },
    "paddle.fluid.layers.log": {
        "alias": [
            "paddle.fluid.layers.nn.log"
        ],
        "update_to": "paddle.log"
    },
    "paddle.fluid.layers.loss.nce": {
        "alias": [
            "paddle.fluid.layers.nce"
        ],
        "update_to": "paddle.static.nn.nce"
    },
    "paddle.fluid.layers.nn.flip": {
        "alias": [
            "paddle.fluid.layers.flip"
        ],
        "update_to": "paddle.flip",
        "args_list": [
            "input",
            "dims",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "dims",
                "axis"
            ]
        ]
    },
    "paddle.fluid.layers.nn.shard_index": {
        "alias": [
            "paddle.fluid.layers.shard_index"
        ],
        "update_to": "paddle.shard_index"
    },
    "paddle.fluid.layers.elementwise_mod": {
        "alias": [
            "paddle.fluid.layers.nn.elementwise_mod"
        ],
        "update_to": "paddle.mod",
        "args_list": [
            "x",
            "y",
            "axis",
            "act",
            "name"
        ],
        "args_change": [
            [
                "axis",
                ""
            ],
            [
                "act",
                ""
            ]
        ],
        "args_warning": {
            "axis": "axis is deleted in paddle.mod",
            "act": "act is deleted in paddle.mod"
        }
    },
     # manual check
    "paddle.fluid.dygraph.nn.Linear": {
        "alias": [
            "paddle.fluid.dygraph.Linear"
        ],
        "update_to": "paddle.nn.Linear",
        "args_list": [
            "input_dim",
            "output_dim",
            "param_attr",
            "bias_attr",
            "act",
            "dtype"
        ],
        "args_change": [
            [
                "input_dim",
                "in_features"
            ],
            [
                "output_dim",
                "out_features"
            ],
            [
                "param_attr",
                "weight_attr"
            ],
            [
                "act",
                ""
            ],
            [
                "dtype",
                ""
            ]
        ],
        "args_warning": {
            "act": "this args is deleted in paddle.nn.Linear",
            "dtype": "this args is deleted in paddle.nn.Linear"
        }
    },
    "paddle.fluid.layers.clip_by_norm": {
        "alias": [
            "paddle.fluid.layers.nn.clip_by_norm"
        ],
        "update_to": "paddle.nn.clip_by_norm"
    },
    "paddle.fluid.regularizer.L2DecayRegularizer": {
        "alias": [
            "paddle.fluid.regularizer.L2Decay"
        ],
        "update_to": "paddle.regularizer.L2Decay",
        "args_list": [
            "regularization_coeff"
        ],
        "args_change": [
            [
                "regularization_coeff",
                "coeff"
            ]
        ]
    },
    "paddle.fluid.layers.chunk_eval": {
        "alias": [
            "paddle.fluid.layers.nn.chunk_eval"
        ],
        "update_to": "paddle.metric.chunk_eval"
    },
    "paddle.fluid.layers.tensor.argsort": {
        "alias": [
            "paddle.fluid.layers.argsort"
        ],
        "update_to": "paddle.argsort",
        "args_list": [
            "input",
            "axis",
            "descending",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ]
        ]
    },
    "paddle.fluid.layers.nn.prelu": {
        "alias": [
            "paddle.fluid.layers.prelu"
        ],
        "update_to": "paddle.static.nn.prelu"
    },
    "paddle.fluid.layers.ops.reciprocal": {
        "alias": [
            "paddle.fluid.layers.reciprocal"
        ],
        "update_to": "paddle.reciprocal"
    },
    "paddle.fluid.layers.nn.logical_xor": {
        "alias": [
            "paddle.fluid.layers.logical_xor"
        ],
        "update_to": "paddle.logical_xor"
    },
    "paddle.fluid.dygraph.dygraph_to_static.create_static_variable_gast_node": {
        "alias": [
            "paddle.fluid.dygraph.dygraph_to_static.variable_trans_func.create_static_variable_gast_node"
        ],
        "update_to": "paddle.jit.dy2static.create_static_variable_gast_node"
    },
    "paddle.fluid.layers.crf_decoding": {
        "alias": [
            "paddle.fluid.layers.nn.crf_decoding"
        ],
        "update_to": "paddle.static.nn.crf_decoding"
    },
    # TODO transformer
    "paddle.fluid.layers.diag": {
        "alias": [
            "paddle.fluid.layers.tensor.diag"
        ],
        "warning": "this api is update to paddle.diag"
    },
    "paddle.fluid.layers.less_than": {
        "alias": [
            "paddle.fluid.layers.control_flow.less_than"
        ],
        "update_to": "paddle.less_than",
        "args_list": [
            "x",
            "y",
            "force_cpu",
            "cond"
        ],
        "args_change": [
            [
                "force_cpu",
                ""
            ],
            [
                "cond",
                ""
            ]
        ],
        "args_warning": {
            "force_cpu": "this args is deleted in this version",
            "cond": "this args is deleted in this version"
        }
    },
    "paddle.fluid.layers.affine_grid": {
        "alias": [
            "paddle.fluid.layers.nn.affine_grid"
        ],
        "update_to": "paddle.nn.functional.affine_grid",
        "args_list": [
            "theta",
            "out_shape",
            "name"
        ],
    },
    "paddle.fluid.layers.nn.spectral_norm": {
        "alias": [
            "paddle.fluid.layers.spectral_norm"
        ],
        "update_to": "paddle.static.nn.spectral_norm"
    },
    "paddle.fluid.layers.transpose": {
        "alias": [
            "paddle.fluid.layers.nn.transpose",
            "paddle.complex.transpose",
            "paddle.complex.tensor.transpose",
            "paddle.complex.tensor.manipulation.transpose"
        ],
        "update_to": "paddle.transpose"
    },
    "paddle.fluid.layers.nn.nonzero": {
        "alias": [
            "paddle.fluid.layers.nonzero"
        ],
        "update_to": "paddle.nonzero",
    },
    "paddle.fluid.layers.switch_case": {
        "alias": [
            "paddle.fluid.layers.control_flow.switch_case"
        ],
        "update_to": "paddle.static.nn.switch_case"
    },
    "paddle.fluid.framework.Variable": {
        "alias": [
            "paddle.fluid.Variable"
        ],
        "update_to": "paddle.static.Variable"
    },
    "paddle.fluid.layers.atan": {
        "alias": [
            "paddle.fluid.layers.ops.atan"
        ],
        "update_to": "paddle.atan"
    },
    "paddle.fluid.dataloader.Dataset": {
        "alias": [
            "paddle.fluid.dataloader.dataset.Dataset"
        ],
        "update_to": "paddle.io.Dataset"
    },
    "paddle.fluid.layers.nn.cross": {
        "alias": [
            "paddle.fluid.layers.cross"
        ],
        "update_to": "paddle.cross",
        "args_list": [
            "input",
            "other",
            "dim"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "other",
                "y"
            ],
            [
                "dim",
                "axis"
            ]
        ]
    },
    "paddle.fluid.layers.case": {
        "alias": [
            "paddle.fluid.layers.control_flow.case"
        ],
        "update_to": "paddle.static.nn.case"
    },
    "paddle.fluid.layers.fill_constant": {
        "alias": [
            "paddle.fluid.layers.tensor.fill_constant"
        ],
        "update_to": "paddle.full",
        "args_list": [
            "shape",
            "dtype",
            "value",
            "force_cpu",
            "out"
        ],
        "args_change": [
            [
                "shape",
                "shape"
            ],
            [
                "dtype",
                ""
            ],
            [
                "value",
                "fill_value"
            ],
            [
                "force_cpu",
                ""
            ],
            [
                "out",
                ""
            ]
        ],
    },
    "paddle.fluid.layers.nn.crop": {
        "alias": [
            "paddle.fluid.layers.crop"
        ],
        "update_to": "paddle.crop",
    },
    "paddle.fluid.layers.shape": {
        "alias": [
            "paddle.fluid.layers.nn.shape"
        ],
        "update_to": "paddle.shape"
    },
    "paddle.fluid.layers.ones": {
        "alias": [
            "paddle.fluid.layers.tensor.ones"
        ],
        "update_to": "paddle.ones",
        "args_list": [
            "shape",
            "dtype",
            "force_cpu"
        ],
        "args_change": [
            [
                "force_cpu",
                ""
            ]
        ]
    },
    # TODO transformer
    "paddle.fluid.dygraph.LayerNorm": {
        "alias": [
            "paddle.fluid.dygraph.nn.LayerNorm"
        ],
        "warning": "this api is update to paddle.nn.LayerNorm",
    },
    "paddle.fluid.optimizer.AdagradOptimizer": {
        "alias": [
            "paddle.fluid.optimizer.Adagrad"
        ],
        "update_to": "paddle.optimizer.Adagrad",
    },
    "paddle.fluid.layers.layer_norm": {
        "alias": [
            "paddle.fluid.layers.nn.layer_norm"
        ],
        "update_to": "paddle.static.nn.layer_norm"
    },
    "paddle.fluid.layers.one_hot": {
        "alias": [
            "paddle.fluid.layers.nn.one_hot",
            "paddle.fluid.one_hot",
            "paddle.fluid.input.one_hot"
        ],
        "warning": "input->x, depth->num_classes, x'elements must less than num_classes."
    },
    "paddle.fluid.layers.argmin": {
        "alias": [
            "paddle.fluid.layers.tensor.argmin"
        ],
        "update_to": "paddle.argmin",
    },
    "paddle.fluid.layers.square": {
        "alias": [
            "paddle.fluid.layers.ops.square"
        ],
        "update_to": "paddle.square"
    },
    "paddle.fluid.layers.io.data": {
        "alias": [
            "paddle.fluid.layers.data"
        ],
        "update_to": "paddle.static.data",
        "args_list": [
            "name",
            "shape",
            "append_batch_size",
            "dtype",
            "lod_level",
            "type",
            "stop_gradient"
        ],
        "args_change": [
            [
                "append_batch_size",
                ""
            ],
            [
                "type",
                ""
            ],
            [
                "stop_gradient",
                ""
            ]
        ],
        "args_warning": {
            "append_batch_size": "This args is deleted in this version.",
            "type": "This args is deleted in this version.",
            "stop_gradient": "This args is deleted in this version.",
        }
    },
    "paddle.fluid.layers.loss.kldiv_loss": {
        "alias": [
            "paddle.fluid.layers.kldiv_loss"
        ],
        "update_to": "paddle.nn.functional.kl_div",
        "args_list": [
            "x",
            "target",
            "reduction",
            "name"
        ],
        "args_change": [
            [
                "x",
                "input"
            ],
            [
                "target",
                "label"
            ]
        ]
    },
    # TODO transformer
    "paddle.fluid.dataloader.BatchSampler": {
        "alias": [
            "paddle.fluid.dataloader.batch_sampler.BatchSampler"
        ],
        "warning": "this api is update to paddle.io.BatchSampler",
    },
    "paddle.fluid.layers.nn.logical_or": {
        "alias": [
            "paddle.fluid.layers.logical_or"
        ],
        "update_to": "paddle.logical_or"
    },
    "paddle.fluid.layers.less_equal": {
        "alias": [
            "paddle.fluid.layers.control_flow.less_equal"
        ],
        "update_to": "paddle.less_equal",
        "args_list": [
            "x",
            "y",
            "cond"
        ],
        "args_change": [
            [
                "cond",
                ""
            ]
        ],
        "args_warning": {
            "cond": "cond is deleted in this version"
        }
    },
    "paddle.fluid.dygraph.parallel.prepare_context": {
        "alias": [
            "paddle.fluid.dygraph.prepare_context"
        ],
        "update_to": "paddle.distributed.prepare_context"
    },
    "paddle.reader.decorator.multiprocess_reader": {
        "alias": [
            "paddle.fluid.io.multiprocess_reader",
            "paddle.reader.multiprocess_reader"
        ],
        "update_to": "paddle.reader.decorator.multiprocess_reader"
    },
    "paddle.fluid.layers.concat": {
        "alias": [
            "paddle.fluid.layers.tensor.concat"
        ],
        "update_to": "paddle.concat",
        "args_list": [
            "input",
            "axis",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ]
        ]
    },
    "paddle.fluid.in_dygraph_mode": {
        "alias": [
            "paddle.fluid.framework.in_dygraph_mode"
        ],
        "update_to": "paddle.in_dynamic_mode"
    },
    "paddle.fluid.dygraph.MSELoss": {
        "alias": [
            "paddle.fluid.dygraph.nn.MSELoss"
        ],
        "update_to": "paddle.nn.MSELoss"
    },
    "paddle.fluid.framework.name_scope": {
        "alias": [
            "paddle.fluid.name_scope"
        ],
        "update_to": "paddle.static.name_scope"
    },
    "paddle.fluid.dygraph.NLLLoss": {
        "alias": [
            "paddle.fluid.dygraph.nn.NLLLoss"
        ],
        "update_to": "paddle.nn.NLLLoss",
    },
    "paddle.fluid.layers.ops.softsign": {
        "alias": [
            "paddle.fluid.layers.softsign"
        ],
        "update_to": "paddle.nn.functional.softsign"
    },
    "paddle.fluid.layers.t": {
        "alias": [
            "paddle.fluid.layers.nn.t"
        ],
        "update_to": "paddle.t"
    },
    "paddle.fluid.layers.nn.selu": {
        "alias": [
            "paddle.fluid.layers.selu"
        ],
        "update_to": "paddle.nn.functional.selu"
    },
    "paddle.fluid.layers.meshgrid": {
        "alias": [
            "paddle.fluid.layers.nn.meshgrid"
        ],
        "update_to": "paddle.meshgrid",
    },
    "paddle.fluid.dygraph.nn.Conv3D": {
        "alias": [
            "paddle.fluid.dygraph.Conv3D"
        ],
        "update_to": "paddle.nn.Conv3D",
        "args_list": [
            "num_channels",
            "num_filters",
            "filter_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "param_attr",
            "bias_attr",
            "use_cudnn",
            "act",
            "dtype"
        ],
        "args_change": [
            [
                "num_channels",
                "in_channels"
            ],
            [
                "num_filters",
                "out_channels"
            ],
            [
                "filter_size",
                "kernel_size"
            ],
            [
                "param_attr",
                "weight_attr"
            ],
            [
                "use_cudnn",
                ""
            ],
            [
                "act",
                ""
            ],
            [
                "dtype",
                ""
            ]
        ],
        "args_warning": {
            "use_cudnn": "this args is deleted in paddle.nn.Conv3d",
            "act": "this args is deleted in paddle.nn.Conv3d",
            "dtype": "this args is deleted in paddle.nn.Conv3d"
        }
    },
    "paddle.fluid.layers.exp": {
        "alias": [
            "paddle.fluid.layers.ops.exp"
        ],
        "update_to": "paddle.exp"
    },
    "paddle.fluid.backward.gradients": {
        "alias": [
            "paddle.fluid.gradients"
        ],
        "update_to": "paddle.static.gradients"
    },
    "paddle.fluid.layers.control_flow.equal": {
        "alias": [
            "paddle.fluid.layers.equal"
        ],
        "update_to": "paddle.equal",
        "args_list": [
            "x",
            "y",
            "cond"
        ],
        "args_change": [
            [
                "cond",
                ""
            ]
        ],
        "args_warning": {
            "cond": "this args is deleted in this version"
        }
    },
    "paddle.fluid.ComplexVariable": {
        "alias": [
            "paddle.fluid.framework.ComplexVariable"
        ],
        "warning": "this api is update to paddle.ComplexTensor",
    },
    "paddle.fluid.layers.split": {
        "alias": [
            "paddle.fluid.layers.nn.split"
        ],
        "update_to": "paddle.split",
        "args_list": [
            "input",
            "num_or_sections",
            "dim",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "dim",
                "axis"
            ]
        ]
    },
    "paddle.fluid.layers.dynamic_decode": {
        "update_to": "paddle.nn.dynamic_decode"
    },
    "paddle.fluid.layers.kron": {
        "alias": [
            "paddle.fluid.layers.tensor.kron"
        ],
        "update_to": "paddle.kron",
        "args_list": [
            "x",
            "y",
            "out",
            "name"
        ],
        "args_change": [
            [
                "out",
                ""
            ]
        ]
    },
    "paddle.fluid.layers.unfold": {
        "alias": [
            "paddle.fluid.layers.nn.unfold"
        ],
        "update_to": "paddle.nn.functional.unfold"
    },
    "paddle.fluid.layers.data_norm": {
        "alias": [
            "paddle.fluid.layers.nn.data_norm"
        ],
        "update_to": "paddle.static.nn.data_norm",
    },
    "paddle.fluid.dygraph.jit.TracedLayer": {
        "alias": [
            "paddle.fluid.dygraph.TracedLayer"
        ],
        "update_to": "paddle.jit.TracedLayer"
    },
    "paddle.fluid.ParamAttr": {
        "alias": [
            "paddle.fluid.param_attr.ParamAttr"
        ],
        "update_to": "paddle.ParamAttr",
    },
    "paddle.fluid.layers.nn.bilinear_tensor_product": {
        "alias": [
            "paddle.fluid.layers.bilinear_tensor_product"
        ],
        "update_to": "paddle.static.nn.bilinear_tensor_product"
    },
    "paddle.fluid.layers.round": {
        "alias": [
            "paddle.fluid.layers.ops.round"
        ],
        "update_to": "paddle.round"
    },
    "paddle.fluid.layers.mean": {
        "alias": [
            "paddle.fluid.layers.nn.mean"
        ],
        "update_to": "paddle.mean",
        "args_list": [
            "x",
            "name"
        ]
    },
    "paddle.fluid.io.save_inference_model": {
        "update_to": "paddle.static.save_inference_model"
    },
    "paddle.fluid.framework.cpu_places": {
        "alias": [
            "paddle.fluid.cpu_places"
        ],
        "update_to": "paddle.static.cpu_places"
    },
    "paddle.fluid.executor.global_scope": {
        "alias": [
            "paddle.fluid.global_scope"
        ],
        "update_to": "paddle.static.global_scope"
    },
    # TODO transformer
    "paddle.fluid.layers.where": {
        "alias": [
            "paddle.fluid.layers.nn.where"
        ],
        "warning": "this api is update to paddle.where",
    },
    # TODO transformer
    "paddle.fluid.layers.batch_norm": {
        "alias": [
            "paddle.fluid.layers.nn.batch_norm"
        ],
        "warning": " this api is update to paddle.nn.functional.batch_norm"
    },
    "paddle.fluid.layers.slice": {
        "alias": [
            "paddle.fluid.layers.nn.slice"
        ],
        "update_to": "paddle.slice"
    },
    "paddle.fluid.layers.addmm": {
        "alias": [
            "paddle.fluid.layers.nn.addmm"
        ],
        "update_to": "paddle.addmm",
        "args_list": [
            "input",
            "x",
            "y",
            "alpha",
            "beta",
            "name"
        ],
    },
    "paddle.fluid.layers.scatter": {
        "alias": [
            "paddle.fluid.layers.nn.scatter"
        ],
        "update_to": "paddle.scatter",
        "args_list": [
            "input",
            "index",
            "updates",
            "name",
            "overwrite"
        ],
        "args_change": [
            [
                "input",
                "x"
            ]
        ]
    },
    "paddle.fluid.layers.nn.index_select": {
        "alias": [
            "paddle.fluid.layers.index_select"
        ],
        "update_to": "paddle.index_select",
        "args_list": [
            "input",
            "index",
            "dim"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "dim",
                "axis"
            ]
        ]
    },
    "paddle.fluid.layers.index_sample": {
        "alias": [
            "paddle.fluid.layers.nn.index_sample"
        ],
        "update_to": "paddle.index_sample"
    },
    "paddle.fluid.WeightNormParamAttr": {
        "alias": [
            "paddle.fluid.param_attr.WeightNormParamAttr"
        ],
        "update_to": "paddle.static.WeightNormParamAttr"
    },
    "paddle.fluid.dygraph.no_grad": {
        "alias": [
            "paddle.fluid.dygraph.base.no_grad"
        ],
        "update_to": "paddle.framework.no_grad"
    },
    # TODO transformer paddle.nn.functional.dropout
    "paddle.fluid.layers.nn.dropout": {
        "alias": [
            "paddle.fluid.layers.dropout"
        ],
        "warning": "this api is update to paddle.nn.functional.dropout",
    },
    "paddle.fluid.layers.cast": {
        "alias": [
            "paddle.fluid.layers.tensor.cast"
        ],
        "update_to": "paddle.cast"
    },
    "paddle.fluid.layers.label_smooth": {
        "alias": [
            "paddle.fluid.layers.nn.label_smooth"
        ],
        "update_to": "paddle.nn.functional.label_smooth",
    },
    "paddle.fluid.framework.default_main_program": {
        "alias": [
            "paddle.fluid.default_main_program"
        ],
        "update_to": "paddle.static.default_main_program"
    },
    "paddle.fluid.program_guard": {
        "alias": [
            "paddle.fluid.framework.program_guard"
        ],
        "update_to": "paddle.static.program_guard"
    },
    "paddle.fluid.layers.Print": {
        "alias": [
            "paddle.fluid.layers.control_flow.Print"
        ],
        "update_to": "paddle.static.Print",
        "args_list": [
            "input",
            "first_n",
            "message",
            "summarize",
            "print_tensor_name",
            "print_tensor_type",
            "print_tensor_shape",
            "print_tensor_lod",
            "print_phase"
        ]
    },
    "paddle.fluid.dataset.InMemoryDataset": {
        "update_to": "paddle.distributed.InMemoryDataset"
    },
    # manual check
    "paddle.fluid.layers.squeeze": {
        "alias": [
            "paddle.fluid.layers.nn.squeeze"
        ],
        "update_to": "paddle.squeeze",
        "args_list": [
            "input",
            "axes",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "axes",
                "axis"
            ]
        ]
    },
    "paddle.fluid.dygraph.BatchNorm": {
        "alias": [
            "paddle.fluid.dygraph.nn.BatchNorm"
        ],
        "update_to": "paddle.nn.BatchNorm"
    },
    "paddle.fluid.layers.elu": {
        "alias": [
            "paddle.fluid.layers.nn.elu"
        ],
        "update_to": "paddle.nn.functional.elu"
    },
    "paddle.fluid.dygraph.nn.Embedding": {
        "alias": [
            "paddle.fluid.dygraph.Embedding"
        ],
        "warning": "this api is update to paddle.nn.Embedding"
    },
    "paddle.fluid.dygraph.Conv2DTranspose": {
        "alias": [
            "paddle.fluid.dygraph.nn.Conv2DTranspose"
        ],
        "warning": "this api is update to paddle.nn.Conv2DTranspose"
    },
    "paddle.fluid.layers.nn.bmm": {
        "alias": [
            "paddle.fluid.layers.bmm"
        ],
        "update_to": "paddle.bmm"
    },
    "paddle.fluid.layers.reduce_sum": {
        "alias": [
            "paddle.fluid.layers.nn.reduce_sum"
        ],
        "update_to": "paddle.sum",
        "args_list": [
            "input",
            "dim",
            "keep_dim",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "dim",
                "axis"
            ],
            [
                "keep_dim",
                "keepdim"
            ]
        ]
    },
    "paddle.fluid.initializer.TruncatedNormalInitializer": {
        "alias": [
            "paddle.fluid.initializer.TruncatedNormal"
        ],
        "warning": "this api is update to paddle.nn.initializer.TruncatedNormal",
    
    },
    "paddle.fluid.layers.maxout": {
        "alias": [
            "paddle.fluid.layers.nn.maxout"
        ],
        "update_to": "paddle.nn.functional.maxout",
    },
    "paddle.fluid.dygraph.parallel.ParallelEnv": {
        "alias": [
            "paddle.fluid.dygraph.ParallelEnv"
        ],
        "update_to": "paddle.distributed.ParallelEnv"
    },
    "paddle.fluid.layers.triu": {
        "alias": [
            "paddle.fluid.layers.tensor.triu"
        ],
        "update_to": "paddle.triu",
        "args_list": [
            "input",
            "diagonal",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ]
        ]
    },
    "paddle.fluid.layers.nn.scatter_nd_add": {
        "alias": [
            "paddle.fluid.layers.scatter_nd_add"
        ],
        "update_to": "paddle.scatter_nd_add",
    },
    "paddle.fluid.ExecutionStrategy": {
        "alias": [
            "paddle.fluid.compiler.ExecutionStrategy"
        ],
        "update_to": "paddle.static.ExecutionStrategy"
    },
    "paddle.fluid.layers.nn.conv3d_transpose": {
        "alias": [
            "paddle.fluid.layers.conv3d_transpose"
        ],
        "warning": "this api is update to paddle.nn.functional.conv3d_transpose",
    },
    "paddle.fluid.io.load_program_state": {
        "update_to": "paddle.static.load_program_state"
    },
    "paddle.fluid.layers.accuracy": {
        "alias": [
            "paddle.fluid.layers.metric_op.accuracy"
        ],
        "update_to": "paddle.metric.accuracy",
    },
    "paddle.fluid.layers.abs": {
        "alias": [
            "paddle.fluid.layers.ops.abs"
        ],
        "update_to": "paddle.abs"
    },
    "paddle.fluid.dygraph.learning_rate_scheduler.NoamDecay": {
        "alias": [
            "paddle.fluid.dygraph.NoamDecay"
        ],
        "warning": "this api is update to paddle.optimizer.lr.NoamDecay",

    },
    "paddle.fluid.layers.erf": {
        "alias": [
            "paddle.fluid.layers.ops.erf"
        ],
        "update_to": "paddle.erf",
    },
    "paddle.fluid.framework.cuda_places": {
        "alias": [
            "paddle.fluid.cuda_places"
        ],
        "update_to": "paddle.static.cuda_places"
    },
    # TODO transformer
    "paddle.fluid.layers.ops.cumsum": {
        "alias": [
            "paddle.fluid.layers.cumsum"
        ],
        "warning": "this api is update to paddle.cumsum",
    },
    "paddle.fluid.layers.nn.log_loss": {
        "alias": [
            "paddle.fluid.layers.log_loss"
        ],
        "update_to": "paddle.nn.functional.log_loss"
    },
    # TODO transformer
    "paddle.fluid.layers.nn.interpolate": {
        "alias": [
            "paddle.fluid.layers.interpolate"
        ],
        "warning": "this api is update to paddle.nn.functional.interpolate",
    },
    "paddle.fluid.dygraph.nn.L1Loss": {
        "alias": [
            "paddle.fluid.dygraph.L1Loss"
        ],
        "update_to": "paddle.nn.L1Loss",
    },
    "paddle.fluid.io.batch": {
        "alias": [
            "paddle.batch"
        ],
        "update_to": "paddle.batch"
    },
    "paddle.fluid.layers.nn.gather_nd": {
        "alias": [
            "paddle.fluid.layers.gather_nd"
        ],
        "update_to": "paddle.gather_nd",
        "args_list": [
            "input",
            "index",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ]
        ]
    },
    "paddle.fluid.layers.nn.row_conv": {
        "alias": [
            "paddle.fluid.layers.row_conv"
        ],
        "update_to": "paddle.static.nn.row_conv"
    },
    "paddle.fluid.layers.unstack": {
        "alias": [
            "paddle.fluid.layers.nn.unstack"
        ],
        "update_to": "paddle.unstack"
    },
    "paddle.fluid.dygraph.learning_rate_scheduler.LambdaDecay": {
        "alias": [
            "paddle.fluid.dygraph.LambdaDecay"
        ],
        "warning": " this api is update to paddle.optimizer.lr.LambdaDecay"
    },
    "paddle.fluid.layers.nn.unique": {
        "alias": [
            "paddle.fluid.layers.unique"
        ],
        "update_to": "paddle.unique",
        "args_list": [
            "x",
            "dtype"
        ],
    },
    "paddle.fluid.dygraph.layers.Layer": {
        "alias": [
            "paddle.fluid.dygraph.Layer"
        ],
        "update_to": "paddle.nn.Layer"
    },
    "paddle.fluid.layers.acos": {
        "alias": [
            "paddle.fluid.layers.ops.acos"
        ],
        "update_to": "paddle.acos"
    },
    "paddle.fluid.layers.nn.roll": {
        "alias": [
            "paddle.fluid.layers.roll"
        ],
        "update_to": "paddle.roll",
        "args_list": [
            "input",
            "shifts",
            "dims"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "dims",
                "axis"
            ]
        ]
    },
    "paddle.fluid.layers.nn.unsqueeze": {
        "alias": [
            "paddle.fluid.layers.unsqueeze"
        ],
        "update_to": "paddle.unsqueeze",
        "args_list": [
            "input",
            "axes",
            "name"
        ],
        "args_change": [
            [
                "input",
                "a"
            ],
            [
                "axes",
                "axis"
            ]
        ]
    },
    "paddle.fluid.dygraph.learning_rate_scheduler.MultiStepDecay": {
        "alias": [
            "paddle.fluid.dygraph.MultiStepDecay"
        ],
        "warning": "this api is update to paddle.optimizer.lr.MultiStepDecay",
    },
    "paddle.fluid.layers.loss.mse_loss": {
        "alias": [
            "paddle.fluid.layers.mse_loss"
        ],
        "update_to": "paddle.nn.functional.mse_loss"
    },
    # manual check
    "paddle.fluid.layers.pad": {
        "alias": [
            "paddle.fluid.layers.nn.pad"
        ],
        "warning": " this api is update to paddle.nn.functional.pad",
    },
    # warning 
    "paddle.fluid.layers.cross_entropy": {
        "alias": [
            "paddle.fluid.layers.loss.cross_entropy"
        ],
        "warning": " this api is update to paddle.nn.functional.cross_entropy"
    },
    "paddle.fluid.layers.tensor.range": {
        "alias": [
            "paddle.fluid.layers.range"
        ],
        "update_to": "paddle.arange",
    },
    "paddle.fluid.layers.arange": {
        "alias": [
            "paddle.fluid.layers.tensor.arange"
        ],
        "update_to": "paddle.arange"
    },
    "paddle.fluid.layers.py_func": {
        "alias": [
            "paddle.fluid.layers.nn.py_func"
        ],
        "update_to": "paddle.static.py_func"
    },
    "paddle.fluid.layers.ops.floor": {
        "alias": [
            "paddle.fluid.layers.floor"
        ],
        "update_to": "paddle.floor"
    },
    "paddle.fluid.layers.nn.elementwise_max": {
        "alias": [
            "paddle.fluid.layers.elementwise_max"
        ],
        "update_to": "paddle.maximum",
        "args_list": [
            "x",
            "y",
            "axis",
            "act",
            "name"
        ],
        "args_change": [
            [
                "act",
                ""
            ]
        ],
        "args_warning": {
            "act": "this api is deleted in this version"
        }
    },
    "paddle.fluid.layers.nn.elementwise_div": {
        "alias": [
            "paddle.fluid.layers.elementwise_div"
        ],
        "update_to": "paddle.divide",
        "args_list": [
            "x",
            "y",
            "axis",
            "act",
            "name"
        ],
        "args_change": [
            [
                "axis",
                ""
            ],
            [
                "act",
                ""
            ]
        ],
        "args_warning": {
            "axis": "axis is deleted in paddle.divide",
            "act": "act is deleted in paddle.divide"
        }
    },    
    # TODO transformer
    "paddle.fluid.layers.loss.margin_rank_loss": {
        "alias": [
            "paddle.fluid.layers.margin_rank_loss"
        ],
        "warning": "this api is update to paddle.nn.functional.margin_ranking_loss"
    },
    # manual check
    "paddle.fluid.layers.greater_equal": {
        "alias": [
            "paddle.fluid.layers.control_flow.greater_equal"
        ],
        "update_to": "paddle.greater_equal",
        "args_list": [
            "x",
            "y",
            "cond"
        ],
        "args_change": [
            [
                "cond",
                ""
            ]
        ],
        "args_warning": {
            "cond": "this args is deleted in this version"
        }
    },
    "paddle.fluid.layers.fc": {
        "alias": [
            "paddle.fluid.layers.nn.fc"
        ],
        "update_to": "paddle.static.nn.fc",
        "args_list": [
            "input",
            "size",
            "num_flatten_dims",
            "param_attr",
            "bias_attr",
            "act",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "param_attr",
                "weight_attr"
            ],
            [
                "act",
                "activation"
            ]
        ],
        "warning": "in static graph, this api is update to paddle.static.nn.fc, in dynamic graph, this api is update to paddle.nn.functional.linear"
    },
    "paddle.fluid.layers.nn.reduce_all": {
        "alias": [
            "paddle.fluid.layers.reduce_all"
        ],
        "update_to": "paddle.all",
        "args_list": [
            "input",
            "dim",
            "keep_dim",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "dim",
                "axis"
            ],
            [
                "keep_dim",
                "keepdim"
            ]
        ]
    },
    # manual check
    "paddle.fluid.layers.nn.reduce_max": {
        "alias": [
            "paddle.fluid.layers.reduce_max"
        ],
        "update_to": "paddle.max",
        "args_list": [
            "input",
            "dim",
            "keep_dim",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "dim",
                "axis"
            ],
            [
                "keep_dim",
                "keepdim"
            ]
        ]
    },
    "paddle.fluid.layers.relu": {
        "alias": [
            "paddle.fluid.layers.nn.relu"
        ],
        "update_to": "paddle.nn.functional.relu"
    },
    "paddle.fluid.layers.nn.dot": {
        "alias": [
            "paddle.fluid.layers.dot"
        ],
        "update_to": "paddle.dot"
    },
    "paddle.fluid.executor.Executor": {
        "alias": [
            "paddle.fluid.Executor"
        ],
        "update_to": "paddle.static.Executor"
    },
    "paddle.fluid.layers.BeamSearchDecoder": {
        "update_to": "paddle.nn.BeamSearchDecoder"
    },
    "paddle.fluid.framework.is_compiled_with_cuda": {
        "alias": [
            "paddle.fluid.is_compiled_with_cuda"
        ],
        "update_to": "paddle.is_compiled_with_cuda"
    },
    "paddle.fluid.layers.nn.leaky_relu": {
        "alias": [
            "paddle.fluid.layers.leaky_relu"
        ],
        "update_to": "paddle.nn.functional.leaky_relu",
        "args_list": [
            "x",
            "alpha",
            "name"
        ],
        "args_change": [
            [
                "alpha",
                "negative_slope"
            ]
        ]
    },
    "paddle.fluid.layers.nn.sign": {
        "alias": [
            "paddle.fluid.layers.sign"
        ],
        "update_to": "paddle.sign",
    },
    "paddle.fluid.dygraph.base.grad": {
        "alias": [
            "paddle.fluid.dygraph.grad"
        ],
        "update_to": "paddle.grad",
    },
    "paddle.fluid.layers.logsumexp": {
        "alias": [
            "paddle.fluid.layers.nn.logsumexp"
        ],
        "update_to": "paddle.logsumexp",
        "args_list": [
            "x",
            "dim",
            "keepdim",
            "out",
            "name"
        ],
        "args_change": [
            [
                "dim",
                "axis"
            ],
            [
                "out",
                ""
            ]
        ],
        "args_warning": {
            "out": "this arg is deleted in this version."
        }
    },
    "paddle.fluid.scope_guard": {
        "alias": [
            "paddle.fluid.executor.scope_guard"
        ],
        "update_to": "paddle.static.scope_guard"
    },
    "paddle.fluid.layers.nn.logical_and": {
        "alias": [
            "paddle.fluid.layers.logical_and"
        ],
        "update_to": "paddle.logical_and"
    },
    "paddle.fluid.layers.ops.cos": {
        "alias": [
            "paddle.fluid.layers.cos"
        ],
        "update_to": "paddle.cos"
    },
    # TODO transformer
    "paddle.fluid.layers.nn.conv2d": {
        "alias": [
            "paddle.fluid.layers.conv2d"
        ],
        "warning": "this api is update to paddle.nn.functional.conv2d",
    },
    "paddle.fluid.layers.nn.group_norm": {
        "alias": [
            "paddle.fluid.layers.group_norm"
        ],
        "update_to": "paddle.static.nn.group_norm"
    },
    "paddle.fluid.initializer.ConstantInitializer": {
        "alias": [
            "paddle.fluid.initializer.Constant"
        ],
        "update_to": "paddle.nn.initializer.Constant",
        "args_list": [
            "value",
            "force_cpu"
        ],
        "args_change": [
            [
                "force_cpu",
                ""
            ]
        ],
        "args_warning": {
            "force_cpu": "this args is deleted in paddle.nn.initializer.Constant"
        }
    },
    "paddle.fluid.layers.nn.reduce_min": {
        "alias": [
            "paddle.fluid.layers.reduce_min"
        ],
        "update_to": "paddle.min",
        "args_list": [
            "input",
            "dim",
            "keep_dim",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "dim",
                "axis"
            ],
            [
                "keep_dim",
                "keepdim"
            ]
        ]
    },
    "paddle.fluid.dygraph.nn.Conv2D": {
        "alias": [
            "paddle.fluid.dygraph.Conv2D"
        ],
        "update_to": "paddle.nn.Conv2D",
        "args_change": [
            [
                "num_channels",
                "in_channels"
            ],
            [
                "num_filters",
                "out_channels"
            ],
            [
                "filter_size",
                "kernel_size"
            ],
            [
                "param_attr",
                "weight_attr"
            ],
            [
                "use_cudnn",
                ""
            ],
            [
                "act",
                ""
            ],
            [
                "dtype",
                ""
            ]
        ],
        "args_warning": {
            "use_cudnn": "this args is deleted in paddle.nn.Conv2d",
            "act": "this args is deleted in paddle.nn.Conv2d",
            "dtype": "this args is deleted in paddle.nn.Conv2d"
        }
    },
    "paddle.fluid.layers.nn.dist": {
        "alias": [
            "paddle.fluid.layers.dist"
        ],
        "update_to": "paddle.dist"
    },
    "paddle.fluid.layers.addcmul": {
        "alias": [
            "paddle.fluid.layers.nn.addcmul"
        ],
        "update_to": "paddle.tensor.math.addcmul",
        "args_list": [
            "input",
            "tensor1",
            "tensor2",
            "value",
            "out",
            "name"
        ],
        "args_change": [
            [
                "out",
                ""
            ]
        ],    
        "args_warning": {
            "out": "this arg is deleted in this version."
        }
    },
    "paddle.fluid.default_startup_program": {
        "alias": [
            "paddle.fluid.framework.default_startup_program"
        ],
        "update_to": "paddle.static.default_startup_program"
    },
    "paddle.fluid.layers.hard_sigmoid": {
        "alias": [
            "paddle.fluid.layers.nn.hard_sigmoid"
        ],
        "update_to": "paddle.nn.functional.hardsigmoid",
        "args_list": [
            "x",
            "slope",
            "offset",
            "name"
        ],
        "args_change": [
            [
                "slope",
                ""
            ],
            [
                "offset",
                ""
            ]
        ],
        "args_warning": {
            "slope": "this arg is deleted in this version.",
            "offset": "this arg is deleted in this version.",
        }
    },
    "paddle.fluid.dygraph.ParameterList": {
        "alias": [
            "paddle.fluid.dygraph.container.ParameterList"
        ],
        "update_to": "paddle.nn.ParameterList"
    },
    "paddle.fluid.layers.expand_as": {
        "alias": [
            "paddle.fluid.layers.nn.expand_as"
        ],
        "update_to": "paddle.expand_as",
        "args_list": [
            "x",
            "target_tensor",
            "name"
        ],
        "args_change": [
            [
                "target_tensor",
                "y"
            ],
        ]
    },
    "paddle.fluid.layers.reduce_mean": {
        "alias": [
            "paddle.fluid.layers.nn.reduce_mean"
        ],
        "update_to": "paddle.mean",
        "args_list": [
            "input",
            "dim",
            "keep_dim",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "dim",
                "axis"
            ],
            [
                "keep_dim",
                "keepdim"
            ]
        ]
    },
    "paddle.fluid.dygraph.learning_rate_scheduler.PolynomialDecay": {
        "alias": [
            "paddle.fluid.dygraph.PolynomialDecay"
        ],
        "warning": "this api is update to paddle.optimizer.lr.PolynomialDecay"
    },
    "paddle.fluid.layers.nn.swish": {
        "alias": [
            "paddle.fluid.layers.swish"
        ],
        "update_to": "paddle.nn.functional.swish",
        "args_list": [
            "x",
            "beta",
            "name"
        ],
        "args_change": [
            [
                "beta",
                ""
            ]
        ],
        "args_warning": {
            "beta": "this arg is deleted in this version."
        }
    },
    "paddle.fluid.layers.tensor.tril": {
        "alias": [
            "paddle.fluid.layers.tril"
        ],
        "update_to": "paddle.tril",
        "args_list": [
            "input",
            "diagonal",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ]
        ]
    },
    "paddle.fluid.metrics.Recall": {
        "update_to": "paddle.metric.Recall",
    },
    "paddle.fluid.io.set_program_state": {
        "update_to": "paddle.static.set_program_state"
    },
    "paddle.fluid.ParallelExecutor": {
        "alias": [
            "paddle.fluid.parallel_executor.ParallelExecutor"
        ],
        "update_to": "paddle.static.ParallelExecutor"
    },
    "paddle.fluid.CUDAPinnedPlace": {
        "update_to": "paddle.CUDAPinnedPlace"
    },
    "paddle.fluid.layers.nn.randperm": {
        "alias": [
            "paddle.fluid.layers.randperm"
        ],
        "update_to": "paddle.randperm",
        "args_list": [
            "n",
            "out",
            "dtype",
            "device",
            "stop_gradient",
            "seed"
        ],
        "args_change": [
            [
                "out",
                ""
            ],
            [
                "device",
                ""
            ],
            [
                "stop_gradient",
                ""
            ],
            [
                "seed",
                ""
            ]
        ],
        "args_warning": {
            "out": "this arg is deleted in this version",
            "device": "this arg is deleted in this version",
            "stop_gradient": "this arg is deleted in this version",
            "seed": "this arg is deleted in this version",
        }
    },
    "paddle.fluid.layers.dice_loss": {
        "alias": [
            "paddle.fluid.layers.nn.dice_loss"
        ],
        "update_to": "paddle.nn.functional.dice_loss"
    },
    "paddle.fluid.layers.distributions.Normal": {
        "alias": [
            "paddle.fluid.layers.Normal"
        ],
        "update_to": "paddle.distribution.Normal"
    },
    "paddle.fluid.layers.log_softmax": {
        "alias": [
            "paddle.fluid.layers.nn.log_softmax"
        ],
        "update_to": "paddle.nn.functional.log_softmax",
        "args_list": [
            "input",
            "axis",
            "dtype",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ]
        ]
    },
    "paddle.fluid.dygraph.nn.GroupNorm": {
        "alias": [
            "paddle.fluid.dygraph.GroupNorm"
        ],
        "update_to": "paddle.nn.GroupNorm",
        "args_list": [
            "channels",
            "groups",
            "epsilon",
            "param_attr",
            "bias_attr",
            "act",
            "data_layout",
            "dtype"
        ],
        "args_change": [
            [
                "channels",
                "num_channels"
            ],
            [
                "groups",
                "num_groups"
            ],
            [
                "param_attr",
                "weight_attr"
            ],
            [
                "act",
                ""
            ],
            [
                "data_layout",
                "data_format"
            ],
            [
                "dtype",
                ""
            ]
        ],
        "args_warning": {
            "act": "this arg is deleted in this version",
            "dtype": "this arg is deleted in this version"
        }

    },
    "paddle.fluid.layers.zeros_like": {
        "alias": [
            "paddle.fluid.layers.tensor.zeros_like"
        ],
        "update_to": "paddle.zeros_like",
        "args_list": [
            "x",
            "out"
        ],
        "args_change": [
            [
                "out",
                ""
            ]
        ],
        "args_warning": {
            "out": "this arg is deleted in this version"
        }
    },
    "paddle.fluid.layers.nn.relu6": {
        "alias": [
            "paddle.fluid.layers.relu6"
        ],
        "warning": "this api is update to paddle.nn.functional.relu6",
    },
    "paddle.fluid.layers.sigmoid_focal_loss": {
        "alias": [
            "paddle.fluid.layers.detection.sigmoid_focal_loss"
        ],
        "update_to": "paddle.nn.functional.sigmoid_focal_loss",
        "args_list": [
            "x",
            "label",
            "fg_num",
            "gamma",
            "alpha"
        ],
        "args_change": [
            [
                "x",
                "logit"
            ],
            [
                "fg_num",
                "normalizer"
            ]
        ],
        "warning": "this api default returned summed loss in this version, for more infomation, please see paddle.nn.functional.sigmoid_focal_loss"
    },
    "paddle.fluid.CUDAPlace": {
        "update_to": "paddle.CUDAPlace"
    },
    "paddle.fluid.layers.tanh": {
        "alias": [
            "paddle.fluid.layers.ops.tanh"
        ],
        "update_to": "paddle.tanh"
    },
    "paddle.fluid.layers.nn.instance_norm": {
        "alias": [
            "paddle.fluid.layers.instance_norm"
        ],
        "warning": "this api is update to paddle.nn.functional.instance_norm",
    },
    "paddle.fluid.layers.ops.sigmoid": {
        "alias": [
            "paddle.fluid.layers.sigmoid"
        ],
        "update_to": "paddle.nn.functional.sigmoid"
    },
    "paddle.fluid.layers.increment": {
        "alias": [
            "paddle.fluid.layers.control_flow.increment"
        ],
        "update_to": "paddle.increment",
        "args_list": [
            "x",
            "value",
            "in_place"
        ],
        "args_change": [
            [
                "in_place",
                ""
            ]
        ],
        "args_warning": {
            "in_place": "this arg is deleted in this version"
        }
    },
    "paddle.fluid.layers.tensor.eye": {
        "alias": [
            "paddle.fluid.layers.eye"
        ],
        "update_to": "paddle.eye",
        "args_list": [
            "num_rows",
            "num_columns",
            "batch_shape",
            "dtype"
        ],
        "args_change": [
            [
                "batch_shape",
                ""
            ]
        ],
        "args_warning": {
            "batch_shape": "this arg is deleted in this version"
        }
    },
    "paddle.fluid.layers.loss.softmax_with_cross_entropy": {
        "alias": [
            "paddle.fluid.layers.softmax_with_cross_entropy"
        ],
        "update_to": "paddle.nn.functional.softmax_with_cross_entropy"
    },
    "paddle.fluid.dygraph.container.Sequential": {
        "alias": [
            "paddle.fluid.dygraph.Sequential"
        ],
        "update_to": "paddle.nn.Sequential"
    },
    "paddle.fluid.dygraph.NaturalExpDecay": {
        "alias": [
            "paddle.fluid.dygraph.learning_rate_scheduler.NaturalExpDecay"
        ],
        "warning": "this api is update to paddle.optimizer.lr.NaturalExpDecay",
    },
    "paddle.fluid.layers.cond": {
        "alias": [
            "paddle.fluid.layers.control_flow.cond"
        ],
        "update_to": "paddle.static.nn.cond"
    },
    "paddle.fluid.layers.elementwise_pow": {
        "alias": [
            "paddle.fluid.layers.nn.elementwise_pow"
        ],
        "update_to": "paddle.pow",
        "args_list": [
            "x",
            "y",
            "axis",
            "act",
            "name"
        ],
        "args_change": [
            [
                "axis",
                ""
            ],
            [
                "act",
                ""
            ]
        ],
        "args_warning": {
            "axis": "this arg is deleted in this version",
            "act": "this arg is deleted in this version"
        }
    },
    "paddle.fluid.layers.rank": {
        "alias": [
            "paddle.fluid.layers.nn.rank"
        ],
        "update_to": "paddle.rank"
    },
    "paddle.fluid.layers.isfinite": {
        "alias": [
            "paddle.fluid.layers.tensor.isfinite"
        ],
        "update_to": "paddle.isfinite",
    },
    "paddle.fluid.layers.nn.allclose": {
        "alias": [
            "paddle.fluid.layers.allclose"
        ],
        "update_to": "paddle.allclose",
        "args_list": [
            "input",
            "other",
            "rtol",
            "atol",
            "equal_nan",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "other",
                "y"
            ]
        ]
    },
    "paddle.fluid.metrics.Accuracy": {
        "update_to": "paddle.metric.Accuracy",
    },
    "paddle.fluid.layers.assign": {
        "alias": [
            "paddle.fluid.layers.tensor.assign"
        ],
        "update_to": "paddle.nn.functional.assign",
    },
    "paddle.fluid.layers.sin": {
        "alias": [
            "paddle.fluid.layers.ops.sin"
        ],
        "update_to": "paddle.sin"
    },
    "paddle.fluid.dygraph.BilinearTensorProduct": {
        "alias": [
            "paddle.fluid.dygraph.nn.BilinearTensorProduct"
        ],
        "update_to": "paddle.nn.layer.BilinearTensorProduct"
    },
    "paddle.fluid.layers.nn.topk": {
        "alias": [
            "paddle.fluid.layers.topk"
        ],
        "update_to": "paddle.topk",
        "args_list": [
            "input",
            "k",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ]
        ]
    },
    "paddle.fluid.dygraph.nn.Dropout": {
        "alias": [
            "paddle.fluid.dygraph.Dropout"
        ],
        "warning": "this api is update to paddle.nn.Dropout",
    },
    "paddle.fluid.layers.control_flow.not_equal": {
        "alias": [
            "paddle.fluid.layers.not_equal"
        ],
        "update_to": "paddle.not_equal",
        "args_list": [
            "x",
            "y",
            "cond"
        ],
        "args_change": [
            [
                "cond",
                ""
            ]
        ],
        "args_warning": {
            "cond": "this args is deleted in this version"
        }
    },
    "paddle.fluid.layers.nn.conv3d": {
        "alias": [
            "paddle.fluid.layers.conv3d"
        ],
        "warning": "this api is update to paddle.nn.functional.conv3d",
    },
    "paddle.fluid.dataset.QueueDataset": {
        "update_to": "paddle.distributed.QueueDataset"
    },
    "paddle.fluid.layers.iou_similarity": {
        "alias": [
            "paddle.fluid.layers.detection.iou_similarity"
        ],
        "update_to": "paddle.nn.functional.iou_similarity"
    },
    "paddle.fluid.metrics.Precision": {
        "update_to": "paddle.metric.Precision",
    },
    "paddle.fluid.layers.loss.npair_loss": {
        "alias": [
            "paddle.fluid.layers.npair_loss"
        ],
        "update_to": "paddle.nn.functional.npair_loss"
    },
    "paddle.fluid.layers.nn.stanh": {
        "alias": [
            "paddle.fluid.layers.stanh"
        ],
        "update_to": "paddle.stanh"
    },
    "paddle.fluid.require_version": {
        "alias": [
            "paddle.fluid.framework.require_version"
        ],
        "update_to": "paddle.utils.require_version"
    },
    "paddle.fluid.layers.LSTMCell": {
        "warning": "this api is update to paddle.nn.LSTMCell",
    },
    "paddle.fluid.layers.nn.conv2d_transpose": {
        "alias": [
            "paddle.fluid.layers.conv2d_transpose"
        ],
        "warning": "this api is update to paddle.nn.functional.conv2d_transpose"
    },
    "paddle.fluid.layers.nn.elementwise_min": {
        "alias": [
            "paddle.fluid.layers.elementwise_min"
        ],
        "update_to": "paddle.minimum",
        "args_list": [
            "x",
            "y",
            "axis",
            "act",
            "name"
        ],
        "args_change": [
            [
                "act",
                ""
            ]
        ],
        "args_warning": {
            "act": "this arg is deleted in this version"
        }
    },
    "paddle.fluid.layers.sqrt": {
        "alias": [
            "paddle.fluid.layers.ops.sqrt"
        ],
        "update_to": "paddle.sqrt"
    },
    "paddle.fluid.regularizer.L1DecayRegularizer": {
        "alias": [
            "paddle.fluid.regularizer.L1Decay"
        ],
        "update_to": "paddle.regularizer.L1Decay",
        "args_list": [
            "regularization_coeff"
        ],
        "args_change": [
            [
                "regularization_coeff",
                "coeff"
            ]
        ]
    },
    "paddle.fluid.layers.ops.thresholded_relu": {
        "alias": [
            "paddle.fluid.layers.thresholded_relu"
        ],
        "update_to": "paddle.nn.functional.thresholded_relu"
    },
    "paddle.fluid.layers.nn.log1p": {
        "alias": [
            "paddle.fluid.layers.log1p"
        ],
        "update_to": "paddle.log1p",
        "args_list": [
            "x",
            "out",
            "name"
        ],
        "args_change": [
            [
                "out",
                ""
            ]
        ],
        "args_warning": {
            "out": "this arg is deleted in this version."
        }
    },
    "paddle.fluid.layers.ops.gelu": {
        "alias": [
            "paddle.fluid.layers.gelu"
        ],
        "update_to": "paddle.nn.functional.gelu",
    },
    "paddle.fluid.layers.gather": {
        "alias": [
            "paddle.fluid.layers.nn.gather"
        ],
        "update_to": "paddle.gather",
        "args_list": [
            "input",
            "index",
            "overwrite"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "overwrite",
                ""
            ]
        ],
        "args_warning": {
            "overwrite": "this args is deleted in this version."
        }
    },
    "paddle.fluid.optimizer.AdamaxOptimizer": {
        "alias": [
            "paddle.fluid.optimizer.Adamax"
        ],
        "update_to": "paddle.optimizer.Adamax",
        "args_list": [
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "parameter_list",
            "regularization",
            "grad_clip",
            "name"
        ],
        "args_change": [
            [
                "parameter_list",
                "parameters"
            ],
            [
                "regularization",
                "weight_decay"
            ]
        ]
    },
    "paddle.fluid.optimizer.AdamOptimizer": {
        "alias": [
            "paddle.fluid.optimizer.Adam"
        ],
        "update_to": "paddle.optimizer.Adam",
        "args_list": [
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "parameter_list",
            "regularization",
            "grad_clip",
            "name",
            "lazy_mode"
        ],
        "args_change": [
            [
                "parameter_list",
                "parameters"
            ],
            [
                "regularization",
                "weight_decay"
            ]
        ]
    },
    "paddle.fluid.layers.ops.rsqrt": {
        "alias": [
            "paddle.fluid.layers.rsqrt"
        ],
        "update_to": "paddle.rsqrt"
    },
    "paddle.fluid.dygraph.DataParallel": {
        "alias": [
            "paddle.fluid.dygraph.parallel.DataParallel"
        ],
        "update_to": "paddle.DataParallel",
    },
    "paddle.fluid.layers.nn.stack": {
        "alias": [
            "paddle.fluid.layers.stack"
        ],
        "update_to": "paddle.stack",
    },
    "paddle.fluid.dygraph.nn.SpectralNorm": {
        "alias": [
            "paddle.fluid.dygraph.SpectralNorm"
        ],
        "update_to": "paddle.nn.SpectralNorm"
    },
    "paddle.fluid.layers.reduce_any": {
        "alias": [
            "paddle.fluid.layers.nn.reduce_any"
        ],
        "update_to": "paddle.any",
        "args_list": [
            "input",
            "dim",
            "keep_dim",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "dim",
                "axis"
            ],
            [
                "keep_dim",
                "keepdim"
            ]
        ]
    },
    "paddle.fluid.layers.GRUCell": {
        "warning": "this api is update to paddle.nn.GRUCell"
    }
}
