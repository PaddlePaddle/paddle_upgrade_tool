change_spec = {
    # TODO list
    # paddle.fluid.dygraph.guard
    # paddle.fluid.unique_name.guard
    # paddle.fluid.layers.data
    # paddle.fluid.layers.Normal
    # paddle.fluid.layers.Uniform
    # paddle.fluid.layers.load
    # manual add
    "paddle.fluid.layers.round": {
        "update_to": "paddle.round"
    },
    # manual add
    "paddle.fluid.layers.kron": {
        "update_to": "paddle.kron",
        "args_list": [
            "x",
            "y",
            "out",
            "name"
        ],
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "y",
                "y"
            ],
            [
                "out",
                ""
            ],
            [
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "out": "out is deleted in paddle.kron"
        }
    },
    # manual add
    "paddle.fluid.layers.trace": {
        "update_to": "paddle.trace",
        "args_list": [
            "input",
            "offset",
            "dim1",
            "dim2",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "offset",
                "offset"
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
                "name",
                "name"
            ]
        ]
    },
    # manual add
    "paddle.fluid.layers.sum": {
        "update_to": "paddle.sum"
    },
    # TODO transformer
    # "paddle.fluid.layers.reshape": {
    #     "update_to": "paddle.reshape"
    # },
    # manual add
    "paddle.fluid.io.load": {
        "alias": ["paddle.fluid.load"],
        "update_to": "paddle.io.load"
    },
    # manual add
    "paddle.fluid.initializer.Normal": {
        "update_to": "paddle.nn.initializer.Normal"
    },
    # manual add
    "paddle.fluid.data": {
        "update_to": "paddle.data"
    },
    "paddle.batch": {
        "alias": [
           "paddle.fluid.io.batch"
        ],
        "update_to": "paddle.batch"
    },
    "paddle.check_import_scipy": {
        "update_to": "paddle.check_import_scipy"
    },
    "paddle.compat.long_type": {
        "update_to": "paddle.compat.long_type"
    },
    "paddle.compat.to_text": {
        "update_to": "paddle.compat.to_text"
    },
    "paddle.compat.to_bytes": {
        "update_to": "paddle.compat.to_bytes"
    },
    "paddle.compat.floor_division": {
        "update_to": "paddle.compat.floor_division"
    },
    "paddle.compat.get_exception_message": {
        "update_to": "paddle.compat.get_exception_message"
    },
    "paddle.complex.transpose": {
        "alias": [
            "paddle.fluid.layers.transpose"
        ],
        "update_to": "paddle.transpose"
    },
    # manual check
    "paddle.complex.matmul": {
        "alias": [
            "paddle.fluid.layers.matmul"
        ],
        "update_to": "paddle.matmul",
        "args_list": [
            "x",
            "y",
            "transpose_x",
            "transpose_y",
            "alpha",
            "name"
        ],
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "y",
                "y"
            ],
            [
                "transpose_x",
                "transpose_x"
            ],
            [
                "transpose_y",
                "transpose_y"
            ],
            [
                "alpha",
                ""
            ],
            [
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "alpha": "This args is deleted in this version."
        }
    },
    "paddle.fluid.Program": {
        "update_to": "paddle.static.Program"
    },
    "paddle.fluid.default_startup_program": {
        "update_to": "paddle.static.default_startup_program"
    },
    "paddle.fluid.default_main_program": {
        "update_to": "paddle.static.default_main_program"
    },
    "paddle.fluid.program_guard": {
        "update_to": "paddle.static.program_guard"
    },
    "paddle.fluid.name_scope": {
        "update_to": "paddle.static.name_scope"
    },
    # FlUID_WARNING
    # "paddle.fluid.cuda_places": {
    #     "update_to": "paddle.fluid.cuda_places"
    # },
    # FlUID_WARNING
    # "paddle.fluid.cpu_places": {
    #     "update_to": "paddle.fluid.cpu_places"
    # },
    # FlUID_WARNING
    # "paddle.fluid.cuda_pinned_places": {
    #     "update_to": "paddle.fluid.cuda_pinned_places"
    # },
    # INCUBATE_WARNING
    # "paddle.fluid.in_dygraph_mode": {
    #     "update_to": "paddle.incubate.hapi.dygraph_layer_patch.in_dygraph_mode"
    # },
    # FlUID_WARNING
    # "paddle.fluid.is_compiled_with_cuda": {
    #     "update_to": "paddle.fluid.is_compiled_with_cuda"
    # },
    # TODO Variable
    # "paddle.fluid.Variable": {
    #     "update_to": "paddle.Variable"
    # },
    # TODO ComplexVariable
    # "paddle.fluid.ComplexVariable": {
    #     "update_to": "paddle.fluid.ComplexVariable",
    #     "args_list": [
    #         "real",
    #         "imag"
    #     ],
    #     "args_change": [
    #         [
    #             "real",
    #             "real"
    #         ],
    #         [
    #             "imag",
    #             "imag"
    #         ]
    #     ]
    # },
    # FlUID_WARNING
    # "paddle.fluid.load_op_library": {
    #     "update_to": "paddle.fluid.load_op_library"
    # },
    # FlUID_WARNING
    # "paddle.fluid.require_version": {
    #     "update_to": "paddle.fluid.require_version"
    # },
    # FlUID_WARNING
    # "paddle.fluid.device_guard": {
    #     "update_to": "paddle.fluid.device_guard"
    # },
    # FlUID_WARNING
    # "paddle.fluid.set_flags": {
    #     "update_to": "paddle.fluid.set_flags"
    # },
    # FlUID_WARNING
    # "paddle.fluid.get_flags": {
    #     "update_to": "paddle.fluid.get_flags"
    # },
    "paddle.fluid.Executor": {
        "update_to": "paddle.static.Executor"
    },
    "paddle.fluid.global_scope": {
        "update_to": "paddle.static.global_scope"
    },
    "paddle.fluid.scope_guard": {
        "update_to": "paddle.static.scope_guard"
    },
    # FlUID_WARNING
    # "paddle.fluid.DistributeTranspiler": {
    #     "alias": [
    #         "paddle.fluid.transpiler.DistributeTranspiler"
    #     ],
    #     "update_to": "paddle.fluid.DistributeTranspiler"
    # },
    # FlUID_WARNING
    # "paddle.fluid.memory_optimize": {
    #     "alias": [
    #         "paddle.fluid.transpiler.memory_optimize"
    #     ],
    #     "update_to": "paddle.fluid.memory_optimize"
    # },
    # FlUID_WARNING
    # "paddle.fluid.release_memory": {
    #     "alias": [
    #         "paddle.fluid.transpiler.release_memory"
    #     ],
    #     "update_to": "paddle.fluid.release_memory"
    # },
    # FlUID_WARNING
    # "paddle.fluid.DistributeTranspilerConfig": {
    #     "alias": [
    #         "paddle.fluid.transpiler.DistributeTranspilerConfig"
    #     ],
    #     "update_to": "paddle.fluid.DistributeTranspilerConfig"
    # },
    "paddle.fluid.ParallelExecutor": {
        "update_to": "paddle.static.ParallelExecutor"
    },
    # FlUID_WARNING
    # "paddle.fluid.create_lod_tensor": {
    #     "update_to": "paddle.fluid.create_lod_tensor"
    # },
    # FlUID_WARNING
    # "paddle.fluid.create_random_int_lodtensor": {
    #     "update_to": "paddle.fluid.create_random_int_lodtensor"
    # },
    # FlUID_WARNING
    # "paddle.fluid.DataFeedDesc": {
    #     "update_to": "paddle.fluid.DataFeedDesc"
    # },
    "paddle.fluid.CompiledProgram": {
        "update_to": "paddle.static.CompiledProgram"
    },
    "paddle.fluid.ExecutionStrategy": {
        "update_to": "paddle.static.ExecutionStrategy"
    },
    "paddle.fluid.BuildStrategy": {
        "update_to": "paddle.static.BuildStrategy"
    },
    "paddle.fluid.gradients": {
        "alias": [
            "paddle.fluid.backward.gradients"
        ],
        "update_to": "paddle.static.gradients"
    },
    # FlUID_WARNING
    # "paddle.fluid.io.save_vars": {
    #     "update_to": "paddle.fluid.io.save_vars"
    # },
    # FlUID_WARNING
    # "paddle.fluid.io.save_params": {
    #     "update_to": "paddle.fluid.io.save_params"
    # },
    # FlUID_WARNING
    # "paddle.fluid.io.save_persistables": {
    #     "update_to": "paddle.fluid.io.save_persistables"
    # },
    # FlUID_WARNING
    # "paddle.fluid.io.load_vars": {
    #     "update_to": "paddle.fluid.io.load_vars"
    # },
    # FlUID_WARNING
    # "paddle.fluid.io.load_params": {
    #     "update_to": "paddle.fluid.io.load_params"
    # },
    # FlUID_WARNING
    # "paddle.fluid.io.load_persistables": {
    #     "update_to": "paddle.fluid.io.load_persistables"
    # },
    "paddle.fluid.io.save_inference_model": {
        "update_to": "paddle.io.save_inference_model"
    },
    "paddle.fluid.io.load_inference_model": {
        "update_to": "paddle.io.load_inference_model"
    },
    # manual check
    "paddle.fluid.io.save": {
        "alias": [
            "paddle.fluid.save"
        ],
        "update_to": "paddle.io.save"
    },
    "paddle.fluid.io.load_program_state": {
        "update_to": "paddle.io.load_program_state"
    },
    "paddle.fluid.io.set_program_state": {
        "update_to": "paddle.io.set_program_state"
    },
    # FlUID_WARNING
    # "paddle.fluid.io.get_program_parameter": {
    #     "update_to": "paddle.fluid.io.get_program_parameter"
    # },
    # FlUID_WARNING
    # "paddle.fluid.io.get_program_persistable_vars": {
    #     "update_to": "paddle.fluid.io.get_program_persistable_vars"
    # },
    # FlUID_WARNING
    # "paddle.fluid.io.PyReader": {
    #     "update_to": "paddle.fluid.io.PyReader"
    # },
    "paddle.fluid.io.DataLoader": {
        "update_to": "paddle.io.DataLoader"
    },
    # FlUID_WARNING
    # "paddle.fluid.io.default_collate_fn": {
    #     "update_to": "paddle.fluid.io.default_collate_fn"
    # },
    "paddle.fluid.io.cache": {
        "alias": [
            "paddle.reader.cache"
        ],
        "update_to": "paddle.io.cache"
    },
    "paddle.fluid.io.map_readers": {
        "alias": [
            "paddle.reader.map_readers"
        ],
        "update_to": "paddle.io.map_readers"
    },
    "paddle.fluid.io.buffered": {
        "alias": [
            "paddle.reader.buffered"
        ],
        "update_to": "paddle.io.buffered"
    },
    "paddle.fluid.io.compose": {
        "alias": [
            "paddle.reader.compose"
        ],
        "update_to": "paddle.io.compose"
    },
    "paddle.fluid.io.chain": {
        "alias": [
            "paddle.reader.chain"
        ],
        "update_to": "paddle.io.chain"
    },
    "paddle.fluid.io.shuffle": {
        "alias": [
            "paddle.reader.shuffle"
        ],
        "update_to": "paddle.shuffle"
    },
    "paddle.fluid.io.firstn": {
        "alias": [
            "paddle.reader.firstn"
        ],
        "update_to": "paddle.io.firstn"
    },
    "paddle.fluid.io.xmap_readers": {
        "alias": [
            "paddle.reader.xmap_readers"
        ],
        "update_to": "paddle.io.xmap_readers"
    },
    "paddle.fluid.io.multiprocess_reader": {
        "alias": [
            "paddle.reader.multiprocess_reader"
        ],
        "update_to": "paddle.reader.multiprocess_reader"
    },
    # manual check initializer: add alias Line309 - Line350
    "paddle.fluid.initializer.Constant": {
        "alias": [
            "paddle.fluid.initializer.ConstantInitializer"
        ],
        "update_to": "paddle.nn.initializer.Constant"
    },
    "paddle.fluid.initializer.TruncatedNormal": {
        "alias": [
            "paddle.fluid.initializer.TruncatedNormalInitializer"
        ],
        "update_to": "paddle.nn.initializer.TruncatedNormal"
    },
    "paddle.fluid.initializer.Xavier": {
        "alias": [
            "paddle.fluid.initializer.XavierInitializer"
        ],
        "update_to": "paddle.nn.initializer.Xavier"
    },
    "paddle.fluid.initializer.Bilinear": {
        "alias": [
            "paddle.fluid.initializer.BilinearInitializer"
        ],
        "update_to": "paddle.nn.initializer.Bilinear"
    },
    "paddle.fluid.initializer.MSRA": {
        "alias": [
            "paddle.fluid.initializer.MSRAInitializer"
        ],
        "update_to": "paddle.nn.initializer.MSRA"
    },
    "paddle.fluid.initializer.Uniform": {
        "alias": [
            "paddle.fluid.initializer.UniformInitializer"
        ],
        "update_to": "paddle.nn.initializer.Uniform"
    },
    "paddle.fluid.initializer.Normal": {
        "alias": [
            "paddle.fluid.initializer.NormalInitializer"
        ],
        "update_to": "paddle.nn.initializer.Normal"
    },
    # FlUID_WARNING
    # "paddle.fluid.initializer.NumpyArrayInitializer": {
    #     "update_to": "paddle.fluid.initializer.NumpyArrayInitializer"
    # },
    # TODO embedding FlUID_WARNING
    # "paddle.fluid.embedding": {
    #     "alias": [
    #         "paddle.fluid.layers.embedding"
    #     ],
    #     "update_to": "paddle.fluid.embedding"
    # },
    # manual check paddle.nn.functional.one_hoe
    "paddle.fluid.one_hot": {
        "alias": [
            "paddle.fluid.layers.one_hot"
        ],
        "warning": "input->x, depth->num_classes, x'elements must less than num_classes."
    },
    # manual check
    "paddle.fluid.layers.log1p": {
        "update_to": "paddle.log1p",
        "args_list": [
            "x",
            "out",
            "name"
        ],
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "out",
                ""
            ],
            [
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "out": "this arg is deleted in this version."
        }
    },
    # manual check
    "paddle.fluid.layers.logsumexp": {
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
                "x",
                "x"
            ],
            [
                "dim",
                "axis"
            ],
            [
                "keepdim",
                "keepdim"
            ],
            [
                "out",
                ""
            ],
            [
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "out": "this arg is deleted in this version."
        }
    },
    # TODO check this api
    # "paddle.fluid.layers.clamp": {
    #     "args_list": [
    #         "input",
    #         "min",
    #         "max",
    #         "output",
    #         "name"
    #     ],
    #     "args_change": [
    #         [
    #             "input",
    #             "input"
    #         ],
    #         [
    #             "min",
    #             "min"
    #         ],
    #         [
    #             "max",
    #             "max"
    #         ],
    #         [
    #             "output",
    #             "output"
    #         ],
    #         [
    #             "name",
    #             "name"
    #         ]
    #     ]
    # },
    # TODO args reorder
    # "paddle.fluid.layers.addmm": {
    #     "update_to": "paddle.addmm",
    #     "args_list": [
    #         "input",
    #         "x",
    #         "y",
    #         "alpha",
    #         "beta",
    #         "name"
    #     ],
    #     "args_change": [
    #         [
    #             "input",
    #             "input"
    #         ],
    #         [
    #             "x",
    #             "x"
    #         ],
    #         [
    #             "y",
    #             "y"
    #         ],
    #         [
    #             "alpha",
    #             "alpha"
    #         ],
    #         [
    #             "beta",
    #             "beta"
    #         ],
    #         [
    #             "name",
    #             "name"
    #         ]
    #     ]
    # },
    # mantul check
    "paddle.fluid.layers.addcmul": {
        "update_to": "paddle.addcmul",
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
                "input",
                "input"
            ],
            [
                "tensor1",
                "tensor1"
            ],
            [
                "tensor2",
                "tensor2"
            ],
            [
                "value",
                "value"
            ],
            [
                "out",
                ""
            ],
            [
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "out": "this arg is deleted in this version."
        }
    },
    "paddle.fluid.layers.bmm": {
        "update_to": "paddle.bmm"
    },
    "paddle.fluid.layers.nonzero": {
        "update_to": "paddle.nonzero"
    },
    # manual ckeck
    "paddle.fluid.layers.index_select": {
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
                "index",
                "index"
            ],
            [
                "dim",
                "axis"
            ],
            [
                "",
                "name",
                "None"
            ]
        ]
    },
    "paddle.fluid.layers.dist": {
        "update_to": "paddle.dist"
    },
    "paddle.fluid.layers.dot": {
        "update_to": "paddle.dot"
    },
    "paddle.fluid.layers.t": {
        "update_to": "paddle.t"
    },
    # manual check
    "paddle.fluid.layers.cross": {
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
            ],
            [
                "",
                "name",
                "None"
            ]
        ]
    },
    # TODO manual check
    # "paddle.fluid.layers.interpolate": {
    #     "update_to": "paddle.nn.functional.interpolate",
    #     "args_list": [
    #         "input",
    #         "out_shape",
    #         "scale",
    #         "name",
    #         "resample",
    #         "actual_shape",
    #         "align_corners",
    #         "align_mode",
    #         "data_format"
    #     ],
    #     "args_change": [
    #         [
    #             "input",
    #             "input"
    #         ],
    #         [
    #             "out_shape",
    #             "out_shape"
    #         ],
    #         [
    #             "scale",
    #             "scale"
    #         ],
    #         [
    #             "name",
    #             "name"
    #         ],
    #         [
    #             "resample",
    #             "resample"
    #         ],
    #         [
    #             "actual_shape",
    #             "actual_shape"
    #         ],
    #         [
    #             "align_corners",
    #             "align_corners"
    #         ],
    #         [
    #             "align_mode",
    #             "align_mode"
    #         ],
    #         [
    #             "data_format",
    #             "data_format"
    #         ]
    #     ]
    # },
    "paddle.fluid.layers.diag_embed": {
        "update_to": "paddle.nn.functional.diag_embed"
    },
    # manual check
    "paddle.fluid.layers.meshgrid": {
        "update_to": "paddle.meshgrid",
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.fc": {
    #     "update_to": "paddle.fluid.layers.fc"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.linear_chain_crf": {
    #     "update_to": "paddle.fluid.layers.linear_chain_crf"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.crf_decoding": {
    #     "update_to": "paddle.fluid.layers.crf_decoding"
    # },
    "paddle.fluid.layers.cos_sim": {
        "update_to": "paddle.metric.cos_sim"
    },
    "paddle.fluid.layers.chunk_eval": {
        "update_to": "paddle.metric.chunk_eval"
    },
    # TODO transformer
    # "paddle.fluid.layers.conv2d": {
    #     "args_list": [
    #         "input",
    #         "num_filters",
    #         "filter_size",
    #         "stride",
    #         "padding",
    #         "dilation",
    #         "groups",
    #         "param_attr",
    #         "bias_attr",
    #         "use_cudnn",
    #         "act",
    #         "name",
    #         "data_format"
    #     ],
    #     "args_change": [
    #         [
    #             "input",
    #             "input"
    #         ],
    #         [
    #             "num_filters",
    #             "num_filters"
    #         ],
    #         [
    #             "filter_size",
    #             "filter_size"
    #         ],
    #         [
    #             "stride",
    #             "stride"
    #         ],
    #         [
    #             "padding",
    #             "padding"
    #         ],
    #         [
    #             "dilation",
    #             "dilation"
    #         ],
    #         [
    #             "groups",
    #             "groups"
    #         ],
    #         [
    #             "param_attr",
    #             "param_attr"
    #         ],
    #         [
    #             "bias_attr",
    #             "bias_attr"
    #         ],
    #         [
    #             "use_cudnn",
    #             "use_cudnn"
    #         ],
    #         [
    #             "act",
    #             "act"
    #         ],
    #         [
    #             "name",
    #             "name"
    #         ],
    #         [
    #             "data_format",
    #             "data_format"
    #         ]
    #     ]
    # },
    # TODO transformer
    # "paddle.fluid.layers.conv3d": {
    #     "args_list": [
    #         "input",
    #         "num_filters",
    #         "filter_size",
    #         "stride",
    #         "padding",
    #         "dilation",
    #         "groups",
    #         "param_attr",
    #         "bias_attr",
    #         "use_cudnn",
    #         "act",
    #         "name",
    #         "data_format"
    #     ],
    #     "args_change": [
    #         [
    #             "input",
    #             "input"
    #         ],
    #         [
    #             "num_filters",
    #             "num_filters"
    #         ],
    #         [
    #             "filter_size",
    #             "filter_size"
    #         ],
    #         [
    #             "stride",
    #             "stride"
    #         ],
    #         [
    #             "padding",
    #             "padding"
    #         ],
    #         [
    #             "dilation",
    #             "dilation"
    #         ],
    #         [
    #             "groups",
    #             "groups"
    #         ],
    #         [
    #             "param_attr",
    #             "param_attr"
    #         ],
    #         [
    #             "bias_attr",
    #             "bias_attr"
    #         ],
    #         [
    #             "use_cudnn",
    #             "use_cudnn"
    #         ],
    #         [
    #             "act",
    #             "act"
    #         ],
    #         [
    #             "name",
    #             "name"
    #         ],
    #         [
    #             "data_format",
    #             "data_format"
    #         ]
    #     ]
    # },
    # TODO transformer paddle.nn.functional.softmax
    "paddle.fluid.layers.softmax": {
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
        ]
    },
    # TODO whether remind users max_pool2d or avg_pool2d
    "paddle.fluid.layers.pool2d": {
        "update_to": "paddle.nn.functional.pool2d"
    },
    "paddle.fluid.layers.pool3d": {
        "update_to": "paddle.nn.functional.pool3d"
    },
    "paddle.fluid.layers.adaptive_pool2d": {
        "update_to": "paddle.nn.functional.adaptive_pool2d"
    },
    "paddle.fluid.layers.adaptive_pool3d": {
        "update_to": "paddle.nn.functional.adaptive_pool3d"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.batch_norm": {
    #     "update_to": "paddle.fluid.layers.batch_norm"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.inplace_abn": {
    #     "update_to": "paddle.fluid.layers.inplace_abn"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.instance_norm": {
    #     "update_to": "paddle.fluid.layers.instance_norm"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.data_norm": {
    #     "update_to": "paddle.fluid.layers.data_norm",
    # },
    # # TODO conv_transpose2d FlUID_WARNING
    # "paddle.fluid.layers.conv2d_transpose": {
    #     "update_to": "paddle.fluid.layers.conv2d_transpose"
    # },
    # # TODO conv_transpose3d FlUID_WARNING
    # "paddle.fluid.layers.conv3d_transpose": {
    #     "update_to": "paddle.fluid.layers.conv3d_transpose"
    # },
    "paddle.fluid.layers.reduce_sum": {
        "update_to": "paddle.reduce_sum"
    },
    "paddle.fluid.layers.reduce_mean": {
        "update_to": "paddle.reduce_mean"
    },
    "paddle.fluid.layers.reduce_max": {
        "update_to": "paddle.reduce_max"
    },
    "paddle.fluid.layers.reduce_min": {
        "update_to": "paddle.reduce_min"
    },
    "paddle.fluid.layers.reduce_prod": {
        "update_to": "paddle.reduce_prod"
    },
    "paddle.fluid.layers.reduce_all": {
        "update_to": "paddle.reduce_all"
    },
    "paddle.fluid.layers.reduce_any": {
        "update_to": "paddle.reduce_any"
    },
    # TODO transformer paddle.nn.functional.dropout
    # "paddle.fluid.layers.dropout": {
    #     "args_list": [
    #         "x",
    #         "dropout_prob",
    #         "is_test",
    #         "seed",
    #         "name",
    #         "dropout_implementation"
    #     ],
    #     "args_change": [
    #         [
    #             "x",
    #             "x"
    #         ],
    #         [
    #             "dropout_prob",
    #             "dropout_prob"
    #         ],
    #         [
    #             "is_test",
    #             "is_test"
    #         ],
    #         [
    #             "seed",
    #             "seed"
    #         ],
    #         [
    #             "name",
    #             "name"
    #         ],
    #         [
    #             "dropout_implementation",
    #             "dropout_implementation"
    #         ]
    #     ]
    # },
    # manual check
    "paddle.fluid.layers.split": {
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
                "num_or_sections",
                "num_or_sections"
            ],
            [
                "dim",
                "axis"
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.ctc_greedy_decoder": {
    #     "update_to": "paddle.fluid.layers.ctc_greedy_decoder"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.l2_normalize": {
    #     "update_to": "paddle.fluid.layers.l2_normalize"
    # },
    # manual check
    "paddle.fluid.layers.topk": {
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
            ],
            [
                "k",
                "k"
            ],
            [
                "",
                "axis",
                "None"
            ],
            [
                "",
                "largest",
                "True"
            ],
            [
                "",
                "sorted",
                "True"
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.im2sequence": {
    #     "update_to": "paddle.fluid.layers.im2sequence"
    # },
    # TODO transformer
    # "paddle.fluid.layers.row_conv": {
    #     "args_list": [
    #         "input",
    #         "future_context_size",
    #         "param_attr",
    #         "act"
    #     ],
    #     "args_change": [
    #         [
    #             "input",
    #             "input"
    #         ],
    #         [
    #             "future_context_size",
    #             "future_context_size"
    #         ],
    #         [
    #             "param_attr",
    #             "param_attr"
    #         ],
    #         [
    #             "act",
    #             "act"
    #         ]
    #     ]
    # },
    "paddle.fluid.layers.multiplex": {
        "update_to": "paddle.multiplex"
    },
    # TODO manual check paddle.nn.functional.norm.layer_norm
    # "paddle.fluid.layers.layer_norm": {
    #     "update_to": "paddle.fluid.layers.layer_norm"
    # },
    # TODO FlUID_WARNING paddle.nn.functional.norm.group_norm
    # "paddle.fluid.layers.group_norm": {
    #     "update_to": "paddle.fluid.layers.group_norm"
    # },
    # TODO FlUID_WARNING paddle.nn.functional.norm.spectral_norm
    # "paddle.fluid.layers.spectral_norm": {
    #     "update_to": "paddle.fluid.layers.spectral_norm"
    # },
    "paddle.fluid.layers.smooth_l1": {
        "update_to": "paddle.nn.functional.smooth_l1"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.autoincreased_step_counter": {
    #     "update_to": "paddle.fluid.layers.autoincreased_step_counter"
    # },
    # manual check
    "paddle.fluid.layers.squeeze": {
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
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    # manual check
    "paddle.fluid.layers.unsqueeze": {
        "update_to": "paddle.unsqueeze",
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
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.lod_reset": {
    #     "update_to": "paddle.fluid.layers.lod_reset"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.lod_append": {
    #     "update_to": "paddle.fluid.layers.lod_append"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.lrn": {
    #     "update_to": "paddle.fluid.layers.lrn"
    # },
    # TODO paddle.nn.functional.pad
    # "paddle.fluid.layers.pad": {
    #     "args_list": [
    #         "x",
    #         "paddings",
    #         "pad_value",
    #         "name"
    #     ],
    #     "args_change": [
    #         [
    #             "x",
    #             "x"
    #         ],
    #         [
    #             "paddings",
    #             "paddings"
    #         ],
    #         [
    #             "pad_value",
    #             "pad_value"
    #         ],
    #         [
    #             "name",
    #             "name"
    #         ]
    #     ]
    # },
    "paddle.fluid.layers.pad_constant_like": {
        "update_to": "paddle.nn.functional.pad_constant_like"
    },
    "paddle.fluid.layers.label_smooth": {
        "update_to": "paddle.nn.functional.label_smooth"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.roi_pool": {
    #     "update_to": "paddle.fluid.layers.roi_pool"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.roi_align": {
    #     "update_to": "paddle.fluid.layers.roi_align"
    # },
    "paddle.fluid.layers.dice_loss": {
        "update_to": "paddle.nn.functional.dice_loss"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.image_resize": {
    #     "update_to": "paddle.fluid.layers.image_resize"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.image_resize_short": {
    #     "update_to": "paddle.fluid.layers.image_resize_short"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.resize_bilinear": {
    #     "update_to": "paddle.fluid.layers.resize_bilinear"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.resize_trilinear": {
    #     "update_to": "paddle.fluid.layers.resize_trilinear"
    # },
    # FlUID_WARNING 
    # "paddle.fluid.layers.resize_nearest": {
    #     "update_to": "paddle.fluid.layers.resize_nearest"
    # },
    # manual check
    "paddle.fluid.layers.gather": {
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
                "index",
                "index"
            ],
            [
                "overwrite",
                ""
            ]
        ],
        "args_warning": {"overwrite": "this args is deleted in this version."}
    },
    # manual check
    "paddle.fluid.layers.gather_nd": {
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
            ],
            [
                "index",
                "index"
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    "paddle.fluid.layers.scatter": {
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
            ],
            [
                "index",
                "index"
            ],
            [
                "updates",
                "updates"
            ],
            [
                "name",
                "name"
            ],
            [
                "overwrite",
                "overwrite"
            ]
        ]
    },
    "paddle.fluid.layers.scatter_nd_add": {
        "update_to": "paddle.scatter_nd_add"
    },
    "paddle.fluid.layers.scatter_nd": {
        "update_to": "paddle.scatter_nd"
    },
    "paddle.fluid.layers.random_crop": {
        "update_to": "paddle.nn.functional.random_crop"
    },
    "paddle.fluid.layers.mean_iou": {
        "update_to": "paddle.metric.mean_iou"
    },
    "paddle.fluid.layers.relu": {
        "update_to": "paddle.nn.functional.relu"
    },
    "paddle.fluid.layers.selu": {
        "update_to": "paddle.nn.functional.selu"
    },
    "paddle.fluid.layers.log": {
        "update_to": "paddle.log"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.crop": {
    #     "update_to": "paddle.fluid.layers.crop"
    # },
    "paddle.fluid.layers.crop_tensor": {
        "update_to": "paddle.crop_tensor"
    },
    "paddle.fluid.layers.elu": {
        "update_to": "paddle.nn.functional.elu"
    },
    # manual check
    "paddle.fluid.layers.relu6": {
        "update_to": "paddle.nn.functional.relu6",
        "args_list": [
            "x",
            "threshold",
            "name"
        ],
        "args_warning": {"threshold": "this args is deleted in this version."}
    },
    "paddle.fluid.layers.pow": {
        "update_to": "paddle.pow",
        "args_list": [
            "x",
            "factor",
            "name"
        ],
        "args_change": [
            [
                "x",
                "input"
            ],
            [
                "factor",
                "exponent"
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    "paddle.fluid.layers.stanh": {
        "update_to": "paddle.stanh"
    },
    "paddle.fluid.layers.hard_sigmoid": {
        "update_to": "paddle.nn.functional.hard_sigmoid"
    },
    "paddle.fluid.layers.swish": {
        "update_to": "paddle.nn.functional.swish"
    },
    # TODO transformer
    # "paddle.fluid.layers.prelu": {
    #     "args_list": [
    #         "x",
    #         "mode",
    #         "param_attr",
    #         "name"
    #     ],
    #     "args_change": [
    #         [
    #             "x",
    #             "x"
    #         ],
    #         [
    #             "mode",
    #             "mode"
    #         ],
    #         [
    #             "param_attr",
    #             "param_attr"
    #         ],
    #         [
    #             "name",
    #             "name"
    #         ]
    #     ]
    # },
    "paddle.fluid.layers.brelu": {
        "update_to": "paddle.nn.functional.brelu"
    },
    # manual check
    "paddle.fluid.layers.leaky_relu": {
        "update_to": "paddle.nn.functional.leaky_relu",
        "args_list": [
            "x",
            "alpha",
            "name"
        ],
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "alpha",
                "negative_slope"
            ],
            [
                "name",
                "name"
            ]
        ],
        "warning": "the alpha -> negative_slope and default 0.02 -> 0.01."
    },
    "paddle.fluid.layers.soft_relu": {
        "update_to": "paddle.nn.functional.soft_relu"
    },
    # TODO transformer
    # "paddle.fluid.layers.flatten": {
    #     "args_list": [
    #         "x",
    #         "axis",
    #         "name"
    #     ],
    #     "args_change": [
    #         [
    #             "x",
    #             "x"
    #         ],
    #         [
    #             "axis",
    #             "axis"
    #         ],
    #         [
    #             "name",
    #             "name"
    #         ]
    #     ]
    # },
    # manual check
    "paddle.fluid.layers.stack": {
        "update_to": "paddle.stack",
    },
    "paddle.fluid.layers.pad2d": {
        "update_to": "paddle.nn.functional.pad2d"
    },
    "paddle.fluid.layers.unstack": {
        "update_to": "paddle.unstack"
    },
    # manual check
    "paddle.fluid.layers.unique": {
        "update_to": "paddle.unique",
        "args_list": [
            "x",
            "dtype"
        ],
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "",
                "return_index",
                "True"    
            ],
            [
                "dtype",
                "dtype"
            ]
        ]
    },
    "paddle.fluid.layers.unique_with_counts": {
        "update_to": "paddle.unique_with_counts"
    },
    # TODO transformer
    # "paddle.fluid.layers.expand": {
    #     "args_list": [
    #         "x",
    #         "expand_times",
    #         "name"
    #     ],
    #     "args_change": [
    #         [
    #             "x",
    #             "x"
    #         ],
    #         [
    #             "expand_times",
    #             "expand_times"
    #         ],
    #         [
    #             "name",
    #             "name"
    #         ]
    #     ]
    # },
    # manual check
    "paddle.fluid.layers.expand_as": {
        "update_to": "paddle.expand_as",
        "args_list": [
            "x",
            "target_tensor",
            "name"
        ],
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "target_tensor",
                "y"
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    "paddle.fluid.layers.scale": {
        "update_to": "paddle.scale"
    },
    # TODO act transformer manual check
    "paddle.fluid.layers.elementwise_max": {
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
                "x",
                "x"
            ],
            [
                "y",
                "y"
            ],
            [
                "axis",
                "axis"
            ],
            [
                "act",
                ""
            ],
            [
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "act": "act is deleted in paddle.maximum"
        }
    },
    # TODO act transformer manual check
    "paddle.fluid.layers.elementwise_min": {
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
                "x",
                "x"
            ],
            [
                "y",
                "y"
            ],
            [
                "axis",
                "axis"
            ],
            [
                "act",
                ""
            ],
            [
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "act": "act is deleted in paddle.maximum"
        }
    },
    "paddle.fluid.layers.elementwise_pow": {
        "update_to": "paddle.elementwise_pow"
    },
    # TODO transformer
    "paddle.fluid.layers.elementwise_mod": {
        "update_to": "paddle.remainder",
        "args_list": [
            "x",
            "y",
            "axis",
            "act",
            "name"
        ],
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "y",
                "y"
            ],
            [
                "axis",
                ""
            ],
            [
                "act",
                ""
            ],
            [
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "axis": "axis is deleted in paddle.remainder",
            "act": "act is deleted in paddle.remainder"
        }
    },
    # TODO transformer manual check
    "paddle.fluid.layers.elementwise_floordiv": {
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
                "x",
                "x"
            ],
            [
                "y",
                "y"
            ],
            [
                "axis",
                ""
            ],
            [
                "act",
                ""
            ],
            [
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "axis": "axis is deleted in paddle.floor_divide",
            "act": "act is deleted in paddle.floor_divide"
        }
    },
    # TODO transformer manual check
    "paddle.fluid.layers.elementwise_div": {
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
                "x",
                "x"
            ],
            [
                "y",
                "y"
            ],
            [
                "axis",
                ""
            ],
            [
                "act",
                ""
            ],
            [
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "axis": "axis is deleted in paddle.divide",
            "act": "act is deleted in paddle.divide"
        }
    },
    # TODO transformer manual check
    "paddle.fluid.layers.elementwise_mul": {
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
                "x",
                "x"
            ],
            [
                "y",
                "y"
            ],
            [
                "axis",
                ""
            ],
            [
                "act",
                ""
            ],
            [
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "axis": "axis is deleted in paddle.multiply",
            "act": "act is deleted in paddle.multiply"
        }
    },
    # TODO transformer manual check
    "paddle.fluid.layers.elementwise_add": {
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
                "x",
                "x"
            ],
            [
                "y",
                "y"
            ],
            [
                "axis",
                ""
            ],
            [
                "act",
                ""
            ],
            [
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "axis": "axis is deleted in paddle.add",
            "act": "act is deleted in paddle.add"
        }
    },
    # TODO 
    "paddle.fluid.layers.elementwise_sub": {
        "warning": "this api is deprecated in paddle2.0"
    },
    # manual check
    "paddle.fluid.layers.uniform_random_batch_size_like": {
        "warning": "uniform_random_batch_size_like is deprecated, please see paddle.uniform."
    },
    # manual check
    "paddle.fluid.layers.gaussian_random": {
        "warning": "gaussian_random is deprecated, please see paddle.normal .",
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.sampling_id": {
    #     "update_to": "paddle.fluid.layers.sampling_id"
    # },
    "paddle.fluid.layers.gaussian_random_batch_size_like": {
        "warning": "gaussian_random_batch_size_like is deprecated, please see paddle.normal .",
    },
    "paddle.fluid.layers.slice": {
        "update_to": "paddle.slice"
    },
    "paddle.fluid.layers.strided_slice": {
        "update_to": "paddle.strided_slice"
    },
    "paddle.fluid.layers.shape": {
        "update_to": "paddle.shape"
    },
    "paddle.fluid.layers.rank": {
        "update_to": "paddle.rank"
    },
    "paddle.fluid.layers.size": {
        "update_to": "paddle.numel"
    },
    "paddle.fluid.layers.logical_and": {
        "update_to": "paddle.logical_and"
    },
    "paddle.fluid.layers.logical_or": {
        "update_to": "paddle.logical_or"
    },
    "paddle.fluid.layers.logical_xor": {
        "update_to": "paddle.logical_xor"
    },
    "paddle.fluid.layers.logical_not": {
        "update_to": "paddle.logical_not"
    },
    # manual check
    "paddle.fluid.layers.clip": {
        "update_to": "paddle.clip",
    },
    "paddle.fluid.layers.clip_by_norm": {
        "update_to": "paddle.nn.clip_by_norm"
    },
    # manual check
    "paddle.fluid.layers.mean": {
        "update_to": "paddle.mean"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.mul": {
    #     "update_to": "paddle.fluid.layers.mul"
    # },
    "paddle.fluid.layers.maxout": {
        "update_to": "paddle.nn.functional.maxout"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.space_to_depth": {
    #     "update_to": "paddle.fluid.layers.space_to_depth"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.affine_grid": {
    #     "update_to": "paddle.fluid.layers.affine_grid"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.affine_channel": {
    #     "update_to": "paddle.fluid.layers.affine_channel"
    # },
    "paddle.fluid.layers.similarity_focus": {
        "update_to": "paddle.nn.functional.similarity_focus"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.hash": {
    #     "update_to": "paddle.fluid.layers.hash"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.grid_sampler": {
    #     "update_to": "paddle.fluid.layers.grid_sampler"
    # },
    "paddle.fluid.layers.log_loss": {
        "update_to": "paddle.nn.functional.log_loss"
    },
    "paddle.fluid.layers.add_position_encoding": {
        "update_to": "paddle.nn.functional.add_position_encoding"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.bilinear_tensor_product": {
    #     "update_to": "paddle.fluid.layers.bilinear_tensor_product"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.merge_selected_rows": {
    #     "update_to": "paddle.fluid.layers.merge_selected_rows"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.get_tensor_from_selected_rows": {
    #     "update_to": "paddle.fluid.layers.get_tensor_from_selected_rows"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.shuffle_channel": {
    #     "update_to": "paddle.fluid.layers.shuffle_channel"
    # },
    "paddle.fluid.layers.temporal_shift": {
        "update_to": "paddle.nn.functional.temporal_shift"
    },
    "paddle.fluid.layers.py_func": {
        "update_to": "paddle.static.py_func"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.psroi_pool": {
    #     "update_to": "paddle.fluid.layers.psroi_pool"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.prroi_pool": {
    #     "update_to": "paddle.fluid.layers.prroi_pool"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.pixel_shuffle": {
    #     "update_to": "paddle.fluid.layers.pixel_shuffle"
    # },
    # "paddle.fluid.layers.fsp_matrix": {
    #     "update_to": "paddle.fluid.layers.fsp_matrix"
    # },
    "paddle.fluid.layers.continuous_value_model": {
        "update_to": "paddle.nn.functional.continuous_value_model"
    },
    # TODO transformer
    # "paddle.fluid.layers.where": {
    #     "args_list": [
    #         "condition"
    #     ],
    #     "args_change": [
    #         [
    #             "condition",
    #             "condition"
    #         ]
    #     ]
    # },
    # manual check
    "paddle.fluid.layers.sign": {
        "update_to": "paddle.sign"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.deformable_conv": {
    #     "update_to": "paddle.fluid.layers.deformable_conv"
    # },
    "paddle.fluid.layers.unfold": {
        "update_to": "paddle.nn.functional.unfold"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.deformable_roi_pooling": {
    #     "update_to": "paddle.fluid.layers.deformable_roi_pooling"
    # },
    "paddle.fluid.layers.filter_by_instag": {
        "update_to": "paddle.nn.functional.filter_by_instag"
    },
    "paddle.fluid.layers.shard_index": {
        "update_to": "paddle.shard_index"
    },
    "paddle.fluid.layers.hard_swish": {
        "update_to": "paddle.nn.functional.hard_swish"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.mish": {
    #     "update_to": "paddle.fluid.layers.mish"
    # },
    "paddle.fluid.layers.gather_tree": {
        "update_to": "paddle.nn.gather_tree"
    },
    # manual check
    "paddle.fluid.layers.uniform_random": {
        "update_to": "paddle.uniform"
    },
    # manual check
    "paddle.fluid.layers.randint": {
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
                "low",
                "low"
            ],
            [
                "high",
                "high"
            ],
            [
                "shape",
                "shape"
            ],
            [
                "out",
                ""
            ],
            [
                "dtype",
                "dtype"
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
            ],
            [
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "out": "this args is deleted in this version",
            "device": "this args is deleted in this version",
            "stop_gradient": "this args is deleted in this version",
            "seed": "this args is deleted in this version"
        }
    },
    # manual check
    "paddle.fluid.layers.randn": {
        "update_to": "paddle.randn",
        "args_list": [
            "shape",
            "out",
            "dtype",
            "device",
            "stop_gradient",
            "name"
        ],
        "args_change": [
            [
                "shape",
                "shape"
            ],
            [
                "out",
                ""
            ],
            [
                "dtype",
                "dtype"
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
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "out": "this args is deleted in this version",
            "device": "this args is deleted in this version",
            "stop_gradient": "this args is deleted in this version"
        }
    },
    # manual check
    "paddle.fluid.layers.randperm": {
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
                "n",
                "n"
            ],
            [
                "out",
                ""
            ],
            [
                "dtype",
                "dtype"
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
            ],
            [
                "",
                "name",
                "None"
            ]
        ]
    },
    "paddle.fluid.layers.allclose": {
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
            ],
            [
                "rtol",
                "rtol"
            ],
            [
                "atol",
                "atol"
            ],
            [
                "equal_nan",
                "equal_nan"
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    # TODO
    "paddle.fluid.layers.elementwise_equal": {
        "warning": "this api 2.0 is not approved."
    },
    "paddle.fluid.layers.flip": {
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
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    # manual check
    "paddle.fluid.layers.roll": {
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
                "shifts",
                "shifts"
            ],
            [
                "dims",
                "axis"
            ],
            [
                "",
                "name",
                "None"
            ]
        ]
    },
    # manual check
    "paddle.fluid.layers.log_softmax": {
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
            ],
            [
                "axis",
                "axis"
            ],
            [
                "dtype",
                "dtype"
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    "paddle.fluid.layers.index_sample": {
        "update_to": "paddle.index_sample"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.read_file": {
    #     "update_to": "paddle.fluid.layers.read_file"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.double_buffer": {
    #     "update_to": "paddle.fluid.layers.double_buffer"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.py_reader": {
    #     "update_to": "paddle.fluid.layers.py_reader"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.create_py_reader_by_data": {
    #     "update_to": "paddle.fluid.layers.create_py_reader_by_data"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.create_tensor": {
    #     "update_to": "paddle.fluid.layers.create_tensor"
    # },
    "paddle.fluid.layers.create_parameter": {
        "update_to": "paddle.create_parameter"
    },
    "paddle.fluid.layers.create_global_var": {
        "update_to": "paddle.create_global_var"
    },
    "paddle.fluid.layers.cast": {
        "update_to": "paddle.cast"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.tensor_array_to_tensor": {
    #     "update_to": "paddle.fluid.layers.tensor_array_to_tensor"
    # },
    "paddle.fluid.layers.concat": {
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
            ],
            [
                "axis",
                "axis"
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    "paddle.fluid.layers.sums": {
        "update_to": "paddle.sums"
    },
    "paddle.fluid.layers.assign": {
        "update_to": "paddle.nn.functional.assign"
    },
    "paddle.fluid.layers.fill_constant_batch_size_like": {
        "warning": "this api in paddle2.0 is paddle.fill_constant."
    },
    # manual check
    "paddle.fluid.layers.fill_constant": {
        "update_to": "paddle.fill_constant",
    },
    # manual check
    "paddle.fluid.layers.argmin": {
        "update_to": "paddle.argmin",
    },
    # manual check
    "paddle.fluid.layers.argmax": {
       "update_to": "paddle.argmax",
    },
    # manual check
    "paddle.fluid.layers.argsort": {
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
            ],
            [
                "axis",
                "axis"
            ],
            [
                "descending",
                "descending"
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    # manual check
    "paddle.fluid.layers.ones": {
        "update_to": "paddle.ones",
        "args_list": [
            "shape",
            "dtype",
            "force_cpu"
        ],
        "args_change": [
            [
                "shape",
                "shape"
            ],
            [
                "dtype",
                "dtype"
            ],
            [
                "force_cpu",
                ""
            ],
            [
                "",
                "name",
                "None",
            ]
        ],
        "args_warning": {
            "force_cpu": "this arg is deleted in this version"
        }
    },
    # manual check
    "paddle.fluid.layers.zeros": {
        "update_to": "paddle.zeros",
        "args_list": [
            "shape",
            "dtype",
            "force_cpu"
        ],
        "args_change": [
            [
                "shape",
                "shape"
            ],
            [
                "dtype",
                "dtype"
            ],
            [
                "force_cpu",
                ""
            ],
            [
                "",
                "name",
                "None"
            ]
        ],
        "args_warning": {
            "force_cpu": "this arg is deleted in this version"
        }
    },
    # manual check
    "paddle.fluid.layers.reverse": {
        "update_to": "paddle.reverse"
    },
    "paddle.fluid.layers.has_inf": {
        "update_to": "paddle.has_inf"
    },
    "paddle.fluid.layers.has_nan": {
        "update_to": "paddle.has_nan"
    },
    "paddle.fluid.layers.isfinite": {
        "update_to": "paddle.isfinite"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.range": {
    #     "update_to": "paddle.fluid.layers.range",
    # },
    "paddle.fluid.layers.linspace": {
        "update_to": "paddle.linspace"
    },
    "paddle.fluid.layers.full_like": {
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
                "fill_value",
                "fill_value"
            ],
            [
                "out",
                ""
            ],
            [
                "dtype",
                "dtype"
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
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "out": "this args is deleted in this version",
            "device": "this args is deleted in this version",
            "stop_gradient": "this args is deleted in this version"
        }
    },
    # manual check
    "paddle.fluid.layers.zeros_like": {
        "update_to": "paddle.zeros_like",
        "args_list": [
            "x",
            "out"
        ],
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "out",
                ""
            ],
            [
                "",
                "dtype",
                "None"
            ],
            [
                "",
                "name",
                "None"
            ]
        ],
        "args_warning": {
            "out": "this args is deleted in this version"
        }
    },
    # manual check
    "paddle.fluid.layers.ones_like": {
        "update_to": "paddle.ones_like",
        "args_list": [
            "x",
            "out"
        ],
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "out",
                ""
            ],
            [
                "",
                "dtype",
                "None"
            ],
            [
                "",
                "name",
                "None"
            ]
        ],
        "args_warning": {
            "out": "this args is deleted in this version"
        }
    },
    # TODO
    "paddle.fluid.layers.diag": {
        "warning": "this api is changed a lot, please use paddle.diag"
    },
    # manual check
    "paddle.fluid.layers.eye": {
        "update_to": "paddle.eye",
        "args_list": [
            "num_rows",
            "num_columns",
            "batch_shape",
            "dtype"
        ],
        "args_change": [
            [
                "num_rows",
                "num_rows"
            ],
            [
                "num_columns",
                "num_columns"
            ],
            [
                "batch_shape",
                ""
            ],
            [
                "dtype",
                "dtype"
            ],
            [
                "",
                "name",
                "None"
            ]
        ],
        "args_warning": {
            "batch_shape": "this args is deleted"
        }
    },
    # manual check
    "paddle.fluid.layers.arange": {
        "update_to": "paddle.arange"
    },
    "paddle.fluid.layers.full": {
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
                "shape",
                "shape"
            ],
            [
                "fill_value",
                "fill_value"
            ],
            [
                "out",
                ""
            ],
            [
                "dtype",
                "dtype"
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
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "out": "this args is deleted in this version",
            "device": "this args is deleted in this version",
            "stop_gradient": "this args is deleted in this version"
        }
    },
    # manual check
    "paddle.fluid.layers.tril": {
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
            ],
            [
                "diagonal",
                "diagonal"
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    # manual check
    "paddle.fluid.layers.triu": {
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
            ],
            [
                "diagonal",
                "diagonal"
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.While": {
    #     "update_to": "paddle.fluid.layers.While"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.Switch": {
    #     "update_to": "paddle.fluid.layers.Switch"
    # },
    "paddle.fluid.layers.increment": {
        "update_to": "paddle.increment"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.array_write": {
    #     "update_to": "paddle.fluid.layers.array_write"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.create_array": {
    #     "update_to": "paddle.fluid.layers.create_array"
    # },
    # manual check
    "paddle.fluid.layers.less_than": {
        "update_to": "paddle.less_than",
        "args_list": [
            "x",
            "y",
            "force_cpu",
            "cond"
        ],
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "y",
                "y"
            ],
            [
                "force_cpu",
                ""
            ],
            [
                "cond",
                ""
            ],
            [
                "",
                "name",
                "None"
            ]
        ],
        "args_warning": {
            "force_cpu": "this args is deleted in this version",
            "cond": "this args is deleted in this version"
        }
    },
    # manual check
    "paddle.fluid.layers.less_equal": {
        "update_to": "paddle.less_equal",
        "args_list": [
            "x",
            "y",
            "cond"
        ],
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "y",
                "y"
            ],
            [
                "cond",
                ""
            ],
            [
                "",
                "name",
                "None"
            ]
        ],
        "args_warning": {
            "cond": "this args is deleted in this version"
        }
    },
    # manual check
    "paddle.fluid.layers.greater_than": {
        "update_to": "paddle.greater_than",
        "args_list": [
            "x",
            "y",
            "cond"
        ],
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "y",
                "y"
            ],
            [
                "cond",
                ""
            ],
            [
                "",
                "name",
                "None"
            ]
        ],
        "args_warning": {
            "cond": "this args is deleted in this version"
        }
    },
    # manual check
    "paddle.fluid.layers.greater_equal": {
        "update_to": "paddle.greater_equal",
        "args_list": [
            "x",
            "y",
            "cond"
        ],
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "y",
                "y"
            ],
            [
                "cond",
                ""
            ],
            [
                "",
                "name",
                "None"
            ]
        ],
        "args_warning": {
            "cond": "this args is deleted in this version"
        }
    },
    # manual check
    "paddle.fluid.layers.equal": {
        "update_to": "paddle.equal",
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "y",
                "y"
            ],
            [
                "cond",
                ""
            ],
            [
                "",
                "name",
                "None"
            ]
        ],
        "args_warning": {
            "cond": "this args is deleted in this version"
        }
    },
    # manual check
    "paddle.fluid.layers.not_equal": {
        "update_to": "paddle.not_equal",
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "y",
                "y"
            ],
            [
                "cond",
                ""
            ],
            [
                "",
                "name",
                "None"
            ]
        ],
        "args_warning": {
            "cond": "this args is deleted in this version"
        }
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.array_read": {
    #     "update_to": "paddle.fluid.layers.array_read"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.array_length": {
    #     "update_to": "paddle.fluid.layers.array_length"
    # },
    "paddle.fluid.layers.cond": {
        "update_to": "paddle.nn.cond"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.IfElse": {
    #     "update_to": "paddle.fluid.layers.IfElse"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.DynamicRNN": {
    #     "update_to": "paddle.fluid.layers.DynamicRNN"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.StaticRNN": {
    #     "update_to": "paddle.fluid.layers.StaticRNN"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.reorder_lod_tensor_by_rank": {
    #     "update_to": "paddle.fluid.layers.reorder_lod_tensor_by_rank"
    # },
    "paddle.fluid.layers.Print": {
        "update_to": "paddle.static.Print"
    },
    "paddle.fluid.layers.is_empty": {
        "update_to": "paddle.is_empty"
    },
    "paddle.fluid.layers.case": {
        "update_to": "paddle.nn.case"
    },
    "paddle.fluid.layers.switch_case": {
        "update_to": "paddle.nn.switch_case"
    },
    "paddle.fluid.layers.while_loop": {
        "update_to": "paddle.nn.while_loop"
    },
    "paddle.fluid.layers.sigmoid": {
        "update_to": "paddle.nn.functional.sigmoid"
    },
    "paddle.fluid.layers.logsigmoid": {
        "update_to": "paddle.nn.functional.logsigmoid"
    },
    "paddle.fluid.layers.exp": {
        "update_to": "paddle.exp"
    },
    "paddle.fluid.layers.tanh": {
        "update_to": "paddle.tanh"
    },
    "paddle.fluid.layers.atan": {
        "update_to": "paddle.atan"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.tanh_shrink": {
    #     "update_to": "paddle.fluid.layers.tanh_shrink"
    # },
    "paddle.fluid.layers.sqrt": {
        "update_to": "paddle.sqrt"
    },
    "paddle.fluid.layers.rsqrt": {
        "update_to": "paddle.rsqrt"
    },
    "paddle.fluid.layers.abs": {
        "update_to": "paddle.abs"
    },
    "paddle.fluid.layers.ceil": {
        "update_to": "paddle.ceil"
    },
    "paddle.fluid.layers.floor": {
        "update_to": "paddle.floor"
    },
    "paddle.fluid.layers.cos": {
        "update_to": "paddle.cos"
    },
    "paddle.fluid.layers.acos": {
        "update_to": "paddle.acos"
    },
    "paddle.fluid.layers.asin": {
        "update_to": "paddle.asin"
    },
    "paddle.fluid.layers.sin": {
        "update_to": "paddle.sin"
    },
    "paddle.fluid.layers.reciprocal": {
        "update_to": "paddle.reciprocal"
    },
    "paddle.fluid.layers.square": {
        "update_to": "paddle.square"
    },
    # TODO threshold=20?
    "paddle.fluid.layers.softplus": {
        "update_to": "paddle.softplus"
    },
    "paddle.fluid.layers.softsign": {
        "update_to": "paddle.nn.functional.softsign"
    },
    # manual check
    "paddle.fluid.layers.softshrink": {
        "update_to": "paddle.softshrink",
        "args_list": [
            "x",
            "alpha"
        ],
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "alpha",
                "threshold"
            ],
            [
                "",
                "name",
                "None"
            ]
        ]
    },
    # manual check
    "paddle.fluid.layers.hard_shrink": {
        "update_to": "paddle.nn.functional.hardshrink"
    },
    # TODO transformer paddle.cumsum
    # "paddle.fluid.layers.cumsum": {
    #     "args_list": [
    #         "x",
    #         "axis",
    #         "exclusive",
    #         "reverse"
    #     ],
    #     "args_change": [
    #         [
    #             "x",
    #             "x"
    #         ],
    #         [
    #             "axis",
    #             "axis"
    #         ],
    #         [
    #             "exclusive",
    #             "exclusive"
    #         ],
    #         [
    #             "reverse",
    #             "reverse"
    #         ]
    #     ]
    # },
    "paddle.fluid.layers.thresholded_relu": {
        "update_to": "paddle.nn.functional.thresholded_relu"
    },
    # manual check
    "paddle.fluid.layers.gelu": {
        "update_to": "paddle.nn.functional.gelu"
    },
    # manual check
    "paddle.fluid.layers.erf": {
        "update_to": "paddle.erf"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.prior_box": {
    #     "update_to": "paddle.fluid.layers.prior_box"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.density_prior_box": {
    #     "update_to": "paddle.fluid.layers.density_prior_box"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.multi_box_head": {
    #     "update_to": "paddle.fluid.layers.multi_box_head"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.bipartite_match": {
    #     "update_to": "paddle.fluid.layers.bipartite_match"
    # },
    "paddle.fluid.layers.target_assign": {
        "update_to": "paddle.nn.functional.target_assign"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.detection_output": {
    #     "update_to": "paddle.fluid.layers.detection_output"
    # },
    "paddle.fluid.layers.ssd_loss": {
        "update_to": "paddle.nn.functional.ssd_loss"
    },
    "paddle.fluid.layers.rpn_target_assign": {
        "update_to": "paddle.nn.functional.rpn_target_assign"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.retinanet_target_assign": {
    #     "update_to": "paddle.fluid.layers.retinanet_target_assign"
    # },
    "paddle.fluid.layers.sigmoid_focal_loss": {
        "update_to": "paddle.nn.functional.sigmoid_focal_loss"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.anchor_generator": {
    #     "update_to": "paddle.fluid.layers.anchor_generator"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.roi_perspective_transform": {
    #     "update_to": "paddle.fluid.layers.roi_perspective_transform"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.generate_proposal_labels": {
    #     "update_to": "paddle.fluid.layers.generate_proposal_labels"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.generate_proposals": {
    #     "update_to": "paddle.fluid.layers.generate_proposals"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.generate_mask_labels": {
    #     "update_to": "paddle.fluid.layers.generate_mask_labels"
    # },
    "paddle.fluid.layers.iou_similarity": {
        "update_to": "paddle.nn.functional.iou_similarity"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.box_coder": {
    #     "update_to": "paddle.fluid.layers.box_coder"
    # },
    "paddle.fluid.layers.polygon_box_transform": {
        "update_to": "paddle.nn.functional.polygon_box_transform"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.yolov3_loss": {
    #     "update_to": "paddle.fluid.layers.yolov3_loss"
    # },
    # "paddle.fluid.layers.yolo_box": {
    #     "update_to": "paddle.fluid.layers.yolo_box"
    # },
    # "paddle.fluid.layers.box_clip": {
    #     "update_to": "paddle.fluid.layers.box_clip"
    # },
    "paddle.fluid.layers.multiclass_nms": {
        "update_to": "paddle.nn.functional.multiclass_nms"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.locality_aware_nms": {
    #     "update_to": "paddle.fluid.layers.locality_aware_nms"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.matrix_nms": {
    #     "update_to": "paddle.fluid.layers.matrix_nms"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.retinanet_detection_output": {
    #     "update_to": "paddle.fluid.layers.retinanet_detection_output"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.distribute_fpn_proposals": {
    #     "update_to": "paddle.fluid.layers.distribute_fpn_proposals"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.box_decoder_and_assign": {
    #     "update_to": "paddle.fluid.layers.box_decoder_and_assign"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.collect_fpn_proposals": {
    #     "update_to": "paddle.fluid.layers.collect_fpn_proposals"
    # },
    "paddle.fluid.layers.accuracy": {
        "update_to": "paddle.metric.accuracy"
    },
    "paddle.fluid.layers.auc": {
        "update_to": "paddle.metric.auc"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.exponential_decay": {
    #     "update_to": "paddle.fluid.layers.exponential_decay"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.natural_exp_decay": {
    #     "update_to": "paddle.fluid.layers.natural_exp_decay"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.inverse_time_decay": {
    #     "update_to": "paddle.fluid.layers.inverse_time_decay"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.polynomial_decay": {
    #     "update_to": "paddle.fluid.layers.polynomial_decay"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.piecewise_decay": {
    #     "update_to": "paddle.fluid.layers.piecewise_decay"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.noam_decay": {
    #     "update_to": "paddle.fluid.layers.noam_decay"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.cosine_decay": {
    #     "update_to": "paddle.fluid.layers.cosine_decay"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.linear_lr_warmup": {
    #     "update_to": "paddle.fluid.layers.linear_lr_warmup"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.Categorical": {
    #     "update_to": "paddle.fluid.layers.Categorical"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.MultivariateNormalDiag": {
    #     "update_to": "paddle.fluid.layers.MultivariateNormalDiag"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_conv": {
    #     "update_to": "paddle.fluid.layers.sequence_conv"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_softmax": {
    #     "update_to": "paddle.fluid.layers.sequence_softmax"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_pool": {
    #     "update_to": "paddle.fluid.layers.sequence_pool"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_concat": {
    #     "update_to": "paddle.fluid.layers.sequence_concat"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_first_step": {
    #     "update_to": "paddle.fluid.layers.sequence_first_step"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_last_step": {
    #     "update_to": "paddle.fluid.layers.sequence_last_step"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_slice": {
    #     "update_to": "paddle.fluid.layers.sequence_slice"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_expand": {
    #     "update_to": "paddle.fluid.layers.sequence_expand"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_expand_as": {
    #     "update_to": "paddle.fluid.layers.sequence_expand_as"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_pad": {
    #     "update_to": "paddle.fluid.layers.sequence_pad"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_unpad": {
    #     "update_to": "paddle.fluid.layers.sequence_unpad"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_reshape": {
    #     "update_to": "paddle.fluid.layers.sequence_reshape"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_scatter": {
    #     "update_to": "paddle.fluid.layers.sequence_scatter"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_enumerate": {
    #     "update_to": "paddle.fluid.layers.sequence_enumerate"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_mask": {
    #     "update_to": "paddle.fluid.layers.sequence_mask"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.sequence_reverse": {
    #     "update_to": "paddle.fluid.layers.sequence_reverse"
    # },
    "paddle.fluid.layers.center_loss": {
        "update_to": "paddle.nn.functional.center_loss"
    },
    "paddle.fluid.layers.bpr_loss": {
        "update_to": "paddle.nn.functional.bpr_loss"
    },
    # TODO transformer
    # "paddle.fluid.layers.cross_entropy": {
    #     "args_list": [
    #         "input",
    #         "label",
    #         "soft_label",
    #         "ignore_index"
    #     ],
    #     "args_change": [
    #         [
    #             "input",
    #             "input"
    #         ],
    #         [
    #             "label",
    #             "label"
    #         ],
    #         [
    #             "soft_label",
    #             "soft_label"
    #         ],
    #         [
    #             "ignore_index",
    #             "ignore_index"
    #         ]
    #     ]
    # },
    "paddle.fluid.layers.square_error_cost": {
        "update_to": "paddle.nn.functional.square_error_cost"
    },
    "paddle.fluid.layers.edit_distance": {
        "update_to": "paddle.nn.functional.edit_distance"
    },
    "paddle.fluid.layers.warpctc": {
        "update_to": "paddle.nn.functional.warpctc"
    },
    # TODO FlUID_WARNING
    # "paddle.fluid.layers.nce": {
    #     "update_to": "paddle.fluid.layers.nce"
    # },
    # TODO transformer paddle.nn.functional.hsigmoid
    # "paddle.fluid.layers.hsigmoid": {
    #     "args_list": [
    #         "input",
    #         "label",
    #         "num_classes",
    #         "param_attr",
    #         "bias_attr",
    #         "name",
    #         "path_table",
    #         "path_code",
    #         "is_custom",
    #         "is_sparse"
    #     ],
    #     "args_change": [
    #         [
    #             "input",
    #             "input"
    #         ],
    #         [
    #             "label",
    #             "label"
    #         ],
    #         [
    #             "num_classes",
    #             "num_classes"
    #         ],
    #         [
    #             "param_attr",
    #             "param_attr"
    #         ],
    #         [
    #             "bias_attr",
    #             "bias_attr"
    #         ],
    #         [
    #             "name",
    #             "name"
    #         ],
    #         [
    #             "path_table",
    #             "path_table"
    #         ],
    #         [
    #             "path_code",
    #             "path_code"
    #         ],
    #         [
    #             "is_custom",
    #             "is_custom"
    #         ],
    #         [
    #             "is_sparse",
    #             "is_sparse"
    #         ]
    #     ]
    # },
    "paddle.fluid.layers.sampled_softmax_with_cross_entropy": {
        "update_to": "paddle.nn.functional.sampled_softmax_with_cross_entropy"
    },
    "paddle.fluid.layers.softmax_with_cross_entropy": {
        "update_to": "paddle.nn.functional.softmax_with_cross_entropy"
    },
    "paddle.fluid.layers.rank_loss": {
        "update_to": "paddle.nn.functional.rank_loss"
    },
    # FlUID_WARNING
    # "paddle.fluid.layers.margin_rank_loss": {
    #     "update_to": "paddle.fluid.layers.margin_rank_loss"
    # },
    "paddle.fluid.layers.sigmoid_cross_entropy_with_logits": {
        "update_to": "paddle.nn.functional.sigmoid_cross_entropy_with_logits"
    },
    "paddle.fluid.layers.teacher_student_sigmoid_loss": {
        "update_to": "paddle.nn.functional.teacher_student_sigmoid_loss"
    },
    "paddle.fluid.layers.huber_loss": {
        "update_to": "paddle.nn.functional.huber_loss"
    },
    # manual check
    "paddle.fluid.layers.kldiv_loss": {
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
            ],
            [
                "reduction",
                "reduction"
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    "paddle.fluid.layers.npair_loss": {
        "update_to": "paddle.nn.functional.npair_loss"
    },
    "paddle.fluid.layers.mse_loss": {
        "update_to": "paddle.nn.functional.mse_loss"
    },
    # TODO define RNNCell
    # "paddle.fluid.layers.RNNCell": {
    #     "args_list": [
    #         ""
    #     ],
    #     "args_change": [
    #         [
    #             "",
    #             ""
    #         ]
    #     ]
    # },
    # TODO define GRUCell
    # "paddle.fluid.layers.GRUCell": {
    #     "args_list": [
    #         "hidden_size",
    #         "param_attr",
    #         "bias_attr",
    #         "gate_activation",
    #         "activation",
    #         "dtype",
    #         "name"
    #     ],
    #     "args_change": [
    #         [
    #             "hidden_size",
    #             "hidden_size"
    #         ],
    #         [
    #             "param_attr",
    #             "param_attr"
    #         ],
    #         [
    #             "bias_attr",
    #             "bias_attr"
    #         ],
    #         [
    #             "gate_activation",
    #             "gate_activation"
    #         ],
    #         [
    #             "activation",
    #             "activation"
    #         ],
    #         [
    #             "dtype",
    #             "dtype"
    #         ],
    #         [
    #             "name",
    #             "name"
    #         ]
    #     ]
    # }, 
    # TODO LSTMCell
    # "paddle.fluid.layers.LSTMCell": {
    #     "args_list": [
    #         "hidden_size",
    #         "param_attr",
    #         "bias_attr",
    #         "gate_activation",
    #         "activation",
    #         "forget_bias",
    #         "dtype",
    #         "name"
    #     ],
    #     "args_change": [
    #         [
    #             "hidden_size",
    #             "hidden_size"
    #         ],
    #         [
    #             "param_attr",
    #             "param_attr"
    #         ],
    #         [
    #             "bias_attr",
    #             "bias_attr"
    #         ],
    #         [
    #             "gate_activation",
    #             "gate_activation"
    #         ],
    #         [
    #             "activation",
    #             "activation"
    #         ],
    #         [
    #             "forget_bias",
    #             "forget_bias"
    #         ],
    #         [
    #             "dtype",
    #             "dtype"
    #         ],
    #         [
    #             "name",
    #             "name"
    #         ]
    #     ]
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.Decoder": {
    #     "update_to": "paddle.fluid.layers.Decoder"
    # },
    # INCUBATE_WARNING
    # "paddle.fluid.layers.BeamSearchDecoder": {
    #     "update_to": "paddle.incubate.hapi.text.BeamSearchDecoder"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.rnn": {
    #     "update_to": "paddle.fluid.layers.rnn"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.dynamic_decode": {
    #     "update_to": "paddle.fluid.layers.dynamic_decode"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.DecodeHelper": {
    #     "update_to": "paddle.fluid.layers.DecodeHelper"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.TrainingHelper": {
    #     "update_to": "paddle.fluid.layers.TrainingHelper"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.GreedyEmbeddingHelper": {
    #     "update_to": "paddle.fluid.layers.GreedyEmbeddingHelper"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.SampleEmbeddingHelper": {
    #     "update_to": "paddle.fluid.layers.SampleEmbeddingHelper"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.BasicDecoder": {
    #     "update_to": "paddle.fluid.layers.BasicDecoder"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.dynamic_lstm": {
    #     "update_to": "paddle.fluid.layers.dynamic_lstm"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.dynamic_lstmp": {
    #     "update_to": "paddle.fluid.layers.dynamic_lstmp"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.dynamic_gru": {
    #     "update_to": "paddle.fluid.layers.dynamic_gru"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.gru_unit": {
    #     "update_to": "paddle.fluid.layers.gru_unit"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.lstm_unit": {
    #     "update_to": "paddle.fluid.layers.lstm_unit"
    # },
    # FlUID_WARNING
    # "paddle.fluid.layers.lstm": {
    #     "update_to": "paddle.fluid.layers.lstm"
    # },
    "paddle.fluid.layers.beam_search": {
        "update_to": "paddle.nn.beam_search"
    },
    "paddle.fluid.layers.beam_search_decode": {
        "update_to": "paddle.nn.beam_search_decode"
    },
    "paddle.fluid.dygraph.Layer": {
        "update_to": "paddle.nn.Layer"
    },
    # manual check
    "paddle.fluid.dygraph.no_grad": {
        "update_to": "paddle.no_grad",
    },
    "paddle.fluid.dygraph.grad": {
        "update_to": "paddle.grad"
    },
    "paddle.fluid.dygraph.enable_dygraph": {
        "alias": [
            "paddle.fluid.enable_dygraph"
        ],
        "update_to": "paddle.disable_static"
    },
    "paddle.fluid.dygraph.disable_dygraph": {
        "alias": [
            "paddle.fluid.disable_dygraph"
        ],
        "update_to": "paddle.enable_static"
    },
    "paddle.fluid.dygraph.enable_imperative": {
        "alias": [
            "paddle.fluid.enable_imperative"
        ],
        "update_to": "paddle.disable_static"
    },
    "paddle.fluid.dygraph.disable_imperative": {
        "alias": [
            "paddle.fluid.disable_imperative"
        ],
        "update_to": "paddle.enable_static"
    },
    # FlUID_WARNING
    # "paddle.fluid.dygraph.enabled": {
    #     "update_to": "paddle.fluid.dygraph.enabled"
    # },
    "paddle.fluid.dygraph.to_variable": {
        "alias": ["paddle.fluid.dygraph.base.to_variable"],
        "update_to": "paddle.to_tensor",
        "args_list": [
            "value",
            "name",
            "zero_copy"
        ],
        "args_change": [
            [
                "value",
                "data"
            ],
            [
                "name",
                ""
            ],
            [
                "zero_copy",
                ""
            ],
            [
                "",
                "dtype",
                "None"
            ],
            [
                "",
                "place",
                "None"
            ],
            [
                "",
                "stop_gradient",
                "True"
            ]
        ],
        "args_warning": {
            "name": "this args is deleted",
            "zero_copy": "this args is deleted"
        }
    },
    "paddle.fluid.dygraph.Sequential": {
        "update_to": "paddle.nn.Sequential"
    },
    "paddle.fluid.dygraph.ParameterList": {
        "update_to": "paddle.nn.ParameterList"
    },
    "paddle.fluid.dygraph.LayerList": {
        "update_to": "paddle.nn.LayerList"
    },
    # TODO act transformer manual check
    "paddle.fluid.dygraph.Conv2D": {
        "alias": ["paddle.fluid.dygraph.nn.Conv2D"],
        "update_to": "paddle.nn.Conv2d",
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
            "use_cudnn": "this args is deleted in paddle.nn.Conv2d",
            "act": "this args is deleted in paddle.nn.Conv2d",
            "dtype": "this args is deleted in paddle.nn.Conv2d"
        }
    },
    # TODO act transformer manual check
    "paddle.fluid.dygraph.Conv3D": {
        "alias": ["paddle.fluid.dygraph.nn.Conv3D"],
        "update_to": "paddle.nn.Conv3d",
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
    # manual check
    "paddle.fluid.dygraph.Pool2D": {
        "update_to": "paddle.nn.Pool2D",
        "args_list": [
            "pool_size",
            "pool_type",
            "pool_stride",
            "pool_padding",
            "global_pooling",
            "use_cudnn",
            "ceil_mode",
            "exclusive"
        ],
        "args_change": [
            [
                "pool_size",
                "pool_size"
            ],
            [
                "pool_type",
                "pool_type"
            ],
            [
                "pool_stride",
                "pool_stride"
            ],
            [
                "pool_padding",
                "pool_padding"
            ],
            [
                "global_pooling",
                "global_pooling"
            ],
            [
                "use_cudnn",
                "use_cudnn"
            ],
            [
                "ceil_mode",
                "ceil_mode"
            ],
            [
                "exclusive",
                "exclusive"
            ],
            [
                "",
                "data_format",
                "HCHW"
            ]
        ]
    },
    "paddle.fluid.dygraph.Linear": {
        "alias": ["paddle.fluid.dygraph.nn.Linear"],
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
    "paddle.fluid.dygraph.BatchNorm": {
        "alias": ["paddle.fluid.dygraph.nn.BatchNorm"],
        "update_to": "paddle.nn.BatchNorm"
    },
    # TODO transformer
    # "paddle.fluid.dygraph.Dropout": {
    #     "args_list": [
    #         "p",
    #         "seed",
    #         "dropout_implementation",
    #         "is_test"
    #     ],
    #     "args_change": [
    #         [
    #             "p",
    #             "p"
    #         ],
    #         [
    #             "seed",
    #             ""
    #         ],
    #         [
    #             "dropout_implementation",
    #             "dropout_implementation"
    #         ],
    #         [
    #             "is_test",
    #             "is_test"
    #         ]
    #     ]
    # },
    "paddle.fluid.dygraph.Embedding": {
        "alias": ["paddle.fluid.dygraph.nn.Embedding"],
        "update_to": "paddle.nn.Embedding"
    },
    # FlUID_WARNING
    # "paddle.fluid.dygraph.GRUUnit": {
    #     "update_to": "paddle.fluid.dygraph.GRUUnit"
    # },
    "paddle.fluid.dygraph.InstanceNorm": {
        "update_to": "paddle.nn.InstanceNorm"
    },
    # TODO transformer paddle.nn.LayerNorm
    # "paddle.fluid.dygraph.LayerNorm": {
    #     "args_list": [
    #         "normalized_shape",
    #         "scale",
    #         "shift",
    #         "epsilon",
    #         "param_attr",
    #         "bias_attr",
    #         "act",
    #         "dtype"
    #     ],
    #     "args_change": [
    #         [
    #             "normalized_shape",
    #             "normalized_shape"
    #         ],
    #         [
    #             "scale",
    #             "scale"
    #         ],
    #         [
    #             "shift",
    #             "shift"
    #         ],
    #         [
    #             "epsilon",
    #             "epsilon"
    #         ],
    #         [
    #             "param_attr",
    #             "param_attr"
    #         ],
    #         [
    #             "bias_attr",
    #             "bias_attr"
    #         ],
    #         [
    #             "act",
    #             "act"
    #         ],
    #         [
    #             "dtype",
    #             "dtype"
    #         ]
    #     ]
    # },
    # FlUID_WARNING
    # "paddle.fluid.dygraph.NCE": {
    #     "update_to": "paddle.fluid.dygraph.NCE"
    # },
    # FlUID_WARNING
    # "paddle.fluid.dygraph.PRelu": {
    #     "update_to": "paddle.fluid.dygraph.PRelu"
    # },
    "paddle.fluid.dygraph.BilinearTensorProduct": {
        "update_to": "paddle.nn.BilinearTensorProduct"
    },
    # TODO ConvTranspose2d
    # "paddle.fluid.dygraph.Conv2DTranspose": {
    #     "update_to": "paddle.fluid.dygraph.Conv2DTranspose"
    # },
    # TODO ConvTranspose3d
    # "paddle.fluid.dygraph.Conv3DTranspose": {
    #     "update_to": "paddle.fluid.dygraph.Conv3DTranspose"
    # },
    # TODO transformer paddle.nn.GroupNorm
    # "paddle.fluid.dygraph.GroupNorm": {
    #     "args_list": [
    #         "channels",
    #         "groups",
    #         "epsilon",
    #         "param_attr",
    #         "bias_attr",
    #         "act",
    #         "data_layout",
    #         "dtype"
    #     ],
    #     "args_change": [
    #         [
    #             "channels",
    #             "channels"
    #         ],
    #         [
    #             "groups",
    #             "groups"
    #         ],
    #         [
    #             "epsilon",
    #             "epsilon"
    #         ],
    #         [
    #             "param_attr",
    #             "param_attr"
    #         ],
    #         [
    #             "bias_attr",
    #             "bias_attr"
    #         ],
    #         [
    #             "act",
    #             "act"
    #         ],
    #         [
    #             "data_layout",
    #             "data_layout"
    #         ],
    #         [
    #             "dtype",
    #             "dtype"
    #         ]
    #     ]
    # },
    "paddle.fluid.dygraph.SpectralNorm": {
        "update_to": "paddle.nn.SpectralNorm"
    },
    # TODO FlUID_WARNING
    # "paddle.fluid.dygraph.TreeConv": {
    #     "update_to": "paddle.fluid.dygraph.TreeConv"
    # },
    "paddle.fluid.dygraph.MSELoss": {
        "update_to": "paddle.nn.MSELoss"
    },
    # manual check
    "paddle.fluid.dygraph.L1Loss": {
        "update_to": "paddle.nn.L1Loss"
    },
    # manual check
    "paddle.fluid.dygraph.NLLLoss": {
        "update_to": "paddle.nn.NLLLoss",
    },
    # manual check
    "paddle.fluid.dygraph.BCELoss": {
        "update_to": "paddle.nn.BCELoss",
    },
    # TODO check
    # "paddle.fluid.dygraph.prepare_context": {
    #     "update_to": "paddle.prepare_context"
    # },
    "paddle.fluid.dygraph.ParallelEnv": {
        "update_to": "paddle.ParallelEnv"
    },
    "paddle.fluid.dygraph.DataParallel": {
        "update_to": "paddle.DataParallel"
    },
    # FlUID_WARNING
    # "paddle.fluid.dygraph.save_dygraph": {
    #     "update_to": "paddle.fluid.dygraph.save_dygraph"
    # },
    # FlUID_WARNING
    # "paddle.fluid.dygraph.load_dygraph": {
    #     "update_to": "paddle.fluid.dygraph.load_dygraph"
    # },
    "paddle.fluid.dygraph.NoamDecay": {
        "update_to": "paddle.NoamDecay"
    },
    "paddle.fluid.dygraph.PiecewiseDecay": {
        "update_to": "paddle.PiecewiseDecay"
    },
    "paddle.fluid.dygraph.NaturalExpDecay": {
        "update_to": "paddle.NaturalExpDecay"
    },
    "paddle.fluid.dygraph.ExponentialDecay": {
        "update_to": "paddle.ExponentialDecay"
    },
    "paddle.fluid.dygraph.InverseTimeDecay": {
        "update_to": "paddle.InverseTimeDecay"
    },
    "paddle.fluid.dygraph.PolynomialDecay": {
        "update_to": "paddle.PolynomialDecay"
    },
    "paddle.fluid.dygraph.CosineDecay": {
        "update_to": "paddle.CosineDecay"
    },
    # TODO define LinearLrWarmup
    # "paddle.fluid.dygraph.LinearLrWarmup": {
    #     "args_list": [
    #         "learning_rate",
    #         "warmup_steps",
    #         "start_lr",
    #         "end_lr",
    #         "begin",
    #         "step",
    #         "dtype"
    #     ],
    #     "args_change": [
    #         [
    #             "learning_rate",
    #             "learning_rate"
    #         ],
    #         [
    #             "warmup_steps",
    #             "warmup_steps"
    #         ],
    #         [
    #             "start_lr",
    #             "start_lr"
    #         ],
    #         [
    #             "end_lr",
    #             "end_lr"
    #         ],
    #         [
    #             "begin",
    #             "begin"
    #         ],
    #         [
    #             "step",
    #             "step"
    #         ],
    #         [
    #             "dtype",
    #             "dtype"
    #         ]
    #     ]
    # },
    # manual check
    "paddle.fluid.dygraph.ReduceLROnPlateau": {
        "update_to": "paddle.optimizer.ReduceLROnPlateau",
        "args_list": [
            "learning_rate",
            "mode",
            "decay_rate",
            "patience",
            "verbose",
            "threshold",
            "threshold_mode",
            "cooldown",
            "min_lr",
            "eps",
            "dtype"
        ],
        "args_change": [
            [
                "learning_rate",
                "learning_rate"
            ],
            [
                "mode",
                "mode"
            ],
            [
                "decay_rate",
                "factor"
            ],
            [
                "patience",
                "patience"
            ],
            [
                "verbose",
                "verbose"
            ],
            [
                "threshold",
                "threshold"
            ],
            [
                "threshold_mode",
                "threshold_mode"
            ],
            [
                "cooldown",
                "cooldown"
            ],
            [
                "min_lr",
                "min_lr"
            ],
            [
                "eps",
                "epsilon"
            ],
            [
                "dtype",
                "dtype"
            ]
        ]
    },
    # FlUID_WARNING
    # "paddle.fluid.dygraph.StepDecay": {
    #     "update_to": "paddle.fluid.dygraph.StepDecay"
    # },
    # FlUID_WARNING
    # "paddle.fluid.dygraph.MultiStepDecay": {
    #     "update_to": "paddle.fluid.dygraph.MultiStepDecay"
    # },
    # FlUID_WARNING
    # "paddle.fluid.dygraph.LambdaDecay": {
    #     "update_to": "paddle.fluid.dygraph.LambdaDecay"
    # },
    "paddle.fluid.dygraph.BackwardStrategy": {
        "update_to": "paddle.BackwardStrategy"
    },
    "paddle.fluid.dygraph.TracedLayer": {
        "update_to": "paddle.jit.TracedLayer"
    },
    "paddle.fluid.dygraph.ProgramTranslator": {
        "update_to": "paddle.jit.ProgramTranslator"
    },
    # FlUID_WARNING
    # "paddle.fluid.transpiler.HashName": {
    #     "update_to": "paddle.fluid.transpiler.HashName"
    # },
    # FlUID_WARNING
    # "paddle.fluid.transpiler.RoundRobin": {
    #     "update_to": "paddle.fluid.transpiler.RoundRobin"
    # },
    # FlUID_WARNING
    # "paddle.fluid.nets.simple_img_conv_pool": {
    #     "update_to": "paddle.fluid.nets.simple_img_conv_pool"
    # },
    # FlUID_WARNING
    # "paddle.fluid.nets.sequence_conv_pool": {
    #     "update_to": "paddle.fluid.nets.sequence_conv_pool"
    # },
    # FlUID_WARNING
    # "paddle.fluid.nets.glu": {
    #     "update_to": "paddle.fluid.nets.glu"
    # },
    # FlUID_WARNING
    # "paddle.fluid.nets.scaled_dot_product_attention": {
    #     "update_to": "paddle.fluid.nets.scaled_dot_product_attention"
    # },
    # FlUID_WARNING
    # "paddle.fluid.nets.img_conv_group": {
    #     "update_to": "paddle.fluid.nets.img_conv_group"
    # },
    "paddle.fluid.optimizer.SGD": {
        "alias": ["paddle.fluid.optimizer.SGDOptimizer"],
        "update_to": "paddle.optimizer.SGD"
    },
    "paddle.fluid.optimizer.Momentum": {
        "alias": ["paddle.fluid.optimizer.MomentumOptimizer"],
        "update_to": "paddle.optimizer.Momentum"
    },
    "paddle.fluid.optimizer.Adagrad": {
        "alias": ["paddle.fluid.optimizer.AdagradOptimizer"],
        "update_to": "paddle.optimizer.Adagrad"
    },
    # manual check
    "paddle.fluid.optimizer.Adam": {
        "alias": ["paddle.fluid.optimizer.AdamOptimizer"],
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
                "learning_rate",
                "learning_rate"
            ],
            [
                "beta1",
                "beta1"
            ],
            [
                "beta2",
                "beta2"
            ],
            [
                "epsilon",
                "epsilon"
            ],
            [
                "parameter_list",
                "parameters"
            ],
            [
                "regularization",
                "weight_decay"
            ],
            [
                "grad_clip",
                "grad_clip"
            ],
            [
                "name",
                "name"
            ],
            [
                "lazy_mode",
                "lazy_mode"
            ]
        ]
    },
    "paddle.fluid.optimizer.Adamax": {
        "alias": ["paddle.fluid.optimizer.AdamaxOptimizer"],
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
                "learning_rate",
                "learning_rate"
            ],
            [
                "beta1",
                "beta1"
            ],
            [
                "beta2",
                "beta2"
            ],
            [
                "epsilon",
                "epsilon"
            ],
            [
                "parameter_list",
                "parameters"
            ],
            [
                "regularization",
                "weight_decay"
            ],
            [
                "grad_clip",
                "grad_clip"
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    "paddle.fluid.optimizer.Dpsgd": {
        "alias": ["paddle.fluid.optimizer.DpsgdOptimizer"],
        "update_to": "paddle.optimizer.Dpsgd"
    },
    "paddle.fluid.optimizer.DecayedAdagrad": {
        "alias": ["paddle.fluid.optimizer.DecayedAdagradOptimizer"],
        "update_to": "paddle.optimizer.DecayedAdagrad"
    },
    "paddle.fluid.optimizer.Ftrl": {
        "alias": ["paddle.fluid.optimizer.FtrlOptimizer"],
        "update_to": "paddle.optimizer.Ftrl"
    },
    # FLUID_WARNING
    # "paddle.fluid.optimizer.RMSPropOptimizer": {
    #     "update_to": "paddle.fluid.optimizer.RMSPropOptimizer"
    # },
    "paddle.fluid.optimizer.Adadelta": {
        "alias": ["paddle.fluid.optimizer.AdadeltaOptimizer"],
        "update_to": "paddle.optimizer.Adadelta"
    },
    "paddle.fluid.optimizer.ModelAverage": {
        "update_to": "paddle.optimizer.ModelAverage"
    },
    "paddle.fluid.optimizer.LarsMomentum": {
        "alias": ["paddle.fluid.optimizer.LarsMomentumOptimizer"],
        "update_to": "paddle.optimizer.LarsMomentum"
    },
    "paddle.fluid.optimizer.DGCMomentumOptimizer": {
        "update_to": "paddle.optimizer.DGCMomentumOptimizer"
    },
    "paddle.fluid.optimizer.LambOptimizer": {
        "update_to": "paddle.optimizer.LambOptimizer"
    },
    "paddle.fluid.optimizer.ExponentialMovingAverage": {
        "update_to": "paddle.optimizer.ExponentialMovingAverage"
    },
    # TODO transformer
    # "paddle.fluid.optimizer.PipelineOptimizer": {
    #     "update_to": "paddle.optimizer.PipelineOptimizer",
    #     "args_list": [
    #         "optimizer",
    #         "cut_list",
    #         "place_list",
    #         "concurrency_list",
    #         "queue_size",
    #         "sync_steps",
    #         "start_cpu_core_id"
    #     ],
    #     "args_change": [
    #         [
    #             "optimizer",
    #             "optimizer"
    #         ],
    #         [
    #             "cut_list",
    #             "cut_list"
    #         ],
    #         [
    #             "place_list",
    #             "place_list"
    #         ],
    #         [
    #             "concurrency_list",
    #             "concurrency_list"
    #         ],
    #         [
    #             "queue_size",
    #             "queue_size"
    #         ],
    #         [
    #             "sync_steps",
    #             "sync_steps"
    #         ],
    #         [
    #             "start_cpu_core_id",
    #             "start_cpu_core_id"
    #         ]
    #     ]
    # },
    "paddle.fluid.optimizer.LookaheadOptimizer": {
        "update_to": "paddle.optimizer.LookaheadOptimizer"
    },
    "paddle.fluid.optimizer.RecomputeOptimizer": {
        "update_to": "paddle.optimizer.RecomputeOptimizer"
    },
    "paddle.fluid.backward.append_backward": {
        "update_to": "paddle.static.append_backward"
    },
    # FlUID_WARNING
    # "paddle.fluid.regularizer.L1Decay": {
    #     "update_to": "paddle.fluid.regularizer.L1Decay"
    # },
    # FlUID_WARNING
    # "paddle.fluid.regularizer.L2Decay": {
    #     "update_to": "paddle.fluid.regularizer.L2Decay"
    # },
    # FlUID_WARNING
    # "paddle.fluid.regularizer.L1DecayRegularizer": {
    #     "update_to": "paddle.fluid.regularizer.L1DecayRegularizer"
    # },
    # FlUID_WARNING
    # "paddle.fluid.regularizer.L2DecayRegularizer": {
    #     "update_to": "paddle.fluid.regularizer.L2DecayRegularizer"
    # },
    # FlUID_WARNING
    # "paddle.fluid.LoDTensor": {
    #     "update_to": "paddle.fluid.LoDTensor"
    # },
    # FlUID_WARNING
    # "paddle.fluid.LoDTensorArray": {
    #     "update_to": "paddle.fluid.LoDTensorArray"
    # },
    "paddle.fluid.CPUPlace": {
        "update_to": "paddle.CPUPlace"
    },
    "paddle.fluid.CUDAPlace": {
        "update_to": "paddle.CUDAPlace"
    },
    "paddle.fluid.CUDAPinnedPlace": {
        "update_to": "paddle.CUDAPinnedPlace"
    },
    "paddle.fluid.Tensor": {
        "update_to": "paddle.Tensor"
    },
    "paddle.fluid.ParamAttr": {
        "alias": ["paddle.fluid.param_attr.ParamAttr"],
        "update_to": "paddle.ParamAttr"
    },
    "paddle.fluid.WeightNormParamAttr": {
        "update_to": "paddle.static.WeightNormParamAttr"
    },
    # FlUID_WARNING
    # "paddle.fluid.DataFeeder": {
    #     "update_to": "paddle.fluid.DataFeeder"
    # },
    # FlUID_WARNING
    # "paddle.fluid.clip.set_gradient_clip": {
    #     "update_to": "paddle.fluid.clip.set_gradient_clip"
    # },
    # FlUID_WARNING
    # "paddle.fluid.clip.ErrorClipByValue": {
    #     "update_to": "paddle.fluid.clip.ErrorClipByValue"
    # },
    "paddle.fluid.clip.GradientClipByValue": {
        "update_to": "paddle.nn.GradientClipByValue"
    },
    "paddle.fluid.clip.GradientClipByNorm": {
        "update_to": "paddle.nn.GradientClipByNorm"
    },
    "paddle.fluid.clip.GradientClipByGlobalNorm": {
        "update_to": "paddle.nn.GradientClipByGlobalNorm"
    },
    # FlUID_WARNING
    # "paddle.fluid.profiler.cuda_profiler": {
    #     "update_to": "paddle.fluid.profiler.cuda_profiler"
    # },
    # FlUID_WARNING
    # "paddle.fluid.profiler.reset_profiler": {
    #     "update_to": "paddle.fluid.profiler.reset_profiler"
    # },
    # FlUID_WARNING
    # "paddle.fluid.profiler.profiler": {
    #     "update_to": "paddle.fluid.profiler.profiler"
    # },
    # FlUID_WARNING
    # "paddle.fluid.profiler.start_profiler": {
    #     "update_to": "paddle.fluid.profiler.start_profiler"
    # },
    # FlUID_WARNING
    # "paddle.fluid.profiler.stop_profiler": {
    #     "update_to": "paddle.fluid.profiler.stop_profiler"
    # },
    # FlUID_WARNING
    # "paddle.fluid.unique_name.generate": {
    #     "update_to": "paddle.fluid.unique_name.generate"
    # },
    # FlUID_WARNING
    # "paddle.fluid.unique_name.switch": {
    #     "update_to": "paddle.fluid.unique_name.switch"
    # },
    # FlUID_WARNING
    # "paddle.fluid.Scope": {
    #     "update_to": "paddle.fluid.Scope"
    # },
    # FlUID_WARNING
    # "paddle.fluid.install_check.run_check": {
    #     "update_to": "paddle.fluid.install_check.run_check"
    # },
    "paddle.reader.ComposeNotAligned": {
        "update_to": "paddle.reader.ComposeNotAligned"
    },
    "paddle.sysconfig.get_include": {
        "update_to": "paddle.sysconfig.get_include"
    },
    "paddle.sysconfig.get_lib": {
        "update_to": "paddle.sysconfig.get_lib"
    },
    "paddle.version.mkl": {
        "update_to": "paddle.version.mkl"
    },
    "paddle.version.show": {
        "update_to": "paddle.version.show"
    }
}
