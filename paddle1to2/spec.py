change_spec = {
    '''
    TODO 2.0 miss api list
    paddle.fluid.dygraph.dygraph_to_static.convert_function_with_cache
    paddle.fluid.initializer.TruncatedNormalInitializer,paddle.fluid.initializer.TruncatedNormal
    paddle.fluid.dygraph.BackwardStrategy,paddle.fluid.dygraph.backward_strategy.BackwardStrategy
    paddle.fluid.dygraph.jit.declarative,paddle.fluid.dygraph.declarative
    '''
    '''
    TODO 2.0 FLUID WARNING
    paddle.fluid.unique_name.generate
    paddle.fluid.dygraph.dygraph_to_static.to_static_variable_gast_node
    paddle.fluid.layers.load
    paddle.fluid.load_op_library
    paddle.fluid.layers.tensor_array_to_tensor
    paddle.fluid.io.get_program_parameter
    paddle.fluid.layers.Switch
    paddle.fluid.dygraph.guard
    paddle.fluid.unique_name.guard
    paddle.fluid.dygraph.dygraph_to_static.DygraphToStaticAst
    paddle.fluid.clip.set_gradient_clip
    paddle.fluid.layers.create_tensor
    paddle.fluid.layers.rnn
    paddle.fluid.DataFeeder
    paddle.fluid.layers.MultivariateNormalDiag
    paddle.fluid.device_guard
    paddle.fluid.layers.sequence_pool
    paddle.fluid.layers.sequence_conv
    paddle.fluid.DistributeTranspilerConfig
    paddle.fluid.dygraph.dygraph_to_static.convert_call
    paddle.fluid.metrics.ChunkEvaluator
    paddle.fluid.evaluator.ChunkEvaluator
    paddle.fluid.metrics.DetectionMAP
    paddle.fluid.layers.While
    paddle.fluid.profiler.stop_profiler
    paddle.fluid.layers.sequence_scatter
    paddle.fluid.dygraph.GRUUnit
    paddle.fluid.dygraph.dygraph_to_static.data_layer_not_check
    paddle.fluid.layers.DynamicRNN
    paddle.fluid.dygraph.StepDecay
    paddle.fluid.nets.img_conv_group
    paddle.fluid.nets.simple_img_conv_pool
    paddle.fluid.layers.layer_function_generator.autodoc
    paddle.fluid.regularizer.L2Decay
    paddle.fluid.create_random_int_lodtensor
    paddle.fluid.dygraph.dygraph_to_static.create_static_variable_gast_node
    paddle.fluid.layers.sum
    paddle.fluid.layers.sequence_unpad
    paddle.fluid.layers.matrix_nms
    paddle.fluid.trainer_factory.TrainerFactory
    paddle.fluid.profiler.start_profiler
    paddle.fluid.io.default_collate_fn
    paddle.fluid.nets.scaled_dot_product_attention
    paddle.fluid.metrics.EditDistance
    paddle.fluid.evaluator.EditDistance
    paddle.fluid.layers.im2sequence
    paddle.fluid.dygraph.dygraph_to_static.NameVisitor
    paddle.fluid.nets.sequence_conv_pool
    paddle.fluid.io.load_params
    paddle.fluid.io.get_program_persistable_vars
    paddle.fluid.layers.create_array
    paddle.fluid.layers.dynamic_lstmp
    paddle.fluid.dygraph.TreeConv
    paddle.fluid.layers.merge_selected_rows
    paddle.fluid.layers.sequence_pad
    paddle.fluid.dygraph.enabled
    paddle.fluid.dygraph.dygraph_to_static.StaticAnalysisVisitor
    paddle.fluid.profiler.cuda_profiler
    paddle.fluid.layers.array_write
    paddle.fluid.layers.Categorical
    paddle.fluid.layers.lod_append
    paddle.fluid.layers.sequence_expand_as
    paddle.fluid.layers.Decoder
    paddle.fluid.transpiler.RoundRobin
    paddle.fluid.layers.sequence_mask
    paddle.fluid.layers.dynamic_decode
    paddle.fluid.layers.ctc_greedy_decoder
    paddle.fluid.regularizer
    paddle.fluid.dygraph.dygraph_to_static.break_continue_transformer.BreakContinueTransformer
    paddle.fluid.layers.lod_reset
    paddle.fluid.cpu_places
    paddle.fluid.layers.reverse
    paddle.fluid.layers.lstm
    paddle.fluid.layers.layer_function_generator.generate_activation_fn
    paddle.fluid.layers.sequence_first_step
    paddle.fluid.layers.DecodeHelper
    paddle.fluid.LoDTensorArray
    paddle.fluid.io.save_params
    paddle.fluid.layers.reorder_lod_tensor_by_rank
    paddle.fluid.set_flags
    paddle.fluid.dataset.InMemoryDataset
    paddle.fluid.layers.create_py_reader_by_data
    paddle.fluid.metrics.CompositeMetric
    paddle.fluid.layers.lstm_unit
    paddle.fluid.layers.sequence_expand
    paddle.fluid.layers.sequence_last_step
    paddle.fluid.cuda_places
    paddle.fluid.device_worker.Section
    paddle.fluid.layers.inplace_abn
    paddle.fluid.dygraph.dygraph_to_static.NodeVarType
    paddle.fluid.layers.array_read
    paddle.fluid.layers.mish
    paddle.fluid.get_flags
    paddle.fluid.initializer.NumpyArrayInitializer
    paddle.fluid.dygraph.LambdaDecay
    paddle.fluid.incubate.fleet.base.mode.Mode
    paddle.fluid.layers.sequence_slice
    paddle.fluid.dygraph.MultiStepDecay
    paddle.fluid.io.load_persistables
    paddle.fluid.trainer_desc.DistMultiTrainer
    paddle.fluid.layers.layer_function_generator.generate_layer_fn
    paddle.fluid.dygraph.NCE
    paddle.fluid.trainer_desc.PipelineTrainer
    paddle.fluid.dygraph.dygraph_to_static_func
    paddle.fluid.layers.py_reader
    paddle.fluid.is_compiled_with_cuda
    paddle.fluid.clip.ErrorClipByValue
    paddle.fluid.layers.IfElse
    paddle.fluid.layers.locality_aware_nms
    paddle.fluid.memory_optimize
    paddle.fluid.Scope
    paddle.fluid.dygraph.dygraph_to_static.convert_to_static
    paddle.fluid.DistributeTranspiler
    paddle.fluid.incubate.data_generator.MultiSlotStringDataGenerator
    paddle.fluid.io.save_vars
    paddle.fluid.install_check.run_check
    paddle.fluid.cuda_pinned_places
    paddle.fluid.layers.StaticRNN
    paddle.fluid.layers.sequence_enumerate
    paddle.fluid.layers.dynamic_lstm
    paddle.fluid.profiler.profiler
    paddle.fluid.layers.sequence_reverse
    paddle.fluid.dygraph.dygraph_to_static.AstNodeWrapper
    paddle.fluid.Tensor
    paddle.fluid.trainer_factory.FetchHandlerMonitor
    paddle.fluid.wrapped_decorator.signature_safe_contextmanager
    paddle.fluid.layers.gru_unit
    paddle.fluid.unique_name
    paddle.fluid.layers.double_buffer
    paddle.fluid.layers.layer_function_generator.templatedoc
    paddle.fluid.io.PyReader
    paddle.fluid.layers.sequence_concat
    paddle.fluid.layers.mul
    paddle.fluid.layers.sampling_id
    paddle.fluid.device_worker.DeviceWorker
    paddle.fluid.trainer_desc.TrainerDesc
    paddle.fluid.profiler.reset_profiler
    paddle.fluid.layers.autoincreased_step_counter
    paddle.fluid.transpiler.collective.GradAllReduce
    paddle.fluid.layers.BasicDecoder
    paddle.fluid.metrics.MetricBase
    paddle.fluid.layers.read_file
    paddle.fluid.layers.array_length
    paddle.fluid.io.load_vars
    paddle.fluid.DataFeedDesc
    paddle.fluid.release_memory
    paddle.fluid.average.WeightedAverage
    paddle.fluid.dataset.QueueDataset
    paddle.fluid.layers.sequence_softmax
    paddle.fluid.dygraph.dygraph_to_static.LoopTransformer
    paddle.fluid.require_version
    paddle.fluid.device_worker.Hogwild
    paddle.fluid.regularizer.L1Decay
    paddle.fluid.layers.linear_chain_crf
    paddle.fluid.wrapped_decorator.wrap_decorator
    paddle.fluid.layers.GreedyEmbeddingHelper
    paddle.fluid.transpiler.collective.LocalSGD
    paddle.fluid.layers.sequence_reshape
    paddle.fluid.create_lod_tensor
    paddle.fluid.unique_name.switch
    paddle.fluid.incubate.data_generator.MultiSlotDataGenerator
    paddle.fluid.transpiler.HashName
    paddle.fluid.install_check
    paddle.fluid.layers.get_tensor_from_selected_rows
    paddle.fluid.io.save_persistables
    paddle.fluid.device_worker.DownpourSGDOPT
    paddle.fluid.device_worker.DownpourSGD
    paddle.fluid.nets.glu
    paddle.fluid.layers.TrainingHelper
    paddle.fluid.trainer_desc.MultiTrainer
    paddle.fluid.layers.dynamic_gru
    paddle.fluid.log_helper.get_logger
    paddle.fluid.layers.SampleEmbeddingHelper
    '''
    # manual check
    "paddle.fluid.dygraph.base.to_variable": {
        "alias": [
            "paddle.fluid.dygraph.to_variable"
        ],
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
            ]
        ]
    },
    # manual check
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
            ]
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
        "update_to": "paddle.nn.while_loop"
    },
    "paddle.fluid.optimizer.LambOptimizer": {
        "update_to": "paddle.optimizer.LambOptimizer"
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
                "input"
            ],
            [
                "offset",
                "offset"
            ],
            [
                "dim1",
                "dim1"
            ],
            [
                "dim2",
                "dim2"
            ],
            [
                "out",
                "out"
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    "paddle.fluid.initializer.Normal": {
        "alias": [
            "paddle.fluid.initializer.NormalInitializer"
        ],
        "update_to": "paddle.nn.initializer.Normal"
    },
    "paddle.fluid.load": {
        "alias": [
            "paddle.fluid.io.load"
        ],
        "update_to": "paddle.io.load"
    },
    "paddle.fluid.layers.nn.similarity_focus": {
        "alias": [
            "paddle.fluid.layers.similarity_focus"
        ],
        "update_to": "paddle.nn.functional.similarity_focus"
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
        ]
    },
    "paddle.fluid.initializer.UniformInitializer": {
        "alias": [
            "paddle.fluid.initializer.Uniform"
        ],
        "update_to": "paddle.nn.initializer.Uniform"
    },
    "paddle.fluid.layers.distributions.Uniform": {
        "alias": [
            "paddle.fluid.layers.Uniform"
        ],
        "update_to": "paddle.distribution.Uniform"
    },
    "paddle.fluid.layers.tensor.argmax": {
        "alias": [
            "paddle.fluid.layers.argmax"
        ],
        "update_to": "paddle.argmax"
    },
    "paddle.fluid.layers.learning_rate_scheduler.cosine_decay": {
        "alias": [
            "paddle.fluid.layers.cosine_decay"
        ],
        "update_to": "paddle.nn.functional.cosine_decay"
    },
    "paddle.fluid.layers.fsp_matrix": {
        "alias": [
            "paddle.fluid.layers.nn.fsp_matrix"
        ],
        "update_to": "paddle.nn.functional.fsp_matrix"
    },
    # TODO transformer
    "paddle.fluid.dygraph.nn.PRelu": {
        "alias": [
            "paddle.fluid.dygraph.PRelu"
        ],
        # "update_to" : "paddle.nn.PReLU"
        "warning": "paddle.fluid.dygraph.PRelu changes a lot, please read paddle.nn.PReLU for update"
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
        "update_to": "paddle.linspace"
    },
    "paddle.fluid.layers.huber_loss": {
        "alias": [
            "paddle.fluid.layers.loss.huber_loss"
        ],
        "update_to": "paddle.nn.functional.huber_loss"
    },
    "paddle.fluid.dygraph.dygraph_to_static.ProgramTranslator": {
        "alias": [
            "paddle.fluid.dygraph.dygraph_to_static.program_translator.ProgramTranslator",
            "paddle.fluid.dygraph.ProgramTranslator"
        ],
        "update_to": "paddle.jit.ProgramTranslator"
    },
    "paddle.fluid.layers.tensor.has_inf": {
        "alias": [
            "paddle.fluid.layers.has_inf"
        ],
        "update_to": "paddle.has_inf"
    },
    "paddle.incubate.hapi.vision.transforms.HueTransform": {
        "alias": [
            "paddle.incubate.hapi.vision.HueTransform",
            "paddle.incubate.hapi.vision.transforms.transforms.HueTransform"
        ],
        "update_to": "paddle.vision.HueTransform"
    },
    # transformer
    "paddle.fluid.layers.expand": {
        "alias": [
            "paddle.fluid.layers.nn.expand"
        ],
        # "update_to": "paddle.expand",
        "warning": "paddle.fluid.layers.expand changes a lot, please read paddle.expand for update"
    },
    "paddle.sysconfig.get_include": {
        "update_to": "paddle.sysconfig.get_include"
    },
    "paddle.fluid.layers.nn.image_resize": {
        "alias": [
            "paddle.fluid.layers.image_resize"
        ],
        "update_to": "paddle.nn.functional.image_resize"
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
        "update_to": "paddle.InverseTimeDecay"
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
    # manual check
    "paddle.fluid.layers.nn.size": {
        "alias": [
            "paddle.fluid.layers.size"
        ],
        "update_to": "paddle.numel",
        "args_list": [
            "input"
        ],
        "args_change": [
            [
                "input",
                "x"
            ]
        ]
    },
    "paddle.fluid.metrics.Auc": {
        "update_to": "paddle.metric.Auc",
        "args_list": [
            "name",
            "curve",
            "num_thresholds"
        ],
        "warning": "the Auc's args order is change to curve, num_thresholds, name."
    },
    "paddle.fluid.BuildStrategy": {
        "alias": [
            "paddle.fluid.compiler.BuildStrategy"
        ],
        "update_to": "paddle.static.BuildStrategy"
    },
    "paddle.fluid.layers.nn.shuffle_channel": {
        "alias": [
            "paddle.fluid.layers.shuffle_channel"
        ],
        "update_to": "paddle.nn.functional.shuffle_channel"
    },
    "paddle.fluid.layers.nn.reshape": {
        "alias": [
            "paddle.fluid.layers.reshape"
        ],
        # "update_to": "paddle.reshape",
        "warning": "paddle.fluid.layers.reshape has changed a lot, please read paddle.reshape for update."
    },
    "paddle.fluid.layers.deformable_roi_pooling": {
        "alias": [
            "paddle.fluid.layers.nn.deformable_roi_pooling"
        ],
        "update_to": "paddle.nn.functional.deformable_roi_pooling"
    },
    # manual check
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
            ]
        ],
        "args_warning": {
            "cond": "this args is deleted in this version"
        }
    },
    # manual check
    "paddle.fluid.dygraph.load_dygraph": {
        "alias": [
            "paddle.fluid.dygraph.checkpoint.load_dygraph"
        ],
        "update_to": "paddle.load",
        "args_list": [
            "model_path",
            "keep_name_table"
        ],
        "args_change": [
            [
                "model_path",
                "model_path"
            ],
            [
                "keep_name_table",
                ""
            ]
        ],
        "args_warning": {
            "keep_name_table": "this args is deleted in this version"
        }
    },
    "paddle.fluid.layers.resize_nearest": {
        "alias": [
            "paddle.fluid.layers.nn.resize_nearest"
        ],
        "update_to": "paddle.nn.functional.resize_nearest"
    },
    "paddle.fluid.reader.DataLoader": {
        "alias": [
            "paddle.fluid.io.DataLoader"
        ],
        "update_to": "paddle.io.DataLoader"
    },
    "paddle.fluid.compiler.CompiledProgram": {
        "alias": [
            "paddle.fluid.CompiledProgram"
        ],
        "update_to": "paddle.static.CompiledProgram"
    },
    "paddle.fluid.optimizer.Adadelta": {
        "alias": [
            "paddle.fluid.optimizer.AdadeltaOptimizer"
        ],
        "update_to": "paddle.optimizer.Adadelta",
        "args_list": [
            "learning_rate",
            "epsilon",
            "rho",
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
    # TODO transformer
    "paddle.fluid.layers.flatten": {
        "alias": [
            "paddle.fluid.layers.nn.flatten"
        ],
        # "update_to": "paddle.flatten",
        "warning": "paddle.fluid.layers.flatten has changed a lot, please read paddle.flatten for update"
    },
    "paddle.fluid.dygraph.nn.BCELoss": {
        "alias": [
            "paddle.fluid.dygraph.BCELoss"
        ],
        "update_to": "paddle.nn.BCELoss"
    },
    "paddle.fluid.dygraph.learning_rate_scheduler.ExponentialDecay": {
        "alias": [
            "paddle.fluid.dygraph.ExponentialDecay"
        ],
        "update_to": "paddle.ExponentialDecay"
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
        "update_to": "paddle.nn.functional.softplus"
    },
    "paddle.fluid.layers.roi_align": {
        "alias": [
            "paddle.fluid.layers.nn.roi_align"
        ],
        "update_to": "paddle.nn.functional.roi_align"
    },
    # manual check
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
        "update_to": "paddle.nn.Pool2D",
        "warning": "paddle.nn.Pool2D add a new arg data_format, default is \"NCHW\""
    },
    "paddle.fluid.io.shuffle": {
        "alias": [
            "paddle.reader.decorator.shuffle",
            "paddle.reader.shuffle"
        ],
        "update_to": "paddle.shuffle"
    },
    "paddle.fluid.layers.mean_iou": {
        "alias": [
            "paddle.fluid.layers.nn.mean_iou"
        ],
        "update_to": "paddle.metric.mean_iou"
    },
    "paddle.fluid.layers.generate_mask_labels": {
        "alias": [
            "paddle.fluid.layers.detection.generate_mask_labels"
        ],
        "update_to": "paddle.nn.functional.vision.generate_mask_labels"
    },
    "paddle.fluid.layers.multiplex": {
        "alias": [
            "paddle.fluid.layers.nn.multiplex"
        ],
        "update_to": "paddle.multiplex"
    },
    "paddle.fluid.layers.roi_perspective_transform": {
        "alias": [
            "paddle.fluid.layers.detection.roi_perspective_transform"
        ],
        "update_to": "paddle.nn.functional.vision.roi_perspective_transform"
    },
    # TODO transformer
    "paddle.fluid.embedding": {
        "alias": [
            "paddle.fluid.input.embedding",
            "paddle.fluid.layers.embedding",
            "paddle.fluid.layers.nn.embedding"
        ],
        # "update_to": "paddle.nn.functional.embedding",
        "warning": "paddle.fluid.embedding has changed a lot, please read paddle.nn.functional.embedding for update"
    },
    "paddle.fluid.io.load_inference_model": {
        "update_to": "paddle.io.load_inference_model"
    },
    "paddle.fluid.layers.nn.adaptive_pool3d": {
        "alias": [
            "paddle.fluid.layers.adaptive_pool3d"
        ],
        "update_to": "paddle.nn.functional.adaptive_pool3d"
    },
    "paddle.fluid.layers.nn.scale": {
        "alias": [
            "paddle.fluid.layers.scale"
        ],
        "update_to": "paddle.scale"
    },
    "paddle.fluid.layers.rpn_target_assign": {
        "alias": [
            "paddle.fluid.layers.detection.rpn_target_assign"
        ],
        "update_to": "paddle.nn.functional.rpn_target_assign"
    },
    "paddle.fluid.layers.nn.add_position_encoding": {
        "alias": [
            "paddle.fluid.layers.add_position_encoding"
        ],
        "update_to": "paddle.nn.functional.add_position_encoding"
    },
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
    "paddle.fluid.clip.GradientClipByNorm": {
        "update_to": "paddle.nn.GradientClipByNorm"
    },
    "paddle.fluid.layers.learning_rate_scheduler.inverse_time_decay": {
        "alias": [
            "paddle.fluid.layers.inverse_time_decay"
        ],
        "update_to": "paddle.nn.functional.learning_rate.inverse_time_decay"
    },
    "paddle.fluid.layers.temporal_shift": {
        "alias": [
            "paddle.fluid.layers.nn.temporal_shift"
        ],
        "update_to": "paddle.nn.functional.temporal_shift"
    },
    "paddle.fluid.layers.lrn": {
        "alias": [
            "paddle.fluid.layers.nn.lrn"
        ],
        "update_to": "paddle.nn.functional.lrn"
    },
    "paddle.fluid.layers.create_parameter": {
        "alias": [
            "paddle.fluid.layers.tensor.create_parameter"
        ],
        "update_to": "paddle.create_parameter"
    },
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
        "update_to": "paddle.is_empty"
    },
    # manual check
    "paddle.fluid.dygraph.Conv3DTranspose": {
        "alias": [
            "paddle.fluid.dygraph.nn.Conv3DTranspose"
        ],
        "update_to": "paddle.nn.ConvTranspose3d",
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
        "update_to": "paddle.crop_tensor"
    },
    # manual check
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
            ],
            [
                "name",
                "name"
            ]
        ]
    },
    "paddle.fluid.layers.nn.resize_trilinear": {
        "alias": [
            "paddle.fluid.layers.resize_trilinear"
        ],
        "update_to": "paddle.nn.functional.resize_trilinear"
    },
    "paddle.fluid.layers.clamp": {
        "alias": [
            "paddle.fluid.layers.nn.clamp"
        ],
        "update_to": "paddle.clip",
        "args_list": [
            "input",
            "min",
            "max",
            "output",
            "name"
        ],
        "args_change": [
            [
                "input",
                "x"
            ],
            [
                "output",
                ""
            ],
            [
                "name",
                "name"
            ]
        ],
        "args_warning": {
            "output": "output is deleted in this version."
        }
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
    # TODO note approve need transformer
    # "paddle.fluid.layers.hsigmoid": {
    #     "alias": [
    #         "paddle.fluid.layers.loss.hsigmoid"
    #     ],
    #     "update_to": "paddle.nn.functional.hsigmoid",
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
    #     ]
    # },
    "paddle.fluid.layers.scatter_nd": {
        "alias": [
            "paddle.fluid.layers.nn.scatter_nd"
        ],
        "update_to": "paddle.scatter_nd"
    },
    "paddle.fluid.dygraph.disable_imperative": {
        "alias": [
            "paddle.fluid.dygraph.base.disable_imperative",
            "paddle.fluid.disable_imperative"
        ],
        "update_to": "paddle.enable_static"
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
        "update_to": "paddle.PiecewiseDecay"
    },
    "paddle.fluid.layers.beam_search": {
        "update_to": "paddle.nn.beam_search"
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
                "x",
                "x"
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
        "update_to": "paddle.strided_slice"
    },
    "paddle.fluid.layers.box_clip": {
        "alias": [
            "paddle.fluid.layers.detection.box_clip"
        ],
        "update_to": "paddle.nn.functional.vision.box_clip"
    },
    "paddle.fluid.layers.log": {
        "alias": [
            "paddle.fluid.layers.nn.log"
        ],
        "update_to": "paddle.log"
    },
    "paddle.fluid.layers.nn.continuous_value_model": {
        "alias": [
            "paddle.fluid.layers.continuous_value_model"
        ],
        "update_to": "paddle.nn.functional.continuous_value_model"
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
    "paddle.fluid.layers.sigmoid_cross_entropy_with_logits": {
        "alias": [
            "paddle.fluid.layers.loss.sigmoid_cross_entropy_with_logits"
        ],
        "update_to": "paddle.nn.functional.sigmoid_cross_entropy_with_logits"
    },
    "paddle.fluid.layers.nn.deformable_conv": {
        "alias": [
            "paddle.fluid.layers.deformable_conv"
        ],
        "update_to": "paddle.static.nn.deformable_conv"
    },
    # manual check
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
    "paddle.fluid.optimizer.Dpsgd": {
        "alias": [
            "paddle.fluid.optimizer.DpsgdOptimizer"
        ],
        "update_to": "paddle.optimizer.Dpsgd"
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
    # manual check
    "paddle.complex.tensor.linalg.matmul": {
        "alias": [
            "paddle.complex.tensor.matmul",
            "paddle.complex.matmul",
            "paddle.fluid.layers.nn.matmul",
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
                "alpha",
                ""
            ]
        ],
        "args_warning": {
            "alpha": "This args is deleted in this version."
        }
    },
    "paddle.fluid.layers.chunk_eval": {
        "alias": [
            "paddle.fluid.layers.nn.chunk_eval"
        ],
        "update_to": "paddle.metric.chunk_eval"
    },
    # manual check
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
    # manual check
    "paddle.fluid.layers.nn.prelu": {
        "alias": [
            "paddle.fluid.layers.prelu"
        ],
        "update_to": "paddle.nn.functional.prelu",
        "args_list": [
            "x",
            "mode",
            "param_attr",
            "name"
        ],
        "args_change": [
            [
                "mode",
                ""
            ],
            [
                "param_attr",
                "weight"
            ]
        ],        
        "args_warning": {
            "mode": "mode is deleted in paddle.nn.functional.prelu"
        }        
    },
    "paddle.fluid.layers.ops.reciprocal": {
        "alias": [
            "paddle.fluid.layers.reciprocal"
        ],
        "update_to": "paddle.reciprocal"
    },
    "paddle.fluid.io.compose": {
        "alias": [
            "paddle.reader.decorator.compose",
            "paddle.reader.compose"
        ],
        "update_to": "paddle.io.compose"
    },
    "paddle.fluid.layers.nn.logical_xor": {
        "alias": [
            "paddle.fluid.layers.logical_xor"
        ],
        "update_to": "paddle.logical_xor"
    },
    "paddle.fluid.layers.crf_decoding": {
        "alias": [
            "paddle.fluid.layers.nn.crf_decoding"
        ],
        "update_to": "paddle.static.nn.crf_decoding"
    },
    # TODO transformer
    # "paddle.fluid.layers.diag": {
    #     "alias": [
    #         "paddle.fluid.layers.tensor.diag"
    #     ],
    #     "update_to": "paddle.diag",
    #     "args_list": [
    #         "diagonal"
    #     ],
    #     "args_change": [
    #         [
    #             "diagonal",
    #             "diagonal"
    #         ]
    #     ]
    # },
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
    # manual check
    "paddle.fluid.layers.affine_grid": {
        "alias": [
            "paddle.fluid.layers.nn.affine_grid"
        ],
        "update_to": "paddle.nn.functional.affine_grid",
        "args_list": [
            "theta",
            "out_shape",
            "name"
        ]
    },
    "paddle.fluid.layers.nn.spectral_norm": {
        "alias": [
            "paddle.fluid.layers.spectral_norm"
        ],
        "update_to": "paddle.static.nn.spectral_norm"
    },
    # manual checkq
    "paddle.complex.tensor.sum": {
        "alias": [
            "paddle.complex.sum",
            "paddle.complex.tensor.math.sum"
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
        "update_to": "paddle.nonzero"
    },
    "paddle.fluid.layers.polygon_box_transform": {
        "alias": [
            "paddle.fluid.layers.detection.polygon_box_transform"
        ],
        "update_to": "paddle.nn.functional.polygon_box_transform"
    },
    "paddle.fluid.layers.ops.hard_shrink": {
        "alias": [
            "paddle.fluid.layers.hard_shrink"
        ],
        "update_to": "paddle.nn.functional.hardshrink",
        "args_list": [
            "x",
            "threshold"
        ],
        "args_change": [
            [
                "x",
                "x"
            ],
            [
                "threshold",
                "threshold"
            ]
        ]
    },
    "paddle.fluid.layers.switch_case": {
        "alias": [
            "paddle.fluid.layers.control_flow.switch_case"
        ],
        "update_to": "paddle.nn.switch_case"
    },
    "paddle.fluid.framework.Variable": {
        "alias": [
            "paddle.fluid.Variable"
        ],
        "update_to": "paddle.Variable"
    },
    "paddle.fluid.layers.atan": {
        "alias": [
            "paddle.fluid.layers.ops.atan"
        ],
        "update_to": "paddle.atan"
    },
    "paddle.fluid.optimizer.RecomputeOptimizer": {
        "update_to": "paddle.optimizer.RecomputeOptimizer"
    },
    "paddle.fluid.dataloader.Dataset": {
        "alias": [
            "paddle.fluid.dataloader.dataset.Dataset"
        ],
        "update_to": "paddle.io.Dataset"
    },
    "paddle.fluid.layers.sampled_softmax_with_cross_entropy": {
        "alias": [
            "paddle.fluid.layers.loss.sampled_softmax_with_cross_entropy"
        ],
        "update_to": "paddle.nn.functional.sampled_softmax_with_cross_entropy"
    },
    "paddle.fluid.layers.detection.generate_proposals": {
        "alias": [
            "paddle.fluid.layers.generate_proposals"
        ],
        "update_to": "paddle.nn.functional.vision.generate_proposals"
    },
    # manual check
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
        "update_to": "paddle.nn.case"
    },
    "paddle.fluid.layers.fill_constant": {
        "alias": [
            "paddle.fluid.layers.tensor.fill_constant"
        ],
        "update_to": "paddle.fill_constant"
    },
    "paddle.fluid.layers.nn.crop": {
        "alias": [
            "paddle.fluid.layers.crop"
        ],
        "update_to": "paddle.crop_tensor"
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
        ],
        "args_warning": {
            "force_cpu": "this arg is deleted in this version"
        }
    },
    "paddle.fluid.layers.noam_decay": {
        "alias": [
            "paddle.fluid.layers.learning_rate_scheduler.noam_decay"
        ],
        "update_to": "paddle.nn.functional.learning_rate.noam_decay"
    },
    # TODO transformer paddle.nn.LayerNorm
    # "paddle.fluid.dygraph.LayerNorm": {
    #     "alias": [
    #         "paddle.fluid.dygraph.nn.LayerNorm"
    #     ],
    #     "update_to": "paddle.nn.LayerNorm",
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
    "paddle.fluid.optimizer.AdagradOptimizer": {
        "alias": [
            "paddle.fluid.optimizer.Adagrad"
        ],
        "update_to": "paddle.optimizer.Adagrad"
    },
    # TODO transformer
    # "paddle.fluid.layers.layer_norm": {
    #     "alias": [
    #         "paddle.fluid.layers.nn.layer_norm"
    #     ],
    #     "update_to": "paddle.nn.functional.norm.layer_norm",
    #     "args_list": [
    #         "input",
    #         "scale",
    #         "shift",
    #         "begin_norm_axis",
    #         "epsilon",
    #         "param_attr",
    #         "bias_attr",
    #         "act",
    #         "name"
    #     ],
    #     "args_change": [
    #         [
    #             "input",
    #             "input"
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
    #             "begin_norm_axis",
    #             "begin_norm_axis"
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
    #             "name",
    #             "name"
    #         ]
    #     ]
    # },
    "paddle.fluid.layers.one_hot": {
        "alias": [
            "paddle.fluid.layers.nn.one_hot",
            "paddle.fluid.one_hot",
            "paddle.fluid.input.one_hot"
        ],
        "warning": "input->x, depth->num_classes, x'elements must less than num_classes."
    },
    "paddle.fluid.initializer.Xavier": {
        "alias": [
            "paddle.fluid.initializer.XavierInitializer"
        ],
        "update_to": "paddle.nn.initializer.Xavier"
    },
    "paddle.fluid.layers.nn.l2_normalize": {
        "alias": [
            "paddle.fluid.layers.l2_normalize"
        ],
        "update_to": "paddle.nn.functional.norm.l2_normalize"
    },
    "paddle.fluid.layers.affine_channel": {
        "alias": [
            "paddle.fluid.layers.nn.affine_channel"
        ],
        "update_to": "paddle.nn.functional.vision.affine_channel"
    },
    "paddle.fluid.layers.argmin": {
        "alias": [
            "paddle.fluid.layers.tensor.argmin"
        ],
        "update_to": "paddle.argmin"
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
    "paddle.fluid.data": {
        "update_to": "paddle.data"
    },
    "paddle.compat.to_bytes": {
        "update_to": "paddle.compat.to_bytes"
    },
    # manual check
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
    "paddle.fluid.layers.nn.uniform_random": {
        "alias": [
            "paddle.fluid.layers.uniform_random"
        ],
        "update_to": "paddle.uniform"
    },
    "paddle.fluid.layers.detection.bipartite_match": {
        "alias": [
            "paddle.fluid.layers.bipartite_match"
        ],
        "update_to": "paddle.nn.functional.vision.bipartite_match"
    },
    "paddle.fluid.layers.nn.elementwise_sub": {
        "alias": [
            "paddle.fluid.layers.elementwise_sub"
        ],
        "update_to": "paddle.elementwise_sub"
    },
    # TODO transformer
    # "paddle.fluid.dataloader.BatchSampler": {
    #     "alias": [
    #         "paddle.fluid.dataloader.batch_sampler.BatchSampler"
    #     ],
    #     "update_to": "paddle.io.BatchSampler",
    #     "args_list": [
    #         "dataset",
    #         "indices",
    #         "shuffle",
    #         "batch_size",
    #         "drop_last"
    #     ],
    #     "args_change": [
    #         [
    #             "dataset",
    #             "dataset"
    #         ],
    #         [
    #             "indices",
    #             "indices"
    #         ],
    #         [
    #             "shuffle",
    #             "shuffle"
    #         ],
    #         [
    #             "batch_size",
    #             "batch_size"
    #         ],
    #         [
    #             "drop_last",
    #             "drop_last"
    #         ]
    #     ]
    # },
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
    # need transformer
    "paddle.fluid.dygraph.LinearLrWarmup": {
        "alias": [
            "paddle.fluid.dygraph.learning_rate_scheduler.LinearLrWarmup"
        ],
        "warning": "this api is update tp paddle.optimizer.LinearLrWarmup",
    },
    "paddle.fluid.layers.loss.teacher_student_sigmoid_loss": {
        "alias": [
            "paddle.fluid.layers.teacher_student_sigmoid_loss"
        ],
        "update_to": "paddle.nn.functional.teacher_student_sigmoid_loss"
    },
    "paddle.fluid.dygraph.parallel.prepare_context": {
        "alias": [
            "paddle.fluid.dygraph.prepare_context"
        ],
        "update_to": "paddle.distributed.prepare_context"
    },
    "paddle.fluid.layers.detection.box_decoder_and_assign": {
        "alias": [
            "paddle.fluid.layers.box_decoder_and_assign"
        ],
        "update_to": "paddle.nn.functional.vision.box_decoder_and_assign"
    },
    "paddle.reader.decorator.multiprocess_reader": {
        "alias": [
            "paddle.fluid.io.multiprocess_reader",
            "paddle.reader.multiprocess_reader"
        ],
        "update_to": "paddle.reader.multiprocess_reader"
    },
    "paddle.fluid.layers.nn.adaptive_pool2d": {
        "alias": [
            "paddle.fluid.layers.adaptive_pool2d"
        ],
        "update_to": "paddle.nn.functional.adaptive_pool2d"
    },
    "paddle.fluid.layers.nn.space_to_depth": {
        "alias": [
            "paddle.fluid.layers.space_to_depth"
        ],
        "update_to": "paddle.nn.functional.vision.space_to_depth"
    },
    # manual check
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
        "update_to": "paddle.nn.NLLLoss"
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
                "axis",
                ""
            ],
            [
                "act",
                ""
            ]
        ],
        "args_warning": {
            "axis": "axis is deleted in paddle.multiply",
            "act": "act is deleted in paddle.multiply"
        }
    },
    "paddle.complex.tensor.math.elementwise_mul": {
        "alias": [
            "paddle.complex.elementwise_mul",
            "paddle.complex.tensor.elementwise_mul"
        ],
        "update_to": "paddle.multiply"
    },
    "paddle.compat.floor_division": {
        "update_to": "paddle.compat.floor_division"
    },
    "paddle.fluid.layers.nn.clip": {
        "alias": [
            "paddle.fluid.layers.clip"
        ],
        "update_to": "paddle.clip"
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
    "paddle.fluid.layers.nn.gaussian_random": {
        "alias": [
            "paddle.fluid.layers.gaussian_random"
        ],
        "update_to": "paddle.normal",
        "args_list": [
            "shape",
            "mean",
            "std",
            "seed",
            "dtype"
        ],
        "args_change": [
            [
                "seed",
                ""
            ]
        ],
        "args_warning": {
            "seed": "this arg is deleted in this version."
        }
    },
    "paddle.fluid.layers.nn.selu": {
        "alias": [
            "paddle.fluid.layers.selu"
        ],
        "update_to": "paddle.nn.functional.selu"
    },
    # manual check
    "paddle.fluid.layers.meshgrid": {
        "alias": [
            "paddle.fluid.layers.nn.meshgrid"
        ],
        "update_to": "paddle.meshgrid"
    },
    "paddle.fluid.dygraph.nn.Conv3D": {
        "alias": [
            "paddle.fluid.dygraph.Conv3D"
        ],
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
    # TODO transformer
    "paddle.fluid.optimizer.PipelineOptimizer": {
        "warning": "this api is update to paddle.optimizer.PipelineOptimizer",
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
    "paddle.fluid.layers.retinanet_detection_output": {
        "alias": [
            "paddle.fluid.layers.detection.retinanet_detection_output"
        ],
        "update_to": "paddle.nn.functional.vision.retinanet_detection_output"
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
        ],
        "args_warning": {
            "out": "out is deleted in paddle.kron"
        }
    },
    "paddle.complex.tensor.math.kron": {
        "alias": [
            "paddle.complex.kron",
            "paddle.complex.tensor.kron"
        ],
        "update_to": "paddle.kron"
    },
    "paddle.fluid.io.ComposeNotAligned": {
        "alias": [
            "paddle.reader.decorator.ComposeNotAligned",
            "paddle.reader.ComposeNotAligned"
        ],
        "update_to": "paddle.reader.ComposeNotAligned"
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
        "update_to": "paddle.ParamAttr"
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
    "paddle.compat.round": {
        "update_to": "paddle.compat.round"
    },
    "paddle.fluid.layers.randn": {
        "alias": [
            "paddle.fluid.layers.nn.randn"
        ],
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
    "paddle.fluid.disable_dygraph": {
        "alias": [
            "paddle.fluid.dygraph.base.disable_dygraph",
            "paddle.fluid.dygraph.disable_dygraph"
        ],
        "update_to": "paddle.enable_static"
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
        "update_to": "paddle.io.save_inference_model"
    },
    "paddle.fluid.layers.collect_fpn_proposals": {
        "alias": [
            "paddle.fluid.layers.detection.collect_fpn_proposals"
        ],
        "update_to": "paddle.nn.functional.vision.collect_fpn_proposals"
    },
    "paddle.fluid.layers.brelu": {
        "alias": [
            "paddle.fluid.layers.nn.brelu"
        ],
        "update_to": "paddle.nn.functional.brelu"
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
    # TO transformer
    "paddle.fluid.layers.batch_norm": {
        "alias": [
            "paddle.fluid.layers.nn.batch_norm"
        ],
        "warning": " this api is update to paddle.nn.functional.batch_norm"
    },
    "paddle.fluid.layers.beam_search_decode": {
        "update_to": "paddle.nn.beam_search_decode"
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
        ]
    },
    # manual check
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
    # manual check
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
    "paddle.fluid.layers.elementwise_equal": {
        "alias": [
            "paddle.fluid.layers.nn.elementwise_equal"
        ],
        "update_to": "paddle.equal"
    },
    "paddle.fluid.WeightNormParamAttr": {
        "alias": [
            "paddle.fluid.param_attr.WeightNormParamAttr"
        ],
        "update_to": "paddle.static.WeightNormParamAttr"
    },
    "paddle.fluid.layers.metric_op.auc": {
        "alias": [
            "paddle.fluid.layers.auc"
        ],
        "update_to": "paddle.metric.auc"
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
    "paddle.fluid.io.buffered": {
        "alias": [
            "paddle.reader.buffered",
            "paddle.reader.decorator.buffered"
        ],
        "update_to": "paddle.io.buffered"
    },
    "paddle.fluid.dygraph.save_dygraph": {
        "alias": [
            "paddle.fluid.dygraph.checkpoint.save_dygraph"
        ],
        "update_to": "paddle.save"
    },
    "paddle.fluid.layers.cast": {
        "alias": [
            "paddle.fluid.layers.tensor.cast"
        ],
        "update_to": "paddle.cast"
    },
    "paddle.fluid.layers.nn.image_resize_short": {
        "alias": [
            "paddle.fluid.layers.image_resize_short"
        ],
        "update_to": "paddle.nn.functional.vision.image_resize_short"
    },
    "paddle.incubate.hapi.vision.SaturationTransform": {
        "alias": [
            "paddle.incubate.hapi.vision.transforms.transforms.SaturationTransform",
            "paddle.incubate.hapi.vision.transforms.SaturationTransform"
        ],
        "update_to": "paddle.vision.SaturationTransform"
    },
    "paddle.fluid.layers.label_smooth": {
        "alias": [
            "paddle.fluid.layers.nn.label_smooth"
        ],
        "update_to": "paddle.nn.functional.label_smooth"
    },
    "paddle.fluid.framework.default_main_program": {
        "alias": [
            "paddle.fluid.default_main_program"
        ],
        "update_to": "paddle.static.default_main_program"
    },
    "paddle.fluid.layers.learning_rate_scheduler.piecewise_decay": {
        "alias": [
            "paddle.fluid.layers.piecewise_decay"
        ],
        "update_to": "paddle.nn.functional.learning_rate.piecewise_decay"
    },
    "paddle.fluid.layers.roi_pool": {
        "alias": [
            "paddle.fluid.layers.nn.roi_pool"
        ],
        "update_to": "paddle.nn.functional.vision.roi_pool"
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
    "paddle.fluid.optimizer.ExponentialMovingAverage": {
        "update_to": "paddle.optimizer.ExponentialMovingAverage"
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
    "paddle.fluid.dygraph.nn.InstanceNorm": {
        "alias": [
            "paddle.fluid.dygraph.InstanceNorm"
        ],
        "update_to": "paddle.nn.InstanceNorm"
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
    "paddle.fluid.layers.nn.soft_relu": {
        "alias": [
            "paddle.fluid.layers.soft_relu"
        ],
        "update_to": "paddle.nn.functional.soft_relu"
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
        "warning": "this api is update to paddle.nn.ConvTranspose2d"
    },
    "paddle.fluid.layers.retinanet_target_assign": {
        "alias": [
            "paddle.fluid.layers.detection.retinanet_target_assign"
        ],
        "update_to": "paddle.nn.functional.vision.retinanet_target_assign"
    },
    "paddle.reader.cache": {
        "alias": [
            "paddle.reader.decorator.cache",
            "paddle.fluid.io.cache"
        ],
        "update_to": "paddle.io.cache"
    },
    "paddle.fluid.optimizer.SGD": {
        "alias": [
            "paddle.fluid.optimizer.SGDOptimizer"
        ],
        "update_to": "paddle.optimizer.SGD",
        "args_list": [
            "learning_rate",
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
                "weight_dacey"
            ]
        ]
    },
    "paddle.fluid.layers.nn.bmm": {
        "alias": [
            "paddle.fluid.layers.bmm"
        ],
        "update_to": "paddle.bmm"
    },
    "paddle.fluid.layers.nn.hash": {
        "alias": [
            "paddle.fluid.layers.hash"
        ],
        "update_to": "paddle.nn.functional.lod.hash"
    },
    # manual check
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
    "paddle.fluid.layers.maxout": {
        "alias": [
            "paddle.fluid.layers.nn.maxout"
        ],
        "update_to": "paddle.nn.functional.maxout"
    },
    "paddle.fluid.dygraph.parallel.ParallelEnv": {
        "alias": [
            "paddle.fluid.dygraph.ParallelEnv"
        ],
        "update_to": "paddle.distributed.ParallelEnv"
    },
    # manual check
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
        "update_to": "paddle.scatter_nd_add"
    },
    "paddle.fluid.ExecutionStrategy": {
        "alias": [
            "paddle.fluid.compiler.ExecutionStrategy"
        ],
        "update_to": "paddle.static.ExecutionStrategy"
    },
    # TODO transformer
    "paddle.fluid.layers.nn.conv3d_transpose": {
        "alias": [
            "paddle.fluid.layers.conv3d_transpose"
        ],
        "warning": "this api is update to paddle.nn.functional.conv_transpose3d",
    },
    "paddle.fluid.io.load_program_state": {
        "update_to": "paddle.io.load_program_state"
    },
    "paddle.fluid.io.xmap_readers": {
        "alias": [
            "paddle.reader.decorator.xmap_readers",
            "paddle.reader.xmap_readers"
        ],
        "update_to": "paddle.io.xmap_readers"
    },
    "paddle.fluid.layers.accuracy": {
        "alias": [
            "paddle.fluid.layers.metric_op.accuracy"
        ],
        "update_to": "paddle.metric.accuracy"
    },
    "paddle.fluid.layers.abs": {
        "alias": [
            "paddle.fluid.layers.ops.abs"
        ],
        "update_to": "paddle.abs"
    },
    "paddle.fluid.dygraph.enable_dygraph": {
        "alias": [
            "paddle.fluid.enable_dygraph",
            "paddle.fluid.dygraph.base.enable_dygraph"
        ],
        "update_to": "paddle.disable_static"
    },
    "paddle.fluid.dygraph.learning_rate_scheduler.NoamDecay": {
        "alias": [
            "paddle.fluid.dygraph.NoamDecay"
        ],
        "update_to": "paddle.NoamDecay"
    },
    "paddle.fluid.layers.erf": {
        "alias": [
            "paddle.fluid.layers.ops.erf"
        ],
        "update_to": "paddle.erf"
    },
    # TODO transformer
    "paddle.fluid.layers.ops.cumsum": {
        "alias": [
            "paddle.fluid.layers.cumsum"
        ],
        "warning": "this api is update to paddle.cumsum",
    },
    "paddle.incubate.hapi.vision.transforms.ContrastTransform": {
        "alias": [
            "paddle.incubate.hapi.vision.transforms.transforms.ContrastTransform",
            "paddle.incubate.hapi.vision.ContrastTransform"
        ],
        "update_to": "paddle.vision.ContrastTransform"
    },
    "paddle.fluid.layers.nn.log_loss": {
        "alias": [
            "paddle.fluid.layers.log_loss"
        ],
        "update_to": "paddle.nn.functional.log_loss"
    },
    "paddle.fluid.layers.resize_bilinear": {
        "alias": [
            "paddle.fluid.layers.nn.resize_bilinear"
        ],
        "update_to": "paddle.nn.functional.vision.resize_bilinear"
    },
    "paddle.fluid.layers.detection.target_assign": {
        "alias": [
            "paddle.fluid.layers.target_assign"
        ],
        "update_to": "paddle.nn.functional.target_assign"
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
        "update_to": "paddle.nn.L1Loss"
    },
    "paddle.fluid.io.batch": {
        "alias": [
            "paddle.batch"
        ],
        "update_to": "paddle.batch"
    },
    # manual check
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
    # TODO transformer
    "paddle.fluid.layers.nn.row_conv": {
        "alias": [
            "paddle.fluid.layers.row_conv"
        ],
        "warning": "this api is update to paddle.nn.functional.row_conv"
    },
    "paddle.fluid.layers.unstack": {
        "alias": [
            "paddle.fluid.layers.nn.unstack"
        ],
        "update_to": "paddle.unstack"
    },
    # manual check
    "paddle.fluid.layers.nn.unique": {
        "alias": [
            "paddle.fluid.layers.unique"
        ],
        "update_to": "paddle.unique",
        "args_list": [
            "x",
            "dtype"
        ]
    },
    "paddle.fluid.dygraph.layers.Layer": {
        "alias": [
            "paddle.fluid.dygraph.Layer"
        ],
        "update_to": "paddle.nn.Layer"
    },
    "paddle.fluid.laqyers.acos": {
        "alias": [
            "paddle.fluid.layers.ops.acos"
        ],
        "update_to": "paddle.acos"
    },
    # manual check
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
    # TODO transformer
    "paddle.fluid.layers.fill_constant_batch_size_like": {
        "alias": [
            "paddle.fluid.layers.tensor.fill_constant_batch_size_like"
        ],
        "warning": "this api is update to paddle.fill_constant"
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
                "x"
            ],
            [
                "axes",
                "axis"
            ]
        ]
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
    # TODO check paddle.text
    # "paddle.fluid.layers.RNNCell": {
    #     "update_to": "paddle.text.RNNCell",
    # },
    # warning
    "paddle.fluid.layers.uniform_random_batch_size_like": {
        "alias": [
            "paddle.fluid.layers.nn.uniform_random_batch_size_like"
        ],
        "warning": " this api is update to paddle.uniform"
    },
    "paddle.fluid.optimizer.Ftrl": {
        "alias": [
            "paddle.fluid.optimizer.FtrlOptimizer"
        ],
        "update_to": "paddle.optimizer.Ftrl"
    },
    "paddle.fluid.layers.learning_rate_scheduler.linear_lr_warmup": {
        "alias": [
            "paddle.fluid.layers.linear_lr_warmup"
        ],
        "update_to": "paddle.nn.functional.learning_rate.linear_lr_warmup"
    },
    "paddle.fluid.layers.detection.anchor_generator": {
        "alias": [
            "paddle.fluid.layers.anchor_generator"
        ],
        "update_to": "paddle.nn.functional.vision.anchor_generator"
    },
    "paddle.fluid.optimizer.LarsMomentumOptimizer": {
        "alias": [
            "paddle.fluid.optimizer.LarsMomentum"
        ],
        "update_to": "paddle.optimizer.LarsMomentum"
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
        "update_to": "paddle.arange"
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
    "paddle.fluid.optimizer.ModelAverage": {
        "update_to": "paddle.optimizer.ModelAverage"
    },
    "paddle.fluid.layers.nn.psroi_pool": {
        "alias": [
            "paddle.fluid.layers.psroi_pool"
        ],
        "update_to": "paddle.nn.functional.vision.psroi_pool"
    },
    "paddle.fluid.layers.yolo_box": {
        "alias": [
            "paddle.fluid.layers.detection.yolo_box"
        ],
        "update_to": "paddle.nn.functional.vision.yolo_box"
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
            "act" : "this api is deleted in this version"
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
        "update_to": "paddle.static.nn.fc"
    },
    "paddle.fluid.layers.loss.bpr_loss": {
        "alias": [
            "paddle.fluid.layers.bpr_loss"
        ],
        "update_to": "paddle.nn.functional.bpr_loss"
    },
    "paddle.fluid.layers.box_coder": {
        "alias": [
            "paddle.fluid.layers.detection.box_coder"
        ],
        "update_to": "paddle.nn.functional.vision.box_coder"
    },
    "paddle.fluid.layers.nn.reduce_all": {
        "alias": [
            "paddle.fluid.layers.reduce_all"
        ],
        "update_to": "paddle.reduce_all"
    },
    "paddle.fluid.layers.learning_rate_scheduler.exponential_decay": {
        "alias": [
            "paddle.fluid.layers.exponential_decay"
        ],
        "update_to": "paddle.nn.functional.learning_rate.exponential_decay"
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
    "paddle.reader.decorator.map_readers": {
        "alias": [
            "paddle.reader.map_readers",
            "paddle.fluid.io.map_readers"
        ],
        "update_to": "paddle.io.map_readers"
    },
    "paddle.fluid.layers.cos_sim": {
        "alias": [
            "paddle.fluid.layers.nn.cos_sim"
        ],
        "update_to": "paddle.metric.cos_sim"
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
        "update_to": "paddle.text.BeamSearchDecoder"
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
    "paddle.fluid.layers.distribute_fpn_proposals": {
        "alias": [
            "paddle.fluid.layers.detection.distribute_fpn_proposals"
        ],
        "update_to": "paddle.nn.functional.distribute_fpn_proposals"
    },
    "paddle.fluid.layers.detection.prior_box": {
        "alias": [
            "paddle.fluid.layers.prior_box"
        ],
        "update_to": "paddle.nn.functional.prior_box"
    },
    # manual check
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
    "paddle.fluid.layers.pool2d": {
        "alias": [
            "paddle.fluid.layers.nn.pool2d"
        ],
        "update_to": "paddle.nn.functional.pool2d"
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
    "paddle.fluid.dygraph.CosineDecay": {
        "alias": [
            "paddle.fluid.dygraph.learning_rate_scheduler.CosineDecay"
        ],
        "update_to": "paddle.CosineDecay"
    },
    "paddle.fluid.layers.rank_loss": {
        "alias": [
            "paddle.fluid.layers.loss.rank_loss"
        ],
        "update_to": "paddle.nn.functional.rank_loss"
    },
    "paddle.fluid.clip.GradientClipByGlobalNorm": {
        "update_to": "paddle.nn.GradientClipByGlobalNorm"
    },
    "paddle.fluid.layers.yolov3_loss": {
        "alias": [
            "paddle.fluid.layers.detection.yolov3_loss"
        ],
        "update_to": "paddle.nn.functional.vision.yolov3_loss"
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
    "paddle.fluid.layers.loss.center_loss": {
        "alias": [
            "paddle.fluid.layers.center_loss"
        ],
        "update_to": "paddle.nn.functional.center_loss"
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
    "paddle.fluid.layers.pad2d": {
        "alias": [
            "paddle.fluid.layers.nn.pad2d"
        ],
        "update_to": "paddle.nn.functional.pad2d"
    },
    # TODO transformer
    "paddle.fluid.layers.nn.conv2d": {
        "alias": [
            "paddle.fluid.layers.conv2d"
        ],
        "warning": "this api is update to paddle.nn.functional.conv2d",
    },
    "paddle.fluid.layers.edit_distance": {
        "alias": [
            "paddle.fluid.layers.loss.edit_distance"
        ],
        "update_to": "paddle.nn.functional.edit_distance"
    },
    "paddle.fluid.optimizer.DGCMomentumOptimizer": {
        "update_to": "paddle.optimizer.DGCMomentumOptimizer"
    },
    "paddle.fluid.layers.nn.group_norm": {
        "alias": [
            "paddle.fluid.layers.group_norm"
        ],
        "update_to": "paddle.static.nn.group_norm"
    },
    "paddle.fluid.initializer.Bilinear": {
        "alias": [
            "paddle.fluid.initializer.BilinearInitializer"
        ],
        "update_to": "paddle.nn.initializer.Bilinear"
    },
    "paddle.fluid.initializer.ConstantInitializer": {
        "alias": [
            "paddle.fluid.initializer.Constant"
        ],
        "update_to": "paddle.nn.initializer.Constant"
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
        "update_to": "paddle.nn.Conv2d",
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
    "paddle.fluid.io.firstn": {
        "alias": [
            "paddle.reader.firstn",
            "paddle.reader.decorator.firstn"
        ],
        "update_to": "paddle.io.firstn"
    },
    # mantul check
    "paddle.fluid.layers.addcmul": {
        "alias": [
            "paddle.fluid.layers.nn.addcmul"
        ],
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
                "out",
                ""
            ]
        ],
        "args_warning": {
            "out": "this arg is deleted in this version."
        }
    },
    "paddle.fluid.layers.logsigmoid": {
        "alias": [
            "paddle.fluid.layers.ops.logsigmoid"
        ],
        "update_to": "paddle.nn.functional.logsigmoid"
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
        "update_to": "paddle.nn.functional.hard_sigmoid"
    },
    "paddle.fluid.dygraph.ParameterList": {
        "alias": [
            "paddle.fluid.dygraph.container.ParameterList"
        ],
        "update_to": "paddle.nn.ParameterList"
    },
    "paddle.fluid.layers.polynomial_decay": {
        "alias": [
            "paddle.fluid.layers.learning_rate_scheduler.polynomial_decay"
        ],
        "update_to": "paddle.nn.functional.learning_rate.polynomial_decay"
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
            ]
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
        "update_to": "paddle.PolynomialDecay"
    },
    "paddle.fluid.layers.nn.swish": {
        "alias": [
            "paddle.fluid.layers.swish"
        ],
        "update_to": "paddle.nn.functional.swish"
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
    "paddle.fluid.dataset.DatasetFactory": {
        "update_to": "paddle.distributed.fleet.DatasetFactory"
    },
    "paddle.fluid.dygraph.learning_rate_scheduler.ReduceLROnPlateau": {
        "alias": [
            "paddle.fluid.dygraph.ReduceLROnPlateau"
        ],
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
                "decay_rate",
                "factor"
            ],
            [
                "eps",
                "epsilon"
            ]
        ]
    },
    "paddle.fluid.metrics.Recall": {
        "update_to": "paddle.metric.Recall",
    },
    "paddle.fluid.io.set_program_state": {
        "update_to": "paddle.io.set_program_state"
    },
    "paddle.fluid.layers.nn.smooth_l1": {
        "alias": [
            "paddle.fluid.layers.smooth_l1"
        ],
        "update_to": "paddle.nn.functional.smooth_l1"
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
        ]
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
    "paddle.fluid.layers.nn.pool3d": {
        "alias": [
            "paddle.fluid.layers.pool3d"
        ],
        "update_to": "paddle.nn.functional.pool3d"
    },
    "paddle.fluid.layers.detection.detection_output": {
        "alias": [
            "paddle.fluid.layers.detection_output"
        ],
        "update_to": "paddle.nn.functional.vision.detection_output"
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
    "paddle.fluid.layers.random_crop": {
        "alias": [
            "paddle.fluid.layers.nn.random_crop"
        ],
        "update_to": "paddle.nn.functional.random_crop"
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
        ]
    },
    "paddle.fluid.optimizer.RMSPropOptimizer": {
        "update_to": "paddle.optimizer.RMSProp",
        "args_list": [
            "learning_rate",
            "rho",
            "epsilon",
            "momentum",
            "centered",
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
        ]
    },
    # manual check
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
        "update_to": "paddle.nn.functional.sigmoid_focal_loss"
    },
    "paddle.fluid.CUDAPlace": {
        "update_to": "paddle.CUDAPlace"
    },
    "paddle.fluid.clip.GradientClipByValue": {
        "update_to": "paddle.nn.GradientClipByValue"
    },
    "paddle.fluid.layers.tanh": {
        "alias": [
            "paddle.fluid.layers.ops.tanh"
        ],
        "update_to": "paddle.tanh"
    },
    # manual check
    "paddle.fluid.layers.nn.instance_norm": {
        "alias": [
            "paddle.fluid.layers.instance_norm"
        ],
        "warning": "this api is update to paddle.nn.functional.instance_norm"
    },
    "paddle.fluid.layers.ops.sigmoid": {
        "alias": [
            "paddle.fluid.layers.sigmoid"
        ],
        "update_to": "paddle.nn.functional.sigmoid"
    },
    "paddle.fluid.layers.sums": {
        "alias": [
            "paddle.fluid.layers.tensor.sums"
        ],
        "update_to": "paddle.sums"
    },
    "paddle.fluid.layers.increment": {
        "alias": [
            "paddle.fluid.layers.control_flow.increment"
        ],
        "update_to": "paddle.increment"
    },
    "paddle.fluid.layers.tensor.eye": {
        "alias": [
            "paddle.fluid.layers.eye"
        ],
        "update_to": "paddle.eye"
    },
    "paddle.check_import_scipy": {
        "update_to": "paddle.check_import_scipy"
    },
    "paddle.fluid.layers.loss.softmax_with_cross_entropy": {
        "alias": [
            "paddle.fluid.layers.softmax_with_cross_entropy"
        ],
        "update_to": "paddle.nn.functional.softmax_with_cross_entropy"
    },
    "paddle.fluid.layers.detection.multiclass_nms": {
        "alias": [
            "paddle.fluid.layers.multiclass_nms"
        ],
        "update_to": "paddle.nn.functional.multiclass_nms"
    },
    "paddle.fluid.dygraph.container.Sequential": {
        "alias": [
            "paddle.fluid.dygraph.Sequential"
        ],
        "update_to": "paddle.nn.Sequential"
    },
    "paddle.fluid.io.chain": {
        "alias": [
            "paddle.reader.chain",
            "paddle.reader.decorator.chain"
        ],
        "update_to": "paddle.io.chain"
    },
    "paddle.fluid.layers.warpctc": {
        "alias": [
            "paddle.fluid.layers.loss.warpctc"
        ],
        "update_to": "paddle.nn.functional.warpctc"
    },
    "paddle.fluid.dygraph.NaturalExpDecay": {
        "alias": [
            "paddle.fluid.dygraph.learning_rate_scheduler.NaturalExpDecay"
        ],
        "update_to": "paddle.NaturalExpDecay"
    },
    "paddle.fluid.layers.nn.grid_sampler": {
        "alias": [
            "paddle.fluid.layers.grid_sampler"
        ],
        "update_to": "paddle.nn.functional.vision.grid_sampler"
    },
    "paddle.fluid.layers.cond": {
        "alias": [
            "paddle.fluid.layers.control_flow.cond"
        ],
        "update_to": "paddle.nn.cond"
    },
    "paddle.fluid.initializer.MSRAInitializer": {
        "alias": [
            "paddle.fluid.initializer.MSRA"
        ],
        "update_to": "paddle.nn.initializer.MSRA"
    },
    "paddle.fluid.layers.detection.density_prior_box": {
        "alias": [
            "paddle.fluid.layers.density_prior_box"
        ],
        "update_to": "paddle.nn.functional.density_prior_box"
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
        ]
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
        "update_to": "paddle.isfinite"
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
    "paddle.sysconfig.get_lib": {
        "update_to": "paddle.sysconfig.get_lib"
    },
    "paddle.fluid.metrics.Accuracy": {
        "update_to": "paddle.metric.Accuracy",
        "args_list": [
            "name"
        ],
        "args_change": [
            [
                "name",
                "name"
            ]
        ]
    },
    "paddle.fluid.optimizer.LookaheadOptimizer": {
        "update_to": "paddle.optimizer.LookaheadOptimizer"
    },
    "paddle.fluid.layers.nn.unique_with_counts": {
        "alias": [
            "paddle.fluid.layers.unique_with_counts"
        ],
        "update_to": "paddle.unique_with_counts"
    },
    "paddle.fluid.layers.tensor.create_global_var": {
        "alias": [
            "paddle.fluid.layers.create_global_var"
        ],
        "update_to": "paddle.create_global_var"
    },
    "paddle.fluid.io.save": {
        "alias": [
            "paddle.fluid.save"
        ],
        "update_to": "paddle.io.save"
    },
    "paddle.fluid.layers.tensor.has_nan": {
        "alias": [
            "paddle.fluid.layers.has_nan"
        ],
        "update_to": "paddle.has_nan"
    },
    "paddle.fluid.layers.assign": {
        "alias": [
            "paddle.fluid.layers.tensor.assign"
        ],
        "update_to": "paddle.nn.functional.assign"
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
        "update_to": "paddle.nn.BilinearTensorProduct"
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
        ]
    },
    # TODO transformer
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
    # TODO transformer
    "paddle.fluid.layers.nn.conv3d": {
        "alias": [
            "paddle.fluid.layers.conv3d"
        ],
        "warning": "this api is update to paddle.nn.functional.conv3d",
    },
    "paddle.fluid.layers.iou_similarity": {
        "alias": [
            "paddle.fluid.layers.detection.iou_similarity"
        ],
        "update_to": "paddle.nn.functional.iou_similarity"
    },
    "paddle.fluid.layers.nn.hard_swish": {
        "alias": [
            "paddle.fluid.layers.hard_swish"
        ],
        "update_to": "paddle.nn.functional.hard_swish"
    },
    # manual ckeck
    "paddle.fluid.layers.gaussian_random_batch_size_like": {
        "alias": [
            "paddle.fluid.layers.nn.gaussian_random_batch_size_like"
        ],
        "warning": "this api is update to paddle.normal"
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
    "paddle.fluid.layers.nn.prroi_pool": {
        "alias": [
            "paddle.fluid.layers.prroi_pool"
        ],
        "update_to": "paddle.nn.functional.vision.prroi_pool"
    },
    "paddle.fluid.layers.gather_tree": {
        "alias": [
            "paddle.fluid.layers.nn.gather_tree"
        ],
        "update_to": "paddle.nn.gather_tree"
    },
    "paddle.fluid.layers.detection.generate_proposal_labels": {
        "alias": [
            "paddle.fluid.layers.generate_proposal_labels"
        ],
        "update_to": "paddle.nn.functional.vision.generate_proposal_labels"
    },
    # TODO 
    "paddle.fluid.layers.LSTMCell": {
        "warning": "this api is update to paddle.nn.LSTMCell"
    },
    # TODO transformer
    "paddle.fluid.layers.nn.conv2d_transpose": {
        "alias": [
            "paddle.fluid.layers.conv2d_transpose"
        ],
        "warning": "this api is update to paddle.nn.functional.conv_transpose2d"
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
            "act": "act is deleted in paddle.maximum"
        }
    },
    "paddle.fluid.layers.sqrt": {
        "alias": [
            "paddle.fluid.layers.ops.sqrt"
        ],
        "update_to": "paddle.sqrt"
    },
    "paddle.fluid.layers.natural_exp_decay": {
        "alias": [
            "paddle.fluid.layers.learning_rate_scheduler.natural_exp_decay"
        ],
        "update_to": "paddle.nn.functional.learning_rate.natural_exp_decay"
    },
    "paddle.fluid.layers.nn.filter_by_instag": {
        "alias": [
            "paddle.fluid.layers.filter_by_instag"
        ],
        "update_to": "paddle.nn.functional.filter_by_instag"
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
    # manual check
    "paddle.fluid.layers.ops.gelu": {
        "alias": [
            "paddle.fluid.layers.gelu"
        ],
        "update_to": "paddle.nn.functional.gelu"
    },
    "paddle.fluid.layers.nn.pad_constant_like": {
        "alias": [
            "paddle.fluid.layers.pad_constant_like"
        ],
        "update_to": "paddle.nn.functional.pad_constant_like"
    },
    "paddle.version.mkl": {
        "update_to": "paddle.version.mkl"
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
        "update_to": "paddle.DataParallel"
    },
    "paddle.fluid.layers.nn.stack": {
        "alias": [
            "paddle.fluid.layers.stack"
        ],
        "update_to": "paddle.stack"
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
        "update_to": "paddle.reduce_any"
    },
    # TODO transformer
    "paddle.fluid.layers.GRUCell": {
        "warning": "this api is update to paddle.nn.GRUCell",
    } 
}