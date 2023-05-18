import ast
import json
import inspect
import functools
from typing import cast
from glob import glob
from shutil import rmtree
from zipfile import ZipFile
from tempfile import mkdtemp
from pathlib import Path, PurePosixPath

import taichi.lang.impl as impl
import taichi.types
from taichi import ndarray, ScalarNdarray, VectorNdarray, MatrixNdarray, Vector, Matrix
from taichi.lang.kernel_impl import Kernel
from taichi.lang.exception import (
    TaichiRuntimeError,
    TaichiCompilationError
)
from taichi.lang.matrix import VectorType, MatrixType
from taichi.types import int32, primitive_types
from taichi.types.ndarray_type import NdarrayType
from taichi.graph import Arg, ArgKind, GraphBuilder
from taichi.aot import Module

from auto_graph.arg_value import ArgValue, IntArgValue, MatrixArgValue, ArrayArgValue
from auto_graph.allocation import Allocation


class Launch:
    def __init__(self, kernel_fn, args):
        super().__init__()
        self.kernel_fn = kernel_fn
        self.args = args


def _cook_dtype_string(dtype):
    if dtype == int:
        return _cook_dtype_string(impl.get_runtime().default_ip)
    elif dtype == taichi.i8:
        return "i8"
    elif dtype == taichi.i16:
        return "i16"
    elif dtype == taichi.i32:
        return "i32"
    elif dtype == taichi.i64:
        return "i64"
    elif dtype == float:
        return _cook_dtype_string(impl.get_runtime().default_fp)
    elif dtype == taichi.f16:
        return "f16"
    elif dtype == taichi.f32:
        return "f32"
    elif dtype == taichi.f64:
        return "f64"
    elif dtype == taichi.u8:
        return "u8"
    elif dtype == taichi.u16:
        return "u16"
    elif dtype == taichi.u32:
        return "u32"
    elif dtype == taichi.u64:
        return "u64"
    else:
        raise TaichiRuntimeError(f"Unsupported dtype")


def auto_graph(fn):
    """Marks a function as a Taichi compute graph.

    A Taichi compute graph is a restricted function written in Python, which helps users
    conveniently export compiled kernels as well as their host logic.

    Args:
        fn (Callable): the Python function to be decorated

    Returns:
        Callable: The decorated function

    Example::

        >>> ndarray_t = ti.types.ndarray(dtype=ti.f32, ndim=1)
        >>>
        >>> @ti.kernel
        >>> def foo(x: ndarray_t, delta: ndarray_t):
        >>>     for i in ti.grouped(x):
        >>>         x[i] = x[i] + delta[i]
        >>>
        >>> @ti.auto_graph
        >>> def run(length: ti.i32, x: ndarray_t):
        >>>     delta = ti.ndarray(dtype=ti.f32, shape=length)
        >>>     foo(x, delta)
    """
    if fn.__qualname__ != fn.__name__:
        raise TaichiCompilationError(f"Taichi auto graph {fn.__name__} must be defined at the top level")
    graph = AutoGraph(fn)

    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        return graph.__call__(*args, **kwargs)

    decorated._is_taichi_graph = True
    decorated._graph = graph
    decorated.compile = graph.compile
    decorated.run = graph.run
    decorated.archive = graph.archive
    return decorated


class AutoGraph:
    def __init__(self, _func):
        self.func = _func
        self.graph_arguments = {}
        self.graph_argument_ids = []
        self.variables = {}
        self.global_kernels = {}
        self.launch_contexts = []
        self.allocated_arrays = []
        self.shape_arguments = []
        self.graph_builder = None
        self.compiled_graph = None

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def extract_arguments(self):
        sig = inspect.signature(self.func)
        if sig.return_annotation not in (inspect.Signature.empty, None):
            raise TaichiCompilationError("Taichi auto-graph do not support return values")
        params = sig.parameters
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise TaichiCompilationError("Taichi auto-graph do not support variable keyword parameters "
                                             "(i.e., **kwargs)")
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TaichiCompilationError("Taichi auto-graph do not support variable positional parameters "
                                             "(i.e., *args)")
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise TaichiCompilationError("Taichi auto-graph do not support keyword parameters")
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise TaichiCompilationError('Taichi auto-graph only support "positional or keyword" parameters')
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                raise TaichiCompilationError(f"Taichi auto-graph `{self.func.__name__}` parameter `{arg_name}` must be "
                                             f"type annotated")
            else:
                if isinstance(annotation, NdarrayType):
                    if isinstance(annotation.dtype, VectorType):
                        raise TaichiCompilationError(f"Taichi vector is not supported in auto-graph")
                    self.graph_arguments[arg_name] = ArrayArgValue(
                        arg_type=ArrayArgValue.Type.GRAPH_VAR,
                        graph_var_name=arg_name,
                        graph_var_ndim=annotation.ndim,
                        graph_var_dtype=annotation.dtype
                    )
                elif isinstance(annotation, VectorType):
                    raise TaichiCompilationError(f"Taichi vector is not supported in auto-graph")
                elif isinstance(annotation, MatrixType):
                    self.graph_arguments[arg_name] = MatrixArgValue(
                        arg_type=MatrixArgValue.Type.GRAPH_VAR,
                        graph_var_name=arg_name,
                        graph_var_n=annotation.n,
                        graph_var_m=annotation.m,
                        graph_var_dtype=annotation.dtype
                    )
                elif id(annotation) in [id(int32), id(int)]:
                    self.graph_arguments[arg_name] = IntArgValue(
                        arg_type=IntArgValue.Type.GRAPH_VAR,
                        graph_var_name=arg_name
                    )
                else:
                    raise TaichiCompilationError(f"Invalid type annotation of Taichi auto-graph: {annotation}")
        for arg_name in self.graph_arguments:
            self.graph_argument_ids.append(id(self.graph_arguments[arg_name]))
            self.variables[arg_name] = self.graph_arguments[arg_name]

    def extract_kernels(self):
        for key, value in self.func.__globals__.items():
            if inspect.isfunction(value) and hasattr(value, '_primal') and isinstance(value._primal, Kernel):
                self.global_kernels[key] = value

    def parse_function_body(self):
        source = inspect.getsource(self.func)
        graph_definition = ast.parse(source).body[0]
        if not isinstance(graph_definition, ast.FunctionDef):
            raise TaichiCompilationError(f"Taichi auto-graph {self.func.__name__} must be defined as a Python function")

        statements = graph_definition.body
        for statement in statements:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                self.parse_kernel_launch(statement.value)
            elif isinstance(statement, ast.Assign):
                if len(statement.targets) != 1:
                    raise TaichiCompilationError(f"More than one target is unsupported in Taichi auto-graph")
                assert isinstance(statement.targets[0], ast.Name)
                if isinstance(statement.value, ast.Call):
                    self.parse_call_assignment(statement)
                elif isinstance(statement.value, ast.Name):
                    self.parse_alias_assignment(statement)
                elif isinstance(statement.value, ast.Constant):
                    self.parse_expression_assignment(statement)
                elif isinstance(statement.value, ast.Subscript):
                    self.parse_expression_assignment(statement)
                elif isinstance(statement.value, ast.BinOp):
                    self.parse_expression_assignment(statement)
                else:
                    raise TaichiCompilationError(f"Assignment value type {type(statement.value)} is unsupported in "
                                                 f"Taichi auto-graph")
            else:
                raise TaichiCompilationError(f"The statement in Taichi auto-graph {self.func.__name__} must be "
                                             f"assignments or kernel launch_contexts (without return value): "
                                             f"\"{ast.unparse(cast(ast.AST, statement))}\"")

    @staticmethod
    def _check_kernel_arguments(parameters, kernel_arguments):
        for parameter, kernel_argument in zip(parameters, kernel_arguments):
            if not kernel_argument.check_match_parameter(parameter):
                return False
        return True

    def parse_kernel_launch(self, node):
        kernel_fn = self.global_kernels[node.func.id]
        parameters = kernel_fn._primal.arguments
        for parameter in parameters:
            if id(parameter.annotation) in [id(int), id(float)]:
                raise TaichiCompilationError(f"Builtin primitive types are not supported in auto-graph")
            if isinstance(parameter.annotation, VectorType):
                raise TaichiCompilationError(f"Taichi vector is not supported in auto-graph")
            if isinstance(parameter.annotation, NdarrayType) and isinstance(parameter.annotation.dtype, VectorType):
                raise TaichiCompilationError(f"Taichi vector is not supported in auto-graph")
        kernel_arguments = []
        for arg in node.args:
            if isinstance(arg, ast.Name):
                if arg.id not in self.variables:
                    raise TaichiCompilationError(f"Undefined variable {arg.id}")
                kernel_arguments.append(self.variables[arg.id])
            elif isinstance(arg, ast.Constant) or isinstance(arg, ast.Subscript) or isinstance(arg, ast.BinOp):
                kernel_arguments.append(self._construct_expression(arg))
            else:
                raise TaichiCompilationError(f"Invalid argument in kernel {kernel_fn.__name__}")
        if not self._check_kernel_arguments(parameters, kernel_arguments):
            raise TaichiCompilationError(f"Argument type error in kernel {kernel_fn.__name__}")
        self.launch_contexts.append(Launch(kernel_fn, kernel_arguments))

    def _visit_attribute(self, node):
        if isinstance(node, ast.Name):
            if node.id in self.func.__globals__:
                return self.func.__globals__[node.id]
            else:
                raise TaichiCompilationError(f"Undefined variable {node.id} in global scope")
        elif isinstance(node, ast.Attribute):
            value = self._visit_attribute(node.value)
            if hasattr(value, node.attr):
                return getattr(value, node.attr)
            else:
                raise TaichiCompilationError(f"Undefined attribute {node.attr} in {value}")

    def _visit_function(self, node):
        if isinstance(node, ast.Attribute):
            return self._visit_attribute(node)
        elif isinstance(node, ast.Call):
            func = self._visit_function(node.func)
            args = []
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    args.append(arg.value)
                elif isinstance(arg, ast.Attribute):
                    args.append(self._visit_attribute(arg))
                elif isinstance(arg, ast.Name) and arg.id == 'int':
                    args.append(impl.get_runtime().default_ip)
                elif isinstance(arg, ast.Name) and arg.id == 'float':
                    args.append(impl.get_runtime().default_fp)
                else:
                    raise TaichiCompilationError(f"Unsupported arg type {type(arg)} in Taichi auto-graph")
            kwargs = {}
            for keyword in node.keywords:
                if isinstance(keyword.value, ast.Constant):
                    kwargs[keyword.arg] = keyword.value.value
                elif isinstance(keyword.value, ast.Attribute):
                    kwargs[keyword.arg] = self._visit_attribute(keyword.value)
                elif isinstance(keyword.value, ast.Name) and keyword.value.id == 'int':
                    kwargs[keyword.arg] = impl.get_runtime().default_ip
                elif isinstance(keyword.value, ast.Name) and keyword.value.id == 'float':
                    kwargs[keyword.arg] = impl.get_runtime().default_fp
                else:
                    raise TaichiCompilationError(f"Unsupported arg type {type(keyword.value)} in Taichi auto-graph")
            return func(*args, **kwargs)
        else:
            raise TaichiCompilationError(f"Unsupported function type {type(node)} in Taichi auto-graph")

    def _cook_node_dtype(self, node):
        if isinstance(node, ast.Attribute):
            return self._visit_attribute(node)
        elif isinstance(node, ast.Call):
            return self._visit_function(node)
        elif isinstance(node, ast.Name) and node.id == 'int':
            return impl.get_runtime().default_ip
        elif isinstance(node, ast.Name) and node.id == 'float':
            return impl.get_runtime().default_fp
        else:
            raise TaichiCompilationError(f"Invalid dtype {ast.unparse(node)}")

    def _cook_node_shape(self, node):
        if isinstance(node, ast.Tuple):
            shape_list = []
            for shape_element in node.elts:
                shape_list.append(self._construct_expression(shape_element))
        else:
            shape_list = [self._construct_expression(node)]
        return shape_list

    @staticmethod
    def _cook_node_integer(node):
        if isinstance(node, ast.Constant):
            return node.value
        else:
            raise TaichiCompilationError(f"n and m must be literal integer values")

    def _cook_allocation_ndarray(self, dtype, shape):
        dtype = self._cook_node_dtype(dtype)
        shape = self._cook_node_shape(shape)
        return Allocation(dtype=dtype, shape=shape)

    def _cook_allocation_scalar_ndarray(self, dtype, arr_shape):
        dtype = self._cook_node_dtype(dtype)
        assert id(dtype) in primitive_types.type_ids
        shape = self._cook_node_shape(arr_shape)
        return Allocation(dtype=dtype, shape=shape)

    def _cook_allocation_vector_ndarray(self, n, dtype, shape):
        n = self._cook_node_integer(n)
        dtype = self._cook_node_dtype(dtype)
        assert id(dtype) in primitive_types.type_ids
        shape = self._cook_node_shape(shape)
        return Allocation(dtype=taichi.types.vector(n=n, dtype=dtype), shape=shape)

    def _cook_allocation_matrix_ndarray(self, n, m, dtype, shape):
        n = self._cook_node_integer(n)
        m = self._cook_node_integer(m)
        dtype = self._cook_node_dtype(dtype)
        assert id(dtype) in primitive_types.type_ids
        shape = self._cook_node_shape(shape)
        return Allocation(dtype=taichi.types.matrix(n=n, m=m, dtype=dtype), shape=shape)

    def parse_call_assignment(self, node):
        func = self._visit_function(node.value.func)
        if func == ndarray:
            args, kwargs = node.value.args, {keyword.arg: keyword.value for keyword in node.value.keywords}
            array = ArrayArgValue(
                arg_type=ArrayArgValue.Type.ALLOC_VAR,
                alloc_var=self._cook_allocation_ndarray(*args, **kwargs)
            )
            self.allocated_arrays.append(array)
            self.variables[node.targets[0].id] = array
        elif func == ScalarNdarray:
            args, kwargs = node.value.args, {keyword.arg: keyword.value for keyword in node.value.keywords}
            array = ArrayArgValue(
                arg_type=ArrayArgValue.Type.ALLOC_VAR,
                alloc_var=self._cook_allocation_scalar_ndarray(*args, **kwargs)
            )
            self.allocated_arrays.append(array)
            self.variables[node.targets[0].id] = array
        elif func == VectorNdarray:
            raise TaichiCompilationError(f"Taichi vector is not supported in auto-graph")
        elif func == MatrixNdarray:
            args, kwargs = node.value.args, {keyword.arg: keyword.value for keyword in node.value.keywords}
            array = ArrayArgValue(
                arg_type=ArrayArgValue.Type.ALLOC_VAR,
                alloc_var=self._cook_allocation_matrix_ndarray(*args, **kwargs)
            )
            self.allocated_arrays.append(array)
            self.variables[node.targets[0].id] = array
        elif func == Vector or isinstance(func, VectorType):
            raise TaichiCompilationError(f"Taichi vector is not supported in auto-graph")
        elif func == Matrix or isinstance(func, MatrixType):
            if len(node.value.args) != 1:
                raise TaichiCompilationError(f"Unsupported argument number for {func}")
            try:
                const_matrix = ast.literal_eval(ast.unparse(node.value.args[0]))
            except Exception:
                raise TaichiCompilationError(f"Argument for {func} must be literal lists")
            self.variables[node.targets[0].id] = MatrixArgValue(
                arg_type=MatrixArgValue.Type.CONST,
                const_value=func(const_matrix)
            )
        else:
            raise TaichiCompilationError(f"Unsupported function call {type(func)} in Taichi auto-graph")

    def parse_alias_assignment(self, node):
        if node.value.id not in self.variables:
            raise TaichiCompilationError(f"Undefined variable {node.value.id}")
        self.variables[node.targets[0].id] = self.variables[node.value.id]

    def _construct_expression(self, node):
        if isinstance(node, ast.Name):
            if node.id not in self.variables:
                raise TaichiCompilationError(f"Undefined variable {node.id}")
            if not isinstance(self.variables[node.id], IntArgValue) and \
                    not isinstance(self.variables[node.id], MatrixArgValue):
                raise TaichiCompilationError(f"Taichi Ndarray is unsupported in binary operation")
            return self.variables[node.id]
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, int):
                raise TaichiCompilationError(f"Literal value type {type(node)} is unsupported in Taichi auto-graph")
            return IntArgValue(
                arg_type=IntArgValue.Type.CONST,
                const_value=node.value
            )
        elif isinstance(node, ast.Subscript):
            if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, int):
                raise TaichiCompilationError(f"Subscript index must be an integer literal value")
            if isinstance(node.value, ast.Attribute) and node.value.attr == 'shape':
                assert isinstance(node.value.value, ast.Name)
                if node.value.value.id not in self.variables:
                    raise TaichiCompilationError(f"Undefined variable {node.value.value.id}")
                array_var = self.variables[node.value.value.id]
                if not isinstance(array_var, ArrayArgValue):
                    raise TaichiCompilationError(f"Subscript is only supported for indexing Taichi Ndarray shapes")
                if node.slice.value < 0 or node.slice.value >= array_var.ndim:
                    raise TaichiCompilationError(f"The index of shape is out of range")
                if id(array_var) in self.graph_argument_ids:
                    shape_argument = IntArgValue(
                        arg_type=IntArgValue.Type.SHAPE_VAR,
                        shape_var_array=array_var,
                        shape_var_dim=node.slice.value
                    )
                    self.shape_arguments.append(shape_argument)
                    return shape_argument
                else:
                    return array_var.shape[node.slice.value]
            else:
                raise TaichiCompilationError(f"Subscript is only supported for indexing Taichi Ndarray shapes")
        elif isinstance(node, ast.BinOp):
            left = self._construct_expression(node.left)
            right = self._construct_expression(node.right)
            assert isinstance(left, IntArgValue) or isinstance(left, MatrixArgValue)

            if type(left) == type(right):
                if isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
                elif isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Div):
                    return left / right
                elif isinstance(node.op, ast.Mod):
                    return left % right
                elif isinstance(node.op, ast.MatMult) and isinstance(left, MatrixArgValue):
                    return left @ right
                else:
                    raise TaichiCompilationError(f"Unsupported binary operator {type(node.op)} between {type(left)}")
            else:
                raise TaichiCompilationError(f"Different types in binary operation: {type(left)} and {type(right)}")
        else:
            raise TaichiCompilationError(f"Value type {type(node)} is unsupported in binary operation")

    def parse_expression_assignment(self, node):
        self.variables[node.targets[0].id] = self._construct_expression(node.value)

    def build_compiled_graph(self):
        self.graph_builder = GraphBuilder()
        for launch_index, launch in enumerate(self.launch_contexts):
            sym_args = []
            for launch_arg_index, launch_arg in enumerate(launch.args):
                sym_name = f"kernel_{launch_index}_arg_{launch_arg_index}"
                if isinstance(launch_arg, IntArgValue):
                    int_type = launch.kernel_fn._primal.arguments[launch_arg_index].annotation
                    sym_arg = Arg(ArgKind.SCALAR, sym_name, int_type)
                elif isinstance(launch_arg, MatrixArgValue):
                    sym_arg = Arg(ArgKind.MATRIX, sym_name, MatrixType(n=launch_arg.n, m=launch_arg.m, ndim=2,
                                                                       dtype=launch_arg.dtype))
                elif isinstance(launch_arg, ArrayArgValue):
                    sym_arg = Arg(ArgKind.NDARRAY, sym_name, launch_arg.dtype, launch_arg.ndim)
                else:
                    raise TaichiCompilationError(f"Invalid type for launch arguments")
                sym_args.append(sym_arg)
            self.graph_builder.dispatch(launch.kernel_fn, *sym_args)
        self.compiled_graph = self.graph_builder.compile()

    def compile(self):
        self.extract_arguments()
        self.extract_kernels()
        self.parse_function_body()
        self.build_compiled_graph()

    def run(self, args):
        if self.compiled_graph is None:
            raise TaichiRuntimeError(f"Please compile the auto-graph first")

        ArgValue.reset_buffer()
        if len(args) != len(self.graph_arguments):
            raise TaichiCompilationError(f"Auto-graph takes {len(self.graph_arguments)} arguments but {len(args)} "
                                         f"were given")
        for graph_argument_name in self.graph_arguments:
            if graph_argument_name not in args:
                raise TaichiCompilationError(f"Graph argument {graph_argument_name} not found in given arguments")
            graph_argument = self.graph_arguments[graph_argument_name]
            if not graph_argument.check_match_instance(args[graph_argument_name]):
                raise TaichiCompilationError(f"Argument type {type(args[graph_argument_name])} does not match the "
                                             f"type of graph argument {graph_argument_name}")
            graph_argument.set_value(args[graph_argument_name])
        for shape_argument in self.shape_arguments:
            graph_array = shape_argument.shape_var_array.value
            shape_argument.set_value(graph_array.shape[shape_argument.shape_var_dim])
        for allocated_array in self.allocated_arrays:
            shape = []
            for shape_item in allocated_array.shape:
                shape.append(shape_item.get_value())
            shape = tuple(shape)
            allocated_array.set_value(taichi.ndarray(dtype=allocated_array.dtype, shape=shape))

        compiled_graph_args = {}
        for launch_index, launch in enumerate(self.launch_contexts):
            for launch_arg_index, launch_arg in enumerate(launch.args):
                sym_name = f"kernel_{launch_index}_arg_{launch_arg_index}"
                compiled_graph_args[sym_name] = launch_arg.get_value()
        self.compiled_graph.run(compiled_graph_args)

    def export_graph_json(self):
        id_to_array_name = {}
        meta_data = {
            "graph_arguments": {},
            "allocated_arrays": {},
            "launches": {}
        }
        for graph_argument_name, graph_argument in self.graph_arguments.items():
            meta_data["graph_arguments"][graph_argument_name] = {}
            graph_argument_entry = meta_data["graph_arguments"][graph_argument_name]
            if isinstance(graph_argument, IntArgValue):
                graph_argument_entry["type"] = "int"
                graph_argument_entry["dtype"] = "i32"  # TODO
            elif isinstance(graph_argument, MatrixArgValue):
                graph_argument_entry["type"] = "matrix"
                graph_argument_entry["n"] = graph_argument.n
                graph_argument_entry["m"] = graph_argument.m
                graph_argument_entry["dtype"] = _cook_dtype_string(graph_argument.dtype)
            elif isinstance(graph_argument, ArrayArgValue):
                id_to_array_name[id(graph_argument)] = graph_argument_name
                graph_argument_entry["type"] = "array"
                graph_argument_entry["ndim"] = graph_argument.ndim
                if id(graph_argument.dtype) in primitive_types.type_ids:
                    graph_argument_entry["n"] = 0
                    graph_argument_entry["m"] = 0
                    graph_argument_entry["dtype"] = _cook_dtype_string(graph_argument.dtype)
                elif isinstance(graph_argument.dtype, MatrixType):
                    graph_argument_entry["n"] = graph_argument.dtype.n
                    graph_argument_entry["m"] = graph_argument.dtype.m
                    graph_argument_entry["dtype"] = _cook_dtype_string(graph_argument.dtype.dtype)
                else:
                    raise TaichiRuntimeError("Unsupported dtype")
        for allocated_array_index, allocated_array in enumerate(self.allocated_arrays):
            allocated_array_name = f"array:{allocated_array_index}"
            meta_data["allocated_arrays"][allocated_array_name] = {}
            allocated_array_entry = meta_data["allocated_arrays"][allocated_array_name]
            id_to_array_name[id(allocated_array)] = allocated_array_name
            if id(allocated_array.dtype) in primitive_types.type_ids:
                allocated_array_entry["n"] = 0
                allocated_array_entry["m"] = 0
                allocated_array_entry["dtype"] = _cook_dtype_string(allocated_array.dtype)
            elif isinstance(allocated_array.dtype, MatrixType):
                allocated_array_entry["n"] = allocated_array.dtype.n
                allocated_array_entry["m"] = allocated_array.dtype.m
                allocated_array_entry["dtype"] = _cook_dtype_string(allocated_array.dtype.dtype)
            else:
                raise TaichiRuntimeError("Unsupported dtype")
            allocated_array_entry["shape"] = [str(shape_item) for shape_item in allocated_array.shape]
        for launch_index, launch in enumerate(self.launch_contexts):
            for launch_arg_index, launch_arg in enumerate(launch.args):
                launch_arg_name = f"kernel_{launch_index}_arg_{launch_arg_index}"
                meta_data["launches"][launch_arg_name] = {}
                launch_arg_entry = meta_data["launches"][launch_arg_name]
                if isinstance(launch_arg, IntArgValue):
                    launch_arg_entry["type"] = "int"
                    launch_arg_entry["dtype"] = "i32"  # TODO
                    launch_arg_entry["value"] = str(launch_arg)
                elif isinstance(launch_arg, MatrixArgValue):
                    launch_arg_entry["type"] = "matrix"
                    launch_arg_entry["dtype"] = _cook_dtype_string(launch_arg.dtype)
                    launch_arg_entry["value"] = str(launch_arg)
                elif isinstance(launch_arg, ArrayArgValue):
                    launch_arg_entry["type"] = "array"
                    launch_arg_entry["value"] = id_to_array_name[id(launch_arg)]
        return json.dumps(meta_data, separators=(',', ':'))

    def archive(self, arch, filepath):
        assert filepath.endswith(".tcm"), "AOT module artifact archive must ends with .tcm"
        tcm_path = Path(filepath).absolute()
        assert tcm_path.parent.exists(), "Output directory doesn't exist"
        temp_dir = mkdtemp(prefix="tcm_")

        # Save AOT module
        mod = Module(arch=arch)
        mod.add_graph('auto_graph', self.compiled_graph)
        mod.save(temp_dir)

        # Save meta-data for auto-graph
        with open(f"{str(PurePosixPath(Path(temp_dir)))}/auto_graph.json", "w") as f:
            f.write(self.export_graph_json())

        # Package to a zip archive and remove the cached files
        with ZipFile(tcm_path, "w") as z:
            for path in glob(f"{temp_dir}/*", recursive=True):
                z.write(path, Path.relative_to(Path(path), temp_dir))
        rmtree(temp_dir)
