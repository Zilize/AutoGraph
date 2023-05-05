import ast
import sys
import inspect
import functools
from typing import cast

from taichi import ndarray, ScalarNdarray, VectorNdarray, MatrixNdarray, Vector, Matrix
from taichi.lang.kernel_impl import Kernel
from taichi.lang.exception import (
    TaichiRuntimeError,
    TaichiRuntimeTypeError,
    TaichiCompilationError
)
from taichi.lang.matrix import VectorType, MatrixType
from taichi.types import int32, ndarray_type

from graph.arg_value import IntArgValue, MatrixArgValue, ArrayArgValue


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
    return decorated


class AutoGraph:
    def __init__(self, _func):
        self.func = _func
        self.graph_arguments = {}
        self.variables = {}
        self.extract_arguments()
        self.global_kernels = {}
        self.extract_kernels()
        self.launches = []
        self.allocated_arrays = []
        self.shape_arguments = []
        self.parse_function_body()

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
                if isinstance(annotation, ndarray_type.NdarrayType):
                    self.graph_arguments[arg_name] = ArrayArgValue(
                        arg_type=ArrayArgValue.Type.GRAPH_VAR,
                        graph_var_name=arg_name,
                        graph_var_ndim=annotation.ndim,
                        graph_var_dtype=annotation.dtype
                    )
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
                elif isinstance(statement.value, ast.Constant):
                    self.parse_const_assignment(statement)
                elif isinstance(statement.value, ast.Name):
                    self.parse_alias_assignment(statement)
                elif isinstance(statement.value, ast.BinOp):
                    self.parse_binary_operation(statement)
                elif isinstance(statement.value, ast.Subscript):
                    self.parse_shape_assignment(statement)
                else:
                    raise TaichiCompilationError(f"Assignment value type {type(statement.value)} is unsupported in "
                                                 f"Taichi auto-graph")
            else:
                raise TaichiCompilationError(f"The statement in Taichi auto-graph {self.func.__name__} must be "
                                             f"assignments or kernel launches (without return value): "
                                             f"\"{ast.unparse(cast(ast.AST, statement))}\"")

    def parse_kernel_launch(self, node):
        pass

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
                else:
                    raise TaichiCompilationError(f"Unsupported arg type {type(arg)} in Taichi auto-graph")
            kwargs = {}
            for keyword in node.keywords:
                if isinstance(keyword.value, ast.Constant):
                    kwargs[keyword.arg] = keyword.value.value
                elif isinstance(keyword.value, ast.Attribute):
                    kwargs[keyword.arg] = self._visit_attribute(keyword.value)
                else:
                    raise TaichiCompilationError(f"Unsupported arg type {type(keyword.value)} in Taichi auto-graph")
            return func(*args, **kwargs)
        else:
            raise TaichiCompilationError(f"Unsupported function type {type(node)} in Taichi auto-graph")

    def parse_call_assignment(self, node):
        func = self._visit_function(node.value.func)
        if func == ndarray:
            pass
        elif func == ScalarNdarray:
            pass
        elif func == VectorNdarray:
            pass
        elif func == MatrixNdarray:
            pass
        elif func == Vector or func == Matrix or isinstance(func, VectorType) or isinstance(func, MatrixType):
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

    def parse_const_assignment(self, node):
        if not isinstance(node.value.value, int):
            raise TaichiCompilationError(f"Literal value assignment of datatype {type(node.value.value)} is "
                                         f"unsupported in Taichi auto-graph")
        self.variables[node.targets[0].id] = IntArgValue(
            arg_type=IntArgValue.Type.CONST,
            const_value=node.value.value
        )

    def parse_alias_assignment(self, node):
        if node.value.id not in self.variables:
            raise TaichiCompilationError(f"Undefined variable {node.value.id}")
        if isinstance(self.variables[node.value.id], IntArgValue):
            self.variables[node.targets[0].id] = IntArgValue(
                arg_type=IntArgValue.Type.ALIAS_VAR,
                alias_var=self.variables[node.value.id]
            )
        elif isinstance(self.variables[node.value.id], MatrixArgValue):
            self.variables[node.targets[0].id] = MatrixArgValue(
                arg_type=MatrixArgValue.Type.ALIAS_VAR,
                alias_var=self.variables[node.value.id]
            )
        elif isinstance(self.variables[node.value.id], ArrayArgValue):
            self.variables[node.targets[0].id] = ArrayArgValue(
                arg_type=ArrayArgValue.Type.ALIAS_VAR,
                alias_var=self.variables[node.value.id]
            )

    def _construct_binary_operation_graph(self, node):
        if isinstance(node, ast.BinOp):
            left = self._construct_binary_operation_graph(node.left)
            right = self._construct_binary_operation_graph(node.right)
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
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, int):
                raise TaichiCompilationError(f"Value type {type(node)} is unsupported in binary operation")
            return IntArgValue(
                arg_type=IntArgValue.Type.CONST,
                const_value=node.value
            )
        elif isinstance(node, ast.Name):
            if node.id not in self.variables:
                raise TaichiCompilationError(f"Undefined variable {node.id}")
            if not isinstance(self.variables[node.id], IntArgValue) and \
                    not isinstance(self.variables[node.id], MatrixArgValue):
                raise TaichiCompilationError(f"Taichi Ndarray is unsupported in binary operation")
            return self.variables[node.id]
        elif isinstance(node, ast.Subscript):
            raise NotImplementedError
        else:
            raise TaichiCompilationError(f"Value type {type(node)} is unsupported in binary operation")

    def parse_binary_operation(self, node):
        self.variables[node.targets[0].id] = self._construct_binary_operation_graph(node.value)

    def _construct_shape_argument(self, node):
        assert isinstance(node, ast.Subscript)
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
            shape_argument = IntArgValue(
                arg_type=IntArgValue.Type.SHAPE_VAR,
                shape_var_array=array_var,
                shape_var_dim=node.slice.value
            )
            self.shape_arguments.append(shape_argument)
            return shape_argument
        else:
            raise TaichiCompilationError(f"Subscript is only supported for indexing Taichi Ndarray shapes")

    def parse_shape_assignment(self, node):
        self.variables[node.targets[0].id] = self._construct_shape_argument(node.value)
