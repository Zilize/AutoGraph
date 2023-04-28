import ast
import sys
import inspect
import functools
from typing import cast

from taichi.lang.kernel_impl import Kernel
from taichi.lang.exception import (
    TaichiRuntimeError,
    TaichiRuntimeTypeError,
    TaichiSyntaxError
)
from taichi.lang.matrix import MatrixType
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
        raise TaichiSyntaxError(f"Taichi auto graph {fn.__name__} must be defined at the top level")
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
        self.extract_arguments()
        self.global_kernels = {}
        self.extract_kernels()
        self.launches = []
        self.variables = {}
        self.allocated_arrays = []
        self.parse_function_body()

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def extract_arguments(self):
        sig = inspect.signature(self.func)
        if sig.return_annotation not in (inspect.Signature.empty, None):
            raise TaichiSyntaxError("Taichi auto graphs do not support return values")
        params = sig.parameters
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise TaichiSyntaxError("Taichi auto graphs do not support variable keyword parameters (i.e., **kwargs)")
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TaichiSyntaxError("Taichi auto graphs do not support variable positional parameters (i.e., *args)")
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise TaichiSyntaxError("Taichi auto graphs do not support keyword parameters")
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise TaichiSyntaxError('Taichi auto graphs only support "positional or keyword" parameters')
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                raise TaichiSyntaxError(
                    f"Taichi auto graph `{self.func.__name__}` parameter `{arg_name}` must be type annotated"
                )
            else:
                if isinstance(annotation, ndarray_type.NdarrayType):
                    raise NotImplementedError
                elif isinstance(annotation, MatrixType):
                    raise NotImplementedError
                elif id(annotation) in [id(int32), id(int)]:
                    raise NotImplementedError
                else:
                    raise TaichiSyntaxError(f"Invalid type annotation of Taichi auto graph: {annotation}")

    def extract_kernels(self):
        for key, value in self.func.__globals__.items():
            if inspect.isfunction(value) and hasattr(value, '_primal') and isinstance(value._primal, Kernel):
                self.global_kernels[key] = value

    def parse_function_body(self):
        source = inspect.getsource(self.func)
        graph_definition = ast.parse(source).body[0]
        if not isinstance(graph_definition, ast.FunctionDef):
            raise TaichiSyntaxError(f"Taichi auto graph {self.func.__name__} must be defined as a Python function")

        statements = graph_definition.body
        for statement in statements:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                self.parse_kernel_call(statement.value)
            elif isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
                self.parse_ndarray_allocation(statement)
            elif isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Constant):
                pass
            elif isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Name):
                pass
            elif isinstance(statement, ast.Assign) and isinstance(statement.value, ast.BinOp):
                self.parse_integer_calculation(statement)
            else:
                raise TaichiSyntaxError(f"The statement in Taichi auto graph {self.func.__name__} must be assignments "
                                        f"or kernel calls: \"{ast.unparse(cast(ast.AST, statement))}\"")

    def parse_kernel_call(self, node):
        pass

    def parse_ndarray_allocation(self, node):
        pass

    def parse_integer_calculation(self, node):
        pass
