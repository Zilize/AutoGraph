from enum import Enum
from graph.dispatch import Launch, Allocation

import taichi as ti
from taichi.graph import ArgKind
from taichi.types import primitive_types
from taichi.lang.exception import TaichiRuntimeTypeError, TaichiCompilationError


class ArgValue:
    def __init__(self):
        pass


class IntArgValue(ArgValue):
    class Type(Enum):
        CONST = 1      # x = 1
        GRAPH_VAR = 2  # x: ti.i32
        SHAPE_VAR = 3  # x = arr.shape[0]
        BINOP_VAR = 4  # x = x + 1

    class Op(Enum):
        ADD = 1
        SUB = 2
        MUL = 3
        DIV = 4
        MOD = 5

    def __init__(self,
                 arg_type: Type,
                 const_value=None,
                 graph_var_name=None,
                 shape_var_array=None,
                 shape_var_dim=None,
                 binop_var_left=None,
                 binop_var_op=None,
                 binop_var_right=None
                 ):
        super().__init__()
        self.arg_type = arg_type
        assert isinstance(self.arg_type, IntArgValue.Type)
        if arg_type == IntArgValue.Type.CONST:
            if const_value is None:
                raise TaichiCompilationError(f"Argument const_value should not be None with Type.CONST")
            self.const_value = const_value
            assert isinstance(const_value, int)
        elif arg_type == IntArgValue.Type.GRAPH_VAR:
            if graph_var_name is None:
                raise TaichiCompilationError(f"Argument var_name should not be None with Type.GRAPH_VAR")
            self.graph_var_name = graph_var_name
        elif arg_type == IntArgValue.Type.SHAPE_VAR:
            if shape_var_array is None or shape_var_dim is None:
                raise TaichiCompilationError(f"Argument shape_var_array and shape_var_dim should not be None with "
                                             f"Type.SHAPE_VAR")
            self.shape_var_array = shape_var_array
            self.shape_var_dim = shape_var_dim
            assert isinstance(self.shape_var_array, ArrayArgValue)
            if self.shape_var_dim < 0 or self.shape_var_dim >= self.shape_var_array.ndim:
                raise TaichiRuntimeTypeError(f"The index of shape is out of range")
        elif arg_type == IntArgValue.Type.BINOP_VAR:
            if binop_var_left is None or binop_var_op is None or binop_var_right is None:
                raise TaichiCompilationError(f"Argument binop_var_left, binop_var_op and binop_var_right should not "
                                             f"be None with Type.BINOP_VAR")
            self.binop_var_left = binop_var_left
            self.binop_var_op = binop_var_op
            self.binop_var_right = binop_var_right
            assert isinstance(self.binop_var_op, IntArgValue.Op)

    def __repr__(self):
        return f"IntArgValue with Type({self.arg_type.name})"

    def __str__(self):
        if self.arg_type == IntArgValue.Type.CONST:
            return str(self.const_value)
        elif self.arg_type == IntArgValue.Type.GRAPH_VAR:
            return self.graph_var_name
        elif self.arg_type == IntArgValue.Type.SHAPE_VAR:
            return f"arr({id(self.shape_var_array)}).shape[{self.shape_var_dim}]"
        elif self.arg_type == IntArgValue.Type.BINOP_VAR:
            op = None
            if self.binop_var_op == IntArgValue.Op.ADD:
                op = '+'
            elif self.binop_var_op == IntArgValue.Op.SUB:
                op = '-'
            elif self.binop_var_op == IntArgValue.Op.MUL:
                op = '*'
            elif self.binop_var_op == IntArgValue.Op.DIV:
                op = '/'
            elif self.binop_var_op == IntArgValue.Op.MOD:
                op = '%'
            return f"({str(self.binop_var_left)}{op}{str(self.binop_var_right)})"

    def __add__(self, other):
        if isinstance(other, IntArgValue):
            return IntArgValue(IntArgValue.Type.BINOP_VAR,
                               binop_var_left=self,
                               binop_var_op=IntArgValue.Op.ADD,
                               binop_var_right=other)
        raise TaichiCompilationError(f"Unsupported operand types for '+': IntArgValue and {type(other).__name__}")

    def __sub__(self, other):
        if isinstance(other, IntArgValue):
            return IntArgValue(IntArgValue.Type.BINOP_VAR,
                               binop_var_left=self,
                               binop_var_op=IntArgValue.Op.SUB,
                               binop_var_right=other)
        raise TaichiCompilationError(f"Unsupported operand types for '-': IntArgValue and {type(other).__name__}")

    def __mul__(self, other):
        if isinstance(other, IntArgValue):
            return IntArgValue(IntArgValue.Type.BINOP_VAR,
                               binop_var_left=self,
                               binop_var_op=IntArgValue.Op.MUL,
                               binop_var_right=other)
        raise TaichiCompilationError(f"Unsupported operand types for '*': IntArgValue and {type(other).__name__}")

    def __truediv__(self, other):
        if isinstance(other, IntArgValue):
            return IntArgValue(IntArgValue.Type.BINOP_VAR,
                               binop_var_left=self,
                               binop_var_op=IntArgValue.Op.DIV,
                               binop_var_right=other)
        raise TaichiCompilationError(f"Unsupported operand types for '/': IntArgValue and {type(other).__name__}")

    def __mod__(self, other):
        if isinstance(other, IntArgValue):
            return IntArgValue(IntArgValue.Type.BINOP_VAR,
                               binop_var_left=self,
                               binop_var_op=IntArgValue.Op.MOD,
                               binop_var_right=other)
        raise TaichiCompilationError(f"Unsupported operand types for '%': IntArgValue and {type(other).__name__}")


class MatrixArgValue(ArgValue):
    class Type(Enum):
        CONST = 1      # x = [[1, 2], [3, 4]]
        GRAPH_VAR = 2  # x: ti.types.matrix(n=2, m=2, dtype=ti.i32)
        BINOP_VAR = 3  # x = x + [[1, 1], [1, 1]]

    class Op(Enum):
        ADD = 1
        SUB = 2
        MUL = 3
        DIV = 4
        MOD = 5
        MATMUL = 6

    def __init__(self,
                 arg_type: Type,
                 const_value=None,
                 graph_var_name=None,
                 graph_var_n=None,
                 graph_var_m=None,
                 graph_var_dtype=None,
                 binop_var_left=None,
                 binop_var_op=None,
                 binop_var_right=None
                 ):
        super().__init__()
        self.arg_type = arg_type
        assert isinstance(self.arg_type, MatrixArgValue.Type)
        if arg_type == MatrixArgValue.Type.CONST:
            if const_value is None:
                raise TaichiCompilationError(f"Argument const_value should not be None with Type.CONST")
            self.const_value = const_value
            assert isinstance(self.const_value, ti.Matrix)
            self.n = self.const_value.n
            self.m = self.const_value.m
            self.dtype = None
        elif arg_type == MatrixArgValue.Type.GRAPH_VAR:
            if graph_var_name is None or graph_var_n is None or graph_var_m is None or graph_var_dtype is None:
                raise TaichiCompilationError(f"Argument graph_var_name, graph_var_n, graph_var_m, graph_var_dtype "
                                             f"should not be None with Type.GRAPH_VAR")
            self.graph_var_name = graph_var_name
            self.n = graph_var_n
            self.m = graph_var_m
            self.dtype = graph_var_dtype
        elif arg_type == MatrixArgValue.Type.BINOP_VAR:
            if binop_var_left is None or binop_var_op is None or binop_var_right is None:
                raise TaichiCompilationError(f"Argument binop_var_left, binop_var_op and binop_var_right should not "
                                             f"be None with Type.BINOP_VAR")
            self.binop_var_left = binop_var_left
            self.binop_var_op = binop_var_op
            self.binop_var_right = binop_var_right
            assert isinstance(self.binop_var_op, MatrixArgValue.Op)
            if self.binop_var_op == MatrixArgValue.Op.MATMUL:
                if self.binop_var_left.m != self.binop_var_right.n:
                    raise TaichiCompilationError(f"Cannot perform matmul operations on matrices with shapes:"
                                                 f"({self.binop_var_left.n}, {self.binop_var_left.m}) and "
                                                 f"({self.binop_var_right.n}, {self.binop_var_right.m})")
            elif self.binop_var_left.n != self.binop_var_right.n or self.binop_var_left.m != self.binop_var_right.m:
                raise TaichiCompilationError(f"Cannot perform element-wise operations on matrices with different "
                                             f"shapes:({self.binop_var_left.n}, {self.binop_var_left.m}) and "
                                             f"({self.binop_var_right.n}, {self.binop_var_right.m})")
            if self.binop_var_op == MatrixArgValue.Op.MATMUL:
                self.n = self.binop_var_left.n
                self.m = self.binop_var_right.m
                self.dtype = None
            else:
                self.n = self.binop_var_left.n
                self.m = self.binop_var_left.m
                self.dtype = None

    def __repr__(self):
        return f"MatrixArgValue with Type({self.arg_type.name}), n({self.n}), m({self.m})"

    def __str__(self):
        if self.arg_type == MatrixArgValue.Type.CONST:
            return str(self.const_value.to_list())
        elif self.arg_type == MatrixArgValue.Type.GRAPH_VAR:
            return self.graph_var_name
        elif self.arg_type == MatrixArgValue.Type.BINOP_VAR:
            op = None
            if self.binop_var_op == MatrixArgValue.Op.ADD:
                op = '+'
            elif self.binop_var_op == MatrixArgValue.Op.SUB:
                op = '-'
            elif self.binop_var_op == MatrixArgValue.Op.MUL:
                op = '*'
            elif self.binop_var_op == MatrixArgValue.Op.DIV:
                op = '/'
            elif self.binop_var_op == MatrixArgValue.Op.MOD:
                op = '%'
            elif self.binop_var_op == MatrixArgValue.Op.MATMUL:
                op = '@'
            return f"({str(self.binop_var_left)}{op}{str(self.binop_var_right)})"

    def __add__(self, other):
        if isinstance(other, MatrixArgValue):
            return MatrixArgValue(MatrixArgValue.Type.BINOP_VAR,
                                  binop_var_left=self,
                                  binop_var_op=MatrixArgValue.Op.ADD,
                                  binop_var_right=other)
        raise TaichiCompilationError(f"Unsupported operand types for '+': MatrixArgValue and {type(other).__name__}")

    def __sub__(self, other):
        if isinstance(other, MatrixArgValue):
            return MatrixArgValue(MatrixArgValue.Type.BINOP_VAR,
                                  binop_var_left=self,
                                  binop_var_op=MatrixArgValue.Op.SUB,
                                  binop_var_right=other)
        raise TaichiCompilationError(f"Unsupported operand types for '-': MatrixArgValue and {type(other).__name__}")

    def __mul__(self, other):
        if isinstance(other, MatrixArgValue):
            return MatrixArgValue(MatrixArgValue.Type.BINOP_VAR,
                                  binop_var_left=self,
                                  binop_var_op=MatrixArgValue.Op.MUL,
                                  binop_var_right=other)
        raise TaichiCompilationError(f"Unsupported operand types for '*': MatrixArgValue and {type(other).__name__}")

    def __truediv__(self, other):
        if isinstance(other, MatrixArgValue):
            return MatrixArgValue(MatrixArgValue.Type.BINOP_VAR,
                                  binop_var_left=self,
                                  binop_var_op=MatrixArgValue.Op.DIV,
                                  binop_var_right=other)
        raise TaichiCompilationError(f"Unsupported operand types for '/': MatrixArgValue and {type(other).__name__}")

    def __mod__(self, other):
        if isinstance(other, MatrixArgValue):
            return MatrixArgValue(MatrixArgValue.Type.BINOP_VAR,
                                  binop_var_left=self,
                                  binop_var_op=MatrixArgValue.Op.MOD,
                                  binop_var_right=other)
        raise TaichiCompilationError(f"Unsupported operand types for '%': MatrixArgValue and {type(other).__name__}")

    def __matmul__(self, other):
        if isinstance(other, MatrixArgValue):
            return MatrixArgValue(MatrixArgValue.Type.BINOP_VAR,
                                  binop_var_left=self,
                                  binop_var_op=MatrixArgValue.Op.MATMUL,
                                  binop_var_right=other)
        raise TaichiCompilationError(f"Unsupported operand types for '@': MatrixArgValue and {type(other).__name__}")


class ArrayArgValue(ArgValue):
    class Type(Enum):
        GRAPH_VAR = 1
        ALLOC_VAR = 2

    def __init__(self,
                 arg_type: Type,
                 graph_var_name=None,
                 graph_var_ndim=None,
                 graph_var_dtype=None,
                 alloc_var=None,
                 alias_var=None):
        super().__init__()
        self.arg_type = arg_type
        assert isinstance(self.arg_type, ArrayArgValue.Type)
        if arg_type == ArrayArgValue.Type.GRAPH_VAR:
            if graph_var_name is None or graph_var_ndim is None or graph_var_dtype is None:
                raise TaichiCompilationError(f"Argument graph_var_name, graph_var_ndim, graph_var_dtype should not be "
                                             f"None with Type.GRAPH_VAR")
            self.graph_var_name = graph_var_name
            self.shape = None
            self.ndim = graph_var_ndim
            self.dtype = graph_var_dtype
        elif arg_type == ArrayArgValue.Type.ALLOC_VAR:
            if alloc_var is None:
                raise TaichiCompilationError(f"Argument alloc_var should not be None with Type.ALLOC_VAR")
            self.alloc_var = alloc_var
            assert isinstance(self.alloc_var, Allocation)
            self.shape = self.alloc_var.shape
            self.ndim = self.alloc_var.ndim
            self.dtype = self.alloc_var.dtype

    def __repr__(self):
        return f"ArrayArgValue with Type({self.arg_type.name})"


if __name__ == '__main__':
    a = IntArgValue(IntArgValue.Type.CONST, const_value=1)
    b = IntArgValue(IntArgValue.Type.CONST, const_value=2)
    x = IntArgValue(IntArgValue.Type.GRAPH_VAR, graph_var_name="x")
    y = IntArgValue(IntArgValue.Type.SHAPE_VAR,
                    shape_var_array=ArrayArgValue(ArrayArgValue.Type.GRAPH_VAR,
                                                  graph_var_name='arr',
                                                  graph_var_ndim=2,
                                                  graph_var_dtype=ti.f32),
                    shape_var_dim=1)
    t0 = (a + x) / (b * y)
    t1 = t0 + a
    print(t0)
    print(t1)

    mat_a = ti.Matrix([[1, 2], [3, 4]])
    mat_b = ti.Matrix([[1, 1], [1, 1]])
    mat_c = ti.Matrix([[1, 1, 1], [2, 2, 2]])
    a = MatrixArgValue(MatrixArgValue.Type.CONST, const_value=mat_a)
    b = MatrixArgValue(MatrixArgValue.Type.CONST, const_value=mat_b)
    c = MatrixArgValue(MatrixArgValue.Type.CONST, const_value=mat_c)
    x = MatrixArgValue(MatrixArgValue.Type.GRAPH_VAR, graph_var_name='x', graph_var_n=2, graph_var_m=2, graph_var_dtype=ti.f32)
    t = (a + b * x) @ c
    print(t)
