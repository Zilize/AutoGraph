from enum import Enum
from auto_graph.allocation import Allocation

import taichi as ti
import numpy as np
from taichi.types import int32
from taichi.types.ndarray_type import NdarrayType
from taichi.lang.matrix import MatrixType
from taichi.lang.exception import TaichiRuntimeTypeError, TaichiCompilationError
from taichi.lang._ndarray import Ndarray
from taichi import ScalarNdarray, VectorNdarray, MatrixNdarray, Vector, Matrix


class ArgValue:
    arg_value_buffer = []

    @staticmethod
    def reset_buffer():
        for arg_value in ArgValue.arg_value_buffer:
            arg_value.reset_value()
        ArgValue.arg_value_buffer.clear()

    def __init__(self):
        self.annotation = None
        self.value = None

    def set_annotation(self, annotation):
        self.annotation = annotation

    def set_value(self, value):
        ArgValue.arg_value_buffer.append(self)
        self.value = value

    def get_value(self):
        raise NotImplementedError

    def reset_value(self):
        self.value = None

    def check_match_parameter(self, parameter):
        param_annotation = parameter.annotation
        if self.annotation == int32 and id(param_annotation) in [id(int), id(int32)]:
            return True
        elif isinstance(self.annotation, MatrixType) and isinstance(param_annotation, MatrixType):
            if self.annotation.n == param_annotation.n and self.annotation.m == param_annotation.m and \
                    self.annotation.dtype == param_annotation.dtype:
                return True
            return False
        elif isinstance(self.annotation, NdarrayType) and isinstance(param_annotation, NdarrayType):
            if self.annotation.ndim != param_annotation.ndim:
                return False
            if isinstance(self.annotation.dtype, MatrixType) and isinstance(param_annotation.dtype, MatrixType):
                if self.annotation.dtype.n == param_annotation.dtype.n and \
                        self.annotation.dtype.m == param_annotation.dtype.m and \
                        self.annotation.dtype.dtype == param_annotation.dtype.dtype:
                    return True
                return False
            elif self.annotation.dtype == param_annotation.dtype:
                return True
            else:
                return False

    def check_match_instance(self, instance):
        if self.annotation == int32 and (isinstance(instance, int) or isinstance(instance, int32)):
            return True
        elif isinstance(self.annotation, MatrixType):
            if not isinstance(instance, Matrix) or isinstance(instance, Vector):
                return False
            if self.annotation.n != instance.n or self.annotation.m != instance.m:
                return False
            return True
        elif isinstance(self.annotation, NdarrayType):
            if not isinstance(instance, Ndarray) or isinstance(instance, VectorNdarray):
                return False
            if isinstance(self.annotation.dtype, MatrixType):
                if not isinstance(instance, MatrixNdarray):
                    return False
                if self.annotation.dtype.n == instance.n and self.annotation.dtype.m == instance.m:
                    return True
                else:
                    return False
            else:
                if not isinstance(instance, ScalarNdarray):
                    return False
                return True


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
        self.set_annotation(int32)  # TODO

    def get_value(self):
        ArgValue.arg_value_buffer.append(self)
        if self.arg_type == IntArgValue.Type.CONST:
            self.value = self.const_value
        elif self.arg_type == IntArgValue.Type.BINOP_VAR:
            if self.value is None:
                left_value = self.binop_var_left.get_value()
                right_value = self.binop_var_right.get_value()
                if self.binop_var_op == IntArgValue.Op.ADD:
                    self.value = left_value + right_value
                elif self.binop_var_op == IntArgValue.Op.SUB:
                    self.value = left_value - right_value
                elif self.binop_var_op == IntArgValue.Op.MUL:
                    self.value = left_value * right_value
                elif self.binop_var_op == IntArgValue.Op.DIV:
                    self.value = left_value / right_value
                elif self.binop_var_op == IntArgValue.Op.MOD:
                    self.value = left_value % right_value
        assert self.value is not None
        return self.value

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.arg_type == IntArgValue.Type.CONST:
            return str(self.const_value)
        elif self.arg_type == IntArgValue.Type.GRAPH_VAR:
            return self.graph_var_name
        elif self.arg_type == IntArgValue.Type.SHAPE_VAR:
            return f"{self.shape_var_array.graph_var_name}{{{self.shape_var_dim}}}"
            # return f"arr({id(self.shape_var_array)}).shape[{self.shape_var_dim}]"
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
            assert isinstance(self.const_value, ti.Matrix) and self.const_value.ndim == 2
            self.n = self.const_value.n
            self.m = self.const_value.m
            if self.const_value.entries.dtype in [np.float32, np.float64]:
                self.dtype = ti.f32
            elif self.const_value.entries.dtype in [np.int32, np.int64]:
                self.dtype = ti.i32
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
            else:
                self.n = self.binop_var_left.n
                self.m = self.binop_var_left.m
            if self.binop_var_left.dtype != self.binop_var_right.dtype:
                raise TaichiCompilationError(f"Different primitive types in matrix binary operation: "
                                             f"{self.binop_var_left.dtype} and {self.binop_var_right.dtype}")
            self.dtype = self.binop_var_left.dtype
        self.set_annotation(MatrixType(n=self.n, m=self.m, ndim=2, dtype=self.dtype))

    def get_value(self):
        ArgValue.arg_value_buffer.append(self)
        if self.arg_type == MatrixArgValue.Type.CONST:
            self.value = self.const_value
        elif self.arg_type == MatrixArgValue.Type.BINOP_VAR:
            if self.value is None:
                left_value = self.binop_var_left.get_value()
                right_value = self.binop_var_right.get_value()
                if self.binop_var_op == MatrixArgValue.Op.ADD:
                    self.value = left_value + right_value
                elif self.binop_var_op == MatrixArgValue.Op.SUB:
                    self.value = left_value - right_value
                elif self.binop_var_op == MatrixArgValue.Op.MUL:
                    self.value = left_value * right_value
                elif self.binop_var_op == MatrixArgValue.Op.DIV:
                    self.value = left_value / right_value
                elif self.binop_var_op == MatrixArgValue.Op.MOD:
                    self.value = left_value % right_value
                elif self.binop_var_op == MatrixArgValue.Op.MATMUL:
                    self.value = left_value @ right_value
        assert self.value is not None
        return self.value

    def __repr__(self):
        return self.__str__()
        # return f"MatrixArgValue with Type({self.arg_type.name}), n({self.n}), m({self.m})"

    def __str__(self):
        if self.arg_type == MatrixArgValue.Type.CONST:
            return str(self.const_value.to_list()).replace(" ", "")
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
                 alloc_var=None):
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
        self.set_annotation(NdarrayType(dtype=self.dtype, ndim=self.ndim))

    def get_value(self):
        assert self.value is not None
        return self.value

    def __repr__(self):
        return f"ArrayArgValue with Type({self.arg_type.name}), Id({id(self)})"


if __name__ == '__main__':
    u = ti.types.matrix(n=1, m=2, dtype=ti.f32)
    v = ti.types.matrix(n=2, m=1, dtype=ti.f32)
    w = ti.types.vector(n=3, dtype=ti.f32)

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
