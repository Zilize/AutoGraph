from enum import Enum

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
        ALIAS_VAR = 4  # x = x_0
        BINOP_VAR = 5  # x = x + 1

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
                 alias_var=None,
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
        elif arg_type == IntArgValue.Type.ALIAS_VAR:
            if alias_var is None:
                raise TaichiCompilationError(f"Argument alias_var should not be None with Type.ALIAS_VAR")
            self.alias_var = alias_var
        elif arg_type == IntArgValue.Type.BINOP_VAR:
            if binop_var_left is None or binop_var_op is None or binop_var_right is None:
                raise TaichiCompilationError(f"Argument binop_var_left, binop_var_op and binop_var_right should not "
                                             f"be None with Type.BINOP_VAR")
            self.binop_var_left = binop_var_left
            self.binop_var_op = binop_var_op
            self.binop_var_right = binop_var_right
            assert isinstance(self.binop_var_op, IntArgValue.Op)

    def __repr__(self):
        return f"IntArgValue with Type: {self.arg_type.name}"

    def __str__(self):
        if self.arg_type == IntArgValue.Type.CONST:
            return str(self.const_value)
        elif self.arg_type == IntArgValue.Type.GRAPH_VAR:
            return self.graph_var_name
        elif self.arg_type == IntArgValue.Type.SHAPE_VAR:
            return f"arr({id(self.shape_var_array)}).shape[{self.shape_var_dim}]"
        elif self.arg_type == IntArgValue.Type.ALIAS_VAR:
            return str(self.alias_var)
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
        pass

    def __sub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __mod__(self, other):
        pass


class MatrixArgValue(ArgValue):
    def __init__(self, n, m, dtype):
        super().__init__()
        self.n = n
        self.m = m
        self.dtype = dtype
        if id(self.dtype) not in primitive_types.type_ids:
            raise TaichiRuntimeTypeError(f"Invalid dtype {self.dtype} of Taichi matrix")


class ArrayArgValue(ArgValue):
    def __init__(self, ndim):
        super().__init__()
        self.ndim = ndim


if __name__ == '__main__':
    a = IntArgValue(IntArgValue.Type.CONST, const_value=1)
