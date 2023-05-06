import taichi as ti


class Dispatch:
    def __init__(self):
        pass


class Launch(Dispatch):
    def __init__(self, kernel_fn, args):
        super().__init__()
        self.kernel_fn = kernel_fn
        self.args = args


class Allocation(Dispatch):
    def __init__(self, dtype, shape):
        super().__init__()
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)
