class Allocation:
    def __init__(self, dtype, shape):
        super().__init__()
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)
