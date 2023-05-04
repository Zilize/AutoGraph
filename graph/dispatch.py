import taichi as ti


class Dispatch:
    def __init__(self):
        pass


class Launch(Dispatch):
    def __init__(self):
        super().__init__()


class Allocation(Dispatch):
    def __init__(self):
        super().__init__()
        self.ndim = None
        self.shape = None
        self.dtype = None
