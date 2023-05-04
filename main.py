import taichi as ti
from graph import auto_graph


def demo():
    a = ti.ndarray(dtype=ti.f32, shape=(3, 4))
    b = ti.ScalarNdarray(dtype=ti.f32, arr_shape=(3, 4))
    c = ti.VectorNdarray(n=4, dtype=ti.f32, shape=(3, 4))
    d = ti.MatrixNdarray(n=3, m=3, dtype=ti.f32, shape=(3, 4))

    print("h")


@ti.kernel
def kernel_delta(arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + 1.0


@ti.kernel
def kernel_update(arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
                  delta: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + delta[i]


@auto_graph
def run_sim(i: int, j: ti.i32, x: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    # v = ti.Vector([1, 2, 3])
    s = x.shape[0]
    _width = (2 + (i + j) * 1) * (5 + 6 / 7)
    _width = _width * 1 + 0
    _delta = 1
    __delta = _delta
    width = (_width / _delta)
    delta = ti.ndarray(dtype=ti.f32, shape=(x.shape[0], width))
    kernel_delta(delta)
    kernel_update(x, delta)


@auto_graph
def run_demo(
        i: int,
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        v: ti.types.vector(n=-1, dtype=ti.f32),
        m: ti.types.matrix(n=3, m=3, dtype=ti.f32),
        nm: ti.types.ndarray(dtype=ti.types.vector(n=3, dtype=ti.f32), ndim=2)
):
    delta = ti.ndarray(dtype=ti.f32, shape=i)
    kernel_delta(delta)
    kernel_update(x, delta)


if __name__ == '__main__':
    ti.init(arch=ti.cpu)

    aaa = ti.Vector([1, 2, 3])

    a = ti.Vector([1, 2, 3])
    print(a)
    b = ti.Matrix([[1, 2, 3], [4, 5, 6]])
    c = ti.Matrix([[1, 2, 3], [4, 5, 6]])
    d = b * c
    print(d)
