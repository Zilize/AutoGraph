import taichi as ti
from graph import auto_graph


def demo():
    a = ti.ndarray(dtype=ti.f32, shape=(3, 4))
    b = ti.ScalarNdarray(dtype=ti.f32, arr_shape=(3, 4))
    c = ti.VectorNdarray(n=4, dtype=ti.f32, shape=(3, 4))
    d = ti.MatrixNdarray(n=3, m=3, dtype=ti.f32, shape=(3, 4))

    print("h")


@ti.kernel
def kernel_primitive(x: ti.f32, y: ti.f16, z: int):
    a = x + y + z


@ti.kernel
def kernel_delta(delta: ti.i32, arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + delta


@ti.kernel
def kernel_update(arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
                  delta: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + delta[i]


@auto_graph
def fool_graph(arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    x = 2
    y = 3
    dt = x + y
    dt_arr = ti.ndarray(dtype=ti.f32, shape=arr.shape[0])
    kernel_delta(dt, dt_arr)
    kernel_update(arr, dt_arr)
    kernel_update(arr, dt_arr)


# @auto_graph
# def run_shape(a: int, arr: ti.types.ndarray(dtype=ti.f32, ndim=2)):
#     arr_a = ti.ndarray(dtype=ti.f32, shape=(a, arr.shape[0]))
#     x = arr_a.shape[0]
#     y = arr_a.shape[1]
#     arr_b = ti.ndarray(dtype=ti.f32, shape=(x, y))
#     arr_c = ti.ndarray(dtype=ti.f32, shape=(10, arr_b.shape[0], arr_b.shape[1], arr.shape[0], arr.shape[1]))


@auto_graph
def run_sim(i: int, j: ti.i32, x: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    a0 = ti.ndarray(dtype=ti.f32, shape=((i + j) * 5, 4))
    a1 = ti.ScalarNdarray(dtype=ti.f32, arr_shape=(3, 4))
    a2 = ti.MatrixNdarray(n=3, m=3, dtype=ti.f32, shape=(3, 4))
    v1 = ti.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v3 = ti.types.matrix(n=2, m=3, dtype=ti.f32)([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # s = x.shape[0]
    # _width = (2 + (i + j) * 1) * (5 + 6 / 7)
    # _width = _width * 1 + 0
    # _delta = 1
    # __delta = _delta
    # width = (_width / _delta) + __delta
    # delta = ti.ndarray(dtype=ti.f32, shape=(x.shape[0], width))
    # kernel_delta(delta)
    # kernel_update(x, delta)


@ti.kernel
def kernel_types(
        a: ti.i32,
        b: ti.i32,
        d: ti.types.matrix(n=2, m=3, dtype=ti.i32),
        e: ti.types.ndarray(dtype=ti.f32, ndim=2),
        g: ti.types.ndarray(dtype=ti.types.matrix(n=2, m=3, dtype=ti.f32), ndim=4)
):
    i = a + b


@auto_graph
def run_types(
        a0: int,
        b0: ti.i32,
        c0: ti.types.matrix(n=2, m=3, dtype=ti.i32),
        d0: ti.types.ndarray(dtype=ti.f32, ndim=2),
        e0: ti.types.ndarray(dtype=ti.types.matrix(n=2, m=3, dtype=ti.f32), ndim=4)
):
    kernel_types(a0, b0, c0, d0, e0)
    a1 = 1
    b1 = 2
    c1 = ti.types.matrix(n=2, m=3, dtype=ti.i32)([[1, 2, 3], [4, 5, 6]])
    d1 = ti.ScalarNdarray(dtype=ti.f32, arr_shape=(a1, 4))
    e1 = ti.MatrixNdarray(n=2, m=3, dtype=ti.f32, shape=(e0.shape[0], e0.shape[1], e0.shape[2], e0.shape[3]))
    kernel_types(a1, b1, c1, d1, e1)


# @auto_graph
# def run_demo(
#         i: int,
#         x: ti.types.ndarray(dtype=ti.f32, ndim=1),
#         m: ti.types.matrix(n=3, m=3, dtype=ti.f32)
# ):
#     delta = ti.ndarray(dtype=ti.f32, shape=i)
#     kernel_delta(delta)
#     kernel_update(x, delta)


if __name__ == '__main__':
    ti.init(arch=ti.cpu, default_ip=ti.i64)
    fool_arr = ti.ndarray(dtype=ti.f32, shape=4)
    fool_graph.run({'arr': fool_arr})
    print(fool_arr.to_numpy())

    # run_types.run({
    #     'a0': 1,
    #     'b0': 2,
    #     'c0': ti.types.matrix(n=2, m=3, dtype=ti.i32)([[1, 2, 3], [4, 5, 6]]),
    #     'd0': ti.ScalarNdarray(dtype=ti.f32, arr_shape=(3, 4)),
    #     'e0': ti.MatrixNdarray(n=2, m=3, dtype=ti.f32, shape=(11, 22, 33, 44))
    # })
