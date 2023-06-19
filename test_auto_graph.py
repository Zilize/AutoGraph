import numpy as np
import taichi as ti
from auto_graph import auto_graph

arch = ti.vulkan
ti.init(arch=arch)


@ti.kernel
def kernel_types(
        a: ti.i32,
        b: ti.types.vector(n=3, dtype=ti.i32),
        c: ti.types.matrix(n=3, m=3, dtype=ti.i32),
        d: ti.types.ndarray(dtype=ti.i32, ndim=2),
        e: ti.types.ndarray(dtype=ti.types.vector(n=3, dtype=ti.i32), ndim=2),
        f: ti.types.ndarray(dtype=ti.types.matrix(n=3, m=3, dtype=ti.i32), ndim=2)
):
    for i in ti.grouped(d):
        d[i] = d[i] + a
    for i in ti.grouped(e):
        e[i] = e[i] + b
    for i in ti.grouped(f):
        f[i] = f[i] + c


@auto_graph
def graph_types(
        a: ti.i32,
        b: ti.types.vector(n=3, dtype=ti.i32),
        c: ti.types.matrix(n=3, m=3, dtype=ti.i32),
        d: ti.types.ndarray(dtype=ti.i32, ndim=2),
        e: ti.types.ndarray(dtype=ti.types.vector(n=3, dtype=ti.i32), ndim=2),
        f: ti.types.ndarray(dtype=ti.types.matrix(n=3, m=3, dtype=ti.i32), ndim=2)
):
    kernel_types(a, b, c, d, e, f)


def test_types():
    a = 1
    b = ti.Vector(np.ones(3), dt=ti.i32)
    c = ti.Matrix(np.ones((3, 3)), dt=ti.i32)
    d = ti.ndarray(dtype=ti.i32, shape=(4, 4))
    e = ti.ndarray(dtype=ti.types.vector(n=3, dtype=ti.i32), shape=(4, 4))
    f = ti.ndarray(dtype=ti.types.matrix(n=3, m=3, dtype=ti.i32), shape=(4, 4))
    graph_types.compile()
    graph_types.run({"a": a, "b": b, "c": c, "d": d, "e": e, "f": f})
    assert d.to_numpy().dtype == np.int32
    assert e.to_numpy().dtype == np.int32
    assert f.to_numpy().dtype == np.int32

    np_d = np.ones((4, 4), dtype=np.int32)
    np_e = np.ones((4, 4, 3), dtype=np.int32)
    np_f = np.ones((4, 4, 3, 3), dtype=np.int32)
    assert np.array_equal(d.to_numpy(), np_d)
    assert np.array_equal(e.to_numpy(), np_e)
    assert np.array_equal(f.to_numpy(), np_f)


@ti.kernel
def kernel_delta_scalar(delta: ti.i32, arr: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + delta


@ti.kernel
def kernel_scalar(delta_arr: ti.types.ndarray(dtype=ti.i32, ndim=2), arr: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + delta_arr[i]


@auto_graph
def graph_scalar_0(delta: ti.i32, arr: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    delta_arr = ti.ndarray(dtype=ti.i32, shape=(arr.shape[0], arr.shape[1]))
    kernel_delta_scalar(delta, delta_arr)
    kernel_scalar(delta_arr, arr)


@auto_graph
def graph_scalar_1(delta: ti.i32, arr: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    delta_arr = ti.ScalarNdarray(dtype=ti.i32, arr_shape=(arr.shape[0], arr.shape[1]))
    kernel_delta_scalar(delta, delta_arr)
    kernel_scalar(delta_arr, arr)


def test_scalar():
    delta = 1
    arr = ti.ndarray(dtype=ti.i32, shape=(3, 4))

    graph_scalar_0.compile()
    graph_scalar_0.run({"delta": delta, "arr": arr})

    np_arr = np.ones((3, 4), dtype=np.int32)
    assert np.array_equal(arr.to_numpy(), np_arr)

    delta = 1
    arr = ti.ndarray(dtype=ti.i32, shape=(3, 4))

    graph_scalar_1.compile()
    graph_scalar_1.run({"delta": delta, "arr": arr})

    np_arr = np.ones((3, 4), dtype=np.int32)
    assert np.array_equal(arr.to_numpy(), np_arr)


@ti.kernel
def kernel_delta_vector(delta: ti.types.vector(n=3, dtype=ti.f32),
                        arr: ti.types.ndarray(dtype=ti.types.vector(n=3, dtype=ti.f32), ndim=2)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + delta


@ti.kernel
def kernel_vector(delta_arr: ti.types.ndarray(dtype=ti.types.vector(n=3, dtype=ti.f32), ndim=2),
                  arr: ti.types.ndarray(dtype=ti.types.vector(n=3, dtype=ti.f32), ndim=2)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + delta_arr[i]


@auto_graph
def graph_vector_0(delta: ti.types.vector(n=3, dtype=ti.f32),
                   arr: ti.types.ndarray(dtype=ti.types.vector(n=3, dtype=ti.f32), ndim=2)):
    delta_arr = ti.ndarray(dtype=ti.types.vector(n=3, dtype=ti.f32), shape=(arr.shape[0], arr.shape[1]))
    kernel_delta_vector(delta, delta_arr)
    kernel_vector(delta_arr, arr)


@auto_graph
def graph_vector_1(delta: ti.types.vector(n=3, dtype=ti.f32),
                   arr: ti.types.ndarray(dtype=ti.types.vector(n=3, dtype=ti.f32), ndim=2)):
    delta_arr = ti.VectorNdarray(n=3, dtype=ti.f32, shape=(arr.shape[0], arr.shape[1]))
    kernel_delta_vector(delta, delta_arr)
    kernel_vector(delta_arr, arr)


@auto_graph
def graph_vector_2(delta: ti.types.vector(n=3, dtype=ti.f32),
                   arr: ti.types.ndarray(dtype=ti.types.vector(n=3, dtype=ti.f32), ndim=2)):
    delta_arr = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=(arr.shape[0], arr.shape[1]))
    kernel_delta_vector(delta, delta_arr)
    kernel_vector(delta_arr, arr)


def test_vector():
    delta = ti.types.vector(n=3, dtype=ti.f32)([1.0, 1.0, 1.0])
    arr = ti.ndarray(dtype=ti.types.vector(n=3, dtype=ti.f32), shape=(3, 4))
    graph_vector_0.compile()
    graph_vector_0.run({"delta": delta, "arr": arr})
    np_arr = np.ones((3, 4, 3), dtype=np.float32)
    assert np.array_equal(arr.to_numpy(), np_arr)

    delta = ti.types.vector(n=3, dtype=ti.f32)([1.0, 1.0, 1.0])
    arr = ti.ndarray(dtype=ti.types.vector(n=3, dtype=ti.f32), shape=(3, 4))
    graph_vector_1.compile()
    graph_vector_1.run({"delta": delta, "arr": arr})
    np_arr = np.ones((3, 4, 3), dtype=np.float32)
    assert np.array_equal(arr.to_numpy(), np_arr)

    delta = ti.types.vector(n=3, dtype=ti.f32)([1.0, 1.0, 1.0])
    arr = ti.ndarray(dtype=ti.types.vector(n=3, dtype=ti.f32), shape=(3, 4))
    graph_vector_2.compile()
    graph_vector_2.run({"delta": delta, "arr": arr})
    np_arr = np.ones((3, 4, 3), dtype=np.float32)
    assert np.array_equal(arr.to_numpy(), np_arr)


@ti.kernel
def kernel_delta_matrix(delta: ti.types.matrix(n=2, m=2, dtype=ti.f32),
                        arr: ti.types.ndarray(dtype=ti.types.matrix(n=2, m=2, dtype=ti.f32), ndim=2)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + delta


@ti.kernel
def kernel_matrix(delta_arr: ti.types.ndarray(dtype=ti.types.matrix(n=2, m=2, dtype=ti.f32), ndim=2),
                  arr: ti.types.ndarray(dtype=ti.types.matrix(n=2, m=2, dtype=ti.f32), ndim=2)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + delta_arr[i]


@auto_graph
def graph_matrix_0(delta: ti.types.matrix(n=2, m=2, dtype=ti.f32),
                   arr: ti.types.ndarray(dtype=ti.types.matrix(n=2, m=2, dtype=ti.f32), ndim=2)):
    delta_arr = ti.ndarray(dtype=ti.types.matrix(n=2, m=2, dtype=ti.f32), shape=(arr.shape[0], arr.shape[1]))
    kernel_delta_matrix(delta, delta_arr)
    kernel_matrix(delta_arr, arr)


@auto_graph
def graph_matrix_1(delta: ti.types.matrix(n=2, m=2, dtype=ti.f32),
                   arr: ti.types.ndarray(dtype=ti.types.matrix(n=2, m=2, dtype=ti.f32), ndim=2)):
    delta_arr = ti.MatrixNdarray(n=2, m=2, dtype=ti.f32, shape=(arr.shape[0], arr.shape[1]))
    kernel_delta_matrix(delta, delta_arr)
    kernel_matrix(delta_arr, arr)


@auto_graph
def graph_matrix_2(delta: ti.types.matrix(n=2, m=2, dtype=ti.f32),
                   arr: ti.types.ndarray(dtype=ti.types.matrix(n=2, m=2, dtype=ti.f32), ndim=2)):
    delta_arr = ti.Matrix.ndarray(n=2, m=2, dtype=ti.f32, shape=(arr.shape[0], arr.shape[1]))
    kernel_delta_matrix(delta, delta_arr)
    kernel_matrix(delta_arr, arr)


def test_matrix():
    delta = ti.types.matrix(n=2, m=2, dtype=ti.f32)([[1.0, 1.0], [1.0, 1.0]])
    arr = ti.ndarray(dtype=ti.types.matrix(n=2, m=2, dtype=ti.f32), shape=(3, 4))
    graph_matrix_0.compile()
    graph_matrix_0.run({"delta": delta, "arr": arr})
    np_arr = np.ones((3, 4, 2, 2), dtype=np.float32)
    assert np.array_equal(arr.to_numpy(), np_arr)

    delta = ti.types.matrix(n=2, m=2, dtype=ti.f32)([[1.0, 1.0], [1.0, 1.0]])
    arr = ti.ndarray(dtype=ti.types.matrix(n=2, m=2, dtype=ti.f32), shape=(3, 4))
    graph_matrix_1.compile()
    graph_matrix_1.run({"delta": delta, "arr": arr})
    np_arr = np.ones((3, 4, 2, 2), dtype=np.float32)
    assert np.array_equal(arr.to_numpy(), np_arr)

    delta = ti.types.matrix(n=2, m=2, dtype=ti.f32)([[1.0, 1.0], [1.0, 1.0]])
    arr = ti.ndarray(dtype=ti.types.matrix(n=2, m=2, dtype=ti.f32), shape=(3, 4))
    graph_matrix_2.compile()
    graph_matrix_2.run({"delta": delta, "arr": arr})
    np_arr = np.ones((3, 4, 2, 2), dtype=np.float32)
    assert np.array_equal(arr.to_numpy(), np_arr)


@ti.kernel
def kernel_tuple(delta: ti.i32, arr: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + delta


@auto_graph
def graph_tuple(arr: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    x, y = arr.shape[0], arr.shape[1]
    kernel_tuple(x + y, arr)
    z = x + y
    kernel_tuple(z, arr)
    delta = z
    kernel_tuple(delta, arr)


def test_tuple():
    arr = ti.ndarray(dtype=ti.i32, shape=(3, 4))
    graph_tuple.compile()
    graph_tuple.run({"arr": arr})
    np_arr = np.ones((3, 4), dtype=np.int32) * 21
    assert np.array_equal(arr.to_numpy(), np_arr)
