import numpy as np
import taichi as ti
from auto_graph import auto_graph

arch = ti.vulkan


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
    ti.init(arch=arch)
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
