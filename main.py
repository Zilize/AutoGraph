import taichi as ti
from auto_graph import auto_graph


@ti.kernel
def kernel(delta: ti.i32, arr: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + delta


@auto_graph
def graph(delta: ti.i32, arr: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    kernel(delta, arr)


if __name__ == '__main__':
    ti.init(arch=ti.vulkan)

    graph.compile()
    graph.archive(ti.vulkan, "auto_graph.tcm")
