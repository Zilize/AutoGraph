import taichi as ti
from auto_graph import auto_graph


@auto_graph
def fool_graph(arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    x = 2
    y = 3
    dt = x + y
    dt_arr = ti.ndarray(dtype=ti.f32, shape=arr.shape[0])
    kernel_delta(dt, dt_arr)
    kernel_update(arr, dt_arr)
    kernel_update(arr, dt_arr)


@ti.kernel
def kernel_delta(delta: ti.i32, arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + delta


@ti.kernel
def kernel_update(arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
                  delta: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + delta[i]


if __name__ == '__main__':
    ti.init(arch=ti.vulkan)
    fool_graph.compile()
    fool_graph.archive(ti.vulkan, "auto_graph.tcm")