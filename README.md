# AutoGraph

AutoGraph is a component for automatically compiling and launching Taichi compute graph. This repository provides a prototype implementation for AutoGraph. The roadmap for the entire project is as follows:

- [x] AutoGraph parsing, just-in-time execution in Python environment
- [x] Export meta-data of the AutoGraph, collaborating with existing compute graph components in Taichi 
- [ ] Automatically deploy and launch AutoGraph in C++ environment, using Taichi C-API

You can add the Python decorator `@auto_graph` to the Python function corresponding to the compute graph. The function will be automatically parsed and transformed into internal data structures for just-in-time execution and kernel export. Currently, AutoGraph does not support control flow (e.g., if-else, for-loop, while-loop), while it supports the following basic and useful features:

- Basic operations on integer and MatrixTypes
- Retrieving and utilizing the shapes of the input Taichi Ndarray
- Allocating various types of Taichi Ndarray
- Launching Taichi kernels without return values

You can try out the just-in-time execution in Python environment through the example in `example.py`:

```python
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
    ti.init(arch=ti.cpu, default_ip=ti.i64)
    fool_arr = ti.ndarray(dtype=ti.f32, shape=4)
    fool_graph.compile()
    fool_graph.run({'arr': fool_arr})
    print(fool_arr.to_numpy())
```