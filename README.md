# AutoGraph

## Introduction

AutoGraph is a component for automatically compiling and launching Taichi compute graph. This repository provides a prototype implementation for AutoGraph. The execution logic of AutoGraph is described as follows:

- AutoGraph parsing, just-in-time execution in Python environment
- Export meta-data of the AutoGraph, collaborating with existing compute graph components in Taichi 
- Automatically deploy and launch AutoGraph in C++ environment, using Taichi C-API

You can add the Python decorator `@auto_graph` to the Python function corresponding to the compute graph. The function will be automatically parsed and transformed into internal data structures for just-in-time execution and kernel export. Currently, AutoGraph does not support control flow (e.g., if-else, for-loop, while-loop), while it supports the following basic and useful features:

- Basic operations on integer
- Retrieving and utilizing the shapes of the input Taichi Ndarray
- Allocating various types of Taichi Ndarray
- Launching Taichi kernels without return values

## Datatype

AutoGraph supports 4 types of Taichi data, each with its limitations shown below.

- **Argument** indicates that the data type can be defined as an argument in Python function decorated with `@auto_graph`.
- **Initialization** means that the data type can be defined or initialized in AutoGraph scope.
- **Operation** indicates that the data type can be directly operated in AutoGraph scope, without the need to invoke a Taichi Kernel to perform the operation.

|    DataType    | Scalar (Int32) | Vector | Matrix | Ndarray |
|:--------------:|:--------------:|:------:|:------:|:-------:|
|    Argument    |       ✓        |   ✓    |   ✓    |    ✓    |
| Initialization |       ✓        |   -    |   -    |    ✓    |
|   Operation    |       ✓        |   -    |   -    |    -    |

## Quick Start

```shell
# Clone the repo
git clone --recursive https://github.com/Zilize/AutoGraph
cd AutoGraph

# Install Taichi and Export Taichi module
pip install taichi
python main.py

# Build and launch the project
cmake -B build
cmake --build build
./build/Demo
```

## Example Code

**Python**: Define Taichi Kernel and AutoGraph, and then export Taichi Module.

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
    ti.init(arch=ti.vulkan)
    fool_graph.compile()
    fool_graph.archive(ti.vulkan, "auto_graph.tcm")
```

**C++**: Load Taichi Module and launch the AutoGraph.

```c++
#include <iostream>
#include <auto_graph.h>
#include <taichi/cpp/taichi.hpp>

int main() {
    ti::Runtime runtime(TI_ARCH_VULKAN);
    auto_graph::AutoGraph graph(runtime, "../auto_graph.tcm");

    auto arr = runtime.allocate_ndarray<float>({5}, {}, true);
    graph["arr"] = arr;
    graph.launch();

    auto arr_data = (const float*)arr.map();
    for (int i = 0; i < 5; ++i) std::cout << arr_data[i] << std::endl;
    arr.unmap();

    return 0;
}
```