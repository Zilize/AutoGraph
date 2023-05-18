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