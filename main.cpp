#include <iostream>
#include <auto_graph.h>
#include <taichi/cpp/taichi.hpp>

int main() {
    ti::Runtime runtime(TI_ARCH_VULKAN);

    auto c0 = runtime.allocate_ndarray<float>({3, 4}, {}, true);
    auto d0 = runtime.allocate_ndarray<float>({2, 2, 2, 2}, {2, 3}, true);
    auto_graph::AutoGraph graph(runtime, "../auto_graph.tcm");
    graph["a0"] = 1;
    graph["b0"] = 2;
    graph["c0"] = c0;
    graph["d0"] = d0;

    graph.launch();
    return 0;
}