#include <iostream>
#include <auto_graph.h>
#include <taichi/cpp/taichi.hpp>

int main() {
    ti::Runtime runtime(TI_ARCH_VULKAN);
//    auto *graph = new auto_graph::AutoGraph(runtime, "../auto_graph.tcm");
    auto_graph::AutoGraph graph(runtime, "../auto_graph.tcm");
//    graph["a0"] = 1;
//    graph["b0"] = 2;
    return 0;
}