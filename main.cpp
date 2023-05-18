#include <iostream>
#include <auto_graph.h>
#include <taichi/cpp/taichi.hpp>

int main() {
    ti::Runtime runtime(TI_ARCH_VULKAN);

    auto arr = runtime.allocate_ndarray<int32_t>({5}, {}, true);

//    ti::AotModule aot_module = runtime.load_aot_module("../auto_graph.tcm");
//    ti::ComputeGraph graph = aot_module.get_compute_graph("auto_graph");
//    graph["kernel_0_arg_0"] = 1;
//    graph["kernel_0_arg_1"] = arr;

    auto_graph::AutoGraph graph(runtime, "../auto_graph.tcm");
    graph["delta"] = 1;
    graph["arr"] = arr;

    graph.launch();
    return 0;
}