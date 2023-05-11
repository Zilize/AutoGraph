#include <iostream>
#include <auto_graph.h>
#include <taichi/cpp/taichi.hpp>

int main() {
    auto *graph = new auto_graph::AutoGraph("../auto_graph.tcm");
    return 0;
}