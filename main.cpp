#include <iostream>
#include <auto_graph.h>

int main() {
    std::cout << auto_graph::load_meta_data("../auto_graph.tcm");
    return 0;
}