#include <zip.h>
#include <taichi/cpp/taichi.hpp>

#include "common.h"
#include "auto_graph.h"

namespace auto_graph {

AutoGraph::AutoGraph(const ti::Runtime &_runtime, const char *archive_path): runtime(const_cast<ti::Runtime *>(&_runtime)),
                     compute_graph(runtime->load_aot_module(archive_path).get_compute_graph("auto_graph")) {
    std::string graph_json_string = load_graph_json(archive_path);
    graph_context = new GraphContext(graph_json_string.c_str());
    for (const auto& graph_argument_context: graph_context->graph_argument_contexts) {
        graph_arguments[graph_argument_context.first] = new GraphArgument(graph_argument_context.second);
    }
}

std::string AutoGraph::load_graph_json(const char *archive_path) {
    zip *archive = zip_open(archive_path, 0, nullptr);
    if (!archive) {
        char err_msg[256];
        std::snprintf(err_msg, sizeof(err_msg), "Cannot open file %s", archive_path);
        throw std::runtime_error(err_msg);
    }

    struct zip_stat st{};
    zip_stat_init(&st);
    zip_stat(archive, "auto_graph.json", 0, &st);

    char *buffer = new char[st.size + 1];
    zip_file *file = zip_fopen(archive, "auto_graph.json", 0);
    zip_fread(file, buffer, st.size);
    zip_fclose(file);
    buffer[st.size] = 0;

    return buffer;
}

// LaunchContext an AutoGraph: 3 steps
// 1. check graph arguments
// 2. allocate arrays
// 3. launch the compiled graphs
void AutoGraph::launch() const {

}

}  // namespace auto_graph
