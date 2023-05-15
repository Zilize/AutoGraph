#include <zip.h>
#include <taichi/cpp/taichi.hpp>

#include "common.h"
#include "auto_graph.h"

namespace auto_graph {

AutoGraph::AutoGraph(const char *archive_path) {
    std::string graph_json_string = load_graph_json(archive_path);
    auto *graph_json = new GraphContext(graph_json_string.c_str());
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

    char *buffer = new char[st.size];
    zip_file *file = zip_fopen(archive, "auto_graph.json", 0);
    zip_fread(file, buffer, st.size);
    zip_fclose(file);

    return buffer;
}

}  // namespace auto_graph
