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
        AUTO_GRAPH_ERROR_FORMAT("Cannot open file %s", archive_path);
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
    for (const auto & graph_argument : graph_arguments) {
        if (!graph_argument.second->is_valid()) {
            AUTO_GRAPH_ERROR_FORMAT("Graph argument %s is not assigned", graph_argument.first.c_str());
        }
        if (graph_argument.second->context->argument_type == CONTEXT_INT) {
            if (graph_argument.second->type != I32) {
                AUTO_GRAPH_ERROR_FORMAT("Input argument %s does not match the context", graph_argument.first.c_str());
            }
        }
        else if (graph_argument.second->context->argument_type == CONTEXT_MATRIX) {
            AUTO_GRAPH_ERROR("Matrix is not supported by AutoGraph now");
        }
        else if (graph_argument.second->context->argument_type == CONTEXT_ARRAY) {
            if (graph_argument.second->type != NDARRAY) {
                AUTO_GRAPH_ERROR_FORMAT("Input argument %s does not match the context", graph_argument.first.c_str());
            }
            if (graph_argument.second->context->ndim != graph_argument.second->value.ndarray.shape.dim_count) {
                AUTO_GRAPH_ERROR_FORMAT("The ndim of the input Ndarray %d does not match the context %d",
                                        graph_argument.second->value.ndarray.shape.dim_count,
                                        graph_argument.second->context->ndim);
            }
            if (graph_argument.second->context->n == 0 && graph_argument.second->context->m == 0) {
                if (graph_argument.second->value.ndarray.elem_shape.dim_count != 0) {
                    AUTO_GRAPH_ERROR("The element type of Ndarray is scalar in context but input argument is not");
                }
            }
            else if (graph_argument.second->value.ndarray.elem_shape.dim_count != 2) {
                AUTO_GRAPH_ERROR("The element type only supports scalar and matrix");
            }
            if (graph_argument.second->context->n != graph_argument.second->value.ndarray.elem_shape.dims[0] ||
                graph_argument.second->context->m != graph_argument.second->value.ndarray.elem_shape.dims[1]) {
                AUTO_GRAPH_ERROR("The element shape of input argument does not match the context");
            }
            if (!check_data_type(graph_argument.second->context->data_type,
                                 graph_argument.second->value.ndarray.elem_type)) {
                AUTO_GRAPH_ERROR("The element data type of input argument does not match the context");
            }
        }
    }
}

}  // namespace auto_graph
