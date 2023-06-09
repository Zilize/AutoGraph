#include <taichi/cpp/taichi.hpp>

#include "zip.h"
#include "common.h"
#include "auto_graph.h"

namespace auto_graph {

AutoGraph::AutoGraph(const ti::Runtime &_runtime, const char *archive_path): runtime(const_cast<ti::Runtime *>(&_runtime)) {
    aot_module = runtime->load_aot_module(archive_path);
    compute_graph = aot_module.get_compute_graph("auto_graph");

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

void AutoGraph::check_graph_arguments() const {
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

int AutoGraph::integer_evaluation(const std::string &expression) const {
    std::smatch match;
    if (std::regex_match(expression, match, operation_pattern)) {
        int left = integer_evaluation(match[1].str());
        int right = integer_evaluation(match[3].str());
        char op = match[2].str()[0];
        switch (op) {
            case '+': return left + right;
            case '-': return left - right;
            case '*': return left * right;
            case '/': return left / right;
            case '%': return left % right;
            default: AUTO_GRAPH_ERROR_FORMAT("Invalid operation %c", op);
        }
    }
    else if (std::regex_match(expression, match, integer_pattern)) {
        int value = std::stoi(match[1]);
        return value;
    }
    else if (std::regex_match(expression, match, argument_pattern)) {
        std::string argument_name = match[1].str();
        auto it = graph_arguments.find(argument_name);
        if (it == graph_arguments.end()) {
            AUTO_GRAPH_ERROR_FORMAT("Graph argument %s not found", argument_name.c_str());
        }
        if (it->second->type != I32) {
            AUTO_GRAPH_ERROR_FORMAT("Graph argument %s is not an integer", argument_name.c_str());
        }
        return it->second->value.i32;
    }
    else if (std::regex_match(expression, match, shape_pattern)) {
        std::string argument_name = match[1].str();
        auto it = graph_arguments.find(argument_name);
        if (it == graph_arguments.end()) {
            AUTO_GRAPH_ERROR_FORMAT("Graph argument %s not found", argument_name.c_str());
        }
        if (it->second->type != NDARRAY) {
            AUTO_GRAPH_ERROR_FORMAT("Graph argument %s is not an Ndarray", argument_name.c_str());
        }
        return (int)it->second->value.ndarray.shape.dims[std::stoi(match[2].str())];  // uint32_t to int
    }
    else {
        AUTO_GRAPH_ERROR_FORMAT("The expression %s does not match any regex", expression.c_str());
    }
}

void AutoGraph::allocate_arrays() {
    for (const auto& allocated_array_context: graph_context->allocated_array_contexts) {
        std::vector<uint32_t> shape, elem_shape;
        for (const auto &shape_str: *(allocated_array_context.second->shape)) {
            shape.emplace_back((uint32_t) integer_evaluation(shape_str));
        }
        if (allocated_array_context.second->n != 0 && allocated_array_context.second->m != 0) {
            elem_shape.emplace_back(allocated_array_context.second->n);
            elem_shape.emplace_back(allocated_array_context.second->m);
        }
        const auto &array_name = allocated_array_context.first;
        const auto &data_type = allocated_array_context.second->data_type;
        if (data_type == CONTEXT_I8) {
            allocated_arrays[array_name] = runtime->allocate_ndarray<int8_t>(shape, elem_shape, true);
        }
        else if (data_type == CONTEXT_I16) {
            allocated_arrays[array_name] = runtime->allocate_ndarray<int16_t>(shape, elem_shape, true);
        }
        else if (data_type == CONTEXT_I32) {
            allocated_arrays[array_name] = runtime->allocate_ndarray<int32_t>(shape, elem_shape, true);
        }
        else if (data_type == CONTEXT_I64) {
            AUTO_GRAPH_ERROR("Ndarray allocation for I64 not supported yet");
        }
        else if (data_type == CONTEXT_F16) {
            AUTO_GRAPH_ERROR("Ndarray allocation for F16 not supported yet");
        }
        else if (data_type == CONTEXT_F32) {
            allocated_arrays[array_name] = runtime->allocate_ndarray<float>(shape, elem_shape, true);
        }
        else if (data_type == CONTEXT_F64) {
            allocated_arrays[array_name] = runtime->allocate_ndarray<double>(shape, elem_shape, true);
        }
        else if (data_type == CONTEXT_U8) {
            allocated_arrays[array_name] = runtime->allocate_ndarray<uint8_t>(shape, elem_shape, true);
        }
        else if (data_type == CONTEXT_U16) {
            allocated_arrays[array_name] = runtime->allocate_ndarray<uint16_t>(shape, elem_shape, true);
        }
        else if (data_type == CONTEXT_U32) {
            allocated_arrays[array_name] = runtime->allocate_ndarray<uint32_t>(shape, elem_shape, true);
        }
        else if (data_type == CONTEXT_U64) {
            AUTO_GRAPH_ERROR("Ndarray allocation for U64 not supported yet");
        }
    }
}

// Launch an AutoGraph: 3 steps
// 1. check graph arguments
// 2. allocate arrays
// 3. launch the compiled graphs
void AutoGraph::launch() {
    check_graph_arguments();
    allocate_arrays();
    for (const auto &launch_context: graph_context->launch_contexts) {
        const auto &argument_name = launch_context.first;
        if (launch_context.second->argument_type == CONTEXT_INT) {
            compute_graph[argument_name] = integer_evaluation(launch_context.second->value);
        }
        else if (launch_context.second->argument_type == CONTEXT_MATRIX) {
            AUTO_GRAPH_ERROR("Matrix not supported by compute graph launch yet");
        }
        else if (launch_context.second->argument_type == CONTEXT_ARRAY) {
            if (graph_arguments.find(launch_context.second->value) != graph_arguments.end()) {
                compute_graph[argument_name] = graph_arguments[launch_context.second->value]->value.ndarray;
            }
            else if (allocated_arrays.find(launch_context.second->value) != allocated_arrays.end()) {
                compute_graph[argument_name] = allocated_arrays[launch_context.second->value];
            }
            else {
                AUTO_GRAPH_ERROR_FORMAT("Ndarray name %s not found", launch_context.second->value.c_str());
            }
        }
    }
    compute_graph.launch();
    runtime->wait();
}

}  // namespace auto_graph
