#include "common.h"
#include "graph_context.h"

#include <utility>

namespace auto_graph {

GraphContext::GraphContext(const char *graph_json): document(rapidjson::Document()) {
    document.Parse(graph_json);

    auto graph_arguments_object = document["graph_arguments"].GetObject();
    for (auto &graph_argument_entry: graph_arguments_object) {
        std::string graph_argument_name(graph_argument_entry.name.GetString());
        auto meta_data_object = graph_argument_entry.value.GetObject();
        ArgumentType type = cook_argument_type(std::string(meta_data_object["type"].GetString()));
        DataType dtype = cook_data_type(std::string(meta_data_object["dtype"].GetString()));

        GraphArgumentContext *graph_argument_context = nullptr;
        if (type == INT) {
            graph_argument_context = new GraphArgumentContext(type, dtype);
        }
        else if (type == MATRIX) {
            int n = meta_data_object["n"].GetInt();
            int m = meta_data_object["m"].GetInt();
            graph_argument_context = new GraphArgumentContext(type, dtype, n=n, m=m);
        }
        else if (type == ARRAY) {
            int n = meta_data_object["n"].GetInt();
            int m = meta_data_object["m"].GetInt();
            int ndim = meta_data_object["ndim"].GetInt();
            graph_argument_context = new GraphArgumentContext(type, dtype, n=n, m=m, ndim=ndim);
        }
        graph_argument_contexts[graph_argument_name] = graph_argument_context;
    }

    auto allocated_arrays_object = document["allocated_arrays"].GetObject();
    for (auto &allocated_array_entry: allocated_arrays_object) {
        std::string allocated_array_name(allocated_array_entry.name.GetString());
        auto meta_data_object = allocated_array_entry.value.GetObject();
        DataType dtype = cook_data_type(std::string(meta_data_object["dtype"].GetString()));
        int n = meta_data_object["n"].GetInt();
        int m = meta_data_object["m"].GetInt();
        auto shape = new std::vector<std::string>();
        for (auto &shape_item: meta_data_object["shape"].GetArray()) {
            shape->emplace_back(shape_item.GetString());
        }
        auto allocated_array_context = new AllocatedArrayContext(dtype, shape, n=n, m=m);
        allocated_array_contexts[allocated_array_name] = allocated_array_context;
    }

    auto launches_object = document["launches"].GetObject();
    for (auto &launch_entry: launches_object) {
        std::string launch_name(launch_entry.name.GetString());
        auto meta_data_object = launch_entry.value.GetObject();
        ArgumentType type = cook_argument_type(std::string(meta_data_object["type"].GetString()));
        std::string value(meta_data_object["value"].GetString());

        LaunchContext *launch_context = nullptr;
        if (type == INT || type == MATRIX) {
            DataType dtype = cook_data_type(std::string(meta_data_object["dtype"].GetString()));
            launch_context = new LaunchContext(type, dtype, value);
        }
        else if (type == ARRAY) {
            launch_context = new LaunchContext(type, NONE, value);
        }
        launch_contexts[launch_name] = launch_context;
    }
}

}
