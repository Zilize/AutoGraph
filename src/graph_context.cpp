#include "common.h"
#include "graph_context.h"

namespace auto_graph {

GraphContext::GraphContext(const char *graph_json): document(rapidjson::Document()) {
    document.Parse(graph_json);

    auto graph_arguments_object = document["graph_arguments"].GetObject();
    for (auto &graph_argument_entry: graph_arguments_object) {
        std::string graph_argument_name(graph_argument_entry.name.GetString());
        auto meta_data_object = graph_argument_entry.value.GetObject();
        ArgumentType type = cook_argument_type(std::string(meta_data_object["type"].GetString()));
        DataType dtype = cook_data_type(std::string(meta_data_object["dtype"].GetString()));

        GraphArgument *graph_argument = nullptr;
        if (type == INT) {
            graph_argument = new GraphArgument(type, dtype);
        }
        else if (type == MATRIX) {
            int n = meta_data_object["n"].GetInt();
            int m = meta_data_object["m"].GetInt();
            graph_argument = new GraphArgument(type, dtype, n=n, m=m);
        }
        else if (type == ARRAY) {
            int n = meta_data_object["n"].GetInt();
            int m = meta_data_object["m"].GetInt();
            int ndim = meta_data_object["ndim"].GetInt();
            graph_argument = new GraphArgument(type, dtype, n=n, m=m, ndim=ndim);
        }
        graph_arguments[graph_argument_name] = graph_argument;
    }
}

}
