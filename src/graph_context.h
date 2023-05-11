#ifndef DEMO_GRAPH_CONTEXT_H
#define DEMO_GRAPH_CONTEXT_H

#include <memory>
#include <unordered_map>
#include <rapidjson/document.h>

#include "graph_argument.h"

namespace auto_graph {

class GraphContext {
public:
    explicit GraphContext(const char *graph_json);

private:
    rapidjson::Document document;
    std::unordered_map<std::string, GraphArgument*> graph_arguments;
};

}

#endif //DEMO_GRAPH_CONTEXT_H
