#ifndef DEMO_GRAPH_CONTEXT_H
#define DEMO_GRAPH_CONTEXT_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <rapidjson/document.h>

#include "common.h"

namespace auto_graph {

struct GraphArgumentContext {
    GraphArgumentContext(const ArgumentType &_argument_type, const DataType &_data_type, const int &_n=0,
                         const int &_m=0, const int &_ndim=0):
                         argument_type(_argument_type), data_type(_data_type), n(_n), m(_m), ndim(_ndim) {}

    ArgumentType argument_type;
    DataType data_type;
    int n, m, ndim;
};

struct AllocatedArrayContext {
    AllocatedArrayContext(const DataType &_data_type, std::vector<std::string>* _shape, const int &_n=0,
                          const int &_m=0): data_type(_data_type), shape(_shape), n(_n), m(_m) {}

    DataType data_type;
    std::vector<std::string> *shape;
    int n, m;
};

struct LaunchContext {
    LaunchContext(const ArgumentType &_argument_type, const DataType &_data_type, std::string _value):
                  argument_type(_argument_type), data_type(_data_type), value(std::move(_value)) {}

    ArgumentType argument_type;
    DataType data_type;
    std::string value;
};

class GraphContext {
public:
    explicit GraphContext(const char *graph_json);

    rapidjson::Document document;
    std::unordered_map<std::string, GraphArgumentContext*> graph_argument_contexts;
    std::unordered_map<std::string, AllocatedArrayContext*> allocated_array_contexts;
    std::unordered_map<std::string, LaunchContext*> launch_contexts;
};

}

#endif //DEMO_GRAPH_CONTEXT_H
