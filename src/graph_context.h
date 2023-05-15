#ifndef DEMO_GRAPH_CONTEXT_H
#define DEMO_GRAPH_CONTEXT_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <rapidjson/document.h>

#include "common.h"

namespace auto_graph {

class GraphArgument {
public:
    explicit GraphArgument(const ArgumentType &_argument_type, const DataType &_data_type, const int &_n=0,
                           const int &_m=0, const int &_ndim=0);

    ArgumentType argument_type;
    DataType data_type;
    int n, m, ndim;
};

class AllocatedArray {
public:
    explicit AllocatedArray(const DataType &_data_type, std::vector<std::string>* _shape, const int &_n=0, const int &_m=0);

    DataType data_type;
    std::vector<std::string> *shape;
    int n, m;
};

class Launch {
public:
    explicit Launch(const ArgumentType &_argument_type, const DataType &_data_type, std::string _value);

    ArgumentType argument_type;
    DataType data_type;
    std::string value;
};

class GraphContext {
public:
    explicit GraphContext(const char *graph_json);

private:
    rapidjson::Document document;
    std::unordered_map<std::string, GraphArgument*> graph_arguments;
    std::unordered_map<std::string, AllocatedArray*> allocated_arrays;
    std::unordered_map<std::string, Launch*> launches;
};

}

#endif //DEMO_GRAPH_CONTEXT_H
