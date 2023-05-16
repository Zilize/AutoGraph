#ifndef DEMO_AUTO_GRAPH_H
#define DEMO_AUTO_GRAPH_H

#include "taichi/cpp/taichi.hpp"
#include "graph_context.h"

namespace auto_graph {

typedef enum ArgumentType {
    I32,
    NDARRAY
} ArgumentType;

typedef union ArgumentValue {
    int32_t i32;
    TiNdArray ndarray;
} ArgumentValue;

class GraphArgument {
    friend class AutoGraph;

public:
    explicit GraphArgument(const GraphArgumentContext *_context): context(_context), valid(false) {}

    inline GraphArgument &operator=(int32_t i32) {
        valid = true;
        type = I32;
        value.i32 = i32;
        return *this;
    }
    inline GraphArgument &operator=(const TiNdArray &ndarray) {
        valid = true;
        type = NDARRAY;
        value.ndarray = ndarray;
        return *this;
    }

    [[nodiscard]] bool is_valid() const {
        return valid;
    }

private:
    const GraphArgumentContext *context;
    bool valid;

    ArgumentType type{};
    ArgumentValue value{};
};

class AutoGraph {
    friend class GraphArgument;

public:
    AutoGraph(const ti::Runtime &_runtime, const char *archive_path);
    void launch() const;

    inline GraphArgument &operator[](const char *name) {
        if (graph_arguments.find(name) == graph_arguments.end()) {
            throw std::runtime_error("Graph argument not found in the context");
        }
        return *graph_arguments[name];
    }
    inline GraphArgument &operator[](const std::string &name) {
        return operator[](name.c_str());
    }

protected:
    static std::string load_graph_json(const char *archive_path);

private:
    GraphContext *graph_context = nullptr;
    ti::Runtime *runtime = nullptr;
    ti::ComputeGraph compute_graph;

    std::unordered_map<std::string, GraphArgument*> graph_arguments{};
};

}

#endif //DEMO_AUTO_GRAPH_H
