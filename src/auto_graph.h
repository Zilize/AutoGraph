#ifndef DEMO_AUTO_GRAPH_H
#define DEMO_AUTO_GRAPH_H

#include "graph_context.h"

namespace auto_graph {

class AutoGraph {
public:
    explicit AutoGraph(const char *archive_path);

protected:
    static std::string load_graph_json(const char *archive_path);
    static void parse_graph_json(const std::string &graph_json);

private:
};

}

#endif //DEMO_AUTO_GRAPH_H
