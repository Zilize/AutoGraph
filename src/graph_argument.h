#ifndef DEMO_GRAPH_ARGUMENT_H
#define DEMO_GRAPH_ARGUMENT_H

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

}

#endif //DEMO_GRAPH_ARGUMENT_H
