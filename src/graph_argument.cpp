#include "graph_argument.h"

namespace auto_graph {

GraphArgument::GraphArgument(const ArgumentType &_argument_type, const DataType &_data_type,
                             const int &_n, const int &_m, const int &_ndim):
                             argument_type(_argument_type), data_type(_data_type), n(_n), m(_m), ndim(_ndim) {}

}
