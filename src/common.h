#ifndef DEMO_COMMON_H
#define DEMO_COMMON_H

#include <string>
#include <stdexcept>

namespace auto_graph {

enum ArgumentType {
    INT,
    MATRIX,
    ARRAY
};

enum DataType {
    NONE,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    U8,
    U16,
    U32,
    U64
};

inline ArgumentType cook_argument_type(const std::string &argument_type) {
    if (argument_type == "int") return INT;
    else if (argument_type == "matrix") return MATRIX;
    else if (argument_type == "array") return ARRAY;
    else throw std::runtime_error("Invalid argument type");
}

inline DataType cook_data_type(const std::string &data_type) {
    if (data_type == "i8") return I8;
    else if (data_type == "i16") return I16;
    else if (data_type == "i32") return I32;
    else if (data_type == "i64") return I64;
    else if (data_type == "f16") return F16;
    else if (data_type == "f32") return F32;
    else if (data_type == "f64") return F64;
    else if (data_type == "u8") return U8;
    else if (data_type == "u16") return U16;
    else if (data_type == "u32") return U32;
    else if (data_type == "u64") return U64;
    else throw std::runtime_error("Invalid data type");
}

}

#endif //DEMO_COMMON_H
