#ifndef DEMO_COMMON_H
#define DEMO_COMMON_H

#include <string>
#include <stdexcept>

#include <taichi/taichi_core.h>

#define AUTO_GRAPH_ERROR(info) \
    do {                       \
        throw std::runtime_error(info); \
    } while (false)

#define AUTO_GRAPH_ERROR_FORMAT(format, ...) \
    do { \
        char err_msg[256]; \
        std::snprintf(err_msg, sizeof(err_msg), format, __VA_ARGS__); \
        throw std::runtime_error(err_msg); \
    } while (false)

namespace auto_graph {

typedef enum ContextArgumentType {
    CONTEXT_INT,
    CONTEXT_MATRIX,
    CONTEXT_ARRAY
} ContextArgumentType;

typedef enum ContextDataType {
    CONTEXT_NONE,
    CONTEXT_I8,
    CONTEXT_I16,
    CONTEXT_I32,
    CONTEXT_I64,
    CONTEXT_F16,
    CONTEXT_F32,
    CONTEXT_F64,
    CONTEXT_U8,
    CONTEXT_U16,
    CONTEXT_U32,
    CONTEXT_U64
} ContextDataType;

inline ContextArgumentType cook_argument_type(const std::string &argument_type) {
    if (argument_type == "int") return CONTEXT_INT;
    else if (argument_type == "matrix") return CONTEXT_MATRIX;
    else if (argument_type == "array") return CONTEXT_ARRAY;
    else throw std::runtime_error("Invalid argument type");
}

inline ContextDataType cook_data_type(const std::string &data_type) {
    if (data_type == "i8") return CONTEXT_I8;
    else if (data_type == "i16") return CONTEXT_I16;
    else if (data_type == "i32") return CONTEXT_I32;
    else if (data_type == "i64") return CONTEXT_I64;
    else if (data_type == "f16") return CONTEXT_F16;
    else if (data_type == "f32") return CONTEXT_F32;
    else if (data_type == "f64") return CONTEXT_F64;
    else if (data_type == "u8") return CONTEXT_U8;
    else if (data_type == "u16") return CONTEXT_U16;
    else if (data_type == "u32") return CONTEXT_U32;
    else if (data_type == "u64") return CONTEXT_U64;
    else throw std::runtime_error("Invalid data type");
}

inline bool check_data_type(const ContextDataType &context_data_type, const TiDataType &ti_data_type) {
    if (context_data_type == CONTEXT_I8 && ti_data_type == TI_DATA_TYPE_I8) return true;
    if (context_data_type == CONTEXT_I16 && ti_data_type == TI_DATA_TYPE_I16) return true;
    if (context_data_type == CONTEXT_I32 && ti_data_type == TI_DATA_TYPE_I32) return true;
    if (context_data_type == CONTEXT_I64 && ti_data_type == TI_DATA_TYPE_I64) return true;
    if (context_data_type == CONTEXT_F16 && ti_data_type == TI_DATA_TYPE_F16) return true;
    if (context_data_type == CONTEXT_F32 && ti_data_type == TI_DATA_TYPE_F32) return true;
    if (context_data_type == CONTEXT_F64 && ti_data_type == TI_DATA_TYPE_F64) return true;
    if (context_data_type == CONTEXT_U8 && ti_data_type == TI_DATA_TYPE_U8) return true;
    if (context_data_type == CONTEXT_U16 && ti_data_type == TI_DATA_TYPE_U16) return true;
    if (context_data_type == CONTEXT_U32 && ti_data_type == TI_DATA_TYPE_U32) return true;
    if (context_data_type == CONTEXT_U64 && ti_data_type == TI_DATA_TYPE_U64) return true;
    return false;
}

}

#endif //DEMO_COMMON_H
