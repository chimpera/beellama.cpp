#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>

struct llama_kvarn_type_desc {
    llama_kvarn_type type;
    const char * name;
    int key_bits;
    int value_bits;
    int group;
};

struct llama_kvarn_tile_layout {
    size_t k_payload_off;
    size_t v_payload_off;
    size_t k_s_col_off;
    size_t k_zp_off;
    size_t k_s_row_off;
    size_t v_s_col_off;
    size_t v_s_row_off;
    size_t v_zp_off;

    size_t k_payload_bytes;
    size_t v_payload_bytes;
    size_t tile_bytes;
};

struct llama_kvarn_runtime_requirements {
    bool standard_attention;
    bool head_dim_128;
    uint32_t n_seq_max;
};

LLAMA_API size_t llama_kvarn_type_count();

LLAMA_API const llama_kvarn_type_desc * llama_kvarn_type_desc_from_name(const char * name);
LLAMA_API const llama_kvarn_type_desc * llama_kvarn_type_desc_from_type(llama_kvarn_type type);

LLAMA_API llama_kvarn_tile_layout llama_kvarn_make_layout(int head_dim, int group, int key_bits, int value_bits);

LLAMA_API size_t  llama_kvarn_packed_bytes(int n_values, int bits);
LLAMA_API void    llama_kvarn_pack_bits(const uint8_t * values, int n_values, int bits, uint8_t * dst);
LLAMA_API uint8_t llama_kvarn_unpack_bits_value(const uint8_t * src, int index, int bits);

LLAMA_API const char * llama_kvarn_validate_runtime(
        const llama_kvarn_params & params,
        const llama_kvarn_runtime_requirements & requirements);

LLAMA_API void llama_kvarn_hadamard_128(float * values);

LLAMA_API void llama_kvarn_quantize_k_tile(
        const float * tile,
        int sinkhorn_iters,
        int bits,
        const llama_kvarn_tile_layout & layout,
        uint8_t * record);

LLAMA_API void llama_kvarn_quantize_v_tile(
        const float * tile,
        int sinkhorn_iters,
        int bits,
        const llama_kvarn_tile_layout & layout,
        uint8_t * record);

LLAMA_API void llama_kvarn_dequantize_k_tile(
        const uint8_t * record,
        int bits,
        const llama_kvarn_tile_layout & layout,
        float * tile);

LLAMA_API void llama_kvarn_dequantize_v_tile(
        const uint8_t * record,
        int bits,
        const llama_kvarn_tile_layout & layout,
        float * tile);
