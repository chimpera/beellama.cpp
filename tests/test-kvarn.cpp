#include "llama-kvarn.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void require(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "test-kvarn: %s\n", msg);
        std::abort();
    }
}

static void test_type_table() {
    struct expected_type {
        const char * name;
        llama_kvarn_type type;
        int key_bits;
        int value_bits;
        int group;
    };

    const expected_type expected[] = {
        { "off",               LLAMA_KVARN_TYPE_DISABLED, 0, 0, 128 },
        { "kvarn_k2v2_g128",   LLAMA_KVARN_K2V2_G128,     2, 2, 128 },
        { "kvarn_k2v3_g128",   LLAMA_KVARN_K2V3_G128,     2, 3, 128 },
        { "kvarn_k2v4_g128",   LLAMA_KVARN_K2V4_G128,     2, 4, 128 },
        { "kvarn_k3v2_g128",   LLAMA_KVARN_K3V2_G128,     3, 2, 128 },
        { "kvarn_k3v3_g128",   LLAMA_KVARN_K3V3_G128,     3, 3, 128 },
        { "kvarn_k3v4_g128",   LLAMA_KVARN_K3V4_G128,     3, 4, 128 },
        { "kvarn_k4v2_g128",   LLAMA_KVARN_K4V2_G128,     4, 2, 128 },
        { "kvarn_k4v3_g128",   LLAMA_KVARN_K4V3_G128,     4, 3, 128 },
        { "kvarn_k4v4_g128",   LLAMA_KVARN_K4V4_G128,     4, 4, 128 },
    };

    require(llama_kvarn_type_count() == 10, "unexpected KVarN type count");

    for (const auto & exp : expected) {
        const llama_kvarn_type_desc * desc = llama_kvarn_type_desc_from_name(exp.name);
        require(desc != nullptr, "expected type name did not parse");
        require(desc->type == exp.type, "parsed type enum mismatch");
        require(desc->key_bits == exp.key_bits, "parsed key bits mismatch");
        require(desc->value_bits == exp.value_bits, "parsed value bits mismatch");
        require(desc->group == exp.group, "parsed group mismatch");

        const llama_kvarn_type_desc * by_type = llama_kvarn_type_desc_from_type(exp.type);
        require(by_type != nullptr, "expected enum did not map to descriptor");
        require(std::string(by_type->name) == exp.name, "enum descriptor name mismatch");
    }

    require(llama_kvarn_type_desc_from_name("kvarn_k5v2_g128") == nullptr, "invalid type parsed");
}

static void test_tile_layout() {
    struct expected_layout {
        llama_kvarn_type type;
        size_t k_payload;
        size_t v_payload;
        size_t tile_bytes;
    };

    const expected_layout expected[] = {
        { LLAMA_KVARN_K2V2_G128, 4096, 4096,  9728 },
        { LLAMA_KVARN_K2V3_G128, 4096, 6144, 11776 },
        { LLAMA_KVARN_K2V4_G128, 4096, 8192, 13824 },
        { LLAMA_KVARN_K3V2_G128, 6144, 4096, 11776 },
        { LLAMA_KVARN_K3V3_G128, 6144, 6144, 13824 },
        { LLAMA_KVARN_K3V4_G128, 6144, 8192, 15872 },
        { LLAMA_KVARN_K4V2_G128, 8192, 4096, 13824 },
        { LLAMA_KVARN_K4V3_G128, 8192, 6144, 15872 },
        { LLAMA_KVARN_K4V4_G128, 8192, 8192, 17920 },
    };

    for (const auto & exp : expected) {
        const llama_kvarn_type_desc * desc = llama_kvarn_type_desc_from_type(exp.type);
        require(desc != nullptr, "layout type descriptor missing");

        const llama_kvarn_tile_layout layout = llama_kvarn_make_layout(128, 128, desc->key_bits, desc->value_bits);
        require(layout.k_payload_bytes == exp.k_payload, "K payload bytes mismatch");
        require(layout.v_payload_bytes == exp.v_payload, "V payload bytes mismatch");
        require(layout.tile_bytes == exp.tile_bytes, "tile bytes mismatch");
        require(layout.k_s_col_off == layout.k_payload_off + layout.k_payload_bytes, "K scale offset mismatch");
        require(layout.v_payload_off == layout.k_s_row_off + 128 * sizeof(uint16_t), "V payload offset mismatch");
        require(layout.v_s_col_off == layout.v_payload_off + layout.v_payload_bytes, "V scale offset mismatch");
        require(layout.tile_bytes % 8 == 0, "tile bytes not 8-byte aligned");
    }
}

static void test_runtime_validation() {
    const llama_kvarn_runtime_requirements supported = {
        /*.standard_attention =*/ true,
        /*.head_dim_128       =*/ true,
        /*.n_seq_max          =*/ 1,
    };

    for (int type = LLAMA_KVARN_K2V2_G128; type <= LLAMA_KVARN_K4V4_G128; ++type) {
        const auto params = llama_kvarn_params_for_type((llama_kvarn_type) type);
        require(llama_kvarn_validate_runtime(params, supported) == nullptr, "valid runtime rejected");
    }

    auto invalid = llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128);
    invalid.key_bits = 3;
    require(llama_kvarn_validate_runtime(invalid, supported) != nullptr, "mismatched preset bits accepted");

    invalid = llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128);
    invalid.sink_tokens = 0;
    require(llama_kvarn_validate_runtime(invalid, supported) != nullptr, "unsupported sink tokens accepted");

    auto requirements = supported;
    requirements.standard_attention = false;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) != nullptr,
            "nonstandard attention accepted");

    requirements = supported;
    requirements.head_dim_128 = false;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) != nullptr,
            "unsupported head dimension accepted");

    requirements = supported;
    requirements.n_seq_max = 2;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) != nullptr,
            "multi-sequence runtime accepted");
}

static void test_pack_roundtrip(int bits) {
    const int n = 257;
    std::vector<uint8_t> values(n);
    for (int i = 0; i < n; ++i) {
        values[i] = uint8_t((i * 7 + 3) & ((1 << bits) - 1));
    }

    std::vector<uint8_t> packed(llama_kvarn_packed_bytes(n, bits), 0);
    llama_kvarn_pack_bits(values.data(), n, bits, packed.data());

    for (int i = 0; i < n; ++i) {
        const uint8_t got = llama_kvarn_unpack_bits_value(packed.data(), i, bits);
        if (got != values[i]) {
            std::fprintf(stderr, "test-kvarn: %d-bit roundtrip mismatch at %d: got %u expected %u\n",
                    bits, i, unsigned(got), unsigned(values[i]));
            std::abort();
        }
    }
}

static void test_hadamard_roundtrip() {
    std::vector<float> values(128);
    std::vector<float> expected(128);
    for (int i = 0; i < 128; ++i) {
        values[i] = std::sin(float(i) * 0.19f) + float(i - 64) * 0.002f;
    }
    expected = values;

    llama_kvarn_hadamard_128(values.data());
    llama_kvarn_hadamard_128(values.data());

    for (int i = 0; i < 128; ++i) {
        require(std::fabs(values[i] - expected[i]) < 1e-5f, "Hadamard roundtrip mismatch");
    }
}

static float tile_rmse(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size(), "RMSE shape mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = double(a[i]) - double(b[i]);
        sum += diff * diff;
    }
    return float(std::sqrt(sum / a.size()));
}

static void test_tile_quantization(llama_kvarn_type type) {
    const auto * desc = llama_kvarn_type_desc_from_type(type);
    require(desc != nullptr, "quantization type descriptor missing");

    const auto layout = llama_kvarn_make_layout(128, 128, desc->key_bits, desc->value_bits);
    std::vector<float> k(128 * 128);
    std::vector<float> v(128 * 128);
    for (int r = 0; r < 128; ++r) {
        for (int c = 0; c < 128; ++c) {
            k[r * 128 + c] =
                std::sin(float(r) * 0.071f) +
                std::cos(float(c) * 0.113f) +
                float((r * 17 + c * 13) % 29 - 14) * 0.015f;
            v[r * 128 + c] =
                std::cos(float(r) * 0.057f) -
                std::sin(float(c) * 0.091f) +
                float((r * 11 + c * 19) % 31 - 15) * 0.012f;
        }
    }

    std::vector<uint8_t> record(layout.tile_bytes, 0);
    llama_kvarn_quantize_k_tile(k.data(), 16, desc->key_bits, layout, record.data());
    llama_kvarn_quantize_v_tile(v.data(), 16, desc->value_bits, layout, record.data());

    std::vector<float> k_dequant(k.size());
    std::vector<float> v_dequant(v.size());
    llama_kvarn_dequantize_k_tile(record.data(), desc->key_bits, layout, k_dequant.data());
    llama_kvarn_dequantize_v_tile(record.data(), desc->value_bits, layout, v_dequant.data());

    for (size_t i = 0; i < k.size(); ++i) {
        require(std::isfinite(k_dequant[i]), "K dequant produced non-finite value");
        require(std::isfinite(v_dequant[i]), "V dequant produced non-finite value");
    }

    const float max_rmse[] = { 0.0f, 0.0f, 0.40f, 0.22f, 0.12f };
    require(tile_rmse(k, k_dequant) < max_rmse[desc->key_bits], "K tile RMSE too high");
    require(tile_rmse(v, v_dequant) < max_rmse[desc->value_bits], "V tile RMSE too high");
}

static void test_cache_ops(enum ggml_backend_dev_type device_type, bool required) {
    ggml_backend_t backend = ggml_backend_init_by_type(device_type, nullptr);
    if (backend == nullptr && !required) {
        return;
    }
    require(backend != nullptr, "failed to initialize requested backend");

    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize ggml context");

    constexpr int bits = 3;
    constexpr int n_tokens = 385;
    constexpr int n_heads = 1;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 384);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, 4);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false);
    ggml_tensor * materialized = ggml_kvarn_materialize(ctx, records, stored, indices, n_tokens, bits, false);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);
    ggml_build_forward_expand(graph, materialized);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate KVarN tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            input[t * 128 + d] =
                std::sin(float(d) * 0.071f) +
                std::cos(float(t) * 0.037f) +
                float((d * 13 + t * 17) % 31 - 15) * 0.01f;
        }
    }
    std::vector<int64_t> idx(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        idx[i] = i;
    }
    std::vector<uint8_t> zeros(ggml_nbytes(stage) + ggml_nbytes(records), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, zeros.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_set(records, zeros.data(), 0, ggml_nbytes(records));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "KVarN graph compute failed");

    std::vector<ggml_fp16_t> output_f16(ggml_nelements(materialized));
    std::vector<float> output(output_f16.size());
    ggml_backend_tensor_get(materialized, output_f16.data(), 0, ggml_nbytes(materialized));
    ggml_fp16_to_fp32_row(output_f16.data(), output.data(), output.size());

    double sink_error = 0.0;
    double compressed_error = 0.0;
    double previous_tail_error = 0.0;
    double live_tail_error = 0.0;
    for (int t = 0; t < n_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            const double diff = double(input[t * 128 + d]) - double(output[t * 128 + d]);
            if (t < 128) {
                sink_error += diff * diff;
            } else if (t < 256) {
                compressed_error += diff * diff;
            } else if (t < 384) {
                previous_tail_error += diff * diff;
            } else {
                live_tail_error += diff * diff;
            }
        }
    }
    sink_error = std::sqrt(sink_error / (128 * 128));
    compressed_error = std::sqrt(compressed_error / (128 * 128));
    previous_tail_error = std::sqrt(previous_tail_error / (128 * 128));
    live_tail_error = std::sqrt(live_tail_error / 128);
    require(sink_error < 0.01, "sink reconstruction error too high");
    require(compressed_error < 0.25, "compressed reconstruction error too high");
    require(previous_tail_error < 0.01, "previous tail reconstruction error too high");
    require(live_tail_error < 0.01, "live tail reconstruction error too high");

    std::vector<uint8_t> record_data(ggml_nbytes(records));
    ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());
    require(std::any_of(record_data.begin(), record_data.end(), [](uint8_t v) { return v != 0; }),
            "completed group was not flushed");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

int main() {
    ggml_backend_load_all();

    test_type_table();
    test_tile_layout();
    test_runtime_validation();
    test_pack_roundtrip(2);
    test_pack_roundtrip(3);
    test_pack_roundtrip(4);
    test_hadamard_roundtrip();

    for (int type = LLAMA_KVARN_K2V2_G128; type <= LLAMA_KVARN_K4V4_G128; ++type) {
        test_tile_quantization((llama_kvarn_type) type);
    }
    test_cache_ops(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    test_cache_ops(GGML_BACKEND_DEVICE_TYPE_GPU, false);

    std::printf("test-kvarn: all tests OK\n");
    return 0;
}
