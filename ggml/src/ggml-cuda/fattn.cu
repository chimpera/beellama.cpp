#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-mma-turbo.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn.cuh"

#include <atomic>

// InnerQ: update the fattn-side inverse scale array from host (all devices)
void turbo_innerq_update_fattn_scales(const float * scale_inv) {
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_innerq_channel_scale_inv_fattn, scale_inv, 128 * sizeof(float));
    }
    cudaSetDevice(cur_device);
}

void turbo_innerq_init_fattn() {
    float ones[128];
    for (int i = 0; i < 128; i++) ones[i] = 1.0f;
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_innerq_channel_scale_inv_fattn, ones, sizeof(ones));
    }
    cudaSetDevice(cur_device);
}

// Q^2 calibration: host-side management
static int q_calibrate_state = 0; // 0=off, 1=collecting, 2=done

void turbo_q_calibrate_init() {
    const char * env = getenv("TURBO_Q_CALIBRATE");
    if (!env || atoi(env) != 1) return;

    double zeros[128] = {};
    int zero = 0, one = 1;
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_q_channel_sq_fattn, zeros, sizeof(zeros));
        cudaMemcpyToSymbol(d_q_channel_count_fattn, &zero, sizeof(zero));
        cudaMemcpyToSymbol(d_q_calibrate_fattn, &one, sizeof(one));
    }
    cudaSetDevice(cur_device);
    q_calibrate_state = 1;
    fprintf(stderr, "TURBO_Q_CALIBRATE: collecting per-position Q^2 statistics\n");
}

void turbo_q_calibrate_finalize() {
    if (q_calibrate_state != 1) return;

    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);

    int zero = 0;
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_q_calibrate_fattn, &zero, sizeof(zero));
    }
    cudaSetDevice(cur_device);

    double sq[128];
    int count;
    cudaMemcpyFromSymbol(sq, d_q_channel_sq_fattn, sizeof(sq));
    cudaMemcpyFromSymbol(&count, d_q_channel_count_fattn, sizeof(count));

    if (count == 0) {
        fprintf(stderr, "TURBO_Q_CALIBRATE: no Q vectors seen, skipping\n");
        q_calibrate_state = 2;
        return;
    }

    // Compute E[Q^2] per position and save as 128 float32 values
    float weights[128];
    fprintf(stderr, "TURBO_Q_CALIBRATE: %d Q groups accumulated\n", count);
    double total = 0;
    for (int i = 0; i < 128; i++) {
        weights[i] = (float)(sq[i] / count);
        total += weights[i];
    }
    float mean = (float)(total / 128.0);
    float maxw = 0, minw = 1e30f;
    for (int i = 0; i < 128; i++) {
        if (weights[i] > maxw) maxw = weights[i];
        if (weights[i] < minw) minw = weights[i];
    }
    fprintf(stderr, "  E[Q^2] mean=%.6f min=%.6f max=%.6f ratio=%.2f\n",
            mean, minw, maxw, maxw / (minw > 1e-10f ? minw : 1e-10f));

    const char * path = "/tmp/q_weights.bin";
    FILE * fp = fopen(path, "wb");
    if (fp) {
        fwrite(weights, sizeof(float), 128, fp);
        fclose(fp);
        fprintf(stderr, "  Saved Q weights to %s (128 floats)\n", path);
    }

    q_calibrate_state = 2;
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if ((turing_mma_available(cc) || amd_wmma_available(cc)) && Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || amd_wmma_available(cc) || Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        if (use_gqa_opt && gqa_ratio > 1) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }

        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

// Turbo MMA fused dispatch: ncols1 selection (mirrors f16 version but calls turbo case).
template <int DKQ, int DV, int ncols2, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 8/ncols2, ncols2, type_K, type_V>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 16/ncols2) {
        ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 16/ncols2, ncols2, type_K, type_V>(ctx, dst);
        return;
    }

    // Turing (sm_75) is capped at ncols=32 -- the kernel has NO_DEVICE_CODE for ncols>32.
    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 32/ncols2, ncols2, type_K, type_V>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 64/ncols2, ncols2, type_K, type_V>(ctx, dst);
}

// Turbo MMA fused dispatch: ncols2 selection based on GQA ratio.
template <int DKQ, int DV, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 8, type_K, type_V>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 4, type_K, type_V>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 2, type_K, type_V>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 1, type_K, type_V>(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// Context-adaptive V alpha: logarithmic scaling based on current KV occupancy.
// 3-bit: alpha = 1.1484 - 0.01443 * ln(n_kv), calibrated on Qwen3.5-27B decode-time KLD sweeps
//   at 8K/16K/32K. Optima: 8K->1.020, 16K->1.005, 32K->1.000.
// 2-bit: alpha = 0.8865 + 0.0195 * ln(n_kv), calibrated on fine-grained 0.005-step decode-time
//   KLD sweeps at 2K/7K/16K/32K on Qwen3.5-27B. Optima: 2K->1.030, 7K->1.065, 16K->1.090, 32K->1.075.
// Override with TURBO_TCQ_DECODE_ALPHA_V env var to force a static alpha (disables adaptive).
static float d_tcq_decode_alpha_v_static = 0.0f; // 0 = use adaptive, >0 = static override
static float d_tcq_decode_alpha_k = 1.0f;       // K decode alpha, static (default 1.0)
static bool d_tcq_decode_alpha_loaded = false;

static inline float tcq_compute_alpha_v(ggml_type v_type, int64_t n_kv) {
    if (d_tcq_decode_alpha_v_static > 0.0f) return d_tcq_decode_alpha_v_static;
    if (n_kv < 1) n_kv = 1;
    const float ln_ctx = logf((float)n_kv);
    if (v_type == GGML_TYPE_TURBO3_TCQ) {
        // v3: 3-point fit from fine-grained KLD sweeps at 8K/16K/32K on Qwen3.5-27B.
        // Per-token: n_kv=512->1.044, 2K->1.024, 8K->1.004, 16K->0.994, 32K->0.984
        // Clamped [0.98, 1.06] -- early tokens (small n_kv) use higher alpha.
        return fmaxf(0.98f, fminf(1.06f, 1.1484f - 0.01443f * ln_ctx));
    } else if (v_type == GGML_TYPE_TURBO2_TCQ) {
        // v2: 4-point fit from fine-grained 0.005-step KLD sweeps at 2K/7K/16K/32K.
        // Per-token: n_kv=512->1.008, 2K->1.035, 7K->1.060, 16K->1.076, 32K->1.089
        return fmaxf(1.00f, fminf(1.12f, 0.8865f + 0.0195f * ln_ctx));
    }
    return 1.0f;
}

static void load_tcq_decode_alpha(int device) {
    static bool loaded[GGML_CUDA_MAX_DEVICES] = {};
    if (loaded[device]) return;
    loaded[device] = true;
    if (!d_tcq_decode_alpha_loaded) {
        d_tcq_decode_alpha_loaded = true;
        const char * sk = getenv("TURBO_TCQ_DECODE_ALPHA_K");
        if (sk) {
            char * end;
            errno = 0;
            float a = strtof(sk, &end);
            if (end != sk && errno == 0 && a > 0.0f && a < 10.0f) {
                d_tcq_decode_alpha_k = a;
                fprintf(stderr, "TCQ decode: K alpha=%.4f\n", a);
            }
        }
        const char * s = getenv("TURBO_TCQ_DECODE_ALPHA_V");
        if (s) {
            char * end;
            errno = 0;
            float a = strtof(s, &end);
            if (end != s && errno == 0 && a > 0.0f && a < 10.0f) {
                d_tcq_decode_alpha_v_static = a;
            }
        }
    }
    if (d_tcq_decode_alpha_v_static > 0.0f) {
        cudaMemcpyToSymbol(d_tcq_decode_alpha_v_fattn, &d_tcq_decode_alpha_v_static, sizeof(float));
        if (device == 0) fprintf(stderr, "TCQ decode: V alpha=%.4f (static override)\n", d_tcq_decode_alpha_v_static);
    } else {
        if (device == 0) fprintf(stderr, "TCQ decode: context-adaptive V alpha enabled\n");
    }
}

// === Turbo prefill: bulk dequant to fp16 + MMA attention ===
// During prefill (Q->ne[1] > 1), dequantize turbo K/V to fp16 temp buffers
// and use the fast MMA kernel instead of the slower vec kernel.

static __global__ void k_turbo2_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO2;
    const int j_in_blk = j % QK_TURBO2;
    const block_turbo2_0 * blk = (const block_turbo2_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t idx = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
    const float val = d_turbo_centroids_2bit_fattn[idx] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo3_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO3;
    const int j_in_blk = j % QK_TURBO3;
    const block_turbo3_0 * blk = (const block_turbo3_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t low2 = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
    const uint8_t hi1  = (blk->signs[j_in_blk / 8] >> (j_in_blk % 8)) & 0x1;
    const float val = d_turbo_centroids_3bit_fattn[low2 | (hi1 << 2)] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo3_tcq_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x; // element index within row (0..ne0-1)
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO3_TCQ;
    const int t = j % QK_TURBO3_TCQ; // element index within 128-element block
    const block_turbo3_tcq * blk = (const block_turbo3_tcq *)src_row + blk_idx;

    const float norm = __half2float(blk->norm) * alpha;

    // Sliding window decode: read 9-bit state from bitstream at bit offset t*3
    const int bit_pos = t * 3;
    const int byte_idx = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
    const int state = (raw >> bit_off) & 0x1FF;
    const float val = d_turbo3_tcq_codebook_fattn[state] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo2_tcq_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO2_TCQ;
    const int t = j % QK_TURBO2_TCQ;
    const block_turbo2_tcq * blk = (const block_turbo2_tcq *)src_row + blk_idx;

    const float norm = __half2float(blk->norm) * alpha;

    // Sliding window decode: read 8-bit state from bitstream at bit offset t*2
    const int bit_pos = t * 2;
    const int byte_idx = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
    const int state = (raw >> bit_off) & 0xFF;
    const float val = d_turbo2_tcq_codebook_fattn[state] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo4_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO4;
    const int j_in_blk = j % QK_TURBO4;
    const block_turbo4_0 * blk = (const block_turbo4_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t idx = (j_in_blk & 1) ? (blk->qs[j_in_blk / 2] >> 4) : (blk->qs[j_in_blk / 2] & 0xF);
    const float val = d_turbo_centroids_4bit_fattn[idx] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

// turbo3 K dequant with inverse FWHT: produces K in original (unrotated) domain
// so Q does NOT need pre-rotation. 128 threads per block, one 128-element storage block per FWHT group.
// In-place 128-point inverse FWHT spread across 128 threads, one element per thread.
// h=1..16 use intra-warp __shfl_xor_sync (no smem, no syncthreads).
// h=32 and h=64 cross warp boundaries, so we fall back to smem with syncs.
// Caller supplies a __shared__ float[128] scratch buffer.
static __device__ __forceinline__ float fwht128_butterfly_inplace(float val, float * smem) {
    const int tid = threadIdx.x;

    // Intra-warp passes: shuffle xor with stride h, no smem, no sync.
    #pragma unroll
    for (int h = 1; h <= 16; h *= 2) {
        const float other = __shfl_xor_sync(0xFFFFFFFFULL, val, h);
        val = (tid & h) ? (other - val) : (val + other);
    }

    // h=32 (cross-warp within block, smem needed)
    smem[tid] = val;
    __syncthreads();
    val = (tid & 32) ? (smem[tid - 32] - val) : (val + smem[tid + 32]);
    __syncthreads();

    // h=64 (cross-warp within block, smem needed)
    smem[tid] = val;
    __syncthreads();
    val = (tid & 64) ? (smem[tid - 64] - val) : (val + smem[tid + 64]);
    __syncthreads();

    return val;
}

// Pair-pack two adjacent threads' fp16 outputs into a single 32-bit half2 store.
// Halves the global store count vs. one __float2half per thread.
static __device__ __forceinline__ void fwht128_store_half(
        float val, half * dst_base) {
    const int tid = threadIdx.x;
    const float neighbor = __shfl_xor_sync(0xFFFFFFFFULL, val, 1);
    if ((tid & 1) == 0) {
        const half2 packed = __floats2half2_rn(val, neighbor);
        *((half2 *)(dst_base + tid)) = packed;
    }
}

static __global__ void k_turbo3_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    // ne0 in elements, FWHT group = 128 elements = one classic turbo3 block.
    const int n_groups = (int)(ne0 / 128);
    constexpr int blocks_per_group = 128 / QK_TURBO3;

    for (int g = 0; g < n_groups; g++) {
        // Element index within the FWHT group
        const int j_in_grp = tid;            // 0..127
        const int blk_in_grp = j_in_grp / QK_TURBO3;  // 0
        const int j_in_blk  = j_in_grp % QK_TURBO3;   // 0..127

        const block_turbo3_0 * blk = (const block_turbo3_0 *)src_row + g * blocks_per_group + blk_in_grp;
        const float norm = __half2float(blk->norm);   // shared across the block

        const uint8_t low2 = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
        const uint8_t hi1  = (blk->signs[j_in_blk / 8] >> (j_in_blk % 8)) & 0x1;
        const float c = d_turbo_centroids_3bit_fattn[low2 | (hi1 << 2)];

        // Inverse FWHT: 5 intra-warp shfl passes + 2 cross-warp smem passes.
        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        // Normalize, apply signs1, undo InnerQ scaling, multiply by norm, cast to fp16
        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + g * 128);
        __syncthreads();
    }
}

// turbo4 K dequant with inverse FWHT: produces K in original (unrotated) domain
// so Q does NOT need pre-rotation. 128 threads per block, loops over 128-element turbo4 blocks.
static __global__ void k_turbo4_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO4);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo4_0 * blk = (const block_turbo4_0 *)src_row + blk_idx;
        const float norm = __half2float(blk->norm);

        const uint8_t idx = (tid & 1) ? (blk->qs[tid / 2] >> 4) : (blk->qs[tid / 2] & 0xF);

        float val = fwht128_butterfly_inplace(d_turbo_centroids_4bit_fattn[idx] * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// turbo2 K dequant with inverse FWHT: produces K in original (unrotated) domain.
// 128 threads per block, loops over 128-element FWHT groups.
// Each group is one turbo2 storage block of 128 elements.
static __global__ void k_turbo2_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_groups = (int)(ne0 / 128);
    constexpr int blocks_per_group = 128 / QK_TURBO2; // 1

    for (int g = 0; g < n_groups; g++) {
        const int j_in_grp = tid;            // 0..127
        const int blk_in_grp = j_in_grp / QK_TURBO2;  // 0
        const int j_in_blk  = j_in_grp % QK_TURBO2;   // 0..127

        const block_turbo2_0 * blk = (const block_turbo2_0 *)src_row + g * blocks_per_group + blk_in_grp;
        const float norm = __half2float(blk->norm);

        const uint8_t idx = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
        const float c = d_turbo_centroids_2bit_fattn[idx];

        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + g * 128);
        __syncthreads();
    }
}

// turbo3_tcq K dequant with inverse FWHT: produces K in original (unrotated) domain.
// 128 threads per block, loops over 128-element TCQ blocks (1 block per FWHT group).
static __global__ void k_turbo3_tcq_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO3_TCQ);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo3_tcq * blk = (const block_turbo3_tcq *)src_row + blk_idx;
        const float norm = __half2float(blk->norm) * alpha;

        // Sliding window decode: read 9-bit state from bitstream at bit offset tid*3
        const int bit_pos = tid * 3;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x1FF;
        const float c = d_turbo3_tcq_codebook_fattn[state];

        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// turbo2_tcq K dequant with inverse FWHT: produces K in original (unrotated) domain.
static __global__ void k_turbo2_tcq_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO2_TCQ);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo2_tcq * blk = (const block_turbo2_tcq *)src_row + blk_idx;
        const float norm = __half2float(blk->norm) * alpha;

        // Sliding window decode: read 8-bit state from bitstream at bit offset tid*2
        const int bit_pos = tid * 2;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        const float c = d_turbo2_tcq_codebook_fattn[state];

        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// q8_0 K dequant to f16 in TKHE layout, matching the turbo K dequant kernels.
// Used at D=512 when K=q8_0 paired with V=turbo: produces (F16, F16) for the FA dispatch
// and bypasses the (Q8_0, TURBO*) D=512 native VEC templates which have buggy SASS on
// sm_120 PTX-JIT for some K/V combos. Q8_0 is in original (unrotated) domain -> output too.
// 1 thread per element, 1 block per (token, head, batch).
static __global__ void k_q8_0_dequant_f16_tkhe(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + row * nb1 + head * nb2;
    const int blk_idx = j / QK8_0;
    const int j_in_blk = j % QK8_0;
    const block_q8_0 * blk = (const block_q8_0 *)src_row + blk_idx;
    const float d = __half2float(blk->d);
    const float val = d * (float)blk->qs[j_in_blk];

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

// Persistent Q rotation buffer per device (shared between prefill and decode paths)
static float * q_rot_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  q_rot_buf_size[GGML_CUDA_MAX_DEVICES] = {};

// Persistent K/V fp16 dequant buffers per device (shared between prefill and decode paths)
static half * kv_dequant_k_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  kv_dequant_k_buf_size[GGML_CUDA_MAX_DEVICES] = {};
static half * kv_dequant_v_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  kv_dequant_v_buf_size[GGML_CUDA_MAX_DEVICES] = {};

// === Safe grow-buffer helper (alloc-before-free) and debug diagnostics ===

template <typename T>
static bool ggml_cuda_try_grow_buffer(
        T ** buf,
        size_t * buf_size,
        size_t required_bytes,
        int device,
        const char * name) {
    if (required_bytes <= *buf_size) {
        return true;
    }

    size_t free_vram = 0;
    size_t total_vram = 0;
    const cudaError_t mem_info_err = cudaMemGetInfo(&free_vram, &total_vram);

    T * new_buf = nullptr;
    cudaError_t err = cudaMalloc((void **) &new_buf, required_bytes);
    if (err != cudaSuccess) {
        if (mem_info_err == cudaSuccess) {
            fprintf(stderr,
                "turbo workspace: cudaMalloc failed for %s on device %d: "
                "requested %.2f MiB, free %.2f MiB / total %.2f MiB, error=%s\n",
                name, device,
                required_bytes / 1024.0 / 1024.0,
                free_vram / 1024.0 / 1024.0,
                total_vram / 1024.0 / 1024.0,
                cudaGetErrorString(err));
        } else {
            fprintf(stderr,
                "turbo workspace: cudaMalloc failed for %s on device %d: "
                "requested %.2f MiB, error=%s\n",
                name, device,
                required_bytes / 1024.0 / 1024.0,
                cudaGetErrorString(err));
        }
        return false;
    }

    if (*buf != nullptr) {
        CUDA_CHECK(cudaFree(*buf));
    }

    *buf = new_buf;
    *buf_size = required_bytes;
    return true;
}

static bool ggml_cuda_turbo_debug_enabled() {
#ifndef NDEBUG
    return true;
#else
    static const bool enabled = getenv("GGML_CUDA_TURBO_DEBUG") != nullptr;
    return enabled;
#endif
}

static void ggml_cuda_turbo_check_last_error(const char * where) {
    if (!ggml_cuda_turbo_debug_enabled()) {
        return;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "turbo FA debug: CUDA error after %s: %s\n", where, cudaGetErrorString(err));
    }
}

static void ggml_cuda_turbo_log_shapes(const char * phase,
                                        const ggml_tensor * Q,
                                        const ggml_tensor * K,
                                        const ggml_tensor * V) {
    if (!ggml_cuda_turbo_debug_enabled()) {
        return;
    }
    fprintf(stderr,
        "turbo FA debug: %s Q type=%s ne=[%ld,%ld,%ld,%ld] "
        "K type=%s ne=[%ld,%ld,%ld,%ld] "
        "V type=%s ne=[%ld,%ld,%ld,%ld]\n",
        phase,
        ggml_type_name(Q->type), (long)Q->ne[0], (long)Q->ne[1], (long)Q->ne[2], (long)Q->ne[3],
        ggml_type_name(K->type), (long)K->ne[0], (long)K->ne[1], (long)K->ne[2], (long)K->ne[3],
        ggml_type_name(V->type), (long)V->ne[0], (long)V->ne[1], (long)V->ne[2], (long)V->ne[3]);
}

// === FWHT rotation kernels for pre-rotate-queries approach ===
// Forward rotation on Q before attention (both prefill and decode paths).
// One block per 128-element group, 128 threads per block.
static __global__ void k_turbo_fwht_forward(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t n_elements) {
    const int64_t offset = blockIdx.x * 128;
    if (offset >= n_elements) return;

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;

    __shared__ float buf[128];

    // InnerQ: apply inverse channel scale to Q before rotation
    float val = src[offset + threadIdx.x] * d_innerq_channel_scale_inv_fattn[threadIdx.x] * s1[threadIdx.x];

    val = fwht128_butterfly_inplace(val, buf);

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    val = val * inv_sqrt_128 * s2[threadIdx.x];
    dst[offset + threadIdx.x] = val;

    // Q^2 calibration: accumulate per-position squared values
    if (d_q_calibrate_fattn) {
        atomicAdd_double(&d_q_channel_sq_fattn[threadIdx.x], (double)(val * val));
        if (threadIdx.x == 0) atomicAdd(&d_q_channel_count_fattn, 1);
    }
}

static bool ggml_cuda_turbo_prefill_attend(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    load_tcq_decode_alpha(ctx.device);
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool turbo_k = K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0 || K->type == GGML_TYPE_TURBO3_TCQ || K->type == GGML_TYPE_TURBO2_TCQ;
    const bool turbo_v = V->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0 || V->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO2_TCQ;

    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    // Reserve all required workspace before any kernel launches or dst->src[] mutations.
    size_t k_size = 0;
    if (turbo_k) {
        const ggml_tensor * k_root = K;
        while (k_root->view_src) k_root = k_root->view_src;
        k_size = (size_t)k_root->ne[0] * k_root->ne[1] * k_root->ne[2] * sizeof(half);
    }
    size_t v_size = 0;
    if (turbo_v) {
        const ggml_tensor * v_root = V;
        while (v_root->view_src) v_root = v_root->view_src;
        v_size = (size_t)v_root->ne[0] * v_root->ne[1] * v_root->ne[2] * sizeof(half);
    }
    size_t q_size = 0;
    const ggml_tensor * Q = dst->src[0];
    const bool turbo_k_needs_rotation = turbo_k && K->type != GGML_TYPE_TURBO4_0 && Q->ne[0] % 128 == 0;
    if (turbo_k_needs_rotation) {
        q_size = ggml_nelements(Q) * sizeof(float);
    }
    if (turbo_k) {
        if (!ggml_cuda_try_grow_buffer(&kv_dequant_k_buf[device], &kv_dequant_k_buf_size[device], k_size, device, "prefill K dequant")) {
            return false;
        }
    }
    if (turbo_v) {
        if (!ggml_cuda_try_grow_buffer(&kv_dequant_v_buf[device], &kv_dequant_v_buf_size[device], v_size, device, "prefill V dequant")) {
            return false;
        }
    }
    if (turbo_k_needs_rotation) {
        if (!ggml_cuda_try_grow_buffer(&q_rot_buf[device], &q_rot_buf_size[device], q_size, device, "prefill Q rotation")) {
            return false;
        }
    }

    half * k_fp16 = nullptr;
    half * v_fp16 = nullptr;

    // Allocate and dequant K to fp16 (turbo2, turbo3, or turbo4)
    if (turbo_k) {
        k_fp16 = kv_dequant_k_buf[device];
        dim3 grid_k(K->ne[1], K->ne[2], K->ne[3]);
        if (K->type == GGML_TYPE_TURBO2_0) {
            k_turbo2_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TURBO3_0) {
            k_turbo3_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TURBO3_TCQ) {
            {
                static bool tcq_fattn_k_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                if (!tcq_fattn_k_cb_loaded[device]) {
                    tcq_fattn_k_cb_loaded[device] = true;
                    const char *cb_path = getenv("TURBO_TCQ_CB");
                    if (cb_path) {
                        float cb[512];
                        FILE *f = fopen(cb_path, "rb");
                        if (f && fread(cb, sizeof(float), 512, f) == 512) {
                            fclose(f);
                            cudaMemcpyToSymbol(d_turbo3_tcq_codebook_fattn, cb, 512*sizeof(float));
                            fprintf(stderr, "TCQ K prefill: loaded codebook from %s (device %d)\n", cb_path, device);
                        } else {
                            if (f) fclose(f);
                            fprintf(stderr, "TCQ K prefill: FAILED to load codebook from %s\n", cb_path);
                        }
                    }
                }
            }
            k_turbo3_tcq_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
        } else if (K->type == GGML_TYPE_TURBO2_TCQ) {
            {
                static bool tcq2_fattn_k_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                if (!tcq2_fattn_k_cb_loaded[device]) {
                    tcq2_fattn_k_cb_loaded[device] = true;
                    const char *cb_path = getenv("TURBO_TCQ_CB2");
                    if (cb_path) {
                        float cb[256];
                        FILE *f = fopen(cb_path, "rb");
                        if (f && fread(cb, sizeof(float), 256, f) == 256) {
                            fclose(f);
                            cudaMemcpyToSymbol(d_turbo2_tcq_codebook_fattn, cb, 256*sizeof(float));
                            fprintf(stderr, "TCQ2 K prefill: loaded 2-bit codebook from %s (device %d)\n", cb_path, device);
                        } else {
                            if (f) fclose(f);
                            fprintf(stderr, "TCQ2 K prefill: FAILED to load codebook from %s\n", cb_path);
                        }
                    }
                }
            }
            k_turbo2_tcq_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
        } else {
            // turbo4 K: inverse FWHT dequant -> produces K in original domain (no Q rotation needed)
            k_turbo4_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        }
        ggml_cuda_turbo_check_last_error("prefill K dequant");
    }

    // Dequant V to fp16 (turbo2, turbo3, or turbo4)
    if (turbo_v) {
        v_fp16 = kv_dequant_v_buf[device];
        dim3 grid_v(V->ne[1], V->ne[2], V->ne[3]);
        if (V->type == GGML_TYPE_TURBO2_0) {
            k_turbo2_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_TURBO3_0) {
            k_turbo3_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_TURBO3_TCQ) {
            // Runtime codebook loading for 3-bit V decode (in case K is a different type)
            {
                static bool tcq_fattn_v_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                if (!tcq_fattn_v_cb_loaded[device]) {
                    tcq_fattn_v_cb_loaded[device] = true;
                    const char *cb_path = getenv("TURBO_TCQ_CB");
                    if (cb_path) {
                        float cb[512];
                        FILE *f = fopen(cb_path, "rb");
                        if (f && fread(cb, sizeof(float), 512, f) == 512) {
                            fclose(f);
                            cudaMemcpyToSymbol(d_turbo3_tcq_codebook_fattn, cb, 512*sizeof(float));
                            fprintf(stderr, "TCQ V decode: loaded 3-bit codebook from %s (device %d)\n", cb_path, device);
                        } else {
                            if (f) fclose(f);
                        }
                    }
                }
            }
            k_turbo3_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]));
        } else if (V->type == GGML_TYPE_TURBO2_TCQ) {
            // Runtime codebook loading for 2-bit V decode (in case K is a different type)
            {
                static bool tcq2_fattn_v_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                if (!tcq2_fattn_v_cb_loaded[device]) {
                    tcq2_fattn_v_cb_loaded[device] = true;
                    const char *cb_path = getenv("TURBO_TCQ_CB2");
                    if (cb_path) {
                        float cb[256];
                        FILE *f = fopen(cb_path, "rb");
                        if (f && fread(cb, sizeof(float), 256, f) == 256) {
                            fclose(f);
                            cudaMemcpyToSymbol(d_turbo2_tcq_codebook_fattn, cb, 256*sizeof(float));
                            fprintf(stderr, "TCQ2 V decode: loaded 2-bit codebook from %s (device %d)\n", cb_path, device);
                        } else {
                            if (f) fclose(f);
                        }
                    }
                }
            }
            k_turbo2_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]));
        } else {
            k_turbo4_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        }
        ggml_cuda_turbo_check_last_error("prefill V dequant");
    }

    // Create fp16 tensor copies on stack
    ggml_tensor K_f16 = *K;
    ggml_tensor V_f16 = *V;

    if (k_fp16) {
        K_f16.type = GGML_TYPE_F16;
        K_f16.data = k_fp16;
        K_f16.nb[0] = sizeof(half);
        K_f16.nb[1] = K->ne[0] * K->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
        K_f16.nb[2] = K->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
        K_f16.nb[3] = K->ne[0] * K->ne[1] * K->ne[2] * sizeof(half);
    }

    if (v_fp16) {
        V_f16.type = GGML_TYPE_F16;
        V_f16.data = v_fp16;
        V_f16.nb[0] = sizeof(half);
        V_f16.nb[1] = V->ne[0] * V->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
        V_f16.nb[2] = V->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
        V_f16.nb[3] = V->ne[0] * V->ne[1] * V->ne[2] * sizeof(half);
    }

// Rotate Q for turbo pre-rotate-queries (only when K is in rotated space)
    // turbo4 K is dequanted via inverse FWHT -> original domain, so Q stays unrotated
    float * q_rotated = nullptr;
    if (turbo_k_needs_rotation) {
        q_rotated = q_rot_buf[device];
        const int64_t n_q_groups = ggml_nelements(Q) / 128;
        k_turbo_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
            (const float *)Q->data, q_rotated, ggml_nelements(Q));
        ggml_cuda_turbo_check_last_error("prefill Q rotation");
    }

    // Temporarily swap src pointers to fp16 K/V and rotated Q
    ggml_tensor * orig_q = dst->src[0];
    ggml_tensor * orig_k = dst->src[1];
    ggml_tensor * orig_v = dst->src[2];

    ggml_tensor Q_rot;
    if (q_rotated) {
        Q_rot = *Q;
        Q_rot.data = q_rotated;
        dst->src[0] = &Q_rot;
    }
    dst->src[1] = k_fp16 ? &K_f16 : orig_k;
    dst->src[2] = v_fp16 ? &V_f16 : orig_v;

    // Dispatch to MMA kernel (sees rotated Q, fp16 K/V, uses tensor cores)
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
    ggml_cuda_turbo_check_last_error("prefill MMA dispatch");

    // Restore original tensor pointers
    dst->src[0] = orig_q;
    dst->src[1] = orig_k;
    dst->src[2] = orig_v;

    // K/V fp16 buffers are persistent (grow-only), no free needed
    return true;
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

#define FATTN_VEC_CASES_ALL_D_512(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)

    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0,   GGML_TYPE_TURBO3_TCQ)
#else
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0,   GGML_TYPE_TURBO3_TCQ)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_WMMA_F16 = 300,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
};

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_TURBO3_TCQ:
        case GGML_TYPE_TURBO2_TCQ:
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:

    // TurboQuant: only the vec kernel has native turbo dequant support.
    // No FATTN_KQ_STRIDE alignment needed -- vec kernel handles arbitrary lengths.
    if (K->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO2_0 ||
        K->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO3_0 ||
        K->type == GGML_TYPE_TURBO4_0 || V->type == GGML_TYPE_TURBO4_0 ||
        K->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO3_TCQ ||
        K->type == GGML_TYPE_TURBO2_TCQ || V->type == GGML_TYPE_TURBO2_TCQ) {
        if (Q->ne[0] <= 512 && Q->ne[0] % 64 == 0)
            return BEST_FATTN_KERNEL_VEC;
        return BEST_FATTN_KERNEL_NONE;
    }

    // D=512: MMA/TILE templates don't support this head_dim, use VEC unconditionally
    if (Q->ne[0] == 512) {
        return BEST_FATTN_KERNEL_VEC;
    }

    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        int gqa_ratio_eff = 1;
        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
            gqa_ratio_eff *= 2;
        }
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use the WMMA kernel if possible:
    if (ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        if (can_use_vector_kernel && Q->ne[1] <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        return BEST_FATTN_KERNEL_WMMA_F16;
    }

    if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && gqa_opt_applies && Q->ne[0] <= 128 && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (Q->ne[1] == 1) {
                    if (!gqa_opt_applies) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            } else {
                if (Q->ne[1] <= 2) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        }
        int gqa_ratio_eff = 1;
        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
            gqa_ratio_eff *= 2;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 8) {
            return BEST_FATTN_KERNEL_TILE; // AMD WMMA is only faster if the full tile width of 16 can be utilized.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use MFMA flash attention for CDNA (MI100+):
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 256 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        const int64_t eff_nq = Q->ne[1] * (gqa_opt_applies ? gqa_ratio : 1);
        // MMA vs tile crossover benchmarked on MI300X @ d32768:
        //   hsk=64  (gqa=4): MMA wins at eff >= 128 (+11%)
        //   hsk=128 (gqa=4): MMA wins at eff >= 128 (+4%)
        if (eff_nq >= (GGML_CUDA_CC_IS_CDNA1(cc) && Q->ne[0] == 64 ? 64 : 128)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        // Fall through to tile kernel for small effective batch sizes.
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // Turbo prefill: dequant to fp16 and use tensor core MMA for batched attention.
    // turbo4 K uses inverse FWHT during dequant -- mixes centroids in float32 shmem before
    // fp16 cast, so precision is fine. turbo2/turbo3 use simple centroid*norm dequant.
    // Set TURBO_PREFILL_VEC=1 to force vec kernel for all turbo types (debug override).
    static const bool turbo_prefill_vec = [] {
        const char * e = getenv("TURBO_PREFILL_VEC");
        if (e) fprintf(stderr, "TURBO_PREFILL_VEC=%s: forcing vec prefill for turbo types\n", e);
        return e != nullptr;
    }();
    const bool turbo_kv = K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0 || K->type == GGML_TYPE_TURBO3_TCQ || K->type == GGML_TYPE_TURBO2_TCQ ||
                          V->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0 || V->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO2_TCQ;

    // Fused MMA turbo: reads raw turbo bytes directly in the MMA kernel, no intermediate fp16 buffers.
    // Phase 1: turbo4_0 matched K/V at D=128. Set GGML_TURBO_MMA_FUSED=0 to disable.
    static const bool turbo_mma_fused = [] {
        const char * e = getenv("GGML_TURBO_MMA_FUSED");
        if (e && atoi(e) == 0) {
            fprintf(stderr, "GGML_TURBO_MMA_FUSED=0: fused turbo MMA kernel disabled\n");
            return false;
        }
        return true;
    }();
    const bool turbo4_matched = K->type == GGML_TYPE_TURBO4_0 && V->type == GGML_TYPE_TURBO4_0;
    if (turbo_mma_fused && turbo4_matched && (Q->ne[0] == 128 || Q->ne[0] == 256) &&
        turing_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        cudaStream_t stream = ctx.stream();
        int device;
        CUDA_CHECK(cudaGetDevice(&device));

        bool fused_workspace_ok = true;
        if (Q->ne[0] % 128 == 0) {
            const size_t q_size = ggml_nelements(Q) * sizeof(float);
            fused_workspace_ok = ggml_cuda_try_grow_buffer(
                &q_rot_buf[device],
                &q_rot_buf_size[device],
                q_size,
                device,
                "fused Q rotation");
        }

        if (fused_workspace_ok) {
            ggml_tensor Q_rot_fused;
            ggml_tensor * orig_q_fused = nullptr;
            if (Q->ne[0] % 128 == 0) {
                const int64_t n_q_groups = ggml_nelements(Q) / 128;
                k_turbo_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
                    (const float *)Q->data, q_rot_buf[device], ggml_nelements(Q));
                ggml_cuda_turbo_check_last_error("fused Q rotation");
                Q_rot_fused = *Q;
                Q_rot_fused.data = q_rot_buf[device];
                orig_q_fused = dst->src[0];
                dst->src[0] = &Q_rot_fused;
            }

            if (Q->ne[0] == 128) {
                ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0>(ctx, dst);
            }
            ggml_cuda_turbo_check_last_error("fused turbo4 MMA dispatch");

            if (orig_q_fused) dst->src[0] = orig_q_fused;
            return;
        }
        // fused workspace allocation failed -- fall through to dequant/native path
    }

    if (turbo_kv && !turbo_prefill_vec && Q->ne[1] > 1 && Q->ne[0] <= 256 && turing_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        // Prefill path: turbo4 K uses inverse FWHT dequant (original domain, no Q rotation),
        // turbo2/3 K uses simple dequant (rotated domain, Q pre-rotated). V un-rotation at graph level.
        const bool did_turbo_prefill = ggml_cuda_turbo_prefill_attend(ctx, dst);
        if (did_turbo_prefill) {
            return;
        }
        // Prefill workspace allocation failed -- fall through to dequant/native path
    }

    load_tcq_decode_alpha(ctx.device);

    // Update VEC __constant__ alpha for context-adaptive mode
    if (d_tcq_decode_alpha_v_static == 0.0f &&
        (V->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO2_TCQ)) {
        float alpha = tcq_compute_alpha_v(V->type, V->ne[1]);
        cudaMemcpyToSymbol(d_tcq_decode_alpha_v_fattn, &alpha, sizeof(float));
    }

    // Load runtime codebooks for TCQ types (needed by both dequant and native VEC paths)
    if (K->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO3_TCQ) {
        static bool tcq3_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
        if (!tcq3_cb_loaded[ctx.device]) {
            tcq3_cb_loaded[ctx.device] = true;
            const char *cb_path = getenv("TURBO_TCQ_CB");
            if (cb_path) {
                float cb[512];
                FILE *f = fopen(cb_path, "rb");
                if (f && fread(cb, sizeof(float), 512, f) == 512) {
                    fclose(f);
                    cudaMemcpyToSymbol(d_turbo3_tcq_codebook_fattn, cb, 512*sizeof(float));
                    fprintf(stderr, "TCQ decode: loaded 3-bit codebook from %s (device %d)\n", cb_path, ctx.device);
                } else {
                    if (f) fclose(f);
                    fprintf(stderr, "TCQ decode: FAILED to load 3-bit codebook from %s\n", cb_path);
                }
            }
        }
    }
    if (K->type == GGML_TYPE_TURBO2_TCQ || V->type == GGML_TYPE_TURBO2_TCQ) {
        static bool tcq2_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
        if (!tcq2_cb_loaded[ctx.device]) {
            tcq2_cb_loaded[ctx.device] = true;
            const char *cb_path = getenv("TURBO_TCQ_CB2");
            if (cb_path) {
                float cb[256];
                FILE *f = fopen(cb_path, "rb");
                if (f && fread(cb, sizeof(float), 256, f) == 256) {
                    fclose(f);
                    cudaMemcpyToSymbol(d_turbo2_tcq_codebook_fattn, cb, 256*sizeof(float));
                    fprintf(stderr, "TCQ decode: loaded 2-bit codebook from %s (device %d)\n", cb_path, ctx.device);
                } else {
                    if (f) fclose(f);
                    fprintf(stderr, "TCQ decode: FAILED to load 2-bit codebook from %s\n", cb_path);
                }
            }
        }
    }

    cudaStream_t stream = ctx.stream();

    // Debug: log shapes at entry to decode/common path
    ggml_cuda_turbo_log_shapes("decode", Q, K, V);

    // Dequant turbo K/V to fp16 for decode: MMA tensor cores on fp16 beat VEC scalar
    // on turbo bits for dense models. Bandwidth savings from turbo's 3/16 footprint are
    // negligible relative to FFN compute (~1% slower native on Qwen3.5-27B, 3090).
    // Set GGML_TURBO_DECODE_NATIVE=1 to force native VEC path (may help bandwidth-limited configs).
    static const bool turbo_decode_native = (getenv("GGML_TURBO_DECODE_NATIVE") != nullptr);
    // Dequant turbo K/V to fp16 for D<=256 (any K/V combo), or D=512 only when BOTH
    // K and V are turbo (Gemma 4 ISWA global layers with K=V -- Bug A2). Mixed q8_0+turbo
    // at D=512 stays on native VEC path because post-dequant q8_0 K + f16 V has no
    // working VEC template at D=512.
    const bool turbo_k_only = K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0 || K->type == GGML_TYPE_TURBO3_TCQ || K->type == GGML_TYPE_TURBO2_TCQ;
    const bool turbo_v_only = V->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0 || V->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO2_TCQ;
    // Mixed f16/q8_0 + turbo at D=512: dequant K (and turbo V) to f16 so FA dispatches as
    // F16/F16 D=512 (which exists). Without this, the native VEC templates needed are
    // either missing (F16<->turbo) or have buggy SASS on sm_120 PTX-JIT (Q8_0<->turbo4 etc).
    const bool k_is_f16_q8_or_turbo = (K->type == GGML_TYPE_F16) || (K->type == GGML_TYPE_Q8_0) || turbo_k_only;
    const bool v_is_f16_q8_or_turbo = (V->type == GGML_TYPE_F16) || (V->type == GGML_TYPE_Q8_0) || turbo_v_only;
    const bool both_dequantable_512 = k_is_f16_q8_or_turbo && v_is_f16_q8_or_turbo;
    const bool do_decode_dequant_candidate = !turbo_decode_native && turbo_kv && (Q->ne[0] <= 256 || (Q->ne[0] <= 512 && both_dequantable_512));

    // Bug #31 exception: when K=turbo2/turbo3 dequant fell back to the rotated kernel (see K
    // dispatch below), K is in WHT-rotated space, not original space, so Q must be pre-rotated.
    // Compute these before any dst->src[] mutations.
    const bool k_needs_dequant = turbo_k_only || (K->type == GGML_TYPE_Q8_0 && Q->ne[0] > 256);
    const bool v_needs_dequant = turbo_v_only || (V->type == GGML_TYPE_Q8_0 && Q->ne[0] > 256);
    const bool turbo_k_any = (K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0 || K->type == GGML_TYPE_TURBO3_TCQ || K->type == GGML_TYPE_TURBO2_TCQ);
    const bool k_t2_use_rotated = (K->type == GGML_TYPE_TURBO2_0) &&
        (V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0 ||
         V->type == GGML_TYPE_Q8_0    || V->type == GGML_TYPE_F16);
    const bool k_t3_use_rotated = (K->type == GGML_TYPE_TURBO3_0) &&
        (V->type == GGML_TYPE_TURBO2_0);
    const bool k_uses_rotated_path = do_decode_dequant_candidate && (
        (k_t2_use_rotated || k_t3_use_rotated));
    // Q rotation is needed when: (a) dequant path + K in rotated domain, or (b) native path + turbo K (rotated domain)
    const bool q_needs_rotation_for_dequant = do_decode_dequant_candidate && k_uses_rotated_path && Q->ne[0] % 128 == 0;

    int device_dec;
    CUDA_CHECK(cudaGetDevice(&device_dec));

    // === Phase 1: Reserve ALL required workspace before any kernel launches or
    //     dst->src[] mutations. This prevents half-entered states where K is
    //     dequanted but V is not, or Q rotation buffer is missing. ===

    bool can_dequant = do_decode_dequant_candidate;

    size_t k_max_bytes = 0;
    if (can_dequant && k_needs_dequant) {
        const ggml_tensor * k_root = K;
        while (k_root->view_src) k_root = k_root->view_src;
        k_max_bytes = (size_t)k_root->ne[0] * k_root->ne[1] * k_root->ne[2] * sizeof(half);
    }
    size_t v_max_bytes = 0;
    if (can_dequant && v_needs_dequant) {
        const ggml_tensor * v_root = V;
        while (v_root->view_src) v_root = v_root->view_src;
        v_max_bytes = (size_t)v_root->ne[0] * v_root->ne[1] * v_root->ne[2] * sizeof(half);
    }
    size_t q_rot_bytes = 0;
    if (can_dequant && q_needs_rotation_for_dequant) {
        q_rot_bytes = ggml_nelements(Q) * sizeof(float);
    }

    if (can_dequant && k_needs_dequant) {
        if (!ggml_cuda_try_grow_buffer(&kv_dequant_k_buf[device_dec], &kv_dequant_k_buf_size[device_dec],
                                       k_max_bytes, device_dec, "decode K dequant")) {
            can_dequant = false;
        }
    }
    if (can_dequant && v_needs_dequant) {
        if (!ggml_cuda_try_grow_buffer(&kv_dequant_v_buf[device_dec], &kv_dequant_v_buf_size[device_dec],
                                       v_max_bytes, device_dec, "decode V dequant")) {
            can_dequant = false;
        }
    }
    if (can_dequant && q_needs_rotation_for_dequant) {
        if (!ggml_cuda_try_grow_buffer(&q_rot_buf[device_dec], &q_rot_buf_size[device_dec],
                                       q_rot_bytes, device_dec, "decode Q rotation (dequant)")) {
            can_dequant = false;
        }
    }

    // === Phase 2: If dequant path is viable, launch kernels and mutate dst->src[] ===
    half * k_fp16_dec = nullptr;
    half * v_fp16_dec = nullptr;
    ggml_tensor K_f16_dec, V_f16_dec;
    ggml_tensor Q_rot_decode;

    const bool do_decode_dequant = can_dequant;

    if (do_decode_dequant) {
        if (k_needs_dequant) {
            k_fp16_dec = kv_dequant_k_buf[device_dec];
            dim3 grid_k(K->ne[1], K->ne[2], K->ne[3]);
            if (K->type == GGML_TYPE_TURBO2_0 && k_t2_use_rotated) {
                k_turbo2_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                    (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
            } else if (K->type == GGML_TYPE_TURBO2_0) {
                k_turbo2_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                    (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
            } else if (K->type == GGML_TYPE_TURBO3_0 && k_t3_use_rotated) {
                k_turbo3_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                    (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
            } else if (K->type == GGML_TYPE_TURBO3_0) {
                k_turbo3_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                    (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
            } else if (K->type == GGML_TYPE_TURBO4_0) {
                k_turbo4_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                    (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
            } else if (K->type == GGML_TYPE_TURBO3_TCQ) {
                k_turbo3_tcq_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                    (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
            } else if (K->type == GGML_TYPE_TURBO2_TCQ) {
                k_turbo2_tcq_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                    (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
            } else if (K->type == GGML_TYPE_Q8_0) {
                k_q8_0_dequant_f16_tkhe<<<grid_k, K->ne[0], 0, stream>>>(
                    (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
            }
            ggml_cuda_turbo_check_last_error("decode K dequant");
            K_f16_dec = *K;
            K_f16_dec.type = GGML_TYPE_F16;
            K_f16_dec.data = k_fp16_dec;
            K_f16_dec.nb[0] = sizeof(half);
            K_f16_dec.nb[1] = K->ne[0] * K->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
            K_f16_dec.nb[2] = K->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
            K_f16_dec.nb[3] = K->ne[0] * K->ne[1] * K->ne[2] * sizeof(half);
            dst->src[1] = &K_f16_dec;
        }
        if (v_needs_dequant) {
            v_fp16_dec = kv_dequant_v_buf[device_dec];
            dim3 grid_v(V->ne[1], V->ne[2], V->ne[3]);
            if (V->type == GGML_TYPE_TURBO2_0) {
                k_turbo2_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                    (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
            } else if (V->type == GGML_TYPE_TURBO3_TCQ) {
                k_turbo3_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                    (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]));
            } else if (V->type == GGML_TYPE_TURBO2_TCQ) {
                k_turbo2_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                    (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]));
            } else if (V->type == GGML_TYPE_TURBO4_0) {
                k_turbo4_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                    (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
            } else if (V->type == GGML_TYPE_TURBO3_0) {
                k_turbo3_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                    (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
            } else if (V->type == GGML_TYPE_Q8_0) {
                k_q8_0_dequant_f16_tkhe<<<grid_v, V->ne[0], 0, stream>>>(
                    (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
            }
            ggml_cuda_turbo_check_last_error("decode V dequant");
            V_f16_dec = *V;
            V_f16_dec.type = GGML_TYPE_F16;
            V_f16_dec.data = v_fp16_dec;
            V_f16_dec.nb[0] = sizeof(half);
            V_f16_dec.nb[1] = V->ne[0] * V->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
            V_f16_dec.nb[2] = V->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
            V_f16_dec.nb[3] = V->ne[0] * V->ne[1] * V->ne[2] * sizeof(half);
            dst->src[2] = &V_f16_dec;
            // Bug A1: nvcc 13 on sm_120a reorders these V_f16_dec.nb[*] stores past the FA
            // dispatcher -> stale strides -> <unused49> garbage. signal_fence is a pure
            // host-compiler barrier (zero machine instructions).
            std::atomic_signal_fence(std::memory_order_seq_cst);
        }

        // Pre-rotate Q for dequant path when K is in rotated domain
        if (q_needs_rotation_for_dequant) {
            const int64_t n_q_groups = ggml_nelements(Q) / 128;
            k_turbo_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
                (const float *)Q->data, q_rot_buf[device_dec], ggml_nelements(Q));
            ggml_cuda_turbo_check_last_error("decode Q rotation (dequant)");
            Q_rot_decode = *Q;
            Q_rot_decode.data = q_rot_buf[device_dec];
            dst->src[0] = &Q_rot_decode;
        }
    }

    // === RAII guard: restore dst->src[] on any exit path ===
    struct fattn_src_restore {
        ggml_tensor * dst;
        ggml_tensor * orig_q;
        ggml_tensor * orig_k;
        ggml_tensor * orig_v;
        ~fattn_src_restore() {
            dst->src[0] = orig_q;
            dst->src[1] = orig_k;
            dst->src[2] = orig_v;
        }
    } src_guard = { dst, dst->src[0], dst->src[1], dst->src[2] };

    // === Native path: Q rotation (only when NOT in dequant path) ===
    // When do_decode_dequant fires, all turbo K types are dequanted via inv-FWHT into
    // ORIGINAL domain -> Q stays unrotated. When decode dequant is skipped (D>256 or
    // GGML_TURBO_DECODE_NATIVE), turbo K is consumed by the native vec turbo dot product,
    // which expects a pre-rotated Q -- so rotate Q in that case.
    const bool turbo_k_in_orig_domain = do_decode_dequant && turbo_k_any && !k_uses_rotated_path;

    // === Fallback: when dequant workspace is unavailable ===
    // Must be checked BEFORE native Q rotation, because the fallback check needs
    // original K/V types in dst->src[] to get correct kernel selection.
    if (!can_dequant && turbo_kv) {
        // Verify original K/V types support a native FA kernel before dispatching.
        // dst->src[] still has original Q/K/V here (RAII guard ensures restoration).
        const best_fattn_kernel fallback_kernel = ggml_cuda_get_best_fattn_kernel(device_dec, dst);
        if (fallback_kernel == BEST_FATTN_KERNEL_VEC) {
            // Native VEC can handle original turbo types directly -- proceed to Q rotation + dispatch.
        } else {
            GGML_ABORT("turbo FA: dequant workspace unavailable and no native VEC fallback for K type=%s V type=%s head_dim=%d",
                        ggml_type_name(K->type), ggml_type_name(V->type), (int)Q->ne[0]);
        }
    }

    // Native Q rotation: happens after fallback check so dst->src[] still has original types
    // for the kernel query, and the switch dispatch sees the right tensor types.
    if (!do_decode_dequant && turbo_k_any && !turbo_k_in_orig_domain && Q->ne[0] % 128 == 0) {
        const size_t q_size = ggml_nelements(Q) * sizeof(float);
        if (!ggml_cuda_try_grow_buffer(&q_rot_buf[device_dec], &q_rot_buf_size[device_dec],
                                       q_size, device_dec, "decode Q rotation (native)")) {
            // Native fallback requires Q rotation -- if workspace is unavailable, it is a
            // fatal configuration error because the native turbo VEC path reads rotated K
            // and must have a pre-rotated Q.
            GGML_ABORT("turbo FA: native VEC fallback selected but Q rotation workspace allocation failed");
        }
        const int64_t n_q_groups = ggml_nelements(Q) / 128;
        k_turbo_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
            (const float *)Q->data, q_rot_buf[device_dec], ggml_nelements(Q));
        ggml_cuda_turbo_check_last_error("decode Q rotation (native)");
        Q_rot_decode = *Q;
        Q_rot_decode.data = q_rot_buf[device_dec];
        dst->src[0] = &Q_rot_decode;
    }

    // === Dispatch ===
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_WMMA_F16:
            ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
    ggml_cuda_turbo_check_last_error("FA dispatch complete");

    // src_guard destructor restores dst->src[] here.
    // K/V fp16 buffers are persistent (grow-only), no free needed

    // Output inverse rotation for turbo V types is handled at graph level
    // (ggml_turbo_wht op in llama-graph.cpp) to maintain CUDA graph compatibility.
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
