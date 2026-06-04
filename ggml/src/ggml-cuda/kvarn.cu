#include "kvarn.cuh"

static constexpr int KVAR_N_DIM = 128;
static constexpr int KVAR_N_TILE_VALUES = KVAR_N_DIM * KVAR_N_DIM;
static constexpr int KVAR_N_SHARED_FLOATS = KVAR_N_TILE_VALUES + 6 * KVAR_N_DIM + 2;
static constexpr int KVAR_N_SHARED_BYTES = KVAR_N_SHARED_FLOATS * sizeof(float);

static __device__ void kvarn_wht_128(float * values) {
    __syncthreads();
    for (int stride = 1; stride < KVAR_N_DIM; stride *= 2) {
        if (threadIdx.x < 64) {
            const int j = (threadIdx.x / stride) * (2 * stride) + (threadIdx.x % stride);
            const float a = values[j];
            const float b = values[j + stride];
            values[j] = a + b;
            values[j + stride] = a - b;
        }
        __syncthreads();
    }
    if (threadIdx.x < KVAR_N_DIM) {
        values[threadIdx.x] *= 0.08838834764831845f;
    }
    __syncthreads();
}

static __device__ float kvarn_std_col(
        const float * tile,
        const float * log_s_col,
        const float * log_s_row,
        int col) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const float sc = expf(log_s_col[col]);
    for (int row = 0; row < KVAR_N_DIM; ++row) {
        const float value = tile[row * KVAR_N_DIM + col] / (sc * expf(log_s_row[row]));
        sum += value;
        sum_sq += value * value;
    }
    const float mean = sum / KVAR_N_DIM;
    return sqrtf(fmaxf((sum_sq - KVAR_N_DIM * mean * mean) / (KVAR_N_DIM - 1), 0.0f));
}

static __device__ float kvarn_std_row(
        const float * tile,
        const float * log_s_col,
        const float * log_s_row,
        int row) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const float sr = expf(log_s_row[row]);
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float value = tile[row * KVAR_N_DIM + col] / (expf(log_s_col[col]) * sr);
        sum += value;
        sum_sq += value * value;
    }
    const float mean = sum / KVAR_N_DIM;
    return sqrtf(fmaxf((sum_sq - KVAR_N_DIM * mean * mean) / (KVAR_N_DIM - 1), 0.0f));
}

static __device__ void kvarn_update_best(
        const float * tile,
        const float * log_s_col,
        const float * log_s_row,
        float * best_col,
        float * best_row,
        float * col_std,
        float * row_std,
        float * best_imbalance,
        float * better) {
    const int i = threadIdx.x;
    col_std[i] = kvarn_std_col(tile, log_s_col, log_s_row, i);
    row_std[i] = kvarn_std_row(tile, log_s_col, log_s_row, i);
    __syncthreads();

    if (i == 0) {
        float col_min = col_std[0];
        float col_max = col_std[0];
        float row_min = row_std[0];
        float row_max = row_std[0];
        for (int j = 1; j < KVAR_N_DIM; ++j) {
            col_min = fminf(col_min, col_std[j]);
            col_max = fmaxf(col_max, col_std[j]);
            row_min = fminf(row_min, row_std[j]);
            row_max = fmaxf(row_max, row_std[j]);
        }
        const float imbalance =
            col_max / fmaxf(col_min, 1e-8f) +
            row_max / fmaxf(row_min, 1e-8f);
        *better = imbalance <= *best_imbalance ? 1.0f : 0.0f;
        if (*better != 0.0f) {
            *best_imbalance = imbalance;
        }
    }
    __syncthreads();

    if (*better != 0.0f) {
        best_col[i] = expf(log_s_col[i]);
        best_row[i] = expf(log_s_row[i]);
    }
    __syncthreads();
}

static __device__ void kvarn_quantize_stage(
        const half * stage,
        uint8_t * record,
        int n_heads,
        int head,
        int stage_group,
        int bits,
        int iterations,
        bool value,
        float * shared) {
    float * tile = shared;
    float * log_s_col = tile + KVAR_N_TILE_VALUES;
    float * log_s_row = log_s_col + KVAR_N_DIM;
    float * best_col = log_s_row + KVAR_N_DIM;
    float * best_row = best_col + KVAR_N_DIM;
    float * col_std = best_row + KVAR_N_DIM;
    float * row_std = col_std + KVAR_N_DIM;
    float * best_imbalance = row_std + KVAR_N_DIM;
    float * better = best_imbalance + 1;

    for (int i = threadIdx.x; i < KVAR_N_TILE_VALUES; i += blockDim.x) {
        const int row = i / KVAR_N_DIM;
        const int col = i % KVAR_N_DIM;
        const int token = value ? row : col;
        const int dim = value ? col : row;
        const int stage_pos = 128 + ((stage_group - 1) & 1) * KVAR_N_DIM + token;
        tile[i] = __half2float(stage[(stage_pos * n_heads + head) * KVAR_N_DIM + dim]);
    }
    log_s_col[threadIdx.x] = 0.0f;
    log_s_row[threadIdx.x] = 0.0f;
    best_col[threadIdx.x] = 1.0f;
    best_row[threadIdx.x] = 1.0f;
    __syncthreads();

    col_std[threadIdx.x] = kvarn_std_col(tile, log_s_col, log_s_row, threadIdx.x);
    row_std[threadIdx.x] = kvarn_std_row(tile, log_s_col, log_s_row, threadIdx.x);
    __syncthreads();
    if (threadIdx.x == 0) {
        float col_min = col_std[0];
        float col_max = col_std[0];
        float row_min = row_std[0];
        float row_max = row_std[0];
        for (int i = 1; i < KVAR_N_DIM; ++i) {
            col_min = fminf(col_min, col_std[i]);
            col_max = fmaxf(col_max, col_std[i]);
            row_min = fminf(row_min, row_std[i]);
            row_max = fmaxf(row_max, row_std[i]);
        }
        *best_imbalance =
            col_max / fmaxf(col_min, 1e-8f) +
            row_max / fmaxf(row_min, 1e-8f);
    }
    __syncthreads();

    for (int iter = 0; iter < iterations; ++iter) {
        const float col = fminf(fmaxf(kvarn_std_col(tile, log_s_col, log_s_row, threadIdx.x), 1e-3f), 1e3f);
        log_s_col[threadIdx.x] = fminf(fmaxf(log_s_col[threadIdx.x] + logf(col), -0.3f), 10.0f);
        __syncthreads();

        const float row = fminf(fmaxf(kvarn_std_row(tile, log_s_col, log_s_row, threadIdx.x), 1e-3f), 1e3f);
        log_s_row[threadIdx.x] = fminf(fmaxf(log_s_row[threadIdx.x] + logf(row), -0.3f), 10.0f);
        __syncthreads();

        kvarn_update_best(tile, log_s_col, log_s_row, best_col, best_row, col_std, row_std, best_imbalance, better);
    }

    const int row = threadIdx.x;
    float lo = 3.402823466e+38F;
    float hi = -3.402823466e+38F;
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float x = tile[row * KVAR_N_DIM + col] / (best_col[col] * best_row[row]);
        lo = fminf(lo, x);
        hi = fmaxf(hi, x);
    }

    const int qmax = (1 << bits) - 1;
    const float scale = fmaxf((hi - lo) / qmax, 1e-10f);
    const int row_bytes = KVAR_N_DIM * bits / 8;
    uint8_t * row_payload = record + row * row_bytes;
    for (int i = 0; i < row_bytes; ++i) {
        row_payload[i] = 0;
    }
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float x = tile[row * KVAR_N_DIM + col] / (best_col[col] * best_row[row]);
        const uint8_t q = (uint8_t) fminf(fmaxf(roundf((x - lo) / scale), 0.0f), (float) qmax);
        const int bit_offset = col * bits;
        for (int bit = 0; bit < bits; ++bit) {
            const int dst_bit = bit_offset + bit;
            row_payload[dst_bit / 8] |= ((q >> bit) & 1u) << (dst_bit % 8);
        }
    }

    const int payload_bytes = KVAR_N_TILE_VALUES * bits / 8;
    half * scale_axis = (half *) (record + payload_bytes);
    half * zp_axis = scale_axis + KVAR_N_DIM;
    half * other_axis = zp_axis + KVAR_N_DIM;
    scale_axis[row] = __float2half_rn(best_row[row] * scale);
    zp_axis[row] = __float2half_rn(best_row[row] * lo);
    other_axis[row] = __float2half_rn(best_col[row]);
    __syncthreads();
}

static __global__ void kvarn_store_kernel(
        const float * current,
        const int64_t * indices,
        half * stage,
        uint8_t * records,
        int n_heads,
        int n_tokens,
        int n_groups,
        int record_bytes,
        int bits,
        int iterations,
        bool value) {
    extern __shared__ float shared[];
    const int head = blockIdx.x;

    for (int token = 0; token < n_tokens; ++token) {
        const int64_t idx = indices[token];
        const int group = (int) (idx / KVAR_N_DIM);
        const int pos = (int) (idx % KVAR_N_DIM);
        if (group >= n_groups) {
            return;
        }

        if (group > 2 && pos == 0) {
            const int flush_group = group - 2;
            uint8_t * record = records + (flush_group * n_heads + head) * record_bytes;
            kvarn_quantize_stage(stage, record, n_heads, head, flush_group, bits, iterations, value, shared);
        }

        shared[threadIdx.x] = current[(token * n_heads + head) * KVAR_N_DIM + threadIdx.x];
        kvarn_wht_128(shared);
        const int stage_pos = group == 0 ? pos : KVAR_N_DIM + ((group - 1) & 1) * KVAR_N_DIM + pos;
        stage[(stage_pos * n_heads + head) * KVAR_N_DIM + threadIdx.x] =
            __float2half_rn(shared[threadIdx.x]);
        __syncthreads();
    }
}

static __device__ uint8_t kvarn_unpack_record(const uint8_t * record, int index, int bits) {
    uint8_t value = 0;
    const int bit_offset = index * bits;
    for (int bit = 0; bit < bits; ++bit) {
        const int src_bit = bit_offset + bit;
        value |= ((record[src_bit / 8] >> (src_bit % 8)) & 1u) << bit;
    }
    return value;
}

static __global__ void kvarn_materialize_kernel(
        const uint8_t * records,
        const half * stage,
        const int64_t * indices,
        half * dst,
        int n_heads,
        int n_kv,
        int n_indices,
        int n_groups,
        int record_bytes,
        int bits,
        bool value) {
    const int token = blockIdx.x / n_heads;
    const int head = blockIdx.x % n_heads;
    const int dim = threadIdx.x;
    __shared__ float rotated[KVAR_N_DIM];
    __shared__ int live_group;

    if (dim == 0) {
        live_group = 0;
        for (int i = 0; i < n_indices; ++i) {
            live_group = max(live_group, (int) (indices[i] / KVAR_N_DIM));
        }
    }
    __syncthreads();

    const int group = token / KVAR_N_DIM;
    const int pos = token % KVAR_N_DIM;
    float x = 0.0f;
    if (group == 0 || (group > 0 && group <= live_group && group + 1 >= live_group)) {
        const int stage_pos = group == 0 ? pos : KVAR_N_DIM + ((group - 1) & 1) * KVAR_N_DIM + pos;
        x = __half2float(stage[(stage_pos * n_heads + head) * KVAR_N_DIM + dim]);
    } else if (group < live_group && group < n_groups) {
        const uint8_t * record = records + (group * n_heads + head) * record_bytes;
        const int row = value ? pos : dim;
        const int col = value ? dim : pos;
        const int payload_bytes = KVAR_N_TILE_VALUES * bits / 8;
        const half * scale_axis = (const half *) (record + payload_bytes);
        const half * zp_axis = scale_axis + KVAR_N_DIM;
        const half * other_axis = zp_axis + KVAR_N_DIM;
        const uint8_t q = kvarn_unpack_record(record, row * KVAR_N_DIM + col, bits);
        x = (q * __half2float(scale_axis[row]) + __half2float(zp_axis[row])) * __half2float(other_axis[col]);
    }

    rotated[dim] = x;
    kvarn_wht_128(rotated);
    dst[(token * n_heads + head) * KVAR_N_DIM + dim] = __float2half_rn(rotated[dim]);
}

void ggml_cuda_op_kvarn_store(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * current = dst->src[0];
    const ggml_tensor * indices = dst->src[1];
    ggml_tensor * stage = dst->src[2];
    ggml_tensor * records = dst->src[3];
    GGML_ASSERT(ggml_is_contiguous(current));
    GGML_ASSERT(ggml_is_contiguous(indices));
    GGML_ASSERT(ggml_is_contiguous(stage));
    GGML_ASSERT(ggml_is_contiguous(records));

    const int bits = ggml_get_op_params_i32(dst, 0);
    const int iterations = ggml_get_op_params_i32(dst, 1);
    const bool value = ggml_get_op_params_i32(dst, 2) != 0;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    CUDA_CHECK(cudaFuncSetAttribute(kvarn_store_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, KVAR_N_SHARED_BYTES));
#endif
    kvarn_store_kernel<<<current->ne[1], KVAR_N_DIM, KVAR_N_SHARED_BYTES, ctx.stream()>>>(
        (const float *) current->data,
        (const int64_t *) indices->data,
        (half *) stage->data,
        (uint8_t *) records->data,
        (int) current->ne[1],
        (int) current->ne[2],
        (int) records->ne[2],
        (int) records->ne[0],
        bits,
        iterations,
        value);
}

void ggml_cuda_op_kvarn_materialize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * records = dst->src[0];
    const ggml_tensor * stage = dst->src[1];
    const ggml_tensor * indices = dst->src[2];
    GGML_ASSERT(ggml_is_contiguous(records));
    GGML_ASSERT(ggml_is_contiguous(stage));
    GGML_ASSERT(ggml_is_contiguous(indices));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int bits = ggml_get_op_params_i32(dst, 0);
    const bool value = ggml_get_op_params_i32(dst, 1) != 0;
    const int blocks = (int) (dst->ne[1] * dst->ne[2]);
    kvarn_materialize_kernel<<<blocks, KVAR_N_DIM, 0, ctx.stream()>>>(
        (const uint8_t *) records->data,
        (const half *) stage->data,
        (const int64_t *) indices->data,
        (half *) dst->data,
        (int) dst->ne[1],
        (int) dst->ne[2],
        (int) indices->ne[0],
        (int) records->ne[2],
        (int) records->ne[0],
        bits,
        value);
}
